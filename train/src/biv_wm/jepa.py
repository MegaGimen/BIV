"""JEPA predictor: (c_t, u) → ẑ in the 2048-d residual space. No tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPred(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )
        self.dim = dim

    def forward(self, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([c, u], dim=-1))


class InverseDyn(nn.Module):
    """(c_t, z*) → û. Same MLP box as JEPAPred; z* is stop-grad at the call site."""

    def __init__(self, dim: int = 2048, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([c, z], dim=-1))


def cosine_align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float().detach(), dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).mean()


def simcse_pair_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    o_texts: list[str],
    temperature: float = 0.05,
) -> torch.Tensor | None:
    """Unsupervised SimCSE on the observation encoder itself.

    ``z1``/``z2`` are two *live* (not stop-grad) encodings of the same
    ``o_texts[i]``, produced by two separate forward passes through the same
    model in train() mode — dropout noise is the only difference, no data
    augmentation needed. Positive pair for row i is (z1[i], z2[i]); negatives
    are every other row's z2 (resp. z1) whose text differs from o_texts[i]
    (character-identical duplicates are excluded the same way collapse_stats
    excludes them, since they are not usable negatives).

    This is the one loss in this file where the *target* side is not
    detached: gradient from the denominator (negative) terms flows back into
    whichever forward pass produced z1/z2, i.e. into the backbone's own
    observation-encoding step. That is the only mechanism here that pushes
    the encoder's own outputs apart — align_loss/inv_loss/bank_nce_loss all
    treat z as a stop-grad target and cannot do this (see AGENTS.md "JEPA
    家族怎么防坍缩"). Returns ``None`` if there are fewer than 2 rows.
    """
    n = int(z1.size(0))
    if n < 2 or z2.size(0) != n or len(o_texts) != n:
        return None
    p1 = F.normalize(z1.float(), dim=-1)
    p2 = F.normalize(z2.float(), dim=-1)
    same = torch.tensor(
        [[o_texts[i] == o_texts[j] for j in range(n)] for i in range(n)],
        dtype=torch.bool,
        device=z1.device,
    )
    eye = torch.eye(n, dtype=torch.bool, device=z1.device)
    dup_mask = same & ~eye  # duplicate-but-not-self: excluded as unusable negatives
    target = torch.arange(n, device=z1.device)

    def _direction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = (a @ b.T) / temperature
        logits = logits.masked_fill(dup_mask, float("-inf"))
        return F.cross_entropy(logits, target)

    return 0.5 * (_direction(p1, p2) + _direction(p2, p1))


def bank_nce_loss(
    pred: torch.Tensor,
    pos_z: torch.Tensor,
    o_texts: list[str],
    bank_z: torch.Tensor,
    bank_o_texts: list[str],
    temperature: float = 0.05,
) -> torch.Tensor | None:
    """JEPA's own guess vs a small, fresh, rolling bank of real observations.

    ``pred`` is live (JEPAPred's output for this micro-batch); ``pos_z`` is
    this micro-batch's own target, already stop-grad at the call site (same
    as align_loss's target). ``bank_z`` is a rolling buffer of *detached*
    recent z* vectors (see ``train_jepa.py``'s bank deque) — cheap, no extra
    forward pass, and refreshed every step so it tracks the current encoder
    instead of going stale like the earlier 256-deep cross-epoch queue did.

    Softmax cross-entropy with the true pair as the positive (index 0), not
    a hand-picked "push cosine to 0" target — that absolute target is what
    fought against LLM embeddings' naturally high (~0.85+) baseline
    similarity and blew up loss_align last time (see AGENTS.md). Bank
    entries whose text matches the current row are masked out as unusable
    negatives, same rule as collapse_stats. Returns ``None`` if the bank is
    still empty or every bank entry happens to be a text duplicate.
    """
    n = int(pred.size(0))
    k = int(bank_z.size(0)) if bank_z.numel() else 0
    if k == 0 or len(o_texts) != n or len(bank_o_texts) != k:
        return None
    p = F.normalize(pred.float(), dim=-1)
    pos = F.normalize(pos_z.float().detach(), dim=-1)
    neg = F.normalize(bank_z.float().detach(), dim=-1)
    pos_logits = (p * pos).sum(dim=-1, keepdim=True) / temperature
    neg_logits = (p @ neg.T) / temperature
    same = torch.tensor(
        [[o_texts[i] == bank_o_texts[j] for j in range(k)] for i in range(n)],
        dtype=torch.bool,
        device=pred.device,
    )
    neg_logits = neg_logits.masked_fill(same, float("-inf"))
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    target = torch.zeros(n, dtype=torch.long, device=pred.device)
    return F.cross_entropy(logits, target)


def _quantile_summary(x: torch.Tensor) -> dict[str, float | int | None]:
    if x.numel() == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    v = x.detach().float().flatten().sort().values
    n = int(v.numel())

    def q(p: float) -> float:
        idx = min(max(int(round((n - 1) * p)), 0), n - 1)
        return float(v[idx])

    return {"n": n, "mean": float(v.mean()), "median": q(0.5), "p90": q(0.9)}


def collapse_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    o_texts: list[str],
) -> dict[str, object]:
    """Collapse check on already-trained rows. No extra forward.

    Paired = pred_i vs own z*_i. Mismatch = pred_i vs z*_j after dropping
    pairs whose observation strings are character-for-character identical
    (those are not usable as negatives). Also report z*-vs-z* and pred-vs-pred
    so a high paired score can be told apart from "everything lives in one cone".
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    n = int(pred.size(0))
    if n != len(o_texts) or n != int(target.size(0)):
        raise ValueError(f"length mismatch pred={pred.size(0)} target={target.size(0)} texts={len(o_texts)}")
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    paired = (p * t).sum(dim=-1)
    sim_pt = p @ t.T
    sim_tt = t @ t.T
    sim_pp = p @ p.T
    mismatch = []
    z_off = []
    pred_off = []
    skipped_same_o = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if o_texts[i] == o_texts[j]:
                skipped_same_o += 1
                continue
            mismatch.append(sim_pt[i, j])
            z_off.append(sim_tt[i, j])
            pred_off.append(sim_pp[i, j])
    mis_t = torch.stack(mismatch) if mismatch else pred.new_empty((0,))
    z_t = torch.stack(z_off) if z_off else pred.new_empty((0,))
    pr_t = torch.stack(pred_off) if pred_off else pred.new_empty((0,))
    out_paired = _quantile_summary(paired)
    out_mis = _quantile_summary(mis_t)
    pmed = out_paired["median"]
    mmed = out_mis["median"]
    if out_mis["n"] == 0:
        verdict = "no_mismatch_pairs"
    elif pmed is None or mmed is None:
        verdict = "unclear"
    elif float(pmed) - float(mmed) >= 0.2:
        verdict = "paired_ahead"
    elif float(pmed) > 0.7 and float(pmed) - float(mmed) < 0.05:
        verdict = "collapse_like"
    else:
        verdict = "unclear"
    return {
        "n": n,
        "skipped_same_o": skipped_same_o,
        "paired": out_paired,
        "mismatch": out_mis,
        "z_self": _quantile_summary(z_t),
        "pred_self": _quantile_summary(pr_t),
        "verdict": verdict,
    }


def format_collapse_line(stats: dict[str, object]) -> str:
    def _cell(block: object, key: str) -> str:
        if not isinstance(block, dict):
            return "na"
        v = block.get(key)
        return "na" if v is None else f"{float(v):.3f}"

    paired = stats.get("paired")
    mis = stats.get("mismatch")
    return (
        f"n={stats.get('n')} skipped_same_o={stats.get('skipped_same_o')} "
        f"paired_med={_cell(paired, 'median')} mismatch_med={_cell(mis, 'median')} "
        f"mismatch_p90={_cell(mis, 'p90')} z_self_med={_cell(stats.get('z_self'), 'median')} "
        f"pred_self_med={_cell(stats.get('pred_self'), 'median')} verdict={stats.get('verdict')}"
    )
