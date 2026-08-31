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


def ranking_nce(
    pred: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    pos_texts: list[str],
    neg_texts: list[str],
    temperature: float = 0.1,
) -> torch.Tensor:
    """Softmax over (own z*, queued z*). Drop queued rows whose observation text matches."""
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        pos = pos.unsqueeze(0)
    if neg.numel() == 0:
        return pred.new_zeros(())
    if neg.ndim == 1:
        neg = neg.unsqueeze(0)
    p = F.normalize(pred.float(), dim=-1)
    pos_n = F.normalize(pos.float().detach(), dim=-1)
    neg_n = F.normalize(neg.float().detach(), dim=-1)
    temp = max(float(temperature), 1e-6)
    losses = []
    for i in range(p.size(0)):
        keep = [j for j, t in enumerate(neg_texts) if t != pos_texts[i]]
        if not keep:
            continue
        pos_sc = (p[i] * pos_n[i]).sum() / temp
        neg_sc = neg_n[keep] @ p[i] / temp
        logits = torch.cat([pos_sc.unsqueeze(0), neg_sc], dim=0)
        target = torch.zeros(1, dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


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
