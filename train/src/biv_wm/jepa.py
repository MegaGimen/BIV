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
    """Abandoned concat IDM. Do not wire into train_jepa.py; use LDAD."""

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


class LDAD(nn.Module):
    """Delta-JEPA latent-difference decoder: Δz → 2048 cond, not [z_t, z_{t+1}]."""

    def __init__(self, dim: int = 2048, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )
        self.dim = dim

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


def cosine_align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stop-grad target. Not Stage 1 L_pred — that must keep z_{t+1} live."""
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float().detach(), dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).mean()


def pred_align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Delta-JEPA eq (2) on the unit sphere. Both sides keep grad."""
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return (p - t).pow(2).sum(dim=-1).mean()


def bank_nce_loss(
    pred: torch.Tensor,
    pos_z: torch.Tensor,
    o_texts: list[str],
    bank_z: torch.Tensor,
    bank_o_texts: list[str],
    temperature: float = 0.05,
) -> torch.Tensor | None:
    """``pred`` vs a small, fresh, rolling bank of real observations.

    ``pred`` is the *live* (gradient-carrying) query; ``pos_z`` is this
    micro-batch's own positive target, stop-grad'd internally regardless of
    what the caller passes in. ``bank_z`` is a rolling buffer of *detached*
    recent z* vectors (see ``train_jepa.py``'s bank deque) — cheap, no extra
    forward pass to build, and refreshed every step so it tracks the current
    encoder instead of going stale like the earlier 256-deep cross-epoch
    queue did.

    Two call sites in ``train_jepa.py``, same function, different anchor:
    - ``loss_bank``: ``pred`` = JEPAPred's output. Trains the predictor to
      discriminate against the bank; never touches the observation encoder
      (``pos_z``/``bank_z`` are both stop-grad).
    - ``loss_simcse``: ``pred`` = a *second, live* encoding of the current
      row's own ``o_ids`` (different dropout draw than the ``pos_z`` view,
      same underlying text — the SimCSE trick, no data augmentation needed).
      This is the one call whose gradient reaches the encoder itself, since
      the anchor here *is* the encoder's own output, not JEPA's guess.

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
    skip_texts: list[str] | None = None,
) -> dict[str, object]:
    """Collapse check on already-trained rows. No extra forward.

    Paired = pred_i vs own z*_i. Mismatch = pred_i vs z*_j after dropping
    pairs whose skip keys are identical (default key = observation string;
    those are not usable as negatives when the target is Enc(o)). Pass a
    unique-per-row ``skip_texts`` for history-mediated Enc(h,a,o) so two
    rows that share stdout still count as negatives. Also report z*-vs-z*
    and pred-vs-pred so a high paired score can be told apart from
    "everything lives in one cone".
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    n = int(pred.size(0))
    if n != len(o_texts) or n != int(target.size(0)):
        raise ValueError(f"length mismatch pred={pred.size(0)} target={target.size(0)} texts={len(o_texts)}")
    keys = o_texts if skip_texts is None else skip_texts
    if len(keys) != n:
        raise ValueError(f"skip_texts={len(keys)} n={n}")
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
            if keys[i] == keys[j]:
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


def cross_source_topk(
    vecs: torch.Tensor,
    sources: list[str],
    *,
    k: int = 20,
) -> dict[str, object]:
    """Highest cosine pairs whose mix sources differ. Same-source pairs dropped.

    ``vecs`` is one side of the Stage 1 graph: left ``[z_t; u]`` (4096) or
    right ``Enc(h,a,o)`` (2048). Returns a quantile summary over every
    cross-source pair plus the top-k ``(i, j, cosine)`` with ``i < j``.
    """
    if vecs.ndim == 1:
        vecs = vecs.unsqueeze(0)
    n = int(vecs.size(0))
    if n != len(sources):
        raise ValueError(f"length mismatch vecs={n} sources={len(sources)}")
    if n < 2 or k < 1:
        return {
            "n": n,
            "n_cross": 0,
            "top": [],
            "cosine": _quantile_summary(vecs.new_empty((0,))),
        }
    x = F.normalize(vecs.float(), dim=-1)
    sim = x @ x.T
    src = list(sources)
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=vecs.device), diagonal=1)
    cross = torch.zeros(n, n, dtype=torch.bool, device=vecs.device)
    for i in range(n):
        for j in range(i + 1, n):
            if src[i] != src[j]:
                cross[i, j] = True
    mask = triu & cross
    n_cross = int(mask.sum().item())
    if n_cross == 0:
        return {
            "n": n,
            "n_cross": 0,
            "top": [],
            "cosine": _quantile_summary(vecs.new_empty((0,))),
        }
    vals = sim[mask]
    filled = sim.masked_fill(~mask, float("-inf"))
    take = min(int(k), n_cross)
    top_v, top_flat = torch.topk(filled.flatten(), take)
    width = n
    top: list[dict[str, object]] = []
    for score, flat in zip(top_v.tolist(), top_flat.tolist()):
        i, j = divmod(int(flat), width)
        if i > j:
            i, j = j, i
        top.append({"i": i, "j": j, "cosine": float(score)})
    return {
        "n": n,
        "n_cross": n_cross,
        "top": top,
        "cosine": _quantile_summary(vals),
    }


def attach_pair_snippets(
    top: list[dict[str, object]],
    *,
    sources: list[str],
    src_index: list[int],
    a_texts: list[str],
    o_texts: list[str],
    snippet: int = 120,
) -> list[dict[str, object]]:
    """Clip action/observation onto top-k index pairs. Never stores history."""
    out: list[dict[str, object]] = []
    for rec in top:
        i = int(rec["i"])
        j = int(rec["j"])
        out.append(
            {
                "cosine": float(rec["cosine"]),
                "i": i,
                "j": j,
                "src_i": sources[i],
                "src_j": sources[j],
                "src_row_i": int(src_index[i]),
                "src_row_j": int(src_index[j]),
                "a_i": _one_line(a_texts[i], snippet),
                "o_i": _one_line(o_texts[i], snippet),
                "a_j": _one_line(a_texts[j], snippet),
                "o_j": _one_line(o_texts[j], snippet),
            }
        )
    return out


def _clip_text(s: str, n: int) -> str:
    s = s if isinstance(s, str) else str(s)
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + f"...[+{len(s) - n} chars]"


def _one_line(s: str, n: int) -> str:
    """Collapse whitespace so a JSON observation cannot wrap the terminal."""
    return _clip_text(" ".join((s if isinstance(s, str) else str(s)).split()), n)


def close_pair_records(
    pred: torch.Tensor,
    target: torch.Tensor,
    o_texts: list[str],
    left_texts: list[str] | None = None,
    *,
    a_texts: list[str] | None = None,
    skip_texts: list[str] | None = None,
    state_lens: list[int] | None = None,
    next_lens: list[int] | None = None,
    threshold: float = 0.7,
    max_pairs: int = 24,
    snippet: int = 120,
    compact: bool = False,
) -> dict[str, object]:
    """Same pairing rules as collapse_stats; keep rows whose cosine >= threshold.

    ``z_self``: two target encodings. ``pred_self``: two pred encodings.
    ``mismatch``: pred_i vs target_j. Duplicate skip keys are dropped.
    ``compact=True`` stores only clipped action/observation (never history).
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    n = int(pred.size(0))
    if n != len(o_texts) or n != int(target.size(0)):
        raise ValueError(f"length mismatch pred={pred.size(0)} target={target.size(0)} texts={len(o_texts)}")
    if left_texts is None:
        left_texts = [""] * n
    elif len(left_texts) != n:
        raise ValueError(f"left_texts={len(left_texts)} n={n}")
    if a_texts is None:
        a_texts = left_texts
    elif len(a_texts) != n:
        raise ValueError(f"a_texts={len(a_texts)} n={n}")
    keys = o_texts if skip_texts is None else skip_texts
    if len(keys) != n:
        raise ValueError(f"skip_texts={len(keys)} n={n}")
    slen = list(state_lens) if state_lens is not None else [0] * n
    nlen = list(next_lens) if next_lens is not None else [0] * n
    if len(slen) != n or len(nlen) != n:
        raise ValueError("state_lens/next_lens length mismatch")
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    sim_pt = (p @ t.T).detach().cpu()
    sim_tt = (t @ t.T).detach().cpu()
    sim_pp = (p @ p.T).detach().cpu()
    z_self: list[dict[str, object]] = []
    pred_self: list[dict[str, object]] = []
    mismatch: list[dict[str, object]] = []

    def _pair_rec(i: int, j: int, cosine: float) -> dict[str, object]:
        rec: dict[str, object] = {
            "i": i,
            "j": j,
            "cosine": cosine,
            "a_i": _one_line(a_texts[i], snippet),
            "o_i": _one_line(o_texts[i], snippet),
            "a_j": _one_line(a_texts[j], snippet),
            "o_j": _one_line(o_texts[j], snippet),
            "state_len_i": int(slen[i]),
            "next_len_i": int(nlen[i]),
            "state_len_j": int(slen[j]),
            "next_len_j": int(nlen[j]),
        }
        if not compact:
            rec["left_i"] = _clip_text(left_texts[i], snippet)
            rec["left_j"] = _clip_text(left_texts[j], snippet)
        return rec

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if keys[i] == keys[j]:
                continue
            rec_z = _pair_rec(i, j, float(sim_tt[i, j]))
            rec_p = _pair_rec(i, j, float(sim_pp[i, j]))
            rec_m = _pair_rec(i, j, float(sim_pt[i, j]))
            if i < j and rec_z["cosine"] >= threshold:
                z_self.append(rec_z)
            if i < j and rec_p["cosine"] >= threshold:
                pred_self.append(rec_p)
            if rec_m["cosine"] >= threshold:
                mismatch.append(rec_m)

    def _top(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        rows.sort(key=lambda r: float(r["cosine"]), reverse=True)
        return rows[:max_pairs]

    return {
        "threshold": threshold,
        "n": n,
        "z_self": _top(z_self),
        "pred_self": _top(pred_self),
        "mismatch": _top(mismatch),
        "n_z_self": len(z_self),
        "n_pred_self": len(pred_self),
        "n_mismatch": len(mismatch),
    }


def format_close_preview(
    pairs: dict[str, object],
    *,
    n_print: int = 8,
    snippet: int = 80,
) -> str:
    """A few one-line close pairs for the terminal. Never prints history."""
    n_print = max(0, int(n_print))
    if n_print == 0:
        return ""
    blocks = (
        ("z_self", "z_next vs z_next"),
        ("pred_self", "z_t vs z_t"),
        ("mismatch", "z_t vs other z_next"),
    )
    lines: list[str] = []
    for key, title in blocks:
        rows = pairs.get(key) or []
        if not isinstance(rows, list):
            rows = []
        total = int(pairs.get(f"n_{key}", len(rows)))
        shown = min(n_print, len(rows))
        lines.append(f"{key} ({title}) showing {shown}/{total}")
        for rec in rows[:shown]:
            if not isinstance(rec, dict):
                continue
            cos = float(rec.get("cosine") or 0.0)
            a_i = _one_line(str(rec.get("a_i") or rec.get("left_i") or ""), snippet)
            o_i = _one_line(str(rec.get("o_i") or ""), snippet)
            a_j = _one_line(str(rec.get("a_j") or rec.get("left_j") or ""), snippet)
            o_j = _one_line(str(rec.get("o_j") or ""), snippet)
            lines.append(f"  {cos:.3f}  a={a_i} o={o_i}  ||  a={a_j} o={o_j}")
    return "\n".join(lines)


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
