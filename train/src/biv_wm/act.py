"""ACT (arXiv:2601.09398) channel-wise activation difference.

Same input to two related models. A channel is one output dimension of a
trainable module. Per channel:

    Δa_i = mean_{t in answer tokens} |a_{i,t}^{m1} − a_{i,t}^{m2}|

This module is the arithmetic only (no model load). The live probe is
``train/scripts/compare_act.py``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

# ACT Figure 1 uses a complementary CDF; 4.5 is the threshold they quote
# for "~99% of channels sit below this" on Qwen2.5 pairs.
CCDF_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.5, 8.0, 16.0)

SKIP_SUBSTR = ("visual", "vision", "mtp.", "rotary")

ATTN_LEAVES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
        "output_gate_proj",
        "in_proj",
    }
)
FFN_PROJ_LEAVES = frozenset({"gate_proj", "up_proj", "down_proj"})
LN_LEAVES = frozenset(
    {
        "input_layernorm",
        "post_attention_layernorm",
        "post_mlp_layernorm",
    }
)


def layer_index(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def hook_kind(name: str) -> str | None:
    """Which ACT bucket a ``named_modules`` path belongs to, or None to skip.

    ACT §3.1 / footnote 1: outputs of trainable modules — attention Q/K/V/O,
    MLP gate/up/down, layer norms, token embedding, ``lm_head``. A channel is
    one output dimension of that module, not a residual-stream coordinate.

    Qwen3.5-35B-A3B is MoE: routed ``experts.*`` projections are skipped
    because a token does not pass every expert (ACT's dense MLP does). The
    shared expert's gate/up/down is the dense-MLP analog. The mixed ``mlp``
    block output is not a projection and is not hooked.
    """
    n = name
    if any(s in n for s in SKIP_SUBSTR):
        return None
    if ".experts." in n:
        return None
    leaf = n.rsplit(".", 1)[-1]
    if leaf in {"embed_tokens", "embed"}:
        return "embed"
    if leaf == "lm_head":
        return "lm_head"
    if leaf in ATTN_LEAVES:
        return "attn"
    if leaf in FFN_PROJ_LEAVES:
        return "ffn"
    if leaf in LN_LEAVES:
        return "ln"
    if leaf == "norm" and "layers" not in n:
        return "ln"
    return None


def token_abs_sum(a: Any, b: Any) -> tuple[list[float], int]:
    """``(sum_t |a_t − b_t|, T)`` for activations ``[T, C]``.

    ACT eq. (2) averages over the pooled answer-token set, so callers should
    add these sums across samples and divide once.
    """
    if hasattr(a, "detach"):
        d = (a.detach().float() - b.detach().float()).abs()
        if d.ndim == 1:
            d = d.unsqueeze(0)
        return [float(v) for v in d.sum(dim=0).reshape(-1).tolist()], int(d.shape[0])
    if not a:
        return [], 0
    t = len(a)
    c = len(a[0])
    out = [0.0] * c
    for row_a, row_b in zip(a, b, strict=True):
        if len(row_a) != c or len(row_b) != c:
            raise ValueError("token_abs_sum: ragged or mismatched width")
        for i in range(c):
            out[i] += abs(float(row_a[i]) - float(row_b[i]))
    return out, t


def mean_from_sum(sum_abs: Sequence[float], n_tokens: int) -> list[float]:
    if n_tokens <= 0:
        return [0.0] * len(sum_abs)
    n = float(n_tokens)
    return [float(x) / n for x in sum_abs]


def channel_delta(a: Any, b: Any) -> list[float]:
    """Mean over the token axis of |a − b|. One-sample case of ACT eq. (2)."""
    s, n = token_abs_sum(a, b)
    return mean_from_sum(s, n)


def add_abs_sum(
    running: dict[str, tuple[list[float], int]],
    key: str,
    sum_abs: Sequence[float],
    n_tokens: int,
) -> None:
    if n_tokens <= 0:
        return
    if key not in running:
        running[key] = ([float(x) for x in sum_abs], int(n_tokens))
        return
    prev, n0 = running[key]
    if len(prev) != len(sum_abs):
        raise ValueError(f"add_abs_sum: width {len(prev)} vs {len(sum_abs)} for {key}")
    running[key] = ([x + float(y) for x, y in zip(prev, sum_abs, strict=True)], n0 + int(n_tokens))


def finalize_running(running: dict[str, tuple[list[float], int]]) -> dict[str, list[float]]:
    return {k: mean_from_sum(s, n) for k, (s, n) in running.items()}


def _sorted(values: Sequence[float]) -> list[float]:
    return sorted(float(v) for v in values)


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    q = min(1.0, max(0.0, p / 100.0))
    idx = min(n - 1, max(0, int(round(q * (n - 1)))))
    return float(sorted_vals[idx])


def channel_stats(delta: Sequence[float]) -> dict[str, Any]:
    """Summarize a pool of Δa_i. ``mean`` is the layer bar (mean over channels)."""
    if not delta:
        return {
            "n_channels": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "top1pct_mean": 0.0,
            "frac_gt": {str(t): 0.0 for t in CCDF_THRESHOLDS},
        }
    s = _sorted(delta)
    n = len(s)
    k = max(1, int(n * 0.01))
    top = s[-k:]
    return {
        "n_channels": n,
        "mean": sum(s) / n,
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p99": _percentile(s, 99),
        "max": s[-1],
        "top1pct_mean": sum(top) / len(top),
        "frac_gt": {str(t): sum(1 for x in s if x > t) / n for t in CCDF_THRESHOLDS},
    }


def ccdf(delta: Sequence[float], thresholds: Iterable[float] = CCDF_THRESHOLDS) -> dict[str, float]:
    vals = [float(v) for v in delta]
    n = len(vals)
    if n == 0:
        return {str(t): 0.0 for t in thresholds}
    return {str(t): sum(1 for x in vals if x > t) / n for t in thresholds}


def bar(value: float, peak: float, width: int = 24) -> str:
    if peak <= 0:
        return "." * width
    n = int(round(width * min(1.0, value / peak)))
    n = max(0, min(width, n))
    return "#" * n + "." * (width - n)


def top_channels(
    named: dict[str, Sequence[float]],
    n: int = 32,
) -> list[dict[str, Any]]:
    """Largest Δa_i across named channel vectors."""
    scored: list[tuple[float, str, int]] = []
    for key, vec in named.items():
        for i, v in enumerate(vec):
            scored.append((float(v), key, i))
    scored.sort(reverse=True)
    out = []
    for val, key, idx in scored[:n]:
        out.append({"key": key, "channel": idx, "delta": val, "layer": layer_index(key)})
    return out
