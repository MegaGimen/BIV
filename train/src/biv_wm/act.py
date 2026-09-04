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
        "q_norm",
        "k_norm",
    }
)


def layer_index(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def hook_kind(name: str) -> str | None:
    """Which ACT bucket a ``named_modules`` path belongs to, or None to skip.

    ACT records outputs of attention Q/K/V/O, MLP gate/up/down, layer
    norms, the token embedding, and ``lm_head``. Qwen3.5-35B-A3B is MoE
    (256 routed experts): those expert projections are skipped so the
    channel pool is not 256× a dense MLP. Shared-expert projections and
    the mixed ``mlp`` output stay.
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
    if leaf == "mlp":
        return "ffn"
    if leaf in LN_LEAVES:
        return "ln"
    if leaf == "norm" and "layers" not in n:
        return "ln"
    return None


def channel_delta(a: Any, b: Any) -> list[float]:
    """Mean over the token axis of |a − b|. ``a``/``b`` are [T, C]."""
    if hasattr(a, "detach"):
        x = (a.detach().float() - b.detach().float()).abs().mean(dim=0)
        return [float(v) for v in x.reshape(-1).tolist()]
    if not a:
        return []
    t = len(a)
    c = len(a[0])
    out = [0.0] * c
    for row_a, row_b in zip(a, b, strict=True):
        if len(row_a) != c or len(row_b) != c:
            raise ValueError("channel_delta: ragged or mismatched width")
        for i in range(c):
            out[i] += abs(float(row_a[i]) - float(row_b[i]))
    return [x / t for x in out]


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
