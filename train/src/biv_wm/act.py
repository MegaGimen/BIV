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
# Denser grid so the JSON can be plotted without re-reading every channel.
ANALYSIS_THRESHOLDS: tuple[float, ...] = (
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    4.5,
    6.0,
    8.0,
    12.0,
    16.0,
    32.0,
)
DEFAULT_TOP_P = 0.01
N_LAYERS = 40

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


def canonical_module_key(name: str) -> str:
    """Strip CausalLM vs ImageTextToText prefixes so AW/Instruct keys match.

    AgentWorld CausalLM: ``model.layers.0.linear_attn.in_proj_qkv``
    Instruct VLM: ``model.language_model.layers.0.linear_attn.in_proj_qkv``
    Both become ``layers.0.linear_attn.in_proj_qkv``.
    """
    n = name
    for p in (
        "model.language_model.",
        "language_model.",
        "model.model.",
        "model.",
    ):
        if n.startswith(p):
            n = n[len(p) :]
            break
    if n == "lm_head" or n.endswith(".lm_head"):
        return "lm_head"
    if n in {"embed_tokens", "embed"} or n.endswith(".embed_tokens"):
        return "embed_tokens"
    if n == "norm" or (n.endswith(".norm") and "layers" not in n):
        return "norm"
    return n


def hook_kind(name: str, *, include_experts: bool = True) -> str | None:
    """Which ACT bucket a ``named_modules`` path belongs to, or None to skip.

    ACT §3.1 / footnote 1: outputs of trainable modules — attention Q/K/V/O,
    MLP gate/up/down, layer norms, token embedding, ``lm_head``. A channel is
    one output dimension of that module, not a residual-stream coordinate.

    Qwen3.5-35B-A3B is MoE: routed ``experts.*`` projections can be included
    (when include_experts=True) or skipped. The shared expert's gate/up/down
    is also FFN. The mixed ``mlp`` block output is not a projection and is not hooked.
    """
    n = name
    if any(s in n for s in SKIP_SUBSTR):
        return None
    if not include_experts and ".experts." in n:
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
    """Summarize a pool of Δa_i (mean / percentiles / tail mass)."""
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


def ranked_channels(named: dict[str, Sequence[float]]) -> list[dict[str, Any]]:
    """Every (module, channel) with Δa_i, sorted descending. Rank is 1-based."""
    rows: list[dict[str, Any]] = []
    for key, vec in named.items():
        kind = hook_kind(key)
        li = layer_index(key)
        for i, v in enumerate(vec):
            rows.append(
                {
                    "key": key,
                    "kind": kind,
                    "layer": li,
                    "channel": int(i),
                    "delta": float(v),
                }
            )
    rows.sort(key=lambda r: r["delta"], reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def top_p_mask(ranked: Sequence[dict[str, Any]], p: float = DEFAULT_TOP_P) -> list[dict[str, Any]]:
    """ACT §4.1: channels whose Δa_i ranks in the top ``p`` fraction."""
    if not ranked:
        return []
    p = min(1.0, max(0.0, float(p)))
    n = max(1, int(round(len(ranked) * p)))
    return list(ranked[: min(n, len(ranked))])


def _pool(
    named: dict[str, Sequence[float]],
    *,
    kinds: set[str] | None = None,
    skip_kinds: set[str] | None = None,
    layer: int | None = None,
) -> list[float]:
    out: list[float] = []
    for key, vec in named.items():
        kind = hook_kind(key)
        if kinds is not None and kind not in kinds:
            continue
        if skip_kinds is not None and kind in skip_kinds:
            continue
        if layer is not None and layer_index(key) != layer:
            continue
        out.extend(float(x) for x in vec)
    return out


def _stats_ccdf(vals: Sequence[float]) -> dict[str, Any]:
    st = channel_stats(vals)
    st["ccdf"] = ccdf(vals, ANALYSIS_THRESHOLDS)
    return st


def analyze_channels(
    named: dict[str, Sequence[float]],
    *,
    p: float = DEFAULT_TOP_P,
    layer_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    """ACT analysis payload: global/kind/layer CCDF, per-module stats, top-p mask."""
    ranked = ranked_channels(named)
    ranked_no_head = [r for r in ranked if r.get("kind") != "lm_head"]
    mask = top_p_mask(ranked, p)
    mask_no_head = top_p_mask(ranked_no_head, p)
    mask_keys = {(r["key"], r["channel"]) for r in mask}

    kinds = ("attn", "ffn", "ln", "embed", "lm_head")
    by_kind = {k: _stats_ccdf(_pool(named, kinds={k})) for k in kinds}

    by_layer: list[dict[str, Any]] = []
    types = list(layer_types or [])
    n_layers = max(N_LAYERS, max((r["layer"] for r in ranked if r["layer"] is not None), default=-1) + 1)
    for i in range(n_layers):
        vals = _pool(named, layer=i)
        st = _stats_ccdf(vals)
        n_in_mask = sum(1 for r in mask if r.get("layer") == i)
        by_layer.append(
            {
                "layer": i,
                "kind": types[i] if i < len(types) else "?",
                "n_in_mask": n_in_mask,
                **st,
            }
        )

    modules: list[dict[str, Any]] = []
    for key, vec in named.items():
        st = channel_stats(vec)
        modules.append(
            {
                "key": key,
                "kind": hook_kind(key),
                "layer": layer_index(key),
                "n_in_mask": sum(1 for i in range(len(vec)) if (key, i) in mask_keys),
                **st,
            }
        )
    modules.sort(key=lambda r: float(r["p99"]), reverse=True)

    threshold = float(mask[-1]["delta"]) if mask else 0.0
    return {
        "p": p,
        "n_channels": len(ranked),
        "n_channels_no_lm_head": len(ranked_no_head),
        "n_mask": len(mask),
        "n_mask_no_lm_head": len(mask_no_head),
        "mask_threshold": threshold,
        "global": _stats_ccdf([r["delta"] for r in ranked]),
        "global_no_lm_head": _stats_ccdf([r["delta"] for r in ranked_no_head]),
        "by_kind": by_kind,
        "by_layer": by_layer,
        "modules": modules,
        "mask": mask,
        "mask_no_lm_head": mask_no_head,
        "ranked": ranked,
    }


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


PARAM_SUFFIXES = (".weight", ".bias")


def param_canonical_key(param_name: str) -> str:
    """Map a safetensors key onto the ACT module key in ``mask.json``.

    ``model.language_model.layers.36.linear_attn.in_proj_qkv.weight``
    → ``layers.36.linear_attn.in_proj_qkv``.
    """
    n = param_name
    for suf in PARAM_SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    return canonical_module_key(n)


def channel_axis(param_name: str, ndim: int) -> int:
    """Which tensor axis is the ACT output channel (one row of the module).

    Linear / ``lm_head``: PyTorch ``weight`` is ``[out, in]``, channel = row.
    RMSNorm / bias: 1-D, channel = index.
    ``embed_tokens``: activation last dim is ``hidden_size``, so column of
    ``[vocab, hidden]`` — the probe hooked the embedding output, not a token row.
    """
    leaf = param_name.rsplit(".", 1)[-1]
    if leaf == "bias" or ndim <= 1:
        return 0
    if param_canonical_key(param_name) == "embed_tokens":
        return ndim - 1
    return 0


def world_param_candidates(instruct_key: str) -> list[str]:
    """Instruct ImageTextToText keys vs AgentWorld CausalLM keys."""
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            out.append(name)

    add(instruct_key)
    prefix = "model.language_model."
    if instruct_key.startswith(prefix):
        add("model." + instruct_key[len(prefix) :])
    elif instruct_key.startswith("model.") and not instruct_key.startswith(prefix):
        add("model.language_model." + instruct_key[len("model.") :])
    return out


def group_mask_channels(rows: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        key = str(row["key"])
        grouped.setdefault(key, []).append(int(row["channel"]))
    return {k: sorted(set(v)) for k, v in grouped.items()}


def load_mask_rows(payload: dict[str, Any], *, no_lm_head: bool) -> list[dict[str, Any]]:
    """Pick ``mask`` or ``mask_no_lm_head`` from compare_act's JSON."""
    if no_lm_head:
        rows = payload.get("mask_no_lm_head")
        if not isinstance(rows, list):
            rows = [
                r
                for r in (payload.get("mask") or [])
                if isinstance(r, dict)
                and r.get("kind") != "lm_head"
                and r.get("key") != "lm_head"
            ]
        return [r for r in rows if isinstance(r, dict)]
    rows = payload.get("mask")
    if not isinstance(rows, list):
        raise ValueError("mask.json has no 'mask' list")
    return [r for r in rows if isinstance(r, dict)]


def blend_masked_channels(
    target: Any,
    source: Any,
    channels: Sequence[int],
    lam: float,
    axis: int = 0,
) -> Any:
    """ACT eq. (masked task vector): ``θ_i ← θ_i^{trg} + λ (θ_i^{abl} − θ_i^{trg})``.

    ``target`` is Instruct (kept on unmasked channels). ``source`` is AgentWorld.
    Only indices in ``channels`` along ``axis`` change. Out-of-range indices skip.
    """
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"blend_masked_channels: shape {tuple(target.shape)} vs {tuple(source.shape)}"
        )
    if not channels:
        return target
    if axis < 0 or axis >= int(target.ndim):
        raise ValueError(f"blend_masked_channels: axis {axis} for ndim {target.ndim}")
    import torch

    n = int(target.shape[axis])
    idx = torch.tensor(list(channels), dtype=torch.long, device=target.device)
    idx = idx[(idx >= 0) & (idx < n)]
    if idx.numel() == 0:
        return target
    sl: list[Any] = [slice(None)] * int(target.ndim)
    sl[axis] = idx
    loc = tuple(sl)
    out = target.clone()
    t = target.to(dtype=torch.float32)
    s = source.to(dtype=torch.float32)
    out[loc] = (t[loc] + float(lam) * (s[loc] - t[loc])).to(dtype=out.dtype)
    return out
