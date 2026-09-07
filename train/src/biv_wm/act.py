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
FFN_PROJ_LEAVES = frozenset(
    {
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "shared_expert_gate",
    }
)
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

    Qwen3.5-35B-A3B is MoE: routed experts are packed 3D Parameters
    (``mlp.experts.gate_up_proj`` / ``down_proj``), not ``experts.0.down_proj``
    Linear children. Capture keys use those Parameter names so merge can
    flatten ``[E, O, I]`` to ``E*O`` channels. Shared-expert gate/up/down and
    the router ``mlp.gate`` are ordinary FFN. The mixed ``mlp`` block output
    is not a projection and is not hooked.
    """
    n = name
    if any(s in n for s in SKIP_SUBSTR):
        return None
    if not include_experts and (".experts." in n or n.endswith(".experts")):
        return None
    if n.endswith(".mlp.gate") or n.endswith(".mlp.gate.weight"):
        return "ffn"
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


def is_packed_experts_module(mod: Any, name: str = "") -> bool:
    """True for Qwen3_5MoeExperts: 3D ``gate_up_proj`` + ``down_proj`` Parameters."""
    if name and any(s in name for s in SKIP_SUBSTR):
        return False
    gu = getattr(mod, "gate_up_proj", None)
    dn = getattr(mod, "down_proj", None)
    return (
        gu is not None
        and dn is not None
        and int(getattr(gu, "ndim", 0) or 0) == 3
        and int(getattr(dn, "ndim", 0) or 0) == 3
    )


def is_packed_expert_param(param_name: str) -> bool:
    n = param_canonical_key(param_name)
    return n.endswith(".experts.gate_up_proj") or n.endswith(".experts.down_proj")


def flatten_packed_expert_weight(tensor: Any) -> Any:
    """``[E, O, I]`` → ``[E*O, I]`` so ACT channel ``e*O + o`` is a matrix row."""
    if int(getattr(tensor, "ndim", 0) or 0) != 3:
        raise ValueError(
            f"flatten_packed_expert_weight: expected 3D, got shape {tuple(tensor.shape)}"
        )
    e, o, i = (int(x) for x in tensor.shape)
    return tensor.reshape(e * o, i)


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
    if hasattr(sum_abs, "detach"):
        n = int(sum_abs.numel()) if hasattr(sum_abs, "numel") else 0
        if n_tokens <= 0:
            return [0.0] * n
        return (sum_abs.detach().float().reshape(-1) / float(n_tokens)).cpu().tolist()
    if n_tokens <= 0:
        return [0.0] * len(sum_abs)
    n = float(n_tokens)
    return [float(x) / n for x in sum_abs]


def channel_delta(a: Any, b: Any) -> list[float]:
    """Mean over the token axis of |a − b|. One-sample case of ACT eq. (2)."""
    s, n = token_abs_sum(a, b)
    return mean_from_sum(s, n)


def add_abs_sum(
    running: dict[str, tuple[Any, int]],
    key: str,
    sum_abs: Sequence[float],
    n_tokens: int,
) -> None:
    if n_tokens <= 0:
        return
    if hasattr(sum_abs, "detach"):
        incoming: Any = sum_abs.detach().float().reshape(-1).cpu()
    else:
        incoming = [float(x) for x in sum_abs]
    if key not in running:
        running[key] = (
            incoming.clone() if hasattr(incoming, "detach") else list(incoming),
            int(n_tokens),
        )
        return
    prev, n0 = running[key]
    if hasattr(prev, "detach"):
        if hasattr(incoming, "detach"):
            vec = incoming.to(dtype=prev.dtype)
        else:
            import torch

            vec = torch.tensor(incoming, dtype=prev.dtype)
        if int(prev.numel()) != int(vec.numel()):
            raise ValueError(
                f"add_abs_sum: width {int(prev.numel())} vs {int(vec.numel())} for {key}"
            )
        running[key] = (prev + vec, n0 + int(n_tokens))
        return
    vals = incoming.tolist() if hasattr(incoming, "tolist") else incoming
    if len(prev) != len(vals):
        raise ValueError(f"add_abs_sum: width {len(prev)} vs {len(vals)} for {key}")
    running[key] = ([x + float(y) for x, y in zip(prev, vals, strict=True)], n0 + int(n_tokens))


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
    empty = {
        "n_channels": 0,
        "mean": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p99": 0.0,
        "max": 0.0,
        "top1pct_mean": 0.0,
        "frac_gt": {str(t): 0.0 for t in CCDF_THRESHOLDS},
    }
    try:
        import numpy as np

        arr = np.asarray(
            delta.detach().float().cpu().numpy() if hasattr(delta, "detach") else delta,
            dtype=np.float64,
        ).reshape(-1)
        n = int(arr.size)
        if n == 0:
            return empty
        s = np.sort(arr)
        k = max(1, int(n * 0.01))
        top = s[-k:]
        return {
            "n_channels": n,
            "mean": float(s.mean()),
            "p50": float(_percentile(s, 50)),
            "p90": float(_percentile(s, 90)),
            "p99": float(_percentile(s, 99)),
            "max": float(s[-1]),
            "top1pct_mean": float(top.mean()),
            "frac_gt": {str(t): float((arr > t).mean()) for t in CCDF_THRESHOLDS},
        }
    except Exception:
        pass
    if hasattr(delta, "numel"):
        n = int(delta.numel())
    else:
        n = len(delta) if delta is not None else 0
    if n == 0:
        return empty
    s = _sorted(delta)
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
    try:
        import numpy as np

        arr = np.asarray(
            delta.detach().float().cpu().numpy() if hasattr(delta, "detach") else delta,
            dtype=np.float64,
        ).reshape(-1)
        n = int(arr.size)
        if n == 0:
            return {str(t): 0.0 for t in thresholds}
        return {str(t): float((arr > float(t)).mean()) for t in thresholds}
    except Exception:
        pass
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
    n_total = 0
    for vec in named.values():
        n_total += int(vec.numel()) if hasattr(vec, "numel") else len(vec)
    if n_total >= 50_000:
        try:
            return _analyze_channels_numpy(named, p=p, layer_types=layer_types)
        except ImportError:
            pass
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


_RANKED_KEEP = 50_000


def _as_np1d(vec: Sequence[float]) -> Any:
    import numpy as np

    if hasattr(vec, "detach"):
        return vec.detach().float().reshape(-1).cpu().numpy()
    return np.asarray(list(vec), dtype=np.float64).reshape(-1)


def _rows_from_flat_index(
    records: list[tuple[str, str | None, int | None, int]],
    offsets: Any,
    flat_idx: Any,
    deltas: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, fi in enumerate(flat_idx, start=1):
        fi = int(fi)
        mod_i = int(offsets.searchsorted(fi, side="right") - 1)
        key, kind, layer, width = records[mod_i]
        ch = fi - int(offsets[mod_i])
        out.append(
            {
                "key": key,
                "kind": kind,
                "layer": layer,
                "channel": int(ch),
                "delta": float(deltas[fi]),
                "rank": rank,
            }
        )
    return out


def _analyze_channels_numpy(
    named: dict[str, Sequence[float]],
    *,
    p: float,
    layer_types: Sequence[str] | None,
) -> dict[str, Any]:
    """Top-p mask without building a Python dict per channel (MoE is ~32M)."""
    import numpy as np

    records: list[tuple[str, str | None, int | None, int]] = []
    parts: list[Any] = []
    for key, vec in named.items():
        arr = _as_np1d(vec)
        records.append((key, hook_kind(key), layer_index(key), int(arr.size)))
        parts.append(arr)
    if not parts:
        z = {
            "p": p,
            "n_channels": 0,
            "n_channels_no_lm_head": 0,
            "n_mask": 0,
            "n_mask_no_lm_head": 0,
            "mask_threshold": 0.0,
            "global": _stats_ccdf([]),
            "global_no_lm_head": _stats_ccdf([]),
            "by_kind": {k: _stats_ccdf([]) for k in ("attn", "ffn", "ln", "embed", "lm_head")},
            "by_layer": [],
            "modules": [],
            "mask": [],
            "mask_no_lm_head": [],
            "ranked": [],
        }
        return z
    deltas = np.concatenate(parts)
    n = int(deltas.size)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for i, rec in enumerate(records):
        offsets[i + 1] = offsets[i] + rec[3]

    p = min(1.0, max(0.0, float(p)))
    n_mask = max(1, int(round(n * p))) if n else 0
    order = np.argsort(-deltas, kind="stable") if n else np.array([], dtype=np.int64)
    top_idx = order[:n_mask]
    mask = _rows_from_flat_index(records, offsets, top_idx, deltas)

    is_head = np.zeros(n, dtype=bool)
    for i, rec in enumerate(records):
        if rec[1] == "lm_head":
            is_head[int(offsets[i]) : int(offsets[i + 1])] = True
    no_head = deltas[~is_head]
    n_no = int(no_head.size)
    n_mask_no = max(1, int(round(n_no * p))) if n_no else 0
    if n_no:
        order_no = np.argsort(-no_head, kind="stable")
        # map back to flat indices
        no_head_flat = np.flatnonzero(~is_head)
        top_no = no_head_flat[order_no[:n_mask_no]]
        mask_no_head = _rows_from_flat_index(records, offsets, top_no, deltas)
    else:
        mask_no_head = []

    keep_n = n if n <= 1_000_000 else max(n_mask, min(n, _RANKED_KEEP))
    ranked = _rows_from_flat_index(records, offsets, order[:keep_n], deltas)

    kinds = ("attn", "ffn", "ln", "embed", "lm_head")
    by_kind = {}
    for k in kinds:
        sl = []
        for rec, part in zip(records, parts, strict=True):
            if rec[1] == k:
                sl.append(part)
        by_kind[k] = _stats_ccdf(np.concatenate(sl) if sl else np.array([]))

    types = list(layer_types or [])
    n_layers = max(N_LAYERS, max((rec[2] for rec in records if rec[2] is not None), default=-1) + 1)
    mask_layer = [0] * n_layers
    for r in mask:
        li = r.get("layer")
        if li is not None and 0 <= int(li) < n_layers:
            mask_layer[int(li)] += 1
    by_layer: list[dict[str, Any]] = []
    for i in range(n_layers):
        sl = [part for rec, part in zip(records, parts, strict=True) if rec[2] == i]
        st = _stats_ccdf(np.concatenate(sl) if sl else np.array([]))
        by_layer.append(
            {
                "layer": i,
                "kind": types[i] if i < len(types) else "?",
                "n_in_mask": mask_layer[i],
                **st,
            }
        )

    from collections import Counter

    mask_count = Counter(str(r["key"]) for r in mask)
    modules: list[dict[str, Any]] = []
    for rec, part in zip(records, parts, strict=True):
        st = channel_stats(part)
        modules.append(
            {
                "key": rec[0],
                "kind": rec[1],
                "layer": rec[2],
                "n_in_mask": int(mask_count[rec[0]]),
                **st,
            }
        )
    modules.sort(key=lambda r: float(r["p99"]), reverse=True)

    def _np_stats(arr: Any) -> dict[str, Any]:
        return _stats_ccdf(arr)

    threshold = float(mask[-1]["delta"]) if mask else 0.0
    return {
        "p": p,
        "n_channels": n,
        "n_channels_no_lm_head": n_no,
        "n_mask": len(mask),
        "n_mask_no_lm_head": len(mask_no_head),
        "mask_threshold": threshold,
        "global": _np_stats(deltas),
        "global_no_lm_head": _np_stats(no_head),
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
