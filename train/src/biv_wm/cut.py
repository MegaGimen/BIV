"""Fish-cut: pick ℓ from probe per-layer stats, decide which checkpoint owns each tensor."""

from __future__ import annotations

import re
from typing import Any

N_LAYERS = 40
GROUP = 4  # 3 Gated DeltaNet + 1 full attention
LM_LAYER_RE = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.")


def decoder_layer_index(key: str) -> int | None:
    """Index in the 40-layer text backbone. Ignores mtp.layers.*."""
    m = LM_LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def group_boundaries(n: int = N_LAYERS, group: int = GROUP) -> list[int]:
    """Legal cut indices: start of a 3+1 block, not 0 or n."""
    return list(range(group, n, group))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pick_cut(per_layer: list[dict[str, Any]]) -> dict[str, Any]:
    """ℓ maximizing mean(Instruct/AW δ-ratio on the tail) − mean(on the front).

    Only group boundaries so a GDN trio and its full-attention layer stay together.
    Probe on 2026-08-30: ℓ=12 (gap 0.049).
    """
    by_i = {int(row["layer"]): row for row in per_layer}
    ratios: list[float] = []
    for i in range(N_LAYERS):
        row = by_i.get(i)
        if not row or row.get("delta_ratio_instruct_over_aw") is None:
            raise ValueError(
                f"per_layer[{i}] missing delta_ratio_instruct_over_aw; run probe.py without --skip-weights"
            )
        ratios.append(float(row["delta_ratio_instruct_over_aw"]))

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for ell in group_boundaries():
        front, back = ratios[:ell], ratios[ell:]
        gap = _mean(back) - _mean(front)
        row = {
            "ell": ell,
            "mean_front": _mean(front),
            "mean_back": _mean(back),
            "gap": gap,
        }
        candidates.append(row)
        if best is None or gap > float(best["gap"]):
            best = row
    assert best is not None
    return {
        "ell": int(best["ell"]),
        "gap": best["gap"],
        "mean_front": best["mean_front"],
        "mean_back": best["mean_back"],
        "rule": (
            "max mean(Instruct/AW delta-ratio)[ℓ:] − mean(...)[:ℓ], "
            f"ℓ in {group_boundaries()} (4-layer GDN+attn groups)"
        ),
        "candidates": candidates,
    }


def tensor_source(key: str, ell: int) -> str:
    """'world' | 'instruct' | 'drop' for one safetensors key."""
    parts = set(key.split("."))
    if "visual" in parts or "vision" in parts or "mtp" in parts:
        return "drop"
    if key == "lm_head.weight" or key.endswith(".lm_head.weight"):
        return "instruct"
    if key.endswith("language_model.norm.weight") or key.endswith("language_model.norm.bias"):
        return "instruct"
    idx = decoder_layer_index(key)
    if idx is not None:
        return "instruct" if idx >= ell else "world"
    return "world"
