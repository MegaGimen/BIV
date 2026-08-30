"""Row-level mean absolute value of a fine-tune delta, as in Layer Swapping (2410.01335).

W_Δ = W_ft − W_pre. For a matrix, take MAV along the last dim (one number per row),
then mean those. For a vector, MAV of the vector.
"""

from __future__ import annotations

from typing import Any


def row_mav(delta: Any) -> float:
    """``delta`` is a torch tensor (ft − base). Returns a Python float."""
    x = delta.detach().float().abs()
    if x.ndim == 0:
        return float(x.item())
    if x.ndim == 1:
        return float(x.mean().item())
    return float(x.mean(dim=-1).mean().item())
