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


def cosine_align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float().detach(), dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).mean()
