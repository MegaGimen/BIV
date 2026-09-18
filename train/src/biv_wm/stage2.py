"""Stage 2 connectors: draft head, scorer, linear W.

Instruct backbone yields c_t from chat(h). The draft head proposes K action
vectors in World space (u^W) plus one mouth vector u^A. Frozen JEPA scores
Pred(z_t, u) for the K drafts and the real Enc(a). W maps u^A into Instruct
lm_head space for teacher-forced command tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftHead(nn.Module):
    """Shared trunk on c_t → K world-action vectors and one mouth vector."""

    def __init__(self, dim: int = 2048, n_drafts: int = 3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_drafts = int(n_drafts)
        self.trunk = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=True),
            nn.GELU(),
            nn.Linear(self.dim, self.dim, bias=True),
        )
        self.to_uw = nn.Linear(self.dim, self.n_drafts * self.dim, bias=True)
        self.to_ua = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, c_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(c_t)
        u_w = self.to_uw(h).view(-1, self.n_drafts, self.dim)
        u_a = self.to_ua(h)
        return u_w, u_a


class Scorer(nn.Module):
    """Score one candidate from (u^W, ẑ). Called on K drafts + 1 real."""

    def __init__(self, dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, 1, bias=True),
        )

    def forward(self, u: torch.Tensor, zhat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u, zhat], dim=-1)).squeeze(-1)


class MouthW(nn.Module):
    """2048→2048 map from draft u^A into Instruct lm_head's doorway."""

    def __init__(self, dim: int = 2048) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def ranking_ce(scores: torch.Tensor, real_index: int | torch.Tensor) -> torch.Tensor:
    """Softmax CE over K+1 candidates. Label is the real branch."""
    if isinstance(real_index, int):
        target = torch.full(
            (scores.size(0),),
            int(real_index),
            device=scores.device,
            dtype=torch.long,
        )
    else:
        target = real_index
    return F.cross_entropy(scores.float(), target)


def mouth_token_ce(
    u_a: torch.Tensor,
    a_ids: torch.Tensor,
    a_mask: torch.Tensor,
    embed: nn.Module,
    mouth_w: nn.Module,
    lm_head: nn.Module,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Teacher-forced command tokens: hidden = u^A + prev embed, then W → lm_head."""
    labels = a_ids.masked_fill(a_mask == 0, -100)
    if not bool((labels != -100).any()):
        return u_a.new_zeros(())
    tok_emb = embed(a_ids)
    zeros = tok_emb.new_zeros(tok_emb.size(0), 1, tok_emb.size(-1))
    prev = torch.cat([zeros, tok_emb[:, :-1]], dim=1)
    hidden = mouth_w(u_a.unsqueeze(1).to(dtype=tok_emb.dtype) + prev)
    mask = labels != -100
    h = hidden[mask]
    y = labels[mask]
    n = h.size(0)
    step = max(int(chunk_size), 1)
    total = h.new_zeros(())
    for i in range(0, n, step):
        logits = lm_head(h[i : i + step])
        total = total + F.cross_entropy(logits.float(), y[i : i + step], reduction="sum")
    return total / n
