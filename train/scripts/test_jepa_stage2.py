#!/usr/bin/env python3
"""Offline checks for Stage 2 draft/scorer/W. Torch optional."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _try_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def test_module_shapes() -> None:
    torch = _try_torch()
    if torch is None:
        return
    from biv_wm.stage2 import DraftHead, MouthW, Scorer, mouth_token_ce, ranking_ce

    b, d, k = 2, 8, 3
    draft = DraftHead(dim=d, n_drafts=k)
    scorer = Scorer(dim=d)
    w = MouthW(dim=d)
    c = torch.randn(b, d)
    u_w, u_a = draft(c)
    assert u_w.shape == (b, k, d)
    assert u_a.shape == (b, d)
    u_star = torch.randn(b, d)
    zhat = torch.randn(b, k + 1, d)
    u_all = torch.cat([u_w, u_star.unsqueeze(1)], dim=1)
    scores = scorer(u_all.reshape(b * (k + 1), d), zhat.reshape(b * (k + 1), d)).view(b, k + 1)
    loss_r = ranking_ce(scores, k)
    assert loss_r.ndim == 0
    embed = torch.nn.Embedding(16, d)
    ids = torch.randint(1, 16, (b, 5))
    mask = torch.ones(b, 5, dtype=torch.long)
    head = torch.nn.Linear(d, 16)
    loss_c = mouth_token_ce(u_a, ids, mask, embed, w, head)
    assert loss_c.ndim == 0
    (loss_r + loss_c).backward()
    assert draft.to_uw.weight.grad is not None
    assert w.proj.weight.grad is not None


def test_real_branch_index() -> None:
    torch = _try_torch()
    if torch is None:
        return
    from biv_wm.stage2 import ranking_ce

    scores = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    loss = ranking_ce(scores, 3)
    assert float(loss) < 0.05


def main() -> None:
    test_module_shapes()
    test_real_branch_index()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
