#!/usr/bin/env python3
"""CPU checks for the Stage 1 collapse diagnostic. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.jepa import bank_nce_loss, collapse_stats, format_collapse_line  # noqa: E402


def test_real_align_not_collapse() -> None:
    n, d = 8, 16
    z = torch.eye(n, d)
    pred = z.clone()
    texts = [f"obs-{i}" for i in range(n)]
    s = collapse_stats(pred, z, texts)
    assert s["verdict"] == "paired_ahead"
    assert s["skipped_same_o"] == 0
    assert isinstance(s["paired"], dict) and s["paired"]["median"] > 0.8
    assert isinstance(s["mismatch"], dict) and s["mismatch"]["median"] < 0.4
    import json

    json.dumps(s)


def test_collapse_like_constant_pred() -> None:
    n, d = 8, 16
    z = torch.ones(n, d)
    pred = torch.ones(n, d)
    texts = [f"obs-{i}" for i in range(n)]
    s = collapse_stats(pred, z, texts)
    assert s["verdict"] == "collapse_like"
    assert abs(float(s["paired"]["median"]) - float(s["mismatch"]["median"])) < 0.02


def test_exact_string_not_used_as_negative() -> None:
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    pred = z.clone()
    texts = ["same", "other", "same"]
    s = collapse_stats(pred, z, texts)
    # pairs (0,2) and (2,0) share the observation string → dropped
    assert s["skipped_same_o"] == 2
    assert s["mismatch"]["n"] == 4


def test_too_few_distinct() -> None:
    z = torch.ones(2, 4)
    pred = torch.ones(2, 4)
    s = collapse_stats(pred, z, ["dup", "dup"])
    assert s["verdict"] == "no_mismatch_pairs"
    assert s["mismatch"]["n"] == 0
    line = format_collapse_line(s)
    assert "verdict=no_mismatch_pairs" in line


def test_bank_nce_loss_live_anchor_gradient_reaches_encoder() -> None:
    """loss_simcse's call site: anchor is a *second live encoding* of the same
    text, pos_z is the first (also live at the call site, but the function
    detaches it internally) — gradient must still reach the anchor tensor,
    since that anchor IS the observation encoder's own output.
    """
    d = 8
    anchor = torch.zeros(1, d, requires_grad=True)
    with torch.no_grad():
        anchor[0, 0] = 1.0
    pos = anchor.detach().clone()  # a separate live encoding, same text, ~same direction
    bank = torch.eye(d)[1:5]
    loss = bank_nce_loss(anchor, pos, ["obs"], bank, [f"neg-{i}" for i in range(4)], temperature=0.05)
    assert loss is not None
    loss.backward()
    assert anchor.grad is not None and torch.any(anchor.grad != 0)


def test_bank_nce_loss_separated_negatives_low_loss() -> None:
    d = 8
    pred = torch.zeros(1, d)
    pred[0, 0] = 1.0
    pos = pred.clone()
    bank = torch.eye(d)[1:5]  # orthogonal to pred/pos
    loss = bank_nce_loss(
        pred, pos, ["obs-real"], bank, [f"obs-neg-{i}" for i in range(4)], temperature=0.05
    )
    assert loss is not None and float(loss) < 0.1


def test_bank_nce_loss_masks_text_duplicates() -> None:
    pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pos = pred.clone()
    bank = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # would look like a perfect negative...
    loss = bank_nce_loss(pred, pos, ["same-text"], bank, ["same-text"], temperature=0.05)
    # ...but it's masked out (same o_text), so this must not blow up into a huge loss.
    assert loss is not None and torch.isfinite(loss) and float(loss) < 0.1


def test_bank_nce_loss_empty_bank_is_none() -> None:
    pred = torch.randn(1, 4)
    assert bank_nce_loss(pred, pred.clone(), ["x"], torch.empty(0, 4), [], temperature=0.05) is None


def main() -> None:
    test_real_align_not_collapse()
    test_collapse_like_constant_pred()
    test_exact_string_not_used_as_negative()
    test_too_few_distinct()
    test_bank_nce_loss_live_anchor_gradient_reaches_encoder()
    test_bank_nce_loss_separated_negatives_low_loss()
    test_bank_nce_loss_masks_text_duplicates()
    test_bank_nce_loss_empty_bank_is_none()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
