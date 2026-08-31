#!/usr/bin/env python3
"""CPU checks for the Stage 1 collapse diagnostic. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.jepa import collapse_stats, format_collapse_line  # noqa: E402


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


def main() -> None:
    test_real_align_not_collapse()
    test_collapse_like_constant_pred()
    test_exact_string_not_used_as_negative()
    test_too_few_distinct()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
