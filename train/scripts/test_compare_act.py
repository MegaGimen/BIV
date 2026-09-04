#!/usr/bin/env python3
"""Offline checks for ACT channel Δa. No GPU, no hub."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.act import (  # noqa: E402
    bar,
    ccdf,
    channel_delta,
    channel_stats,
    hook_kind,
    layer_index,
    top_channels,
)

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_act import build_layers, format_summary  # noqa: E402


def test_channel_delta_mean() -> None:
    a = [[0.0, 2.0], [4.0, 6.0]]
    b = [[0.0, 0.0], [0.0, 0.0]]
    d = channel_delta(a, b)
    assert d == [2.0, 4.0], d


def test_hook_kind_skips_experts() -> None:
    assert hook_kind("language_model.layers.3.self_attn.q_proj") == "attn"
    assert hook_kind("language_model.layers.0.linear_attn.in_proj_qkv") == "attn"
    assert hook_kind("language_model.layers.1.mlp.shared_expert.down_proj") == "ffn"
    assert hook_kind("language_model.layers.1.mlp.experts.17.down_proj") is None
    assert hook_kind("model.visual.patch_embed") is None
    assert hook_kind("lm_head") == "lm_head"
    assert hook_kind("language_model.embed_tokens") == "embed"
    assert hook_kind("language_model.norm") == "ln"


def test_layer_index() -> None:
    assert layer_index("language_model.layers.12.residual") == 12
    assert layer_index("embed") is None
    assert layer_index("lm_head") is None


def test_ccdf_and_stats() -> None:
    delta = [0.1] * 99 + [10.0]
    st = channel_stats(delta)
    assert st["n_channels"] == 100
    assert abs(st["mean"] - (0.1 * 99 + 10.0) / 100) < 1e-9
    frac = ccdf(delta)
    assert abs(frac["4.5"] - 0.01) < 1e-12
    assert frac["16.0"] == 0.0


def test_bar_and_summary() -> None:
    assert bar(0.0, 1.0, width=4) == "...."
    assert bar(1.0, 1.0, width=4) == "####"
    residual = {f"layers.{i}.residual": [float(i + 1)] * 4 for i in range(3)}
    modules = {
        "language_model.layers.0.self_attn.q_proj": [2.0, 2.0],
        "language_model.layers.0.mlp.shared_expert.down_proj": [0.5, 0.5],
    }
    layers = build_layers(residual, modules, ["linear", "linear", "full_attention"])
    report = {
        "method": "test",
        "paths": {"world": "/w", "instruct": "/i"},
        "prompt_note": "test",
        "n_tokens": 8,
        "n_answer_tokens": 2,
        "decoded_answer": "gone",
        "layers": layers,
        "special": {"embed": channel_stats([0.2, 0.4])},
        "ccdf_all": ccdf([0.2, 5.0]),
        "ccdf_no_lm_head": ccdf([0.2]),
        "top_channels": top_channels(residual, n=3),
        "skipped": [],
    }
    text = format_summary(report)
    assert "compare_act.py" in text
    assert "ACT_Δa" in text
    assert "残差流激活差最大层" in text
    assert "L2=" in text


def main() -> None:
    test_channel_delta_mean()
    test_hook_kind_skips_experts()
    test_layer_index()
    test_ccdf_and_stats()
    test_bar_and_summary()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
