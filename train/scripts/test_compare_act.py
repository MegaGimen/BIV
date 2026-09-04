#!/usr/bin/env python3
"""Offline checks for ACT channel Δa. No GPU, no hub."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.act import (  # noqa: E402
    add_abs_sum,
    bar,
    ccdf,
    channel_delta,
    channel_stats,
    finalize_running,
    hook_kind,
    layer_index,
    token_abs_sum,
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


def test_token_weighted_across_samples() -> None:
    """ACT eq. (2) pools answer tokens, not mean-of-per-sample-means."""
    running: dict[str, tuple[list[float], int]] = {}
    s1, n1 = token_abs_sum([[2.0]], [[0.0]])
    add_abs_sum(running, "q_proj", s1, n1)
    s2, n2 = token_abs_sum([[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]])
    add_abs_sum(running, "q_proj", s2, n2)
    got = finalize_running(running)["q_proj"]
    assert abs(got[0] - 0.5) < 1e-12, got
    assert abs(channel_delta([[2.0]], [[0.0]])[0] - 2.0) < 1e-12


def test_hook_kind_act_modules_only() -> None:
    assert hook_kind("language_model.layers.3.self_attn.q_proj") == "attn"
    assert hook_kind("language_model.layers.0.linear_attn.in_proj_qkv") == "attn"
    assert hook_kind("language_model.layers.1.mlp.shared_expert.down_proj") == "ffn"
    assert hook_kind("language_model.layers.1.mlp.experts.17.down_proj") is None
    assert hook_kind("language_model.layers.1.mlp") is None
    assert hook_kind("language_model.layers.3.self_attn.q_norm") is None
    assert hook_kind("model.visual.patch_embed") is None
    assert hook_kind("lm_head") == "lm_head"
    assert hook_kind("language_model.embed_tokens") == "embed"
    assert hook_kind("language_model.norm") == "ln"
    assert hook_kind("layers.0.residual") is None


def test_layer_index() -> None:
    assert layer_index("language_model.layers.12.self_attn.q_proj") == 12
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
    modules = {
        "language_model.layers.0.self_attn.q_proj": [2.0, 2.0],
        "language_model.layers.0.mlp.shared_expert.down_proj": [0.5, 0.5],
        "language_model.layers.2.self_attn.o_proj": [4.0, 4.0, 4.0, 4.0],
    }
    layers = build_layers(modules, ["linear", "linear", "full_attention"])
    report = {
        "method": "test",
        "paths": {"world": "/w", "instruct": "/i"},
        "prompt_note": "test",
        "n_samples": 1,
        "n_tokens": 8,
        "n_answer_tokens": 2,
        "decoded_answer": "gone",
        "layers": layers,
        "special": {"embed": channel_stats([0.2, 0.4])},
        "ccdf_all": ccdf([0.2, 5.0]),
        "ccdf_no_lm_head": ccdf([0.2]),
        "top_channels": top_channels(modules, n=3),
        "skipped": [],
    }
    text = format_summary(report)
    assert "compare_act.py" in text
    assert "ACT_Δa" in text
    assert "模块通道 Δa 最大层" in text
    assert "残差流" not in text
    assert "L2=" in text
    assert layers[0]["act"]["n_channels"] == 4
    assert layers[2]["act"]["mean"] == 4.0


def main() -> None:
    test_channel_delta_mean()
    test_token_weighted_across_samples()
    test_hook_kind_act_modules_only()
    test_layer_index()
    test_ccdf_and_stats()
    test_bar_and_summary()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
