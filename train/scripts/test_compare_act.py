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
    analyze_channels,
    channel_delta,
    finalize_running,
    hook_kind,
    layer_index,
    token_abs_sum,
    top_p_mask,
)

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_act import format_summary  # noqa: E402


def test_channel_delta_mean() -> None:
    a = [[0.0, 2.0], [4.0, 6.0]]
    b = [[0.0, 0.0], [0.0, 0.0]]
    d = channel_delta(a, b)
    assert d == [2.0, 4.0], d


def test_token_weighted_across_samples() -> None:
    running: dict[str, tuple[list[float], int]] = {}
    s1, n1 = token_abs_sum([[2.0]], [[0.0]])
    add_abs_sum(running, "q_proj", s1, n1)
    s2, n2 = token_abs_sum([[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]])
    add_abs_sum(running, "q_proj", s2, n2)
    got = finalize_running(running)["q_proj"]
    assert abs(got[0] - 0.5) < 1e-12, got


def test_hook_kind_act_modules_only() -> None:
    assert hook_kind("language_model.layers.3.self_attn.q_proj") == "attn"
    assert hook_kind("language_model.layers.1.mlp.shared_expert.down_proj") == "ffn"
    assert hook_kind("language_model.layers.1.mlp") is None
    assert hook_kind("lm_head") == "lm_head"
    assert layer_index("language_model.layers.12.self_attn.q_proj") == 12


def test_top_p_mask_and_analyze() -> None:
    named = {
        "language_model.layers.0.self_attn.q_proj": [0.1, 10.0, 0.2, 0.3],
        "language_model.layers.2.mlp.shared_expert.down_proj": [0.05, 0.06],
        "lm_head": [0.01, 9.0],
    }
    analysis = analyze_channels(named, p=0.25, layer_types=["linear", "linear", "full_attention"])
    ranked = analysis["ranked"]
    assert ranked[0]["key"].endswith("q_proj")
    assert ranked[0]["channel"] == 1
    assert ranked[0]["delta"] == 10.0
    mask = top_p_mask(ranked, 0.25)
    assert len(mask) == max(1, int(round(len(ranked) * 0.25)))
    assert analysis["by_kind"]["attn"]["n_channels"] == 4
    assert analysis["by_kind"]["ffn"]["n_channels"] == 2
    assert analysis["by_kind"]["lm_head"]["n_channels"] == 2
    assert analysis["n_channels"] == 8
    # 0.25 * 8 = 2 → mask is the two largest (10.0 and 9.0)
    assert analysis["n_mask"] == 2
    assert {r["key"] for r in analysis["mask"]} == {
        "language_model.layers.0.self_attn.q_proj",
        "lm_head",
    }
    no_head = analysis["mask_no_lm_head"]
    assert all(r["kind"] != "lm_head" for r in no_head)


def test_summary_is_channel_not_layer_cut() -> None:
    named = {
        "language_model.layers.0.self_attn.q_proj": [2.0, 2.0],
        "language_model.layers.2.self_attn.o_proj": [8.0, 8.0, 8.0, 8.0],
        "lm_head": [0.1, 0.2],
    }
    analysis = analyze_channels(named, p=0.25, layer_types=["linear", "linear", "full_attention"])
    analysis.pop("ranked")
    report = {
        "method": "test",
        "paths": {"world": "/w", "instruct": "/i"},
        "prompt_note": "test",
        "n_samples": 1,
        "n_tokens": 8,
        "n_answer_tokens": 2,
        "decoded_answer": "gone",
        "skipped": [],
        **analysis,
    }
    text = format_summary(report)
    assert "compare_act.py" in text
    assert "top-p%" in text
    assert "全局 CCDF" in text
    assert "按模块种类" in text
    assert "ACT 掩码" in text
    assert "不是按层切" in text
    assert "残差流" not in text
    assert "ACT_Δa" not in text


def main() -> None:
    test_channel_delta_mean()
    test_token_weighted_across_samples()
    test_hook_kind_act_modules_only()
    test_top_p_mask_and_analyze()
    test_summary_is_channel_not_layer_cut()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
