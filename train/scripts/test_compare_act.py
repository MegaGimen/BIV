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
    canonical_module_key,
    channel_delta,
    finalize_running,
    hook_kind,
    layer_index,
    param_canonical_key,
    token_abs_sum,
    top_p_mask,
)

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_act import (  # noqa: E402
    _language_model_only,
    _module_exec_device,
    encode_prompt,
    format_summary,
)


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


def test_canonical_module_key_aligns_backbones() -> None:
    aw = "model.layers.0.linear_attn.in_proj_qkv"
    inst = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    assert canonical_module_key(aw) == canonical_module_key(inst)
    assert canonical_module_key(aw) == "layers.0.linear_attn.in_proj_qkv"
    assert canonical_module_key("model.embed_tokens") == canonical_module_key(
        "model.language_model.embed_tokens"
    )
    assert canonical_module_key("lm_head") == "lm_head"
    assert layer_index(canonical_module_key(inst)) == 0
    assert param_canonical_key("model.language_model.layers.0.linear_attn.in_proj_qkv.weight") == (
        canonical_module_key(aw)
    )


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


def test_module_exec_device_skips_meta() -> None:
    class _Dev:
        def __init__(self, typ: str) -> None:
            self.type = typ

    class _P:
        def __init__(self, typ: str) -> None:
            self.device = _Dev(typ)
            self.dtype = "bf16"

    class _Hook:
        execution_device = "cuda:0"

    class _Mod:
        def __init__(self) -> None:
            self._hf_hook = _Hook()

        def parameters(self):
            yield _P("meta")

    got = _module_exec_device(_Mod())
    assert str(got) == "cuda:0", got


def test_language_model_only_flag() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        d = Path(raw)
        assert _language_model_only(d) is False
        (d / "config.json").write_text('{"language_model_only": true}', encoding="utf-8")
        assert _language_model_only(d) is True


def test_encode_prompt_string_chat_template() -> None:
    """Qwen tokenizers may return a string from apply_chat_template; ids must be ints."""

    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = [f"{m['role']}:{m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("assistant:")
            return "\n".join(parts)

        def __call__(self, text, truncation=False, add_special_tokens=True):
            return {"input_ids": [ord(c) % 40 + 1 for c in text]}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]

    msgs = [
        {"role": "user", "content": "ls"},
        {"role": "assistant", "content": "a.txt"},
        {"role": "user", "content": "rm a.txt"},
        {"role": "assistant", "content": "gone"},
    ]
    ids, pos, _note = encode_prompt(_Tok(), text=None, messages=msgs, token_mode="answer")
    assert ids and all(isinstance(x, int) for x in ids), ids[:8]
    assert pos and all(isinstance(i, int) for i in pos)
    assert pos[-1] == len(ids) - 1
    assert all(isinstance(ids[i], int) for i in pos)


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
    test_canonical_module_key_aligns_backbones()
    test_hook_kind_act_modules_only()
    test_top_p_mask_and_analyze()
    test_module_exec_device_skips_meta()
    test_language_model_only_flag()
    test_encode_prompt_string_chat_template()
    test_summary_is_channel_not_layer_cut()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
