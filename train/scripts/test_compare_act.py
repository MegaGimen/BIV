#!/usr/bin/env python3
"""Offline checks for ACT channel Δa. No GPU, no hub."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.act import (  # noqa: E402
    add_abs_sum,
    analyze_channels,
    canonical_module_key,
    channel_axis,
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
    thin_answer_pos,
    use_dual_gpus,
    verify_moe_coverage,
)


def test_channel_delta_mean() -> None:
    a = [[0.0, 2.0], [4.0, 6.0]]
    b = [[0.0, 0.0], [0.0, 0.0]]
    d = channel_delta(a, b)
    assert d == [2.0, 4.0], d


def test_thin_answer_pos() -> None:
    assert thin_answer_pos(list(range(10)), None) == list(range(10))
    assert thin_answer_pos(list(range(10)), 10) == list(range(10))
    assert thin_answer_pos(list(range(10)), 1) == [9]
    got = thin_answer_pos(list(range(100)), 4)
    assert got[0] == 0 and got[-1] == 99
    assert len(got) == 4


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
    assert hook_kind("language_model.layers.1.mlp.gate") == "ffn"
    assert hook_kind("language_model.layers.1.mlp.shared_expert_gate") == "ffn"
    assert hook_kind("language_model.layers.1.mlp.experts.gate_up_proj") == "ffn"
    assert hook_kind("language_model.layers.1.mlp.experts.0.down_proj") == "ffn"
    assert (
        hook_kind("language_model.layers.1.mlp.experts.0.down_proj", include_experts=False)
        is None
    )
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


def test_use_dual_gpus() -> None:
    assert use_dual_gpus(2, "auto") is True
    assert use_dual_gpus(1, "auto") is False
    assert use_dual_gpus(2, "cuda:0") is False


def test_load_jsonl_chats_dir(tmp_path: Path) -> None:
    from compare_act import load_jsonl_chats

    d = tmp_path / "mix"
    w1 = d / "wm_code"
    w1.mkdir(parents=True)
    row = {
        "messages": [
            {"role": "user", "content": "ls"},
            {"role": "assistant", "content": "a.txt"},
            {"role": "user", "content": "rm a.txt"},
            {"role": "assistant", "content": "gone"},
        ]
    }
    import json
    (w1 / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_jsonl_chats(d, 10)
    assert len(loaded) == 1
    assert loaded[0][-1]["content"] == "gone"


def test_encode_prompt_string_chat_template() -> None:

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

    # Test truncation preserves answer
    ids_trunc, pos_trunc, note_trunc = encode_prompt(
        _Tok(), text=None, messages=msgs, token_mode="answer", max_length=10
    )
    assert len(ids_trunc) <= 10
    assert len(pos_trunc) > 0
    assert "truncated to 10" in note_trunc


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


def test_packed_moe_is_not_hooked() -> None:
    """Qwen3.5 stores 256 experts as 3D Parameters, not experts.0.down_proj."""
    packed = "model.layers.1.mlp.experts.gate_up_proj"
    down = "model.layers.1.mlp.experts.down_proj"
    experts_mod = "model.layers.1.mlp.experts"
    router = "model.layers.1.mlp.gate.weight"
    shared = "model.layers.1.mlp.shared_expert.down_proj.weight"
    gate = "model.layers.1.mlp.shared_expert_gate.weight"
    assert hook_kind(packed) == "ffn"
    # leaf is down_proj so hook_kind says ffn — ActCapture still does not
    # register a Linear named experts.down_proj; packed capture uses the 3D Parameter.
    assert hook_kind(down) == "ffn"
    assert hook_kind(experts_mod) is None
    assert hook_kind(param_canonical_key(router)) == "ffn"
    assert hook_kind(param_canonical_key(gate)) == "ffn"
    assert hook_kind(param_canonical_key(shared)) == "ffn"
    assert hook_kind(experts_mod, include_experts=False) is None
    assert hook_kind("layers.1.mlp.experts.gate_up_proj", include_experts=False) is None
    assert channel_axis(packed, 3) == 0
    from probe_act_moe import classify_weight_key, would_register_hook

    assert classify_weight_key(packed) == "packed_expert_gate_up"
    assert classify_weight_key(down) == "packed_expert_down"
    assert classify_weight_key(shared) == "shared_expert_ffn"
    assert classify_weight_key(router) == "router"
    assert classify_weight_key(gate) == "shared_expert_gate"
    assert would_register_hook(down) is False
    assert would_register_hook(shared) is True


def test_verify_moe_coverage_reads_files(tmp_path: Path | None = None) -> None:
    import tempfile

    root = tmp_path if tmp_path is not None else Path(tempfile.mkdtemp(prefix="biv_act_chk_"))
    bad = root / "bad"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "report.json").write_text(
        json.dumps(
            {
                "n_channels": 712576,
                "by_kind": {"attn": {"n_channels": 544640}, "ln": {"n_channels": 165888}},
                "modules": [{"key": "layers.0.linear_attn.in_proj_qkv", "kind": "attn"}],
            }
        ),
        encoding="utf-8",
    )
    (bad / "mask.json").write_text(
        json.dumps({"mask": [{"key": "layers.0.linear_attn.in_proj_qkv", "channel": 1, "kind": "attn"}]}),
        encoding="utf-8",
    )
    failed = verify_moe_coverage(bad, include_experts=True)
    assert failed["ok"] is False
    names = {c["name"]: c["ok"] for c in failed["checks"]}
    assert names["ffn_in_by_kind"] is False
    assert names["modules_packed_gate_up"] is False

    good = root / "good"
    good.mkdir(parents=True, exist_ok=True)
    (good / "report.json").write_text(
        json.dumps(
            {
                "n_channels": 32_000_000,
                "by_kind": {"ffn": {"n_channels": 31_000_000}, "attn": {"n_channels": 500_000}},
                "modules": [
                    {"key": "layers.0.mlp.shared_expert.down_proj", "kind": "ffn"},
                    {"key": "layers.0.mlp.experts.gate_up_proj", "kind": "ffn"},
                    {"key": "layers.0.mlp.experts.down_proj", "kind": "ffn"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (good / "mask.json").write_text(
        json.dumps(
            {
                "mask": [
                    {"key": "layers.0.mlp.shared_expert.down_proj", "channel": 0, "kind": "ffn"},
                    {"key": "layers.0.mlp.experts.gate_up_proj", "channel": 3, "kind": "ffn"},
                    {"key": "layers.0.mlp.experts.down_proj", "channel": 7, "kind": "ffn"},
                ]
            }
        ),
        encoding="utf-8",
    )
    passed = verify_moe_coverage(good, include_experts=True)
    assert passed["ok"] is True, passed


def test_as_btc_keeps_moe_2d() -> None:
    try:
        import torch
    except Exception:
        return
    from compare_act import _as_btc

    t3 = _as_btc(torch.zeros(1, 4, 8))
    t2 = _as_btc(torch.zeros(4, 8))
    assert t3 is not None and tuple(t3.shape) == (1, 4, 8)
    assert t2 is not None and tuple(t2.shape) == (1, 4, 8)


def main() -> None:
    test_channel_delta_mean()
    test_thin_answer_pos()
    test_token_weighted_across_samples()
    test_canonical_module_key_aligns_backbones()
    test_hook_kind_act_modules_only()
    test_packed_moe_is_not_hooked()
    test_verify_moe_coverage_reads_files()
    test_as_btc_keeps_moe_2d()
    test_top_p_mask_and_analyze()
    test_module_exec_device_skips_meta()
    test_language_model_only_flag()
    test_use_dual_gpus()
    test_encode_prompt_string_chat_template()
    test_summary_is_channel_not_layer_cut()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
