#!/usr/bin/env python3
"""Offline checks for ACT masked merge into Instruct. No GPU, no hub."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.act import (  # noqa: E402
    blend_masked_channels,
    channel_axis,
    group_mask_channels,
    load_mask_rows,
    param_canonical_key,
    world_param_candidates,
)


def test_param_canonical_key() -> None:
    inst = "model.language_model.layers.36.linear_attn.in_proj_qkv.weight"
    aw = "model.layers.36.linear_attn.in_proj_qkv.weight"
    assert param_canonical_key(inst) == param_canonical_key(aw)
    assert param_canonical_key(inst) == "layers.36.linear_attn.in_proj_qkv"
    assert param_canonical_key("lm_head.weight") == "lm_head"
    assert param_canonical_key("model.language_model.embed_tokens.weight") == "embed_tokens"
    assert param_canonical_key("model.norm.weight") == "norm"


def test_world_param_candidates() -> None:
    inst = "model.language_model.layers.0.self_attn.q_proj.weight"
    cands = world_param_candidates(inst)
    assert inst in cands
    assert "model.layers.0.self_attn.q_proj.weight" in cands


def test_channel_axis_linear_vs_embed() -> None:
    assert channel_axis("lm_head.weight", 2) == 0
    assert channel_axis("layers.0.self_attn.q_proj.weight", 2) == 0
    assert channel_axis("embed_tokens.weight", 2) == 1
    assert channel_axis("layers.0.input_layernorm.weight", 1) == 0
    assert channel_axis("layers.0.self_attn.q_proj.bias", 1) == 0


def test_load_mask_rows_no_lm_head() -> None:
    payload = {
        "mask": [
            {"key": "lm_head", "channel": 1, "kind": "lm_head"},
            {"key": "layers.0.norm", "channel": 3, "kind": "ln"},
        ],
        "mask_no_lm_head": [{"key": "layers.0.norm", "channel": 3, "kind": "ln"}],
    }
    full = load_mask_rows(payload, no_lm_head=False)
    slim = load_mask_rows(payload, no_lm_head=True)
    assert len(full) == 2
    assert [r["key"] for r in slim] == ["layers.0.norm"]
    grouped = group_mask_channels(full)
    assert grouped["lm_head"] == [1]


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def test_blend_linear_rows() -> None:
    torch = _torch()
    if torch is None:
        print("skip test_blend_linear_rows (no torch)", flush=True)
        return
    target = torch.zeros(4, 3)
    source = torch.ones(4, 3)
    out = blend_masked_channels(target, source, [1, 3], lam=0.4, axis=0)
    assert float(out[0, 0]) == 0.0
    assert abs(float(out[1, 0]) - 0.4) < 1e-6
    assert abs(float(out[3, 2]) - 0.4) < 1e-6
    assert float(out[2, 1]) == 0.0


def test_blend_embed_columns() -> None:
    torch = _torch()
    if torch is None:
        print("skip test_blend_embed_columns (no torch)", flush=True)
        return
    target = torch.zeros(5, 4)
    source = torch.ones(5, 4)
    out = blend_masked_channels(target, source, [2], lam=1.0, axis=1)
    assert float(out[0, 0]) == 0.0
    assert float(out[4, 2]) == 1.0
    assert float(out[1, 3]) == 0.0


def test_merge_act_two_tiny_shards(tmp_path: Path | None = None) -> None:
    torch = _torch()
    if torch is None:
        print("skip test_merge_act_two_tiny_shards (no torch)", flush=True)
        return
    from safetensors.torch import save_file

    root = tmp_path if tmp_path is not None else Path("/tmp/biv_act_merge_test")
    if tmp_path is None:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="biv_act_"))

    world_dir = root / "world"
    agent_dir = root / "agent"
    out_dir = root / "out"
    world_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    w_key = "model.layers.0.linear_attn.in_proj_qkv.weight"
    a_key = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    world_t = torch.ones(4, 8)
    agent_t = torch.zeros(4, 8)
    save_file({w_key: world_t}, str(world_dir / "model.safetensors"))
    save_file({a_key: agent_t}, str(agent_dir / "model.safetensors"))
    (world_dir / "config.json").write_text("{}", encoding="utf-8")
    (agent_dir / "config.json").write_text('{"architectures":["X"]}\n', encoding="utf-8")
    (agent_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    mask_path = root / "mask.json"
    mask_path.write_text(
        json.dumps(
            {
                "p": 0.01,
                "mask": [
                    {
                        "key": "layers.0.linear_attn.in_proj_qkv",
                        "channel": 1,
                        "kind": "attn",
                    }
                ],
                "mask_no_lm_head": [
                    {
                        "key": "layers.0.linear_attn.in_proj_qkv",
                        "channel": 1,
                        "kind": "attn",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    merge_dir = Path(__file__).resolve().parents[2] / "merge"
    if str(merge_dir) not in sys.path:
        sys.path.insert(0, str(merge_dir))
    import importlib.util

    spec = importlib.util.spec_from_file_location("biv_merge_act", merge_dir / "act.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = mod.merge_act(
        world_dir=world_dir,
        agent_dir=agent_dir,
        out_dir=out_dir,
        mask_path=mask_path,
        lam=0.4,
        no_lm_head=False,
    )
    assert meta["n_tensors_patched"] == 1
    assert meta["n_rows_written"] == 1
    assert "n_copied_shards" in meta
    assert "n_rewritten_shards" in meta
    from safetensors import safe_open

    with safe_open(str(out_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        got = f.get_tensor(a_key)
    assert abs(float(got[1, 0]) - 0.4) < 1e-5
    assert float(got[0, 0]) == 0.0
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "tokenizer_config.json").is_file()


def main() -> None:
    test_param_canonical_key()
    test_world_param_candidates()
    test_channel_axis_linear_vs_embed()
    test_load_mask_rows_no_lm_head()
    test_blend_linear_rows()
    test_blend_embed_columns()
    test_merge_act_two_tiny_shards()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
