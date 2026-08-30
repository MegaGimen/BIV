#!/usr/bin/env python3
"""Offline checks for fish-cut ℓ and (h,a,o) split. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.cut import pick_cut, tensor_source
from biv_wm.hao import split_hao


def test_pick_cut_step_at_12() -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "layer": i,
                "delta_ratio_instruct_over_aw": 0.1 if i < 12 else 0.3,
            }
        )
    picked = pick_cut(rows)
    assert picked["ell"] == 12, picked
    assert abs(picked["gap"] - 0.2) < 1e-9
    gaps = {int(r["ell"]): r["gap"] for r in picked["candidates"]}
    assert gaps[12] >= gaps[8]
    assert gaps[12] >= gaps[16]


def test_tensor_source() -> None:
    ell = 12
    assert tensor_source("model.language_model.embed_tokens.weight", ell) == "world"
    assert tensor_source("model.language_model.layers.11.mlp.gate_proj.weight", ell) == "world"
    assert tensor_source("model.language_model.layers.12.self_attn.q_proj.weight", ell) == "instruct"
    assert tensor_source("lm_head.weight", ell) == "instruct"
    assert tensor_source("model.language_model.norm.weight", ell) == "instruct"
    assert tensor_source("mtp.layers.0.self_attn.q_proj.weight", ell) == "drop"
    assert tensor_source("model.visual.patch_embed.weight", ell) == "drop"


def test_split_hao() -> None:
    msgs = [
        {"role": "system", "content": "wm"},
        {"role": "user", "content": '{"tool":"ls"}'},
        {"role": "assistant", "content": '{"output":"a"}'},
        {"role": "user", "content": '{"tool":"rm","arguments":{"path":"a.txt"}}'},
        {"role": "assistant", "content": '{"output":"gone","isError":false}'},
    ]
    h, a, o = split_hao(msgs)
    assert h == msgs[:3]
    assert a["content"].startswith('{"tool":"rm"')
    assert "gone" in o["content"]
    assert split_hao([]) is None
    assert split_hao([{"role": "user", "content": "x"}]) is None


def main() -> None:
    test_pick_cut_step_at_12()
    test_tensor_source()
    test_split_hao()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
