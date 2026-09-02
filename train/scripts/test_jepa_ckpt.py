#!/usr/bin/env python3
"""Offline checks for Muse-style JEPA checkpoint names/rotation. No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.ckpt import (  # noqa: E402
    canonical_lora_key,
    ckpt_complete,
    epoch_end_name,
    find_latest_ckpt,
    parse_ckpt_name,
    rolling_name,
    rotate_rolling,
    write_trainer_state,
)


def _fake_ckpt(root: Path, name: str, *, with_jepa: bool = True) -> Path:
    p = root / name
    p.mkdir(parents=True)
    write_trainer_state(p, epoch=0, global_step=1)
    (p / "adapter_model.safetensors").write_bytes(b"x")
    if with_jepa:
        (p / "jepa.pt").write_bytes(b"x")
    return p


def test_canonical_lora_key() -> None:
    saved = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    live = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    fsdp = "_fsdp_wrapped_module." + live
    assert canonical_lora_key(saved) == canonical_lora_key(live) == canonical_lora_key(fsdp)


def test_names() -> None:
    assert rolling_name(0, 25) == "checkpoint-e0-s25"
    assert epoch_end_name(1, 100) == "checkpoint-epoch1-end-s100"
    assert parse_ckpt_name("checkpoint-e0-s25") == (0, 25, 1)
    assert parse_ckpt_name("checkpoint-epoch2-end-s200") == (2, 200, 2)
    assert parse_ckpt_name("checkpoint-25") == (0, 25, 0)
    assert parse_ckpt_name("step-100") == (0, 100, 0)
    assert parse_ckpt_name("final") is None


def test_rotate_keeps_epoch_end(tmp: Path) -> None:
    _fake_ckpt(tmp, "checkpoint-e0-s25")
    _fake_ckpt(tmp, "checkpoint-e0-s50")
    _fake_ckpt(tmp, "checkpoint-e0-s75")
    _fake_ckpt(tmp, "checkpoint-e0-s100")
    _fake_ckpt(tmp, "checkpoint-epoch1-end-s100")
    removed = rotate_rolling(tmp, 3)
    assert "checkpoint-e0-s25" in removed
    names = {p.name for p in tmp.iterdir() if p.is_dir()}
    assert "checkpoint-epoch1-end-s100" in names
    assert "checkpoint-e0-s25" not in names
    assert len([n for n in names if n.startswith("checkpoint-e0-")]) == 3


def test_find_latest_epoch_then_step(tmp: Path) -> None:
    _fake_ckpt(tmp, "checkpoint-e0-s900")
    _fake_ckpt(tmp, "checkpoint-e1-s10")
    picked = find_latest_ckpt(tmp)
    assert picked is not None
    assert picked.name == "checkpoint-e1-s10"


def test_find_latest_prefers_epoch_end(tmp: Path) -> None:
    _fake_ckpt(tmp, "checkpoint-e0-s100")
    _fake_ckpt(tmp, "checkpoint-epoch1-end-s100")
    picked = find_latest_ckpt(tmp)
    assert picked is not None
    assert picked.name == "checkpoint-epoch1-end-s100"


def test_incomplete_skipped(tmp: Path) -> None:
    _fake_ckpt(tmp, "checkpoint-e0-s25", with_jepa=False)
    assert ckpt_complete(tmp / "checkpoint-e0-s25") is False
    assert find_latest_ckpt(tmp) is None


def test_adapter_only_ok_without_jepa(tmp: Path) -> None:
    _fake_ckpt(tmp, "checkpoint-e0-s25", with_jepa=False)
    assert ckpt_complete(tmp / "checkpoint-e0-s25", require_jepa=False) is True
    picked = find_latest_ckpt(tmp, require_jepa=False)
    assert picked is not None
    assert picked.name == "checkpoint-e0-s25"


def test_trainer_state(tmp: Path) -> None:
    p = tmp / "c"
    p.mkdir()
    write_trainer_state(p, epoch=1, global_step=50, extra={"max_length": 8192})
    data = json.loads((p / "trainer_state.json").read_text(encoding="utf-8"))
    assert data["epoch"] == 1
    assert data["global_step"] == 50
    assert data["max_length"] == 8192


def main() -> None:
    import tempfile

    test_canonical_lora_key()
    test_names()
    with tempfile.TemporaryDirectory() as d:
        test_rotate_keeps_epoch_end(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_find_latest_prefers_epoch_end(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_find_latest_epoch_then_step(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_incomplete_skipped(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_adapter_only_ok_without_jepa(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_trainer_state(Path(d))
    print("ok", flush=True)


if __name__ == "__main__":
    main()
