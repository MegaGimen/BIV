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
    capture_rng_state,
    ckpt_complete,
    copy_optimizer_state,
    epoch_end_name,
    find_latest_ckpt,
    parse_ckpt_name,
    plain_cpu_state_dict,
    plain_cpu_tensor,
    restore_rng_state,
    rolling_name,
    rotate_rolling,
    train_state_name,
    tree_map_local_cpu,
    write_trainer_state,
)


def _fake_ckpt(root: Path, name: str, *, with_jepa: bool = True) -> Path:
    p = root / name
    p.mkdir(parents=True)
    write_trainer_state(p, epoch=0, global_step=1)
    (p / "adapter_model.safetensors").write_bytes(b"x")
    if with_jepa:
        (p / "jepa.pt").write_bytes(b"x")
        (p / "ldad.pt").write_bytes(b"x")
    return p


class _CpuDev:
    type = "cpu"


class _FakeTensor:
    device = _CpuDev()

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def detach(self) -> _FakeTensor:
        return self

    def contiguous(self) -> _FakeTensor:
        return self

    def clone(self) -> str:
        return self.tag

    def to(self, _device: object) -> _FakeTensor:
        return self


class _FakeDTensor:
    def __init__(self, tag: str) -> None:
        self._inner = _FakeTensor(tag)

    def full_tensor(self) -> _FakeTensor:
        return self._inner


def test_plain_cpu_state_dict() -> None:
    sd = {"net.0.weight": _FakeDTensor("full"), "meta": 1}
    out = plain_cpu_state_dict(sd)
    assert out["net.0.weight"] == "full"
    assert out["meta"] == 1
    assert plain_cpu_tensor(_FakeTensor("plain")) == "plain"


def test_canonical_lora_key() -> None:
    saved = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    live = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    fsdp = "_fsdp_wrapped_module." + live
    ckpt = (
        "base_model.model.model.layers.0._checkpoint_wrapped_module."
        "linear_attn.in_proj_a.lora_A.weight"
    )
    ckpt_live = "base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.default.weight"
    assert canonical_lora_key(saved) == canonical_lora_key(live) == canonical_lora_key(fsdp)
    assert canonical_lora_key(ckpt) == canonical_lora_key(ckpt_live)


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


def test_train_state_optional(tmp: Path) -> None:
    p = _fake_ckpt(tmp, "checkpoint-e0-s25")
    assert ckpt_complete(p) is True
    assert not (p / train_state_name(0)).exists()
    assert train_state_name(1) == "train_state.rank1.pt"


def test_rng_python_roundtrip() -> None:
    import random

    random.seed(0)
    capture = capture_rng_state()
    first = random.random()
    restore_rng_state(capture)
    second = random.random()
    assert first == second


def test_tree_map_local_cpu() -> None:
    tree = {"a": _FakeTensor("cpu-a"), "b": [1, {"c": _FakeTensor("cpu-c")}]}
    out = tree_map_local_cpu(tree)
    assert out["a"] == "cpu-a"
    assert out["b"][0] == 1
    assert out["b"][1]["c"] == "cpu-c"


def _try_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def test_rng_roundtrip() -> None:
    torch = _try_torch()
    if torch is None:
        return
    torch.manual_seed(0)
    capture = capture_rng_state()
    first = torch.rand(4)
    restore_rng_state(capture)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_optimizer_roundtrip() -> None:
    torch = _try_torch()
    if torch is None:
        return
    from torch import nn

    torch.manual_seed(1)
    m = nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = m(torch.randn(3, 4)).sum()
    loss.backward()
    opt.step()
    blob = tree_map_local_cpu(opt.state_dict())
    m2 = nn.Linear(4, 2)
    m2.load_state_dict(m.state_dict())
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    assert copy_optimizer_state(opt2, blob) is True
    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"])
        assert torch.allclose(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"])


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
    test_plain_cpu_state_dict()
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
    with tempfile.TemporaryDirectory() as d:
        test_train_state_optional(Path(d))
    test_rng_python_roundtrip()
    test_tree_map_local_cpu()
    test_rng_roundtrip()
    test_optimizer_roundtrip()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
