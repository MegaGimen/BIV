"""Checkpoint names/rotation matching Muse Glimmer (train_muse_trl.py).

Rolling mid-run: ``checkpoint-e{epoch}-s{step}`` (keep newest ``save_total_limit``).
Epoch-end permanent: ``checkpoint-epoch{N}-end-s{step}`` (never rotated).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

ROLLING_RE = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
EPOCH_END_RE = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
HF_DIGIT_RE = re.compile(r"^checkpoint-(\d+)$")
LEGACY_STEP_RE = re.compile(r"^step-(\d+)$")


def rolling_name(epoch: int, step: int) -> str:
    return f"checkpoint-e{int(epoch)}-s{int(step)}"


def epoch_end_name(epoch: int, step: int) -> str:
    """``epoch`` is 1-based completed-epoch index, same as Muse ``round(state.epoch)``."""
    return f"checkpoint-epoch{int(epoch)}-end-s{int(step)}"


def parse_ckpt_name(name: str) -> tuple[int, int, int] | None:
    """``(epoch, step, kind)`` with kind 0=digit/legacy, 1=rolling, 2=epoch-end."""
    m = EPOCH_END_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), 2
    m = ROLLING_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), 1
    m = HF_DIGIT_RE.match(name)
    if m:
        return 0, int(m.group(1)), 0
    m = LEGACY_STEP_RE.match(name)
    if m:
        return 0, int(m.group(1)), 0
    return None


def canonical_lora_key(name: str) -> str:
    """Strip FSDP prefixes and PEFT ``.default.`` so save keys match live names."""
    out = name
    for pfx in ("_fsdp_wrapped_module.", "module."):
        while out.startswith(pfx):
            out = out[len(pfx) :]
    for src, dst in (
        (".lora_A.default.", ".lora_A."),
        (".lora_B.default.", ".lora_B."),
        (".lora_embedding_A.default.", ".lora_embedding_A."),
        (".lora_embedding_B.default.", ".lora_embedding_B."),
    ):
        out = out.replace(src, dst)
    return out


def ckpt_complete(path: Path, *, require_jepa: bool = True) -> bool:
    """trainer_state.json + weights. Mediated Stage 1 also needs jepa.pt and ldad.pt."""
    if not path.is_dir():
        return False
    if not (path / "trainer_state.json").is_file():
        return False
    has_weights = (
        (path / "adapter_model.safetensors").is_file()
        or (path / "pytorch_model_fsdp.bin").is_file()
        or any(path.glob("*.safetensors"))
    )
    if not has_weights:
        return False
    if require_jepa and (
        not (path / "jepa.pt").is_file() or not (path / "ldad.pt").is_file()
    ):
        return False
    return True


def find_latest_ckpt(out_dir: Path, *, require_jepa: bool = True) -> Path | None:
    """Newest complete ckpt. Rank key ``(epoch, step, kind)`` — epoch first, then step.

    Same as Muse / daemon. Rolling ``checkpoint-e{epoch}-s{step}`` uses the 0-based
    epoch index from the training loop; epoch-end uses a 1-based completed epoch,
    so it ranks above that epoch's rolling dirs.
    """
    if not out_dir.is_dir():
        return None
    best: tuple[int, int, int, Path] | None = None
    for p in out_dir.iterdir():
        parsed = parse_ckpt_name(p.name)
        if parsed is None or not ckpt_complete(p, require_jepa=require_jepa):
            continue
        epoch, step, kind = parsed
        key = (epoch, step, kind)
        if best is None or key > (best[0], best[1], best[2]):
            best = (epoch, step, kind, p)
    return None if best is None else best[3]


def rotate_rolling(out: Path, limit: int, *, log: Callable[[str], None] | None = None) -> list[str]:
    """Delete oldest rolling (and leftover digit/step-*) dirs past ``limit``. Epoch-end kept."""
    if limit is None or int(limit) <= 0 or not out.is_dir():
        return []
    rolling: list[tuple[int, Path]] = []
    for p in out.iterdir():
        if not p.is_dir():
            continue
        m = ROLLING_RE.match(p.name)
        if m:
            rolling.append((int(m.group(2)), p))
            continue
        m2 = HF_DIGIT_RE.match(p.name) or LEGACY_STEP_RE.match(p.name)
        if m2:
            rolling.append((int(m2.group(1)), p))
    rolling.sort(key=lambda t: t[0])
    removed: list[str] = []
    while len(rolling) > int(limit):
        _, victim = rolling.pop(0)
        removed.append(victim.name)
        if log is not None:
            log(f"rotate: remove {victim.name}")
        shutil.rmtree(victim, ignore_errors=True)
    return removed


def write_trainer_state(path: Path, *, epoch: int, global_step: int, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    if extra:
        payload.update(extra)
    (path / "trainer_state.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
