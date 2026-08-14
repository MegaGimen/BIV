"""Checkpoint pick helpers (same ranking as train_daemon.sh / train_muse_trl)."""

from __future__ import annotations

import re
from pathlib import Path

_ROLLING_CKPT_RE = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
_EPOCH_END_CKPT_RE = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
_HF_DIGIT_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_ckpt(out_dir: Path) -> Path | None:
    """Newest complete ckpt under ``out_dir``.

    Rank key ``(epoch, step, kind)`` — epoch dominates step:
      checkpoint-{step}                    → epoch 0, kind 0
      checkpoint-e{epoch}-s{step}          → kind 1
      checkpoint-epoch{epoch}-end-s{step}  → kind 2
    """
    if not out_dir.is_dir():
        return None
    best: tuple[int, int, int, Path] | None = None
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        epoch = step = None
        kind = 0
        m = _EPOCH_END_CKPT_RE.match(name)
        if m:
            epoch, step, kind = int(m.group(1)), int(m.group(2)), 2
        else:
            m = _ROLLING_CKPT_RE.match(name)
            if m:
                epoch, step, kind = int(m.group(1)), int(m.group(2)), 1
            else:
                m = _HF_DIGIT_CKPT_RE.match(name)
                if m:
                    epoch, step, kind = 0, int(m.group(1)), 0
        if epoch is None or step is None:
            continue
        if not (p / "trainer_state.json").is_file():
            continue
        if not (
            (p / "adapter_model.safetensors").is_file()
            or (p / "pytorch_model_fsdp.bin").is_file()
            or any(p.glob("*.safetensors"))
        ):
            continue
        key = (epoch, step, kind)
        if best is None or key > (best[0], best[1], best[2]):
            best = (epoch, step, kind, p)
    return None if best is None else best[3]


def resolve_ckpt(
    ckpt: str | Path | None,
    *,
    search_dir: Path | None = None,
) -> Path | None:
    """Resolve ``--ckpt``: path, or ``auto`` under ``search_dir``."""
    if ckpt is None or str(ckpt).strip() in {"", "null", "None"}:
        return None
    raw = str(ckpt).strip()
    if raw.lower() == "auto":
        if search_dir is None:
            raise SystemExit("--ckpt auto requires --ckpt-search-dir (train output_dir)")
        picked = find_latest_ckpt(search_dir)
        if picked is None:
            raise SystemExit(f"--ckpt auto: no complete checkpoint under {search_dir}")
        return picked
    path = Path(raw)
    if not path.is_absolute():
        # Prefer cwd-relative; callers often pass train-relative paths.
        path = path.resolve()
    if not path.is_dir():
        raise SystemExit(f"--ckpt must be an existing checkpoint directory: {path}")
    return path
