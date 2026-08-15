#!/usr/bin/env python3
"""Backfill Trainer log_history from the latest Muse checkpoint into TensorBoard.

Picks the newest complete checkpoint under ``output_dir`` (same ranking as
``train_daemon.sh`` / ``train_muse_trl._find_latest_ckpt``), reads
``trainer_state.json`` → ``log_history``, and writes scalars as one TB run.

Examples:
  # Auto: resolve out_dir from config + --max-length/--choice, latest ckpt → /root/tf-logs
  python scripts/tmplog.py --max-length 65536 --choice 1

  python scripts/tmplog.py --output-dir outputs/muse_glimmer_wm_mix_ml65536_c1
  python scripts/tmplog.py --ckpt outputs/.../checkpoint-e0-s1725 --log-dir /root/tf-logs

Env:
  LOGGING_DIR / TF_LOGS   default log dir (else /root/tf-logs)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"

_ROLLING_CKPT_RE = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
_EPOCH_END_CKPT_RE = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
_HF_DIGIT_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

_SKIP_KEYS = frozenset({"epoch", "step", "total_flos"})


def _resolve(raw: str | Path) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _resolve_out_dir(*, config: Path, max_length: int, choice: int) -> Path:
    import yaml

    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    base = "outputs/muse_glimmer_wm_mix"
    try:
        cfg = yaml.safe_load(text) or {}
        base = str((cfg.get("train") or {}).get("output_dir") or base)
    except Exception:
        m = re.search(r"(?m)^\s*output_dir:\s*(\S+)", text)
        if m:
            base = m.group(1).strip().strip("'\"")
    out = f"{base}_ml{int(max_length)}_c{int(choice)}"
    return _resolve(out)


def _find_latest_ckpt(out_dir: Path) -> Path | None:
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


def _scalar_tag(key: str) -> str:
    if key == "loss":
        return "train/loss"
    if key.startswith("eval_"):
        return "eval/" + key[len("eval_") :]
    if key.startswith("train_"):
        return "train/" + key[len("train_") :]
    return f"train/{key}"


def _write_tb(hist: list[dict[str, Any]], *, log_dir: Path, run_name: str) -> int:
    from torch.utils.tensorboard import SummaryWriter

    run_dir = log_dir / run_name
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    n = 0
    for row in hist:
        step = row.get("step")
        if step is None:
            continue
        step_i = int(step)
        for k, v in row.items():
            if k in _SKIP_KEYS or k == "step":
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            writer.add_scalar(_scalar_tag(k), float(v), step_i)
            n += 1
    writer.flush()
    writer.close()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--choice", type=int, default=1)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Train output_dir containing checkpoint-* (overrides max-length/choice).",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Explicit checkpoint dir (skip auto latest).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard root (default: LOGGING_DIR / TF_LOGS / /root/tf-logs).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="TB run subdirectory name (default: backfill_<ckpt_name>).",
    )
    args = p.parse_args()

    raw_log = (
        str(args.log_dir)
        if args.log_dir is not None
        else (os.environ.get("LOGGING_DIR") or os.environ.get("TF_LOGS") or "/root/tf-logs")
    )
    log_root = Path(raw_log)
    if not log_root.is_absolute():
        log_root = _resolve(log_root)

    if args.ckpt is not None:
        ckpt = _resolve(args.ckpt)
    else:
        if args.output_dir is not None:
            out_dir = _resolve(args.output_dir)
        else:
            if args.max_length is None:
                raise SystemExit("Need --output-dir, or --max-length [--choice], or --ckpt")
            out_dir = _resolve_out_dir(
                config=_resolve(args.config),
                max_length=int(args.max_length),
                choice=int(args.choice),
            )
        print(f"[tmplog] output_dir={out_dir}", flush=True)
        ckpt = _find_latest_ckpt(out_dir)
        if ckpt is None:
            raise SystemExit(f"No complete checkpoint with trainer_state.json under {out_dir}")

    state_path = ckpt / "trainer_state.json"
    if not state_path.is_file():
        raise SystemExit(f"Missing {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    hist = state.get("log_history") or []
    if not isinstance(hist, list) or not hist:
        raise SystemExit(f"Empty log_history in {state_path}")

    run_name = args.run_name or f"backfill_{ckpt.name}"
    print(f"[tmplog] ckpt={ckpt}", flush=True)
    print(f"[tmplog] log_history rows={len(hist)} global_step={state.get('global_step')}", flush=True)
    print(f"[tmplog] → {log_root / run_name}", flush=True)

    n = _write_tb(hist, log_dir=log_root, run_name=run_name)
    print(f"[tmplog] wrote {n} scalars (run={run_name})", flush=True)
    print(f"[tmplog] tensorboard --logdir {log_root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
