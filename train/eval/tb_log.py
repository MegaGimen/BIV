"""TensorBoard helpers for post-train agent / WM eval."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_STEP_RE = re.compile(r"(?:^|[-_])s(\d+)(?:$|[-_])", re.I)
_DIGIT_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def default_log_root() -> Path:
    raw = os.environ.get("LOGGING_DIR") or os.environ.get("TF_LOGS") or "/root/tf-logs"
    p = Path(raw)
    return p if p.is_absolute() else (Path.cwd() / p)


def step_from_arm(arm: str, ckpt: Path | None = None, step: int | None = None) -> int:
    """Use explicit step, else parse from arm/ckpt name for TB x-axis."""
    if step is not None:
        return int(step)
    name = ckpt.name if ckpt is not None else arm
    m = _STEP_RE.search(name)
    if m:
        return int(m.group(1))
    m = _DIGIT_CKPT_RE.match(name)
    if m:
        return int(m.group(1))
    return 0


def next_run_dir(log_root: Path, *, prefix: str) -> Path:
    """``{n}_{prefix}`` under log_root (same ordinal style as Muse train TB)."""
    log_root.mkdir(parents=True, exist_ok=True)
    n = sum(1 for p in log_root.iterdir() if p.is_dir()) + 1
    run = log_root / f"{n}_{prefix}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_agent_eval_tb(
    rows: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    arm: str,
    ckpt: Path | None = None,
    step: int | None = None,
    log_root: Path | None = None,
    run_name: str | None = None,
    suites_meta: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write Harbor suite scores as TensorBoard scalars. Returns run dir."""
    from torch.utils.tensorboard import SummaryWriter

    root = log_root or default_log_root()
    step_i = step_from_arm(arm, ckpt, step)
    safe_arm = re.sub(r"[^\w.\-]+", "_", arm)[:80]
    prefix = run_name or f"eval_agent_{safe_arm}_s{step_i}"
    run_dir = next_run_dir(root, prefix=prefix)

    writer = SummaryWriter(log_dir=str(run_dir))
    ref = (meta.get("muse_glimmer_30b_high_reasoning") or {}) if isinstance(meta, dict) else {}
    suites_meta = suites_meta or {}

    n = 0
    for r in rows:
        suite = str(r.get("suite") or "")
        if not suite:
            continue
        score = r.get("score_percent")
        if isinstance(score, (int, float)):
            writer.add_scalar(f"eval_agent/{suite}/score_percent", float(score), step_i)
            n += 1
        n_trials = r.get("n_trials")
        if isinstance(n_trials, (int, float)):
            writer.add_scalar(f"eval_agent/{suite}/n_trials", float(n_trials), step_i)
            n += 1
        rc = r.get("returncode")
        if isinstance(rc, (int, float)):
            writer.add_scalar(f"eval_agent/{suite}/returncode", float(rc), step_i)
            n += 1

        meta_key = (suites_meta.get(suite) or {}).get("meta_key") or suite
        mscore = (ref.get(meta_key) or {}).get("score")
        if isinstance(score, (int, float)) and isinstance(mscore, (int, float)):
            delta = float(score) - float(mscore)
            writer.add_scalar(f"eval_agent/{suite}/delta_vs_meta", delta, step_i)
            writer.add_scalar(f"eval_agent/{suite}/meta_score", float(mscore), step_i)
            n += 2

    # One scalar that TB can chart across arms at the same step.
    scores = [
        float(r["score_percent"])
        for r in rows
        if isinstance(r.get("score_percent"), (int, float))
    ]
    if scores:
        writer.add_scalar("eval_agent/mean_score_percent", sum(scores) / len(scores), step_i)
        n += 1

    writer.flush()
    writer.close()
    print(
        f"[eval-tb] wrote {n} scalars → {run_dir} (step={step_i})",
        flush=True,
    )
    return run_dir
