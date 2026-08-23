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


class AgentEvalTbSession:
    """Open one TB run for a Harbor ``test.py`` session.

    * **Live** (while trials finish): ``eval_agent_live/<suite>/…`` with
      x-axis = completed trials (updates as Harbor writes ``trial_result.json``).
    * **Final** (per suite / end): ``eval_agent/<suite>/…`` at training ckpt step
      (cross-arm comparison on the train x-axis).
    """

    def __init__(
        self,
        *,
        meta: dict[str, Any],
        arm: str,
        ckpt: Path | None = None,
        step: int | None = None,
        log_root: Path | None = None,
        run_name: str | None = None,
        suites_meta: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.meta = meta if isinstance(meta, dict) else {}
        self.suites_meta = suites_meta or {}
        self.step_i = step_from_arm(arm, ckpt, step)
        safe_arm = re.sub(r"[^\w.\-]+", "_", arm)[:80]
        prefix = run_name or f"eval_agent_{safe_arm}_s{self.step_i}"
        self.run_dir = next_run_dir(log_root or default_log_root(), prefix=prefix)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self._live_last_n: dict[str, int] = {}
        self._final_rows: list[dict[str, Any]] = []
        print(
            f"[eval-tb] live session → {self.run_dir} (final x=ckpt step {self.step_i})",
            flush=True,
        )

    def log_live(self, suite: str, scores: dict[str, Any]) -> None:
        """Flush running pass-rate; x-axis = ``n_trials`` completed so far."""
        n_trials = scores.get("n_trials")
        score = scores.get("score_percent")
        if not isinstance(n_trials, int) or n_trials <= 0:
            return
        if self._live_last_n.get(suite) == n_trials:
            return
        self._live_last_n[suite] = n_trials
        if isinstance(score, (int, float)):
            self.writer.add_scalar(
                f"eval_agent_live/{suite}/score_percent", float(score), n_trials
            )
        self.writer.add_scalar(
            f"eval_agent_live/{suite}/n_trials", float(n_trials), n_trials
        )
        self.writer.flush()
        print(
            f"[eval-tb] live {suite}: score%={score} n_trials={n_trials}",
            flush=True,
        )

    def log_suite_final(self, row: dict[str, Any]) -> None:
        """Write final suite scalars at training ckpt step (may call mid-session)."""
        suite = str(row.get("suite") or "")
        if not suite:
            return
        self._final_rows.append(row)
        ref = self.meta.get("muse_glimmer_30b_high_reasoning") or {}
        step_i = self.step_i
        score = row.get("score_percent")
        if isinstance(score, (int, float)):
            self.writer.add_scalar(
                f"eval_agent/{suite}/score_percent", float(score), step_i
            )
        n_trials = row.get("n_trials")
        if isinstance(n_trials, (int, float)):
            self.writer.add_scalar(
                f"eval_agent/{suite}/n_trials", float(n_trials), step_i
            )
        rc = row.get("returncode")
        if isinstance(rc, (int, float)):
            self.writer.add_scalar(
                f"eval_agent/{suite}/returncode", float(rc), step_i
            )
        meta_key = (self.suites_meta.get(suite) or {}).get("meta_key") or suite
        mscore = (ref.get(meta_key) or {}).get("score")
        if isinstance(score, (int, float)) and isinstance(mscore, (int, float)):
            delta = float(score) - float(mscore)
            self.writer.add_scalar(f"eval_agent/{suite}/delta_vs_meta", delta, step_i)
            self.writer.add_scalar(
                f"eval_agent/{suite}/meta_score", float(mscore), step_i
            )
        self.writer.flush()

    def finalize(self) -> Path:
        scores = [
            float(r["score_percent"])
            for r in self._final_rows
            if isinstance(r.get("score_percent"), (int, float))
        ]
        if scores:
            self.writer.add_scalar(
                "eval_agent/mean_score_percent",
                sum(scores) / len(scores),
                self.step_i,
            )
        self.writer.flush()
        self.writer.close()
        print(
            f"[eval-tb] closed → {self.run_dir} "
            f"(suites={len(self._final_rows)}, step={self.step_i})",
            flush=True,
        )
        return self.run_dir


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
    """One-shot write (compat). Prefer ``AgentEvalTbSession`` for live updates."""
    sess = AgentEvalTbSession(
        meta=meta,
        arm=arm,
        ckpt=ckpt,
        step=step,
        log_root=log_root,
        run_name=run_name,
        suites_meta=suites_meta,
    )
    for r in rows:
        sess.log_suite_final(r)
    return sess.finalize()
