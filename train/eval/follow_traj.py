"""Live-follow Harbor agent trajectory.json (ATIF) while a trial runs."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


def _short(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    t = text.strip()
    if len(t) <= limit:
        return t
    return t[:limit] + f"\n… [{len(t) - limit} more chars]"


def format_step(step: dict[str, Any], *, task: str, path: Path) -> str:
    sid = step.get("step_id")
    src = step.get("source")
    lines = [
        f"\n======== {task}  step={sid}  source={src}  ========",
        f"file: {path}",
    ]
    msg = step.get("message")
    if msg:
        lines.append("--- message ---")
        lines.append(_short(str(msg)))
    for tc in step.get("tool_calls") or []:
        name = tc.get("function_name") or tc.get("name")
        args = tc.get("arguments") or {}
        lines.append(f"--- tool {name} ---")
        if isinstance(args, dict) and "keystrokes" in args:
            lines.append(_short(str(args.get("keystrokes")), 800))
            if "duration" in args:
                lines.append(f"(duration={args['duration']})")
        else:
            lines.append(_short(json.dumps(args, ensure_ascii=False), 800))
    obs = step.get("observation")
    if obs is not None:
        if isinstance(obs, dict):
            body = obs.get("content") or obs.get("output") or obs
            lines.append("--- observation ---")
            lines.append(_short(json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body, 1500))
        else:
            lines.append("--- observation ---")
            lines.append(_short(str(obs), 1500))
    return "\n".join(lines)


def discover_traj_files(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(jobs_dir.rglob("agent/trajectory.json"))


def follow_jobs_dir(
    jobs_dir: Path,
    *,
    stop_event: threading.Event | None = None,
    poll_s: float = 1.5,
    printer: Callable[[str], None] | None = None,
) -> None:
    """Poll until stop_event; print newly appended ATIF steps."""
    emit = printer or (lambda s: print(s, flush=True))
    seen: dict[str, int] = {}
    announced: set[str] = set()
    stop = stop_event or threading.Event()

    emit(f"[follow-traj] watching under {jobs_dir}")
    while not stop.is_set():
        for path in discover_traj_files(jobs_dir):
            key = str(path)
            task = path.parent.parent.name
            if key not in announced:
                announced.add(key)
                emit(f"[follow-traj] found {path}")
                pane = path.parent / "terminus_2.pane"
                if pane.is_file():
                    emit(f"[follow-traj] terminal pane: {pane}")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            steps = data.get("steps") or []
            prev = seen.get(key, 0)
            if len(steps) > prev:
                for step in steps[prev:]:
                    if isinstance(step, dict):
                        emit(format_step(step, task=task, path=path))
                seen[key] = len(steps)
        stop.wait(poll_s)
    emit("[follow-traj] stopped")


def start_follow_thread(jobs_dir: Path, *, poll_s: float = 1.5) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    th = threading.Thread(
        target=follow_jobs_dir,
        kwargs={"jobs_dir": jobs_dir, "stop_event": stop, "poll_s": poll_s},
        name="follow-traj",
        daemon=True,
    )
    th.start()
    return stop, th


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "Usage: python -m eval.follow_traj <jobs_dir|trial_dir|trajectory.json>\n"
            "  Follows **/agent/trajectory.json and prints new steps.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    target = Path(args[0]).resolve()
    if target.name == "trajectory.json":
        jobs_dir = target.parent.parent.parent  # …/job/trial/agent/trajectory.json
    elif (target / "agent" / "trajectory.json").is_file():
        jobs_dir = target.parent
    else:
        jobs_dir = target
    try:
        follow_jobs_dir(jobs_dir)
    except KeyboardInterrupt:
        print("\n[follow-traj] interrupted", flush=True)


if __name__ == "__main__":
    main()
