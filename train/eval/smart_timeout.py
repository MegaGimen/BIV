"""Scale Harbor agent timeout from measured LLM tok/s vs a 40 tok/s canonical.

Original ``task.toml`` ``[agent] timeout_sec`` is the budget at ~40 tok/s.
After each finished trial we accumulate output tokens / API wall time. If the
session rate sits in 30–50 tok/s, multiplier stays 1. Outside that band,
multiplier is ``40 / v`` so a slower model gets more wall clock and a faster
one gets less. Already-running trials keep the timeout they started with;
new trials read the JSON this module writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_TPS = 40.0
BAND_LO = 30.0
BAND_HI = 50.0
MULT_MIN = 0.25
MULT_MAX = 10.0

STATE_NAME = "biv_timeout.json"


def state_path(job_dir: Path) -> Path:
    return job_dir / STATE_NAME


def multiplier_from_tps(v: float) -> float:
    if v <= 0:
        return 1.0
    if BAND_LO <= v <= BAND_HI:
        return 1.0
    return max(MULT_MIN, min(MULT_MAX, CANONICAL_TPS / v))


def multiplier_reason(v: float | None, mult: float) -> str:
    if v is None:
        return "no LLM timings yet, keep task.toml (×1)"
    if BAND_LO <= v <= BAND_HI:
        return f"session {v:.3f} tok/s in [{BAND_LO:g}, {BAND_HI:g}], keep task.toml (×1)"
    return (
        f"session {v:.3f} tok/s outside [{BAND_LO:g}, {BAND_HI:g}], "
        f"× = {CANONICAL_TPS:g}/{v:.3f} = {mult:g}  "
        f"(later trials get that scale)"
    )


def trial_llm_stats(trial: dict[str, Any]) -> tuple[int, float] | None:
    """Return (output_tokens, api_seconds) or None if the trial never called the LLM."""
    ar = trial.get("agent_result")
    if not isinstance(ar, dict):
        return None
    out = int(ar.get("n_output_tokens") or 0)
    meta = ar.get("metadata") if isinstance(ar.get("metadata"), dict) else {}
    times = meta.get("api_request_times_msec") or []
    if out <= 0 or not times:
        return None
    sec = sum(float(t) for t in times) / 1000.0
    if sec <= 0:
        return None
    return out, sec


def _exception_type(trial: dict[str, Any]) -> str | None:
    ei = trial.get("exception_info")
    if not isinstance(ei, dict):
        return None
    raw = ei.get("type") or ei.get("exception_type")
    return str(raw) if raw else None


def _trial_result_file(child: Path) -> Path | None:
    for name in ("result.json", "trial_result.json"):
        p = child / name
        if p.is_file():
            return p
    return None


def empty_state() -> dict[str, Any]:
    return {
        "n_trials": 0,
        "output_tokens": 0,
        "api_sec": 0.0,
        "v": None,
        "multiplier": 1.0,
        "canonical_tps": CANONICAL_TPS,
        "band": [BAND_LO, BAND_HI],
        "events": [],
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in state.items() if k != "events"}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_state()
    if not isinstance(raw, dict):
        return empty_state()
    out = empty_state()
    out.update(raw)
    out["events"] = []
    return out


def refresh_from_job(
    job_dir: Path,
    *,
    seen: set[str] | None = None,
    seed: bool = False,
) -> dict[str, Any] | None:
    """Recompute session tok/s from finished ``result.json`` files.

    ``seen`` tracks result-file paths already announced. ``seed=True`` fills
    ``seen`` from existing files without emitting events (resume start).
    Returns state when the session rate changes or new trials appear.
    """
    path = state_path(job_dir)
    prev = load_state(path)
    n_out = 0
    api_sec = 0.0
    n_trials = 0
    events: list[dict[str, Any]] = []

    for child in sorted(job_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        result_path = _trial_result_file(child)
        if result_path is None:
            continue
        try:
            trial = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trial, dict):
            continue
        key = str(result_path)
        is_new = seen is not None and key not in seen
        if seen is not None:
            seen.add(key)

        stats = trial_llm_stats(trial)
        task = trial.get("task_name") or child.name
        if stats is not None:
            out, sec = stats
            n_out += out
            api_sec += sec
            n_trials += 1
            if is_new and not seed:
                events.append(
                    {
                        "kind": "llm",
                        "trial": child.name,
                        "task": task,
                        "tok_s": round(out / sec, 3),
                        "output_tokens": out,
                        "api_sec": round(sec, 3),
                    }
                )
        elif is_new and not seed:
            events.append(
                {
                    "kind": "infra",
                    "trial": child.name,
                    "task": task,
                    "exception": _exception_type(trial),
                }
            )

    v: float | None = (n_out / api_sec) if api_sec > 0 and n_out > 0 else None
    mult = 1.0 if v is None else multiplier_from_tps(v)
    state = {
        "n_trials": n_trials,
        "output_tokens": n_out,
        "api_sec": round(api_sec, 3),
        "v": None if v is None else round(v, 3),
        "multiplier": round(mult, 4),
        "canonical_tps": CANONICAL_TPS,
        "band": [BAND_LO, BAND_HI],
        "reason": multiplier_reason(v, round(mult, 4)),
        "events": events,
    }
    changed = (
        state["n_trials"] != prev.get("n_trials")
        or state["multiplier"] != prev.get("multiplier")
        or state["v"] != prev.get("v")
    )
    if changed:
        write_state(path, state)
    if seed:
        write_state(path, state)
        state["events"] = []
        return state
    if not changed and not events:
        return None
    return state
