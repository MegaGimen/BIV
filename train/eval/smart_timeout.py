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


def empty_state() -> dict[str, Any]:
    return {
        "n_trials": 0,
        "output_tokens": 0,
        "api_sec": 0.0,
        "v": None,
        "multiplier": 1.0,
        "canonical_tps": CANONICAL_TPS,
        "band": [BAND_LO, BAND_HI],
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
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
    return out


def refresh_from_job(job_dir: Path) -> dict[str, Any] | None:
    """Recompute session tok/s from finished ``result.json`` files.

    Returns the new state when the multiplier or trial count changed, else None.
    """
    path = state_path(job_dir)
    prev = load_state(path)
    n_out = 0
    api_sec = 0.0
    n_trials = 0
    for child in job_dir.iterdir():
        if not child.is_dir():
            continue
        result_path = child / "result.json"
        if not result_path.is_file():
            alt = child / "trial_result.json"
            result_path = alt if alt.is_file() else result_path
        if not result_path.is_file():
            continue
        try:
            trial = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trial, dict):
            continue
        stats = trial_llm_stats(trial)
        if stats is None:
            continue
        out, sec = stats
        n_out += out
        api_sec += sec
        n_trials += 1

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
    }
    if (
        state["n_trials"] == prev.get("n_trials")
        and state["multiplier"] == prev.get("multiplier")
        and state["v"] == prev.get("v")
    ):
        return None
    write_state(path, state)
    return state
