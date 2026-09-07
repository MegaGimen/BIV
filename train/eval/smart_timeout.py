"""Scale Harbor agent timeout from measured LLM tok/s vs a 40 tok/s canonical.

Each finished LLM trial writes ``biv_toks.json`` next to ``result.json``.
Session speed is the unweighted mean of those per-trial rates. Resume
recomputes from every sidecar (and backfills missing ones from
``result.json``). With no LLM trial yet, session speed is 42 tok/s.

Original ``task.toml`` ``[agent] timeout_sec`` is the budget at ~40 tok/s.
If the session rate sits in 30–50 tok/s, multiplier stays 1. Outside that
band, multiplier is ``40 / v``. Already-running trials keep the timeout
they started with; new trials read the JSON this module writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_TPS = 40.0
DEFAULT_SEED_TPS = 42.0
BAND_LO = 30.0
BAND_HI = 50.0
MULT_MIN = 0.25
MULT_MAX = 10.0

STATE_NAME = "biv_timeout.json"
TRIAL_TOKS_NAME = "biv_toks.json"


def state_path(job_dir: Path) -> Path:
    return job_dir / STATE_NAME


def trial_toks_path(trial_dir: Path) -> Path:
    return trial_dir / TRIAL_TOKS_NAME


def multiplier_from_tps(v: float) -> float:
    if v <= 0:
        return 1.0
    if BAND_LO <= v <= BAND_HI:
        return 1.0
    return max(MULT_MIN, min(MULT_MAX, CANONICAL_TPS / v))


def multiplier_reason(v: float | None, mult: float, *, n_trials: int) -> str:
    if n_trials <= 0 or v is None:
        return (
            f"no LLM timings yet, seed {DEFAULT_SEED_TPS:g} tok/s "
            f"(×{mult:g})"
        )
    if BAND_LO <= v <= BAND_HI:
        return (
            f"mean {v:.3f} tok/s over {n_trials} trial(s) "
            f"in [{BAND_LO:g}, {BAND_HI:g}], keep task.toml (×1)"
        )
    return (
        f"mean {v:.3f} tok/s over {n_trials} trial(s) "
        f"outside [{BAND_LO:g}, {BAND_HI:g}], "
        f"× = {CANONICAL_TPS:g}/{v:.3f} = {mult:g}  "
        f"(later trials get that scale)"
    )


def trial_llm_stats(trial: dict[str, Any]) -> tuple[int, float] | None:
    """Return (output_tokens, api_seconds) when the trial called the LLM."""
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


def trial_toks_record(
    trial_dir: Path,
    trial: dict[str, Any],
) -> dict[str, Any] | None:
    """Per-trial tok/s sidecar. Write/refresh ``biv_toks.json`` from result."""
    stats = trial_llm_stats(trial)
    if stats is None:
        return None
    out, sec = stats
    rec = {
        "trial": trial_dir.name,
        "task": trial.get("task_name") or trial_dir.name,
        "tok_s": round(out / sec, 3),
        "output_tokens": out,
        "api_sec": round(sec, 3),
    }
    path = trial_toks_path(trial_dir)
    try:
        path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return rec


def empty_state() -> dict[str, Any]:
    return {
        "n_trials": 0,
        "output_tokens": 0,
        "api_sec": 0.0,
        "v": DEFAULT_SEED_TPS,
        "multiplier": 1.0,
        "canonical_tps": CANONICAL_TPS,
        "seed_default": DEFAULT_SEED_TPS,
        "band": [BAND_LO, BAND_HI],
        "trials": [],
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


def mean_trial_tok_s(records: list[dict[str, Any]]) -> float | None:
    rates = [float(r["tok_s"]) for r in records if r.get("tok_s")]
    if not rates:
        return None
    return sum(rates) / len(rates)


def refresh_from_job(
    job_dir: Path,
    *,
    seen: set[str] | None = None,
    seed: bool = False,
) -> dict[str, Any] | None:
    """Recompute mean tok/s from per-trial sidecars / ``result.json``.

    ``seen`` tracks result-file paths already announced. ``seed=True`` fills
    ``seen`` from existing files without emitting events (resume start).
    """
    path = state_path(job_dir)
    prev = load_state(path)
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    # Fresh Harbor jobs: run_spec only mkdir's jobs_dir; Harbor creates
    # job_dir after spawn. Seed must not crash on the empty path.
    children = (
        sorted(job_dir.iterdir(), key=lambda p: p.name) if job_dir.is_dir() else []
    )
    for child in children:
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

        rec = trial_toks_record(child, trial)
        task = trial.get("task_name") or child.name
        if rec is not None:
            records.append(rec)
            if is_new and not seed:
                events.append({"kind": "llm", **rec})
        elif is_new and not seed:
            events.append(
                {
                    "kind": "infra",
                    "trial": child.name,
                    "task": task,
                    "exception": _exception_type(trial),
                }
            )

    n_trials = len(records)
    n_out = sum(int(r.get("output_tokens") or 0) for r in records)
    api_sec = sum(float(r.get("api_sec") or 0.0) for r in records)
    mean = mean_trial_tok_s(records)
    v = DEFAULT_SEED_TPS if mean is None else mean
    mult = multiplier_from_tps(v)
    state = {
        "n_trials": n_trials,
        "output_tokens": n_out,
        "api_sec": round(api_sec, 3),
        "v": round(v, 3),
        "multiplier": round(mult, 4),
        "canonical_tps": CANONICAL_TPS,
        "seed_default": DEFAULT_SEED_TPS,
        "band": [BAND_LO, BAND_HI],
        "reason": multiplier_reason(mean, round(mult, 4), n_trials=n_trials),
        "trials": records,
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
