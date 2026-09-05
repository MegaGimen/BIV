#!/usr/bin/env python3
"""CPU checks: per-trial tok/s sidecars and resume mean."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from eval.smart_timeout import (  # noqa: E402
    DEFAULT_SEED_TPS,
    refresh_from_job,
    trial_toks_path,
)


def _llm_result(out: int, msec: list[float]) -> dict:
    return {
        "task_name": "t",
        "agent_result": {
            "n_output_tokens": out,
            "metadata": {"api_request_times_msec": msec},
        },
    }


def test_empty_job_seeds_42() -> None:
    with tempfile.TemporaryDirectory() as d:
        job = Path(d)
        state = refresh_from_job(job, seed=True)
        assert state is not None
        assert state["v"] == DEFAULT_SEED_TPS
        assert state["n_trials"] == 0
        assert state["multiplier"] == 1.0


def test_mean_of_trial_rates_and_sidecar() -> None:
    with tempfile.TemporaryDirectory() as d:
        job = Path(d)
        a = job / "task-a"
        b = job / "task-b"
        a.mkdir()
        b.mkdir()
        # 100 tok / 2 s = 50 tok/s
        (a / "result.json").write_text(
            json.dumps(_llm_result(100, [2000.0])), encoding="utf-8"
        )
        # 100 tok / 10 s = 10 tok/s
        (b / "result.json").write_text(
            json.dumps(_llm_result(100, [10000.0])), encoding="utf-8"
        )
        state = refresh_from_job(job, seed=True)
        assert state is not None
        assert state["n_trials"] == 2
        assert state["v"] == 30.0
        assert trial_toks_path(a).is_file()
        assert trial_toks_path(b).is_file()
        rec_a = json.loads(trial_toks_path(a).read_text(encoding="utf-8"))
        assert rec_a["tok_s"] == 50.0


def test_resume_keeps_mean_when_new_trial_arrives() -> None:
    with tempfile.TemporaryDirectory() as d:
        job = Path(d)
        a = job / "old"
        a.mkdir()
        (a / "result.json").write_text(
            json.dumps(_llm_result(100, [2000.0])), encoding="utf-8"
        )
        seen: set[str] = set()
        seeded = refresh_from_job(job, seen=seen, seed=True)
        assert seeded is not None
        assert seeded["v"] == 50.0
        b = job / "new"
        b.mkdir()
        (b / "result.json").write_text(
            json.dumps(_llm_result(100, [10000.0])), encoding="utf-8"
        )
        nxt = refresh_from_job(job, seen=seen)
        assert nxt is not None
        assert nxt["v"] == 30.0
        assert nxt["n_trials"] == 2
        assert any(ev.get("trial") == "new" for ev in nxt["events"])


if __name__ == "__main__":
    test_empty_job_seeds_42()
    test_mean_of_trial_rates_and_sidecar()
    test_resume_keeps_mean_when_new_trial_arrives()
    print("ok")
