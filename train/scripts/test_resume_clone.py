#!/usr/bin/env python3
"""CPU checks: --resume copies Harbor jobs into a new stamp without mutating src."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from eval.run_harbor import clone_resume_jobs  # noqa: E402


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n", encoding="utf-8")


def test_clone_retargets_jobs_dir_and_leaves_src() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src_stamp = root / "old_stamp"
        src_job = src_stamp / "qwen-act_terminal_bench_2_1"
        trial = src_job / "caffe-cifar-10__abc"
        _write(
            src_job / "config.json",
            {
                "job_name": "qwen-act_terminal_bench_2_1",
                "jobs_dir": str(src_stamp),
                "n_attempts": 3,
            },
        )
        _write(
            trial / "config.json",
            {
                "trial_name": "caffe-cifar-10__abc",
                "trials_dir": str(src_job),
            },
        )
        _write(trial / "result.json", {"exception_info": {"exception_type": "BuildException"}})
        marker = trial / "keep.txt"
        marker.write_text("infra-fail\n", encoding="utf-8")

        dest_stamp = root / "new_stamp"
        cloned = clone_resume_jobs([src_job], dest_stamp)
        assert len(cloned) == 1
        dest_job = cloned[0]
        assert dest_job == dest_stamp / src_job.name

        src_cfg = json.loads((src_job / "config.json").read_text(encoding="utf-8"))
        assert src_cfg["jobs_dir"] == str(src_stamp)
        src_trial = json.loads((trial / "config.json").read_text(encoding="utf-8"))
        assert src_trial["trials_dir"] == str(src_job)
        assert marker.read_text(encoding="utf-8") == "infra-fail\n"

        dest_cfg = json.loads((dest_job / "config.json").read_text(encoding="utf-8"))
        assert dest_cfg["jobs_dir"] == str(dest_stamp.resolve())
        dest_trial = json.loads(
            (dest_job / "caffe-cifar-10__abc" / "config.json").read_text(encoding="utf-8")
        )
        assert dest_trial["trials_dir"] == str(dest_job.resolve())
        copied = dest_job / "caffe-cifar-10__abc" / "keep.txt"
        assert copied.read_text(encoding="utf-8") == "infra-fail\n"
        copied.write_text("mutated-copy\n", encoding="utf-8")
        assert marker.read_text(encoding="utf-8") == "infra-fail\n"

        meta = json.loads((dest_stamp / "resume_from.json").read_text(encoding="utf-8"))
        assert meta["jobs"][0]["src"] == str(src_job.resolve())
        assert meta["jobs"][0]["dest"] == str(dest_job.resolve())


if __name__ == "__main__":
    test_clone_retargets_jobs_dir_and_leaves_src()
    print("ok")
