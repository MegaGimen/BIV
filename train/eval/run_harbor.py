"""Harbor job builders for Meta-aligned Muse agent benches."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = EVAL_ROOT.parent
META_REF_PATH = EVAL_ROOT / "meta_reference.json"

# Suite id → Harbor dataset + agent (three-way align).
SUITES: dict[str, dict[str, Any]] = {
    "terminal_bench_2_1": {
        "dataset": "terminal-bench/terminal-bench-2-1",
        "agent": "terminus-2",
        "meta_key": "terminal_bench_2_1",
        "default_k": 3,
        "uses_terminus_kwargs": True,
    },
    "swe_bench_verified": {
        "dataset": "swe-bench/swe-bench-verified",
        "agent": "mini-swe-agent",
        "meta_key": "swe_bench_verified",
        "default_k": 4,
        "uses_terminus_kwargs": False,
    },
    "swe_bench_pro": {
        "dataset": "scale-ai/swe-bench-pro",
        "agent": "mini-swe-agent",
        "meta_key": "swe_bench_pro",
        "default_k": 4,
        "uses_terminus_kwargs": False,
    },
}

DEFAULT_SUITES = tuple(SUITES.keys())


def load_meta_reference() -> dict[str, Any]:
    return json.loads(META_REF_PATH.read_text(encoding="utf-8"))


def harbor_bin() -> str:
    """Prefer train/.venv-eval harbor, else PATH."""
    local = TRAIN_ROOT / ".venv-eval" / "bin" / "harbor"
    if local.is_file():
        return str(local)
    which = shutil.which("harbor")
    if which:
        return which
    raise SystemExit(
        "harbor not found. Create train/.venv-eval (Python ≥3.12) and:\n"
        "  uv pip install --python .venv-eval/bin/python -r requirements-eval.txt"
    )


def normalize_model_name(model: str) -> str:
    """LiteLLM/Harbor expect provider/model; default openai/ for local OpenAI-compat."""
    m = model.strip()
    if "/" not in m:
        return f"openai/{m}"
    return m


@dataclass
class HarborRunSpec:
    suite: str
    dataset: str
    agent: str
    model: str
    base_url: str | None
    api_key: str
    env: str
    n_attempts: int
    n_concurrent: int
    jobs_dir: Path
    job_name: str
    temperature: float
    reasoning_effort: str
    top_p: float
    top_k: int
    include_task_names: list[str] = field(default_factory=list)
    n_tasks: int | None = None
    timeout_multiplier: float = 1.0
    debug: bool = False
    raw_trajectory: bool = False

    def build_cmd(self) -> list[str]:
        cmd = [
            harbor_bin(),
            "run",
            "-d",
            self.dataset,
            "-a",
            self.agent,
            "-m",
            self.model,
            "-e",
            self.env,
            "-k",
            str(self.n_attempts),
            "-n",
            str(self.n_concurrent),
            "-o",
            str(self.jobs_dir),
            "--job-name",
            self.job_name,
            "--timeout-multiplier",
            str(self.timeout_multiplier),
            "-y",
        ]
        if self.debug:
            cmd.append("--debug")
        for name in self.include_task_names:
            cmd.extend(["-i", name])
        if self.n_tasks is not None:
            cmd.extend(["-l", str(self.n_tasks)])

        if self.base_url:
            base = self.base_url.rstrip("/")
            cmd.extend(["--ae", f"OPENAI_BASE_URL={base}"])
            cmd.extend(["--ae", f"OPENAI_API_BASE={base}"])
        cmd.extend(["--ae", f"OPENAI_API_KEY={self.api_key}"])

        if SUITES[self.suite].get("uses_terminus_kwargs"):
            if self.base_url:
                cmd.extend(["--ak", f"api_base={self.base_url.rstrip('/')}"])
            cmd.extend(["--ak", f"temperature={self.temperature}"])
            cmd.extend(["--ak", f"reasoning_effort={self.reasoning_effort}"])
            cmd.extend(
                [
                    "--ak",
                    "llm_call_kwargs="
                    + json.dumps({"top_p": self.top_p, "top_k": self.top_k}),
                ]
            )
            if self.raw_trajectory:
                cmd.extend(["--ak", 'trajectory_config={"raw_content": true}'])
        else:
            cmd.extend(["--ak", f"reasoning_effort={self.reasoning_effort}"])
        return cmd


def make_spec(
    suite: str,
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    env: str,
    jobs_dir: Path,
    job_name: str,
    n_attempts: int | None = None,
    n_concurrent: int = 4,
    include_task_names: list[str] | None = None,
    n_tasks: int | None = None,
    sampling: dict[str, Any] | None = None,
    debug: bool = False,
    raw_trajectory: bool = False,
) -> HarborRunSpec:
    if suite not in SUITES:
        raise SystemExit(f"Unknown suite {suite!r}; choose from {list(SUITES)}")
    meta = load_meta_reference()
    samp = sampling or meta.get("sampling") or {}
    info = SUITES[suite]
    k = int(n_attempts if n_attempts is not None else info["default_k"])
    return HarborRunSpec(
        suite=suite,
        dataset=str(info["dataset"]),
        agent=str(info["agent"]),
        model=normalize_model_name(model),
        base_url=base_url,
        api_key=api_key or "EMPTY",
        env=env,
        n_attempts=k,
        n_concurrent=int(n_concurrent),
        jobs_dir=jobs_dir,
        job_name=job_name,
        temperature=float(samp.get("temperature", 1.0)),
        reasoning_effort=str(samp.get("reasoning", "high")),
        top_p=float(samp.get("top_p", 0.95)),
        top_k=int(samp.get("top_k", 64)),
        include_task_names=list(include_task_names or []),
        n_tasks=n_tasks,
        debug=debug,
        raw_trajectory=raw_trajectory,
    )


def start_score_poll_thread(
    job_dir: Path,
    on_update: Callable[[dict[str, Any]], None],
    *,
    interval_s: float = 15.0,
) -> tuple[threading.Event, threading.Thread]:
    """Poll ``trial_result.json`` / ``result.json`` while Harbor runs."""
    stop = threading.Event()

    def _loop() -> None:
        last_n = -1
        while not stop.wait(interval_s):
            scores = parse_job_score(job_dir)
            n = scores.get("n_trials")
            if isinstance(n, int) and n > 0 and n != last_n:
                last_n = n
                try:
                    on_update(scores)
                except Exception as e:  # noqa: BLE001 — never kill Harbor for TB
                    print(f"[harbor] score poll callback failed: {e!r}", flush=True)

    th = threading.Thread(
        target=_loop, name=f"harbor-score-{job_dir.name}", daemon=True
    )
    th.start()
    return stop, th


def run_spec(
    spec: HarborRunSpec,
    *,
    dry_run: bool = False,
    follow_traj: bool = False,
    on_score_update: Callable[[dict[str, Any]], None] | None = None,
    score_poll_interval_s: float = 15.0,
) -> dict[str, Any]:
    cmd = spec.build_cmd()
    printable = " ".join(shlex.quote(c) for c in cmd)
    result: dict[str, Any] = {
        "suite": spec.suite,
        "dataset": spec.dataset,
        "agent": spec.agent,
        "model": spec.model,
        "cmd": cmd,
        "cmd_str": printable,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    env = os.environ.copy()
    if spec.base_url:
        env["OPENAI_BASE_URL"] = spec.base_url.rstrip("/")
        env["OPENAI_API_BASE"] = spec.base_url.rstrip("/")
    env.setdefault("OPENAI_API_KEY", spec.api_key)

    spec.jobs_dir.mkdir(parents=True, exist_ok=True)
    job_dir = spec.jobs_dir / spec.job_name
    print(f"  job_dir: {job_dir}", flush=True)
    print(
        f"  live traj (another tty): python -m eval.follow_traj {job_dir}",
        flush=True,
    )

    stops: list[threading.Event] = []
    if follow_traj:
        from eval.follow_traj import start_follow_thread

        stop, _th = start_follow_thread(job_dir)
        stops.append(stop)
    if on_score_update is not None:
        stop, _th = start_score_poll_thread(
            job_dir,
            on_score_update,
            interval_s=score_poll_interval_s,
        )
        stops.append(stop)

    try:
        proc = subprocess.run(cmd, cwd=str(TRAIN_ROOT), env=env, check=False)
    finally:
        for stop in stops:
            stop.set()
        if stops:
            time.sleep(0.2)

    result["returncode"] = proc.returncode
    result["job_dir"] = str(job_dir)
    result.update(parse_job_score(job_dir))
    if on_score_update is not None:
        try:
            on_score_update(
                {
                    "score_percent": result.get("score_percent"),
                    "n_trials": result.get("n_trials"),
                    "pass_at_1": result.get("pass_at_1"),
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"[harbor] final score callback failed: {e!r}", flush=True)
    if proc.returncode != 0:
        result["error"] = f"harbor exited {proc.returncode}"
    return result


def parse_job_score(job_dir: Path) -> dict[str, Any]:
    """Extract mean pass rate (%) from a Harbor job directory."""
    out: dict[str, Any] = {"score_percent": None, "n_trials": None, "pass_at_1": None}
    if not job_dir.is_dir():
        return out

    candidates = [
        job_dir / "result.json",
        job_dir / "job_result.json",
        *sorted(job_dir.glob("**/result.json")),
    ]
    data = None
    used = None
    for c in candidates:
        if c.is_file():
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                used = c
                break
            except json.JSONDecodeError:
                continue

    if data is None:
        rewards: list[float] = []
        for tr in job_dir.glob("**/trial_result.json"):
            try:
                t = json.loads(tr.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            vr = (t.get("verifier_result") or {}) if isinstance(t, dict) else {}
            rew = vr.get("rewards") if isinstance(vr, dict) else None
            if isinstance(rew, dict):
                val = rew.get("reward")
                if val is None and rew:
                    val = next(iter(rew.values()))
                if isinstance(val, (int, float)):
                    rewards.append(float(val))
        if rewards:
            mean = sum(1.0 for r in rewards if r >= 1.0 - 1e-9) / len(rewards)
            out["score_percent"] = round(100.0 * mean, 2)
            out["n_trials"] = len(rewards)
            out["pass_at_1"] = mean
        return out

    out["result_json"] = str(used)
    stats = data.get("stats") or data.get("job_stats") or data
    evals: dict[str, Any] = {}
    if isinstance(stats, dict):
        evals = stats.get("evals") or {}
        out["n_trials"] = stats.get("n_completed_trials")
    if isinstance(evals, dict) and evals:
        first = next(iter(evals.values()))
        if isinstance(first, dict):
            pak = first.get("pass_at_k") or {}
            if isinstance(pak, dict) and pak:
                p1 = pak.get(1, pak.get("1"))
                if isinstance(p1, (int, float)):
                    out["pass_at_1"] = float(p1)
                    out["score_percent"] = round(100.0 * float(p1), 2)
            if out["n_trials"] is None:
                out["n_trials"] = first.get("n_trials")
            if out["score_percent"] is None:
                rs = first.get("reward_stats") or {}
                if isinstance(rs, dict):
                    block = rs.get("reward") or (next(iter(rs.values())) if rs else None)
                    if isinstance(block, dict) and block:
                        total = sum(
                            len(v) for v in block.values() if isinstance(v, list)
                        )
                        passed = 0
                        for val, trials in block.items():
                            try:
                                fv = float(val)
                            except (TypeError, ValueError):
                                continue
                            if fv >= 1.0 - 1e-9 and isinstance(trials, list):
                                passed += len(trials)
                        if total > 0:
                            mean = passed / total
                            out["pass_at_1"] = mean
                            out["score_percent"] = round(100.0 * mean, 2)
                            out["n_trials"] = total
    return out
