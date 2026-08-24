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

# Terminus-2: task.toml only sets wall-clock agent timeout_sec (often 900), NOT max_turns.
# Harbor default max_turns is ~1e6 (effectively unlimited). Common when people do set a
# limit: Harbor docs example=100; AgentCompass TB2.1 harness default=300.
DEFAULT_TERMINUS_MAX_TURNS = 300
# Stretch task timeouts so slow remote vLLM is not killed by 900s wall-clock first.
DEFAULT_AGENT_TIMEOUT_MULTIPLIER = 100.0
# Terminus/LiteLLM fallback is 1e6 if the model is unmapped; that makes vLLM
# reject long chat.completions with HTTP 400. Match serve max-model-len.
DEFAULT_MAX_MODEL_LEN = 65536
# Completion budget must be < window. vLLM 400s ``max_tokens=65536`` even on
# a "hi" prompt (prompt_tokens + max_tokens > max_model_len). Also leave
# headroom: LiteLLM token_counter(openai/qwen-merge) is not the Qwen
# tokenizer and ignores the chat template, so Harbor's "free tokens" is high.
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_TOKEN_COUNT_MARGIN = 4096


def terminus_model_info(max_model_len: int) -> dict[str, int]:
    n = int(max_model_len)
    output = min(DEFAULT_MAX_OUTPUT_TOKENS, max(1024, n // 4))
    margin = min(DEFAULT_TOKEN_COUNT_MARGIN, max(0, n // 16))
    inp = max(output + 1024, n - output - margin)
    return {
        "max_input_tokens": inp,
        "max_output_tokens": output,
        "max_tokens": n,
    }


def _inject_model_info(container: dict[str, Any], info: dict[str, int]) -> bool:
    kwargs = container.get("kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
        container["kwargs"] = kwargs
    if kwargs.get("model_info") == info:
        return False
    kwargs["model_info"] = info
    return True


def apply_model_info_to_job_config(job_dir: Path, max_model_len: int) -> None:
    """Patch a saved Harbor job so resume picks up the real vLLM window.

    ``harbor job resume`` does not accept ``--ak``; it rereads ``config.json``.
    Planned vs existing ``TrialConfig`` must stay equal, so the same
    ``model_info`` is written into the job config, every trial config, and
    ``lock.json``.
    """
    info = terminus_model_info(max_model_len)
    n_files = 0

    cfg_path = job_dir / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    changed = False
    for agent in raw.get("agents") or []:
        if isinstance(agent, dict) and _inject_model_info(agent, info):
            changed = True
    if changed:
        cfg_path.write_text(
            json.dumps(raw, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        n_files += 1

    n_trials = 0
    for trial_cfg in job_dir.glob("*/config.json"):
        trial = json.loads(trial_cfg.read_text(encoding="utf-8"))
        agent = trial.get("agent")
        if isinstance(agent, dict) and _inject_model_info(agent, info):
            trial_cfg.write_text(
                json.dumps(trial, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            n_trials += 1
            n_files += 1

    lock_path = job_dir / "lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_changed = False
        for trial in lock.get("trials") or []:
            if not isinstance(trial, dict):
                continue
            agent = trial.get("agent")
            if isinstance(agent, dict) and _inject_model_info(agent, info):
                lock_changed = True
        if lock_changed:
            lock_path.write_text(
                json.dumps(lock, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            n_files += 1

    info = terminus_model_info(max_model_len)
    print(
        f"[harbor] set model_info {info} in {n_files} file(s) "
        f"under {job_dir} (trial configs={n_trials})",
        flush=True,
    )


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
    agent_timeout_multiplier: float | None = DEFAULT_AGENT_TIMEOUT_MULTIPLIER
    max_turns: int | None = DEFAULT_TERMINUS_MAX_TURNS
    max_model_len: int | None = DEFAULT_MAX_MODEL_LEN
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
        if self.agent_timeout_multiplier is not None:
            cmd.extend(
                ["--agent-timeout-multiplier", str(self.agent_timeout_multiplier)]
            )
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
            if self.max_turns is not None:
                cmd.extend(["--ak", f"max_turns={int(self.max_turns)}"])
            if self.max_model_len is not None:
                cmd.extend(
                    [
                        "--ak",
                        "model_info="
                        + json.dumps(terminus_model_info(self.max_model_len)),
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
    timeout_multiplier: float = 1.0,
    agent_timeout_multiplier: float | None = DEFAULT_AGENT_TIMEOUT_MULTIPLIER,
    max_turns: int | None = DEFAULT_TERMINUS_MAX_TURNS,
    max_model_len: int | None = DEFAULT_MAX_MODEL_LEN,
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
        timeout_multiplier=float(timeout_multiplier),
        agent_timeout_multiplier=agent_timeout_multiplier,
        max_turns=max_turns,
        max_model_len=max_model_len,
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


def infer_suite_from_job_dir(job_dir: Path) -> str | None:
    """Map a Harbor job dir name / config to our suite id."""
    name = job_dir.name
    for suite in sorted(SUITES.keys(), key=len, reverse=True):
        if name == suite or name.endswith(f"_{suite}"):
            return suite
    cfg_path = job_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    datasets: list[str] = []
    for d in cfg.get("datasets") or []:
        if isinstance(d, dict):
            path = d.get("path") or d.get("name")
            if isinstance(path, str):
                datasets.append(path)
    for suite, meta in SUITES.items():
        if meta["dataset"] in datasets:
            return suite
    return None


def resolve_resume_job_dirs(
    path: Path,
    *,
    suites: list[str] | None = None,
) -> list[Path]:
    """Accept one Harbor job dir (has ``config.json``) or an agent_eval root."""
    p = path if path.is_absolute() else (TRAIN_ROOT / path)
    p = p.resolve()
    if not p.exists():
        raise SystemExit(f"--resume path does not exist: {p}")
    if (p / "config.json").is_file():
        suite = infer_suite_from_job_dir(p)
        if suites and suite is not None and suite not in suites:
            raise SystemExit(
                f"--resume job {p.name} is suite={suite}, not in --suite {suites}"
            )
        return [p]
    kids = sorted(
        d for d in p.iterdir() if d.is_dir() and (d / "config.json").is_file()
    )
    if not kids:
        raise SystemExit(
            f"--resume: no Harbor job dirs (with config.json) under {p}\n"
            "Pass a suite job dir or the agent_eval stamp root."
        )
    if suites:
        picked: list[Path] = []
        for d in kids:
            s = infer_suite_from_job_dir(d)
            if s in suites:
                picked.append(d)
        if not picked:
            raise SystemExit(
                f"--resume: no job dirs matching --suite {suites} under {p}"
            )
        return picked
    return kids


# Prepended to Harbor's PYTHONPATH so sitecustomize.py can patch Terminus.
# qemu-alpine-ssh ships tmux 3.1c, which rejects ``new-session -e`` (tmux ≥3.2).
_HARBOR_RUNTIME_DIR = EVAL_ROOT / "harbor_runtime"


def _harbor_subprocess_env(
    *,
    base_url: str | None,
    api_key: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    runtime = str(_HARBOR_RUNTIME_DIR)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        runtime if not existing else f"{runtime}{os.pathsep}{existing}"
    )
    if base_url:
        env["OPENAI_BASE_URL"] = base_url.rstrip("/")
        env["OPENAI_API_BASE"] = base_url.rstrip("/")
    if api_key:
        env.setdefault("OPENAI_API_KEY", api_key)
    else:
        env.setdefault("OPENAI_API_KEY", "EMPTY")
    return env


def _execute_harbor(
    cmd: list[str],
    *,
    job_dir: Path,
    env: dict[str, str],
    result: dict[str, Any],
    dry_run: bool = False,
    follow_traj: bool = False,
    on_score_update: Callable[[dict[str, Any]], None] | None = None,
    score_poll_interval_s: float = 15.0,
) -> dict[str, Any]:
    printable = " ".join(shlex.quote(c) for c in cmd)
    result["cmd"] = cmd
    result["cmd_str"] = printable
    result["dry_run"] = dry_run
    result["job_dir"] = str(job_dir)
    if dry_run:
        return result

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
        # Push current partial score immediately (useful on resume).
        try:
            on_score_update(parse_job_score(job_dir))
        except Exception as e:  # noqa: BLE001
            print(f"[harbor] initial score callback failed: {e!r}", flush=True)
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


def run_spec(
    spec: HarborRunSpec,
    *,
    dry_run: bool = False,
    follow_traj: bool = False,
    on_score_update: Callable[[dict[str, Any]], None] | None = None,
    score_poll_interval_s: float = 15.0,
) -> dict[str, Any]:
    cmd = spec.build_cmd()
    result: dict[str, Any] = {
        "suite": spec.suite,
        "dataset": spec.dataset,
        "agent": spec.agent,
        "model": spec.model,
        "resumed": False,
    }
    if dry_run:
        result["cmd"] = cmd
        result["cmd_str"] = " ".join(shlex.quote(c) for c in cmd)
        result["dry_run"] = True
        result["job_dir"] = str(spec.jobs_dir / spec.job_name)
        return result

    spec.jobs_dir.mkdir(parents=True, exist_ok=True)
    job_dir = spec.jobs_dir / spec.job_name
    return _execute_harbor(
        cmd,
        job_dir=job_dir,
        env=_harbor_subprocess_env(base_url=spec.base_url, api_key=spec.api_key),
        result=result,
        dry_run=False,
        follow_traj=follow_traj,
        on_score_update=on_score_update,
        score_poll_interval_s=score_poll_interval_s,
    )


def resume_job(
    job_dir: Path,
    *,
    dry_run: bool = False,
    follow_traj: bool = False,
    on_score_update: Callable[[dict[str, Any]], None] | None = None,
    score_poll_interval_s: float = 15.0,
    filter_error_types: list[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_model_len: int | None = DEFAULT_MAX_MODEL_LEN,
) -> dict[str, Any]:
    """Continue an interrupted Harbor job via ``harbor job resume -p``."""
    job_dir = job_dir if job_dir.is_absolute() else (TRAIN_ROOT / job_dir)
    job_dir = job_dir.resolve()
    cfg = job_dir / "config.json"
    if not cfg.is_file():
        raise SystemExit(f"resume: missing config.json in {job_dir}")
    if max_model_len is not None and not dry_run:
        apply_model_info_to_job_config(job_dir, max_model_len)

    suite = infer_suite_from_job_dir(job_dir) or "unknown"
    meta = SUITES.get(suite) or {}
    cmd = [harbor_bin(), "job", "resume", "-p", str(job_dir)]
    # Harbor default is CancelledError; only pass -f when caller overrides.
    if filter_error_types is not None:
        for err in filter_error_types:
            cmd.extend(["-f", err])

    result: dict[str, Any] = {
        "suite": suite,
        "dataset": meta.get("dataset"),
        "agent": meta.get("agent"),
        "model": None,
        "resumed": True,
        "filter_error_types": filter_error_types,
    }
    try:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        agents = raw.get("agents") or []
        if agents and isinstance(agents[0], dict):
            result["model"] = agents[0].get("model_name") or agents[0].get("model")
            result["agent"] = result["agent"] or agents[0].get("name")
    except json.JSONDecodeError:
        pass

    return _execute_harbor(
        cmd,
        job_dir=job_dir,
        env=_harbor_subprocess_env(base_url=base_url, api_key=api_key),
        result=result,
        dry_run=dry_run,
        follow_traj=follow_traj,
        on_score_update=on_score_update,
        score_poll_interval_s=score_poll_interval_s,
    )


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
