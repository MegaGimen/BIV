#!/usr/bin/env python3
"""Muse Glimmer agent eval: Harbor (this host, Docker) → remote AutoDL vLLM.

Split of roles (do not confuse):
  • AutoDL GPU: ``bash scripts/serve_muse_vllm.sh``
      (default: **latest** LoRA ckpt; ``--base`` for no adapter)
  • This machine: Harbor ``--env docker`` + ``python scripts/test.py``
      No local checkpoint path — only picks remote model id.

Default Harbor model id is ``muse-lora`` (matches serve LoRA name).
Use ``--base`` to hit ``Muse-Glimmer-30B`` when AutoDL served without LoRA.

TB arm/step (optional): copy from AutoDL serve banner, or::

  export MUSE_EVAL_ARM=checkpoint-e0-s2150
  export MUSE_EVAL_STEP=2150

Resume an interrupted Harbor run (skips finished trials)::

  python scripts/test.py --resume outputs/agent_eval/<stamp>_<arm>
  python scripts/test.py --resume outputs/agent_eval/.../<arm>_terminal_bench_2_1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.env_check import check_environment, format_report  # noqa: E402
from eval.run_harbor import (  # noqa: E402
    DEFAULT_AGENT_TIMEOUT_MULTIPLIER,
    DEFAULT_SUITES,
    DEFAULT_TERMINUS_MAX_TURNS,
    SUITES,
    load_meta_reference,
    make_spec,
    resolve_resume_job_dirs,
    resume_job,
    run_spec,
)

# Must match scripts/serve_muse_vllm.sh LORA_NAME / SERVED_BASE defaults.
DEFAULT_BASE_MODEL = "Muse-Glimmer-30B"
DEFAULT_LORA_MODEL = "muse-lora"
# AutoDL custom-service forward (override with $MUSE_BASE_URL if the instance changes).
DEFAULT_REMOTE_URL = (
    "https://u741253-d2n6-518972c0.westd.seetacloud.com:8443/v1"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base",
        action="store_true",
        help=f"Request base model id '{DEFAULT_BASE_MODEL}' "
        "(AutoDL must have been started with serve_muse_vllm.sh --base).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Override model id (default: '{DEFAULT_LORA_MODEL}', "
        f"or '{DEFAULT_BASE_MODEL}' with --base).",
    )
    p.add_argument(
        "--arm",
        type=str,
        default=None,
        help="Label for jobs/TB (default: $MUSE_EVAL_ARM or muse-lora/base). "
        "Prefer the ckpt folder name from AutoDL serve banner.",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="TensorBoard x-axis step (default: $MUSE_EVAL_STEP, else parse --arm, else 0).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("MUSE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_REMOTE_URL,
        help="Remote vLLM OpenAI base URL "
        f"(default: $MUSE_BASE_URL or {DEFAULT_REMOTE_URL}).",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    p.add_argument(
        "--suite",
        action="append",
        dest="suites",
        choices=list(SUITES.keys()),
    )
    p.add_argument(
        "--env",
        type=str,
        default=os.environ.get("HARBOR_ENV", "docker"),
        help="Harbor sandbox backend on THIS host (default: docker).",
    )
    p.add_argument("--n-attempts", "-k", type=int, default=None)
    p.add_argument(
        "--n-concurrent",
        "-n",
        type=int,
        default=int(os.environ.get("HARBOR_N_CONCURRENT", "4")),
    )
    p.add_argument("--include-task", action="append", dest="include_tasks", default=None)
    p.add_argument(
        "--n-tasks",
        "-l",
        type=int,
        default=None,
        help="Max tasks from the suite (Harbor -l). Use 1 for smoke.",
    )
    p.add_argument("--jobs-dir", type=Path, default=None)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume interrupted Harbor job(s): path to one suite job dir "
        "(has config.json), or an agent_eval stamp root containing them. "
        "Skips finished trials; default Harbor filter drops CancelledError.",
    )
    p.add_argument(
        "--filter-error-type",
        "-f",
        action="append",
        dest="filter_error_types",
        default=None,
        help="On --resume: remove trials with this exception type before "
        "continuing (repeatable). Omit to keep Harbor default (CancelledError).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--print-serve-cmd",
        action="store_true",
        help="Print AutoDL serve_muse_vllm.sh hint and exit.",
    )
    p.add_argument(
        "--follow-traj",
        action="store_true",
        help="Stream agent trajectory.json steps to stdout while Harbor runs.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Pass Harbor --debug (more verbose job logs).",
    )
    p.add_argument(
        "--raw-traj",
        action="store_true",
        help="Terminus: dump raw LLM responses into trajectory "
        "(--ak trajectory_config raw_content).",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Terminus-2 LLM round limit (--ak max_turns). "
        "Default 300 (not set by TB task.toml; Harbor alone defaults to ~1e6). "
        "Pass 0 to leave Harbor unlimited.",
    )
    p.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help="Harbor --agent-timeout-multiplier (default 100 so 900s tasks "
        "become ~25h wall-clock and max_turns is the real stop).",
    )
    p.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Harbor --timeout-multiplier for non-agent phases (default 1.0).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard root for bench scores "
        "(default: $LOGGING_DIR / $TF_LOGS / /root/tf-logs). "
        "Pass --no-tensorboard to skip.",
    )
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not write Harbor scores to TensorBoard.",
    )
    return p.parse_args()


def _normalize_base_url(url: str) -> str:
    u = url.rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _serve_hint(*, base: bool) -> str:
    if base:
        return "bash scripts/serve_muse_vllm.sh --base"
    return "bash scripts/serve_muse_vllm.sh   # default: latest LoRA ckpt"


def _print_summary_table(rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    ref = meta.get("muse_glimmer_30b_high_reasoning") or {}
    print("\n=== Summary (our % vs Meta Muse Glimmer-30B) ===", flush=True)
    hdr = (
        f"{'suite':<22} {'ours%':>8} {'meta%':>8} {'delta':>8} "
        f"{'scaffold':<16} {'k':>3} {'n':>6}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in rows:
        suite = r["suite"]
        meta_key = (SUITES.get(suite) or {}).get("meta_key") or suite
        mscore = (ref.get(meta_key) or {}).get("score")
        ours = r.get("score_percent")
        delta = None
        if isinstance(ours, (int, float)) and isinstance(mscore, (int, float)):
            delta = round(float(ours) - float(mscore), 2)
        print(
            f"{suite:<22} "
            f"{'-' if ours is None else f'{ours:.2f}':>8} "
            f"{'-' if mscore is None else f'{mscore:.1f}':>8} "
            f"{'-' if delta is None else f'{delta:+.2f}':>8} "
            f"{str(r.get('agent', '')):<16} "
            f"{r.get('n_attempts', '-'):>3} "
            f"{r.get('n_trials') or '-':>6}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    meta = load_meta_reference()
    suites = list(args.suites) if args.suites else list(DEFAULT_SUITES)
    base_url = _normalize_base_url(args.base_url)

    use_base = bool(args.base)
    model_id = args.model
    if model_id is None:
        model_id = DEFAULT_BASE_MODEL if use_base else DEFAULT_LORA_MODEL

    arm = (
        args.arm
        or os.environ.get("MUSE_EVAL_ARM")
        or ("base" if use_base else DEFAULT_LORA_MODEL)
    )
    step: int | None = args.step
    if step is None:
        env_step = os.environ.get("MUSE_EVAL_STEP")
        if env_step is not None and str(env_step).strip() != "":
            try:
                step = int(env_step)
            except ValueError:
                step = None

    resume_jobs: list[Path] | None = None
    if args.resume is not None:
        # With --suite, only resume matching jobs under a stamp root.
        resume_jobs = resolve_resume_job_dirs(
            args.resume,
            suites=list(args.suites) if args.suites else None,
        )

    # max_turns: None CLI → default 300; 0 → unlimited (omit --ak max_turns)
    if args.max_turns is None:
        max_turns: int | None = DEFAULT_TERMINUS_MAX_TURNS
    elif int(args.max_turns) <= 0:
        max_turns = None
    else:
        max_turns = int(args.max_turns)
    agent_timeout_mult = (
        DEFAULT_AGENT_TIMEOUT_MULTIPLIER
        if args.agent_timeout_multiplier is None
        else float(args.agent_timeout_multiplier)
    )

    print("=== Muse Glimmer agent eval (Harbor@this-host + remote vLLM) ===", flush=True)
    print(f"  dry_run:   {args.dry_run}", flush=True)
    if resume_jobs is not None:
        print(f"  resume:    {[str(p) for p in resume_jobs]}", flush=True)
        print(
            f"  filter_err:{args.filter_error_types or '(harbor default: CancelledError)'}",
            flush=True,
        )
    else:
        print(f"  suites:    {suites}", flush=True)
    print(f"  base_url:  {base_url}", flush=True)
    print(f"  harbor_env:{args.env}", flush=True)
    print(f"  model_id:  {model_id}", flush=True)
    print(f"  arm:       {arm}", flush=True)
    print(f"  tb_step:   {step if step is not None else '(from arm name or 0)'}", flush=True)
    print(
        f"  max_turns: {max_turns if max_turns is not None else '(unlimited)'} "
        f"(terminus only; not from task.toml)",
        flush=True,
    )
    print(
        f"  agent_timeout_mult: {agent_timeout_mult} "
        f"(task agent timeout_sec × this)",
        flush=True,
    )
    print(
        "  NOTE: no local --ckpt; AutoDL picks latest LoRA by default:\n"
        f"    {_serve_hint(base=use_base)}",
        flush=True,
    )

    if args.print_serve_cmd:
        print(_serve_hint(base=use_base), flush=True)
        return

    env_report = check_environment()
    print(format_report(env_report), flush=True)
    if args.env == "docker" and env_report.get("checks", {}).get("docker") != "ok":
        print(
            "[test] ERROR: --env docker but docker check failed. "
            "Fix Docker on this host, or pass --env <other>.",
            flush=True,
        )
        if not args.dry_run:
            raise SystemExit(2)
    if args.env == "e2b" and not os.environ.get("E2B_API_KEY"):
        print(
            "[test] WARNING: --env e2b but E2B_API_KEY unset.",
            flush=True,
        )

    if resume_jobs is not None:
        out_root = resume_jobs[0].parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = args.jobs_dir
        if out_root is None:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in arm)[:80]
            out_root = ROOT / "outputs" / "agent_eval" / f"{stamp}_{safe}"
        elif not out_root.is_absolute():
            out_root = ROOT / out_root
        out_root.mkdir(parents=True, exist_ok=True)

    tb_sess = None
    log_root: Path | None = None
    if (not args.dry_run) and (not args.no_tensorboard):
        try:
            from eval.tb_log import AgentEvalTbSession, default_log_root

            log_root = args.log_dir
            if log_root is None:
                log_root = default_log_root()
            elif not log_root.is_absolute():
                log_root = ROOT / log_root
            tb_sess = AgentEvalTbSession(
                meta=meta,
                arm=arm,
                step=step,
                log_root=log_root,
                suites_meta=SUITES,
            )
        except Exception as e:  # noqa: BLE001 — bench continues without TB
            print(f"[test] WARN TensorBoard session open failed: {e!r}", flush=True)
            tb_sess = None

    rows: list[dict[str, Any]] = []
    if resume_jobs is not None:
        for job_dir in resume_jobs:
            suite = job_dir.name  # overwritten from result
            print(f"\n--- resume {job_dir} ---", flush=True)

            def _on_score(scores: dict[str, Any], _job: Path = job_dir) -> None:
                if tb_sess is None:
                    return
                from eval.run_harbor import infer_suite_from_job_dir

                s = infer_suite_from_job_dir(_job) or _job.name
                tb_sess.log_live(s, scores)

            result = resume_job(
                job_dir,
                dry_run=args.dry_run,
                follow_traj=args.follow_traj,
                on_score_update=_on_score if tb_sess is not None else None,
                filter_error_types=args.filter_error_types,
                base_url=base_url,
                api_key=args.api_key,
            )
            suite = str(result.get("suite") or job_dir.name)
            result["arm"] = arm
            result["step"] = step
            result["model_id"] = result.get("model") or model_id
            result["env"] = args.env
            rows.append(result)
            if tb_sess is not None:
                try:
                    tb_sess.log_suite_final(result)
                except Exception as e:  # noqa: BLE001
                    print(f"[test] WARN TB suite final failed: {e!r}", flush=True)
            print(f"  cmd: {result['cmd_str']}", flush=True)
            if not args.dry_run:
                print(
                    f"  score%={result.get('score_percent')} "
                    f"n_trials={result.get('n_trials')} rc={result.get('returncode')}",
                    flush=True,
                )
    else:
        for suite in suites:
            spec = make_spec(
                suite,
                model=model_id,
                base_url=base_url,
                api_key=args.api_key,
                env=args.env,
                jobs_dir=out_root,
                job_name=f"{arm}_{suite}",
                n_attempts=args.n_attempts,
                n_concurrent=args.n_concurrent,
                include_task_names=args.include_tasks,
                n_tasks=args.n_tasks,
                sampling=meta.get("sampling"),
                debug=args.debug,
                raw_trajectory=args.raw_traj,
                timeout_multiplier=float(args.timeout_multiplier),
                agent_timeout_multiplier=agent_timeout_mult,
                max_turns=max_turns,
            )
            print(
                f"\n--- suite={suite} agent={spec.agent} dataset={spec.dataset} ---",
                flush=True,
            )

            def _on_score(scores: dict[str, Any], _suite: str = suite) -> None:
                if tb_sess is not None:
                    tb_sess.log_live(_suite, scores)

            result = run_spec(
                spec,
                dry_run=args.dry_run,
                follow_traj=args.follow_traj,
                on_score_update=_on_score if tb_sess is not None else None,
            )
            result["arm"] = arm
            result["step"] = step
            result["model_id"] = model_id
            result["n_attempts"] = spec.n_attempts
            result["env"] = args.env
            rows.append(result)
            if tb_sess is not None:
                try:
                    tb_sess.log_suite_final(result)
                except Exception as e:  # noqa: BLE001
                    print(f"[test] WARN TB suite final failed: {e!r}", flush=True)
            print(f"  cmd: {result['cmd_str']}", flush=True)
            if not args.dry_run:
                print(
                    f"  score%={result.get('score_percent')} "
                    f"n_trials={result.get('n_trials')} rc={result.get('returncode')}",
                    flush=True,
                )

    summary = {
        "arm": arm,
        "step": step,
        "model_id": model_id,
        "base_url": base_url,
        "env": args.env,
        "dry_run": args.dry_run,
        "resumed": resume_jobs is not None,
        "resume_paths": [str(p) for p in resume_jobs] if resume_jobs else None,
        "serve_hint": _serve_hint(base=use_base),
        "meta_reference": meta.get("muse_glimmer_30b_high_reasoning"),
        "rows": rows,
        "env_check": env_report,
    }
    path = out_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {path}", flush=True)
    _print_summary_table(rows, meta)

    if tb_sess is not None:
        try:
            tb_dir = tb_sess.finalize()
            summary["tensorboard_run"] = str(tb_dir)
            path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"[test] TensorBoard: tensorboard --logdir {log_root}", flush=True)
        except Exception as e:  # noqa: BLE001 — bench must still succeed if TB fails
            print(f"[test] WARN TensorBoard finalize failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
