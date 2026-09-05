#!/usr/bin/env python3
"""Agent eval: Harbor (this host, Docker) → remote AutoDL vLLM.

Muse Glimmer-30B::

    AutoDL:  bash scripts/serve_muse_vllm.sh
    Here:    python scripts/test.py            # muse-lora
             python scripts/test.py --base     # Muse-Glimmer-30B

Qwen3.5 ACT merge (Instruct + AgentWorld mask rows)::

    AutoDL :6006  python merge/eval.py --act --max-model-len 32768
    AutoDL :6008  python merge/eval.py --base --port 6008 --max-model-len 32768
    Here:         python scripts/test.py --act
                  python scripts/test.py --act-instruct

Harbor ``--env daytona`` (cloud sandbox). No local checkpoint path.
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
    DAYTONA_DEFAULT_N_CONCURRENT,
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
    "https://u741253-ltqs-498e6237.westd.seetacloud.com:8443/v1"
)
# This instance: :6006 ACT merge, :6008 stock Instruct.
DEFAULT_ACT_MODEL = "qwen-act"
DEFAULT_ACT_INSTRUCT_MODEL = "Qwen3.5-35B-A3B"
DEFAULT_ACT_URL = (
    "https://u741253-tujr-a523480e.westd.seetacloud.com:8443/v1"
)
DEFAULT_ACT_INSTRUCT_URL = (
    "https://uu741253-tujr-a523480e.westd.seetacloud.com:8443/v1"
)

SUITE_ALIASES = {
    "terminal-bench-2.1": "terminal_bench_2_1",
    "terminal-bench-2-1": "terminal_bench_2_1",
    "tb2.1": "terminal_bench_2_1",
    "tb21": "terminal_bench_2_1",
}


def _suite_id(value: str) -> str:
    return SUITE_ALIASES.get(value, value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base",
        action="store_true",
        help=f"Request Muse base model id '{DEFAULT_BASE_MODEL}' "
        "(AutoDL must have been started with serve_muse_vllm.sh --base).",
    )
    p.add_argument(
        "--act",
        action="store_true",
        help=f"Harbor → ACT-merged Instruct '{DEFAULT_ACT_MODEL}' "
        f"at :6006 ({DEFAULT_ACT_URL}). Default suite is TB 2.1.",
    )
    p.add_argument(
        "--act-instruct",
        action="store_true",
        help=f"Harbor → stock Instruct '{DEFAULT_ACT_INSTRUCT_MODEL}' "
        f"at :6008 ({DEFAULT_ACT_INSTRUCT_URL}). Default suite is TB 2.1.",
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
        default=None,
        help="Remote vLLM OpenAI base URL "
        "(default: --act → $ACT_BASE_URL / :6006 mapping; "
        "--act-instruct → $ACT_INSTRUCT_URL / :6008 mapping; "
        f"else $MUSE_BASE_URL or {DEFAULT_REMOTE_URL}).",
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
        type=_suite_id,
        choices=list(SUITES.keys()),
    )
    p.add_argument(
        "--env",
        type=str,
        default=os.environ.get("HARBOR_ENV", "daytona"),
        help="Harbor sandbox backend (default: daytona).",
    )
    p.add_argument("--n-attempts", "-k", type=int, default=None)
    p.add_argument(
        "--n-concurrent",
        "-n",
        type=int,
        default=None,
        help="Parallel trials. Default 16 on daytona, 4 on docker. "
        "Override with -n or $HARBOR_N_CONCURRENT.",
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
        "Default: omit (Harbor unlimited). Pass a positive int to cap, "
        "or 0 for the same omit behavior.",
    )
    p.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help="Harbor --agent-timeout-multiplier (default 100 so 900s tasks "
        "become ~25h wall-clock).",
    )
    p.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Harbor --timeout-multiplier for non-agent phases (default 1.0).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Tell Terminus/LiteLLM the vLLM context window. "
        "Default 32768 for --act / --act-instruct (matches merge/eval.py), "
        "65536 for Muse. Unmapped openai/<name> otherwise falls back to 1e6 "
        "and vLLM returns 400. On --resume this is written into config.json.",
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


def _serve_hint(*, act: bool = False, act_instruct: bool = False, base: bool = False) -> str:
    if act_instruct:
        return (
            "python merge/eval.py --base --port 6008 --max-model-len 32768"
        )
    if act:
        return "python merge/eval.py --act --max-model-len 32768"
    if base:
        return "bash scripts/serve_muse_vllm.sh --base"
    return "bash scripts/serve_muse_vllm.sh   # default: latest LoRA ckpt"


def _resolve_eval_target(args: argparse.Namespace) -> dict[str, Any]:
    if args.act and args.act_instruct:
        raise SystemExit("pick one of --act / --act-instruct")
    if args.base and (args.act or args.act_instruct):
        raise SystemExit("--base is Muse Glimmer; use --act-instruct for stock Qwen Instruct")

    use_base = bool(args.base)
    act = bool(args.act)
    act_instruct = bool(args.act_instruct)

    if args.base_url:
        base_url = args.base_url
    elif act:
        base_url = os.environ.get("ACT_BASE_URL") or DEFAULT_ACT_URL
    elif act_instruct:
        base_url = os.environ.get("ACT_INSTRUCT_URL") or DEFAULT_ACT_INSTRUCT_URL
    else:
        base_url = (
            os.environ.get("MUSE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_REMOTE_URL
        )

    model_id = args.model
    if model_id is None:
        if act:
            model_id = DEFAULT_ACT_MODEL
        elif act_instruct:
            model_id = DEFAULT_ACT_INSTRUCT_MODEL
        else:
            model_id = DEFAULT_BASE_MODEL if use_base else DEFAULT_LORA_MODEL

    if args.max_model_len is not None:
        max_model_len = int(args.max_model_len)
    elif os.environ.get("HARBOR_MAX_MODEL_LEN"):
        max_model_len = int(os.environ["HARBOR_MAX_MODEL_LEN"])
    elif act or act_instruct:
        max_model_len = 32768
    else:
        max_model_len = 65536

    if args.suites:
        suites = list(args.suites)
    elif act or act_instruct:
        suites = ["terminal_bench_2_1"]
    else:
        suites = list(DEFAULT_SUITES)

    default_arm = (
        DEFAULT_ACT_MODEL
        if act
        else (
            "instruct"
            if act_instruct
            else ("base" if use_base else DEFAULT_LORA_MODEL)
        )
    )
    return {
        "use_base": use_base,
        "act": act,
        "act_instruct": act_instruct,
        "base_url": _normalize_base_url(base_url),
        "model_id": model_id,
        "max_model_len": max_model_len,
        "suites": suites,
        "default_arm": default_arm,
    }


def _print_summary_table(rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    ref = meta.get("muse_glimmer_30b_high_reasoning") or {}
    print("\n=== Summary (our % vs Meta Muse Glimmer-30B) ===", flush=True)
    hdr = (
        f"{'suite':<22} {'ours%':>8} {'meta%':>8} {'delta':>8} "
        f"{'scaffold':<16} {'k':>3} {'n':>6} {'infra':>6}"
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
            f"{r.get('n_trials') or '-':>6} "
            f"{r.get('n_infra_excluded') or 0:>6}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    if args.n_concurrent is None:
        env_n = os.environ.get("HARBOR_N_CONCURRENT")
        if env_n:
            args.n_concurrent = int(env_n)
        elif args.env == "daytona":
            args.n_concurrent = DAYTONA_DEFAULT_N_CONCURRENT
        else:
            args.n_concurrent = 4
    meta = load_meta_reference()
    target = _resolve_eval_target(args)
    suites = target["suites"]
    base_url = target["base_url"]
    use_base = target["use_base"]
    model_id = target["model_id"]
    args.max_model_len = target["max_model_len"]
    serve_hint = _serve_hint(
        act=target["act"],
        act_instruct=target["act_instruct"],
        base=use_base,
    )

    arm = args.arm or os.environ.get("MUSE_EVAL_ARM") or target["default_arm"]
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

    # max_turns: None CLI → omit --ak max_turns; 0 → same; >0 → cap
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

    title = "ACT merge agent eval" if (target["act"] or target["act_instruct"]) else "Muse Glimmer agent eval"
    print(f"=== {title} (Harbor@this-host + remote vLLM) ===", flush=True)
    print(f"  dry_run:   {args.dry_run}", flush=True)
    if resume_jobs is not None:
        print(f"  resume:    {[str(p) for p in resume_jobs]}", flush=True)
        print(
            f"  filter_err:{args.filter_error_types or '(harbor default: CancelledError)'}",
            flush=True,
        )
        print(f"  n_concurrent: {args.n_concurrent}", flush=True)
    else:
        print(f"  suites:    {suites}", flush=True)
        print(f"  n_concurrent: {args.n_concurrent}", flush=True)
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
    print(f"  max_model_len: {args.max_model_len}", flush=True)
    print(
        f"  agent_timeout_mult: {agent_timeout_mult} "
        f"(task agent timeout_sec × this)",
        flush=True,
    )
    print(
        "  NOTE: no local --ckpt; AutoDL serve:\n"
        f"    {serve_hint}",
        flush=True,
    )

    if args.print_serve_cmd:
        print(serve_hint, flush=True)
        return

    env_report = check_environment(env=args.env)
    print(format_report(env_report), flush=True)
    if not env_report.get("ok"):
        print("[test] ERROR: env check failed.", flush=True)
        if args.env == "docker":
            print(
                "[test] --env docker needs a working Docker daemon.",
                flush=True,
            )
        elif args.env == "daytona":
            print(
                "[test] --env daytona needs DAYTONA_API_KEY and "
                "pip install 'harbor[daytona]' in .venv-eval.",
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
                max_model_len=args.max_model_len,
                n_concurrent=args.n_concurrent,
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
                max_model_len=args.max_model_len,
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
        "serve_hint": serve_hint,
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
