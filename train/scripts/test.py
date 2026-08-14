#!/usr/bin/env python3
"""Muse Glimmer agent eval: Harbor (this host, Docker) → remote AutoDL vLLM.

Split of roles (do not confuse):
  • AutoDL GPU: bash scripts/serve_muse_vllm.sh [--ckpt …]  # :6006 → 公网 :8443
  • This machine: Harbor --env docker + python scripts/test.py
    calls the remote OpenAI-compatible API (default public URL below).

--ckpt on test.py only selects Harbor model id `muse-lora` (must match the
LoRA registered by serve_muse_vllm.sh on AutoDL). It does not load weights here.
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

from eval.ckpt import resolve_ckpt  # noqa: E402
from eval.env_check import check_environment, format_report  # noqa: E402
from eval.load_muse import resolve_model_path  # noqa: E402
from eval.run_harbor import (  # noqa: E402
    DEFAULT_SUITES,
    SUITES,
    load_meta_reference,
    make_spec,
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
        "--ckpt",
        type=str,
        default=None,
        help="Same ckpt as AutoDL serve_muse_vllm.sh. Sets Harbor model id to "
        f"'{DEFAULT_LORA_MODEL}'. Omit → base '{DEFAULT_BASE_MODEL}'.",
    )
    p.add_argument(
        "--ckpt-search-dir",
        type=Path,
        default=None,
        help="For --ckpt auto (train output_dir). Only used to resolve/print "
        "the path; weights still load on AutoDL.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Override model id (default: {DEFAULT_BASE_MODEL} or "
        f"{DEFAULT_LORA_MODEL} when --ckpt is set).",
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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml",
    )
    p.add_argument(
        "--print-serve-cmd",
        action="store_true",
        help="Print AutoDL serve_muse_vllm.sh command for this --ckpt and exit.",
    )
    return p.parse_args()


def _default_ckpt_search_dir(args: argparse.Namespace) -> Path:
    if args.ckpt_search_dir is not None:
        p = args.ckpt_search_dir
        return p if p.is_absolute() else (ROOT / p)
    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    if cfg_path.is_file():
        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            od = (data.get("train") or {}).get("output_dir")
            if od:
                p = Path(str(od))
                return p if p.is_absolute() else (ROOT / p)
        except Exception:
            pass
    return ROOT / "outputs" / "muse_glimmer_wm_mix"


def _arm_label(ckpt: Path | None) -> str:
    return "base" if ckpt is None else ckpt.name


def _normalize_base_url(url: str) -> str:
    u = url.rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _serve_cmd(model_path: Path | None, ckpt: Path | None) -> str:
    parts = ["bash scripts/serve_muse_vllm.sh"]
    if model_path is not None:
        parts.append(f"--model-path {model_path}")
    if ckpt is not None:
        parts.append(f"--ckpt {ckpt}")
    return " ".join(parts)


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
        meta_key = SUITES[suite]["meta_key"]
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

    ckpt_path: Path | None = None
    if args.ckpt:
        # Harbor host often has no adapter files; --ckpt still selects muse-lora.
        ckpt_path = resolve_ckpt(
            args.ckpt,
            search_dir=_default_ckpt_search_dir(args),
            require_local=False,
        )
        if ckpt_path is not None and not ckpt_path.is_dir():
            print(
                f"  NOTE: local ckpt path missing ({ckpt_path}); "
                f"still requesting remote model id '{DEFAULT_LORA_MODEL}'.",
                flush=True,
            )

    model_id = args.model
    if model_id is None:
        model_id = DEFAULT_LORA_MODEL if ckpt_path is not None else DEFAULT_BASE_MODEL

    model_path = None
    try:
        model_path = resolve_model_path(None, train_root=ROOT)
    except SystemExit:
        pass

    print("=== Muse Glimmer agent eval (Harbor@this-host + remote vLLM) ===", flush=True)
    print(f"  dry_run:   {args.dry_run}", flush=True)
    print(f"  suites:    {suites}", flush=True)
    print(f"  base_url:  {base_url}", flush=True)
    print(f"  harbor_env:{args.env}", flush=True)
    print(f"  model_id:  {model_id}", flush=True)
    if ckpt_path is None:
        print(
            "  ckpt:      <none> → AutoDL vLLM should serve base "
            f"(model id {DEFAULT_BASE_MODEL})",
            flush=True,
        )
    else:
        print(f"  ckpt:      {ckpt_path}", flush=True)
        print(
            f"  NOTE: on AutoDL start vLLM with the SAME ckpt "
            f"(LoRA id '{DEFAULT_LORA_MODEL}'):",
            flush=True,
        )
        print(f"    {_serve_cmd(model_path, ckpt_path)}", flush=True)

    if args.print_serve_cmd:
        print(_serve_cmd(model_path, ckpt_path), flush=True)
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

    arm = _arm_label(ckpt_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.jobs_dir
    if out_root is None:
        out_root = ROOT / "outputs" / "agent_eval" / f"{stamp}_{arm}"
    elif not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
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
        )
        print(
            f"\n--- suite={suite} agent={spec.agent} dataset={spec.dataset} ---",
            flush=True,
        )
        result = run_spec(spec, dry_run=args.dry_run)
        result["arm"] = arm
        result["ckpt"] = str(ckpt_path) if ckpt_path else None
        result["model_id"] = model_id
        result["n_attempts"] = spec.n_attempts
        result["env"] = args.env
        rows.append(result)
        print(f"  cmd: {result['cmd_str']}", flush=True)
        if not args.dry_run:
            print(
                f"  score%={result.get('score_percent')} "
                f"n_trials={result.get('n_trials')} rc={result.get('returncode')}",
                flush=True,
            )

    summary = {
        "arm": arm,
        "ckpt": str(ckpt_path) if ckpt_path else None,
        "model_id": model_id,
        "base_url": base_url,
        "env": args.env,
        "dry_run": args.dry_run,
        "serve_hint": _serve_cmd(model_path, ckpt_path),
        "meta_reference": meta.get("muse_glimmer_30b_high_reasoning"),
        "rows": rows,
        "env_check": env_report,
    }
    path = out_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {path}", flush=True)
    _print_summary_table(rows, meta)


if __name__ == "__main__":
    main()
