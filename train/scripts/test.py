#!/usr/bin/env python3
"""Muse Glimmer external agent eval — runs on the GPU train host.

Loads Muse locally (real weights):
  - no --ckpt  → base Muse-Glimmer-30B
  - --ckpt PATH|auto → base + that PEFT/TRL LoRA checkpoint

Starts a local OpenAI-compatible server on 127.0.0.1, then runs Harbor
(Terminal-Bench 2.1 / SWE Verified / SWE Pro). No remote --base-url required.

Examples (on the GPU train server):
  source .venv-muse/bin/activate   # torch + transformers + peft
  # Harbor: use .venv-eval if separate, or pip install -r requirements-eval.txt

  python scripts/test.py --dry-run
  python scripts/test.py
  python scripts/test.py --ckpt outputs/.../checkpoint-e1-s50
  python scripts/test.py --ckpt auto
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.ckpt import resolve_ckpt  # noqa: E402
from eval.env_check import check_environment, format_report  # noqa: E402
from eval.load_muse import pick_muse_python, resolve_model_path  # noqa: E402
from eval.run_harbor import (  # noqa: E402
    DEFAULT_SUITES,
    SUITES,
    load_meta_reference,
    make_spec,
    run_spec,
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
        help="PEFT checkpoint dir (or 'auto'). Loaded into Muse for this run. "
        "Omit = base Muse-Glimmer.",
    )
    p.add_argument(
        "--ckpt-search-dir",
        type=Path,
        default=None,
        help="Train output_dir for --ckpt auto.",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Base Muse weights (default: yaml model_dir / outputs/models/Muse-Glimmer-30B).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MUSE_MODEL", "Muse-Glimmer-30B"),
        help="Served model id (default Muse-Glimmer-30B). Rarely change.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MUSE_SERVE_PORT", "8000")),
        help="Local OpenAI shim port (default 8000).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional: reuse an already-running local/remote OpenAI URL "
        "(skips starting the shim). Default: start local server with loaded weights.",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key passed to Harbor (local shim accepts EMPTY).",
    )
    p.add_argument(
        "--suite",
        action="append",
        dest="suites",
        choices=list(SUITES.keys()),
        help="Suite to run (repeatable). Default: all three.",
    )
    p.add_argument(
        "--env",
        type=str,
        default=os.environ.get("HARBOR_ENV", "docker"),
        help="Harbor sandbox: docker (default) or e2b.",
    )
    p.add_argument("--n-attempts", "-k", type=int, default=None)
    p.add_argument(
        "--n-concurrent",
        "-n",
        type=int,
        default=int(os.environ.get("HARBOR_N_CONCURRENT", "4")),
    )
    p.add_argument(
        "--include-task",
        action="append",
        dest="include_tasks",
        default=None,
        help="Harbor -i task filter (smoke).",
    )
    p.add_argument("--jobs-dir", type=Path, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Env check + print commands; do not load model / start server / run Harbor jobs.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml",
    )
    p.add_argument(
        "--serve-wait-sec",
        type=int,
        default=3600,
        help="Max seconds to wait for local /health after loading Muse (default 3600).",
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


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_health(url: str, timeout_sec: int) -> None:
    health = url.rstrip("/") + "/health"
    # also try /v1 under parent
    candidates = [health, url.rstrip("/") + "/../health"]
    # normalize: base is http://127.0.0.1:8000/v1 → health at http://127.0.0.1:8000/health
    if url.rstrip("/").endswith("/v1"):
        candidates = [url.rstrip("/")[:-3].rstrip("/") + "/health"]
    t0 = time.time()
    last_err = ""
    while time.time() - t0 < timeout_sec:
        for h in candidates:
            try:
                with urllib.request.urlopen(h, timeout=5) as r:
                    if r.status == 200:
                        print(f"[test] server ready ← {h}", flush=True)
                        return
            except Exception as e:
                last_err = str(e)
        time.sleep(2)
    raise SystemExit(f"Local Muse server not healthy within {timeout_sec}s ({last_err})")


def _start_local_server(
    *,
    model_path: Path,
    ckpt: Path | None,
    port: int,
    served_name: str,
    wait_sec: int,
) -> subprocess.Popen:
    if not _port_free(port):
        raise SystemExit(
            f"port {port} busy. Free it or pass --port / --base-url to reuse an existing server."
        )
    py = pick_muse_python(ROOT)
    cmd = [
        py,
        "-m",
        "eval.serve_openai",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-name",
        served_name,
    ]
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    print(f"[test] starting local Muse server:\n  {' '.join(cmd)}", flush=True)
    log_path = ROOT / "outputs" / "agent_eval" / f"serve_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}/v1"
    try:
        _wait_health(base, wait_sec)
    except SystemExit:
        proc.send_signal(signal.SIGTERM)
        raise
    print(f"[test] serve log → {log_path}", flush=True)
    return proc


def _stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


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

    print("=== Muse Glimmer agent eval (GPU host, local weights) ===", flush=True)
    print(f"  dry_run: {args.dry_run}", flush=True)
    print(f"  suites:  {suites}", flush=True)

    env_report = check_environment()
    print(format_report(env_report), flush=True)

    model_path = None
    try:
        model_path = resolve_model_path(args.model_path, train_root=ROOT)
        print(f"  model_path: {model_path}", flush=True)
    except SystemExit as e:
        if args.dry_run:
            print(f"  model_path: <unresolved> ({e})", flush=True)
        else:
            raise

    ckpt_path: Path | None = None
    if args.ckpt:
        ckpt_path = resolve_ckpt(args.ckpt, search_dir=_default_ckpt_search_dir(args))
        print(f"  ckpt:       {ckpt_path}  (WILL LOAD into Muse)", flush=True)
    else:
        print("  ckpt:       <none> → load base Muse-Glimmer", flush=True)

    arm = _arm_label(ckpt_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.jobs_dir
    if out_root is None:
        out_root = ROOT / "outputs" / "agent_eval" / f"{stamp}_{arm}"
    elif not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
        print(f"  would_serve: {base_url}", flush=True)
        rows: list[dict[str, Any]] = []
        for suite in suites:
            spec = make_spec(
                suite,
                model=args.model,
                base_url=base_url,
                api_key=args.api_key,
                env=args.env,
                jobs_dir=out_root,
                job_name=f"{arm}_{suite}",
                n_attempts=args.n_attempts,
                n_concurrent=args.n_concurrent,
                include_task_names=args.include_tasks,
                sampling=meta.get("sampling"),
            )
            result = run_spec(spec, dry_run=True)
            result["arm"] = arm
            result["ckpt"] = str(ckpt_path) if ckpt_path else None
            result["n_attempts"] = spec.n_attempts
            rows.append(result)
            print(f"  cmd: {result['cmd_str']}", flush=True)
        (out_root / "summary.json").write_text(
            json.dumps(
                {"arm": arm, "dry_run": True, "ckpt": str(ckpt_path) if ckpt_path else None, "rows": rows},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _print_summary_table(rows, meta)
        return

    if not env_report.get("ok"):
        raise SystemExit("Environment check failed (Harbor/Docker). Fix then retry.")

    assert model_path is not None
    serve_proc: subprocess.Popen | None = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        print(
            f"  base_url: {base_url} (external; --ckpt is NOT applied by this process)",
            flush=True,
        )
        if ckpt_path is not None:
            print(
                "  WARNING: you passed --ckpt but also --base-url. "
                "Weights are whatever that URL already serves.",
                flush=True,
            )
    else:
        serve_proc = _start_local_server(
            model_path=model_path,
            ckpt=ckpt_path,
            port=args.port,
            served_name=args.model,
            wait_sec=args.serve_wait_sec,
        )
        base_url = f"http://127.0.0.1:{args.port}/v1"
        print(f"  base_url: {base_url} (local shim; ckpt loaded={ckpt_path is not None})", flush=True)

    rows = []
    try:
        for suite in suites:
            job_name = f"{arm}_{suite}"
            spec = make_spec(
                suite,
                model=args.model,
                base_url=base_url,
                api_key=args.api_key,
                env=args.env,
                jobs_dir=out_root,
                job_name=job_name,
                n_attempts=args.n_attempts,
                n_concurrent=args.n_concurrent,
                include_task_names=args.include_tasks,
                sampling=meta.get("sampling"),
            )
            print(
                f"\n--- suite={suite} agent={spec.agent} dataset={spec.dataset} ---",
                flush=True,
            )
            result = run_spec(spec, dry_run=False)
            result["arm"] = arm
            result["ckpt"] = str(ckpt_path) if ckpt_path else None
            result["env"] = args.env
            result["n_attempts"] = spec.n_attempts
            rows.append(result)
            print(
                f"  score%={result.get('score_percent')} "
                f"n_trials={result.get('n_trials')} rc={result.get('returncode')}",
                flush=True,
            )
    finally:
        _stop_server(serve_proc)

    summary_path = out_root / "summary.json"
    payload = {
        "arm": arm,
        "ckpt": str(ckpt_path) if ckpt_path else None,
        "model_path": str(model_path),
        "model": args.model,
        "base_url": base_url,
        "local_serve": serve_proc is not None or args.base_url is None,
        "env": args.env,
        "meta_reference": meta.get("muse_glimmer_30b_high_reasoning"),
        "sampling": meta.get("sampling"),
        "rows": rows,
        "env_check": env_report,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}", flush=True)
    _print_summary_table(rows, meta)


if __name__ == "__main__":
    main()
