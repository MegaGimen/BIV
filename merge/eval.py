#!/usr/bin/env python3
"""Launch vLLM for the Chat Vector merge (or plain Qwen3.5 Instruct).

Default: merged AgentWorld+Instruct at merge/output/chatvector, served as
``qwen-merge``. ``--base`` serves stock Qwen3.5-35B-A3B as ``Qwen3.5-35B-A3B``.

Does not reuse .venv-muse (Muse-patched vLLM). Needs a recent vLLM that
loads Qwen3.5-35B-A3B.

This host (Harbor) is unchanged. Download weights first, merge, then serve::

    python merge/download.py
    python merge/merge.py
    python merge/eval.py
    python merge/eval.py --base

    cd train && source .venv-eval/bin/activate
    python scripts/test.py --model qwen-merge --suite terminal-bench-2.1
    python scripts/test.py --model Qwen3.5-35B-A3B --suite terminal-bench-2.1

``scripts/test.py --base`` still means Muse-Glimmer-30B. Do not use it here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_MERGE_DIR = Path(__file__).resolve().parent
if str(_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_DIR))

from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_CACHE,
    ROOT,
    existing_or_spec,
    has_config,
)

DEFAULT_MERGED = ROOT / "merge" / "output" / "chatvector"
DEFAULT_PORT = 6006
MERGE_NAME = "qwen-merge"
BASE_NAME = "Qwen3.5-35B-A3B"


def log(msg: str) -> None:
    print(msg, flush=True)


def _path_is_muse(path: Path) -> bool:
    muse = (ROOT / "train" / ".venv-muse").resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == muse or muse in resolved.parents or str(muse) in str(resolved)


def refuse_muse_venv() -> None:
    venv = os.environ.get("VIRTUAL_ENV", "")
    if _path_is_muse(Path(sys.prefix)) or _path_is_muse(Path(sys.executable)) or (
        venv and _path_is_muse(Path(venv))
    ):
        raise SystemExit(
            "eval.py cannot run inside train/.venv-muse (Muse-patched vLLM).\n"
            "  bash merge/install_env.sh\n"
            "  source train/.venv/bin/activate\n"
            "  python merge/eval.py --max-model-len 65536"
        )


def find_vllm() -> list[str]:
    """Prefer train/.venv (or repo .venv). Never launch Muse-patched vLLM."""
    refuse_muse_venv()
    for venv in (ROOT / "train" / ".venv", ROOT / ".venv"):
        vllm_bin = venv / "bin" / "vllm"
        if vllm_bin.is_file() and os.access(vllm_bin, os.X_OK):
            log(f"using {vllm_bin}")
            return [str(vllm_bin)]
    muse_hint = str((ROOT / "train" / ".venv-muse").resolve())
    which = shutil.which("vllm")
    if which:
        resolved = str(Path(which).resolve())
        if muse_hint in resolved:
            raise SystemExit(
                f"PATH vllm is Muse ({which}). Install stock vLLM:\n"
                "  bash merge/install_env.sh"
            )
        return [which]
    raise SystemExit(
        "vLLM not found. Create train/.venv first:\n"
        "  bash merge/install_env.sh"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base",
        action="store_true",
        help=f"Serve stock Instruct ({DEFAULT_AGENT}) as {BASE_NAME}",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override weights path / hub id "
        f"(default: {DEFAULT_MERGED} or {DEFAULT_AGENT} with --base)",
    )
    p.add_argument(
        "--served-model-name",
        default=None,
        help=f"OpenAI model id Harbor should request "
        f"(default: {MERGE_NAME} or {BASE_NAME} with --base)",
    )
    p.add_argument("--host", default=os.environ.get("VLLM_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", str(DEFAULT_PORT))))
    p.add_argument("--tp", type=int, default=int(os.environ.get("VLLM_TP", "1")))
    p.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768")),
        help="Context length. TB first pass: 32768, not 262144.",
    )
    p.add_argument(
        "--dtype",
        default=os.environ.get("VLLM_DTYPE", "bfloat16"),
        help="vLLM --dtype (bfloat16 | float16 | auto | fp8)",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=int(os.environ.get("VLLM_MAX_NUM_SEQS", "256")),
        help="Concurrent sequences. Qwen3.5 GDN/Mamba needs this below KV/Mamba blocks.",
    )
    p.add_argument("--vllm-bin", default=None, help="Path to vllm executable")
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
        help="Hub backend when --base / --model is a repo id",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the vLLM command and exit",
    )
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args after -- forwarded to vLLM",
    )
    return p.parse_args()


def build_cmd(args: argparse.Namespace) -> tuple[list[str], str, str]:
    cache_dir = DEFAULT_CACHE if DEFAULT_CACHE.is_absolute() else (ROOT / DEFAULT_CACHE)
    if args.base:
        model = existing_or_spec(args.model or DEFAULT_AGENT, cache_dir)
        served = args.served_model_name or BASE_NAME
    else:
        model = args.model or str(DEFAULT_MERGED)
        served = args.served_model_name or MERGE_NAME
        local = Path(model)
        if not local.is_absolute():
            candidate = ROOT / local
            if has_config(candidate):
                local = candidate
                model = str(candidate.resolve())
            elif has_config(DEFAULT_MERGED) and args.model is None:
                model = str(DEFAULT_MERGED.resolve())
                local = DEFAULT_MERGED
        if local.exists() and not has_config(local):
            raise SystemExit(
                f"Merged checkpoint not ready: {local}\n"
                "Run: python merge/merge.py"
            )
        if args.model is None and not has_config(Path(model)):
            raise SystemExit(
                f"Merged checkpoint not ready: {model}\n"
                "Run: python merge/merge.py"
            )

    if args.vllm_bin:
        launcher = [args.vllm_bin]
    else:
        launcher = find_vllm()

    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [
        *launcher,
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        served,
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--trust-remote-code",
        *extra,
    ]
    return cmd, model, served


def harbor_hint(served: str) -> str:
    return (
        "On the Harbor host (this machine's train/scripts/test.py, unchanged):\n"
        f"  python scripts/test.py --model {served} --suite terminal-bench-2.1\n"
        "Do not pass --base on test.py (that still selects Muse-Glimmer-30B)."
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    cmd, model, served = build_cmd(args)

    if args.source in {"modelscope", "ms"} and not Path(model).exists():
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")

    log("=== Chat Vector vLLM ===")
    log(f"  model:  {model}")
    log(f"  served: {served}")
    log(f"  bind:   {args.host}:{args.port}")
    log(f"  tp={args.tp}  max_model_len={args.max_model_len}  dtype={args.dtype}")
    log("  cmd:    " + " ".join(cmd))
    log(harbor_hint(served))

    if args.dry_run:
        return

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
