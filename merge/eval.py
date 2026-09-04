#!/usr/bin/env python3
"""Launch vLLM for a Qwen3.5 Instruct checkpoint (ACT merge, Chat Vector, or stock).

Default: ACT-merged Instruct at merge/output/act, served as ``qwen-act``.
``--chatvector`` serves merge/output/chatvector as ``qwen-merge``.
``--base`` serves stock Qwen3.5-35B-A3B as ``Qwen3.5-35B-A3B`` (use ``--port 6008``).

Does not reuse .venv-muse (Muse-patched vLLM). Needs a recent vLLM that
loads Qwen3.5-35B-A3B.

This host (Harbor) is unchanged. Download weights first, merge, then serve::

    python merge/download.py
    python train/scripts/compare_act.py
    python merge/act.py
    python merge/eval.py --act --max-model-len 32768
    python merge/eval.py --base --port 6008 --max-model-len 32768

    cd train && source .venv-eval/bin/activate
    python scripts/test.py --act --suite terminal_bench_2_1
    python scripts/test.py --act-instruct --suite terminal_bench_2_1

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

DEFAULT_ACT = ROOT / "merge" / "output" / "act"
DEFAULT_CHATVECTOR = ROOT / "merge" / "output" / "chatvector"
DEFAULT_PORT = 6006
INSTRUCT_PORT = 6008
ACT_NAME = "qwen-act"
MERGE_NAME = "qwen-merge"
BASE_NAME = "Qwen3.5-35B-A3B"
# Qwen3.5 GDN/Mamba: each decode sequence needs one Mamba cache block.
# vLLM / AutoDL stock max_num_seqs is 1024; at 64K + 0.90 util that is more
# blocks than fit (~700–800), so CUDA graph capture aborts. Harbor TB only
# runs 4 concurrent trials.
GDN_SAFE_MAX_NUM_SEQS = 64
VLLM_STOCK_MAX_NUM_SEQS = 1024


def log(msg: str) -> None:
    print(msg, flush=True)


def _mkdir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".biv_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def flashinfer_workspace_candidates() -> list[Path]:
    """AutoDL often has no /root/.cache (or a dangling symlink). Prefer autodl-tmp."""
    out: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)

    explicit = os.environ.get("FLASHINFER_WORKSPACE_DIR", "").strip()
    if explicit:
        add(Path(explicit))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        add(Path(xdg) / "flashinfer")
    autodl = Path("/root/autodl-tmp")
    if autodl.is_dir():
        add(autodl / ".cache" / "flashinfer")
    add(Path.home() / ".cache" / "flashinfer")
    add(Path("/tmp/flashinfer"))
    return out


def prepare_serve_env() -> Path:
    """vLLM enumerates FlashInfer even when FLASH_ATTN wins; import mkdirs the workspace.

    Sampler FlashInfer is off (Blackwell JIT false-fails). Attention stays FLASH_ATTN.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    for dest in flashinfer_workspace_candidates():
        if _mkdir_writable(dest):
            os.environ["FLASHINFER_WORKSPACE_DIR"] = str(dest)
            return dest
    raise SystemExit(
        "No writable FlashInfer workspace. Set FLASHINFER_WORKSPACE_DIR to a real directory."
    )


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
        "--act",
        action="store_true",
        help=f"Serve ACT-merged Instruct ({DEFAULT_ACT}) as {ACT_NAME} "
        "(default if neither --base nor --chatvector)",
    )
    p.add_argument(
        "--chatvector",
        action="store_true",
        help=f"Serve Chat Vector merge ({DEFAULT_CHATVECTOR}) as {MERGE_NAME}",
    )
    p.add_argument(
        "--base",
        action="store_true",
        help=f"Serve stock Instruct ({DEFAULT_AGENT}) as {BASE_NAME} "
        f"(pair with --port {INSTRUCT_PORT} on AutoDL)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override weights path / hub id "
        f"(default: {DEFAULT_ACT}, {DEFAULT_CHATVECTOR} with --chatvector, "
        f"or {DEFAULT_AGENT} with --base)",
    )
    p.add_argument(
        "--served-model-name",
        default=None,
        help=f"OpenAI model id Harbor should request "
        f"(default: {ACT_NAME}, {MERGE_NAME} with --chatvector, "
        f"or {BASE_NAME} with --base)",
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
        default=None,
        help="Concurrent sequences (default: 64). Qwen3.5 GDN/Mamba cannot use "
        "vLLM's stock 1024 — that exceeds Mamba cache blocks at 64K. "
        "Harbor TB only needs ~4.",
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


def resolve_max_num_seqs(cli: int | None) -> int:
    """Keep GDN CUDA-graph capture below available Mamba cache blocks.

    AutoDL images and stock vLLM set ``VLLM_MAX_NUM_SEQS=1024``. Merge may
    have been started with an explicit 256; a later ``--base`` launch then
    inherits 1024 and dies at capture.
    """
    source = "default"
    if cli is not None:
        n = int(cli)
        source = "cli"
    else:
        raw = os.environ.get("VLLM_MAX_NUM_SEQS")
        if raw:
            n = int(raw)
            source = "env"
        else:
            n = GDN_SAFE_MAX_NUM_SEQS
    if n >= VLLM_STOCK_MAX_NUM_SEQS:
        log(
            f"max_num_seqs={n} ({source}) is vLLM/AutoDL stock; Qwen3.5 GDN "
            f"cannot CUDA-graph that many sequences at 64K. "
            f"Using {GDN_SAFE_MAX_NUM_SEQS}."
        )
        return GDN_SAFE_MAX_NUM_SEQS
    if n > 256:
        log(
            f"max_num_seqs={n} ({source}) is high for Qwen3.5 GDN Mamba cache; "
            f"clamping to {GDN_SAFE_MAX_NUM_SEQS}."
        )
        return GDN_SAFE_MAX_NUM_SEQS
    return n


def serve_mode(args: argparse.Namespace) -> str:
    flags = [name for name in ("act", "chatvector", "base") if getattr(args, name)]
    if len(flags) > 1:
        raise SystemExit("pick one of --act / --chatvector / --base")
    if args.base:
        return "base"
    if args.chatvector:
        return "chatvector"
    return "act"


def _resolve_merged(path: Path, how: str) -> str:
    local = path
    if not local.is_absolute():
        candidate = ROOT / local
        if has_config(candidate):
            local = candidate
    if local.exists() and not has_config(local):
        raise SystemExit(f"Merged checkpoint not ready: {local}\n{how}")
    if not has_config(local):
        raise SystemExit(f"Merged checkpoint not ready: {local}\n{how}")
    return str(local.resolve())


def build_cmd(args: argparse.Namespace) -> tuple[list[str], str, str]:
    cache_dir = DEFAULT_CACHE if DEFAULT_CACHE.is_absolute() else (ROOT / DEFAULT_CACHE)
    mode = serve_mode(args)
    if mode == "base":
        model = existing_or_spec(args.model or DEFAULT_AGENT, cache_dir)
        served = args.served_model_name or BASE_NAME
    elif mode == "chatvector":
        model = _resolve_merged(
            Path(args.model) if args.model else DEFAULT_CHATVECTOR,
            "Run: python merge/merge.py",
        )
        served = args.served_model_name or MERGE_NAME
    else:
        model = _resolve_merged(
            Path(args.model) if args.model else DEFAULT_ACT,
            "Run: python merge/act.py",
        )
        served = args.served_model_name or ACT_NAME

    if args.vllm_bin:
        launcher = [args.vllm_bin]
    else:
        launcher = find_vllm()

    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]

    args.max_num_seqs = resolve_max_num_seqs(args.max_num_seqs)

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


def harbor_hint(served: str, mode: str) -> str:
    if mode == "base":
        flag = "--act-instruct"
    elif mode == "chatvector":
        flag = f"--model {served}"
    else:
        flag = "--act"
    return (
        "On the Harbor host (local Docker, train/.venv-eval):\n"
        f"  python scripts/test.py {flag} --suite terminal_bench_2_1\n"
        "Do not pass test.py --base (that still selects Muse-Glimmer-30B).\n"
        f"  :{DEFAULT_PORT} → ACT / merge   :{INSTRUCT_PORT} → stock Instruct"
    )


def main() -> None:
    args = parse_args()
    workspace = prepare_serve_env()
    mode = serve_mode(args)
    cmd, model, served = build_cmd(args)

    if args.source in {"modelscope", "ms"} and not Path(model).exists():
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")

    title = {
        "act": "ACT merge vLLM",
        "chatvector": "Chat Vector vLLM",
        "base": "Instruct vLLM",
    }[mode]
    log(f"=== {title} ===")
    log(f"  model:  {model}")
    log(f"  served: {served}")
    log(f"  bind:   {args.host}:{args.port}")
    log(
        f"  tp={args.tp}  max_model_len={args.max_model_len}  "
        f"max_num_seqs={args.max_num_seqs}  dtype={args.dtype}"
    )
    log(f"  attn:   {os.environ.get('VLLM_ATTENTION_BACKEND')}  "
        f"flashinfer={workspace}")
    log("  cmd:    " + " ".join(cmd))
    log(harbor_hint(served, mode))

    if args.dry_run:
        return

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
