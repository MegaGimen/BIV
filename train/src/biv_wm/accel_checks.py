"""Hard requirements for GPU accel stacks used by BIV train entrypoints."""

from __future__ import annotations

import sys


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(1)


def require_flash_attn(*, allow_kernels_only: bool = False) -> None:
    """Require ``flash_attn`` (hub ``kernels`` alone is not enough for transformers)."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        _fail(f"torch import failed: {e}")

    print(f"  torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    fa_ok = False
    try:
        import flash_attn
        from flash_attn import flash_attn_func  # noqa: F401

        fa_ok = True
        print(
            f"  flash_attn package OK: {getattr(flash_attn, '__version__', '?')}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  flash_attn package MISSING/broken: {e}", flush=True)

    ker_ok = False
    try:
        import kernels
        from kernels import get_kernel

        print(f"  kernels={getattr(kernels, '__version__', '?')}", flush=True)
        k = get_kernel("kernels-community/flash-attn2")
        if getattr(k, "flash_attn_func", None) is None and getattr(
            k, "flash_attn_varlen_func", None
        ) is None:
            print("  hub kernel loaded but missing flash_attn_func", flush=True)
        else:
            ker_ok = True
            print("  hub kernel kernels-community/flash-attn2 OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  hub kernel load failed: {e}", flush=True)

    if fa_ok:
        if not ker_ok:
            print(
                "  note: hub kernel not loadable yet; proceeding with flash_attn only.",
                flush=True,
            )
        print("  FlashAttention check passed.", flush=True)
        return

    if allow_kernels_only and ker_ok:
        print("  FlashAttention check passed via hub kernels only.", flush=True)
        return

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    whl = (
        f"flash_attn-2.8.3+cu130torch2.13-{py}-{py}-linux_x86_64.whl"
    )
    rel = (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
        f"releases/download/v0.9.47/{whl}"
    )
    _fail(
        "\nERROR: FlashAttention is REQUIRED and the Python package is not installed.\n"
        "  `hf download kernels-community/flash-attn2` only fills the HF cache;\n"
        "  it does NOT register the `flash_attn` module that transformers checks.\n\n"
        "  Prefer a prebuilt wheel matching torch/CUDA/Python (example below for\n"
        f"  torch2.13+cu130 + {py}; pick another asset if versions differ):\n"
        f"    REL={rel}\n"
        "    pip install -U packaging ninja einops\n"
        "    pip install --no-cache-dir \"https://ghfast.top/${REL}\" \\\n"
        "      || pip install --no-cache-dir \"https://gh-proxy.com/${REL}\" \\\n"
        "      || pip install --no-cache-dir \"${REL}\"\n\n"
        "  Fallback (needs nvcc == torch.version.cuda):\n"
        "    pip install flash-attn --no-build-isolation\n\n"
        "  Verify:\n"
        "    python -c \"import flash_attn; print(flash_attn.__version__)\"\n"
        "  Refusing to start without FA (no sdpa fallback).\n"
    )


def require_fla_stack() -> None:
    """Require flash-linear-attention + causal-conv1d (Qwen3-Next / Qwen3.5-9B)."""
    fla_ok = False
    try:
        import fla

        print(f"  fla OK: {getattr(fla, '__version__', '?')}", flush=True)
        fla_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  fla MISSING/broken: {e}", flush=True)

    cc_ok = False
    try:
        import causal_conv1d

        print(
            f"  causal_conv1d OK: {getattr(causal_conv1d, '__version__', '?')}",
            flush=True,
        )
        cc_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  causal_conv1d MISSING/broken: {e}", flush=True)

    if fla_ok and cc_ok:
        print("  FLA + causal-conv1d check passed.", flush=True)
        return

    _fail(
        "\nERROR: flash-linear-attention + causal-conv1d are REQUIRED for "
        "Qwen3-Coder-Next / Qwen3.5-9B (GatedDeltaNet / Unsloth fast path).\n"
        "  Without them training falls back to a slow torch path or may misbehave.\n\n"
        "  Install:\n"
        "    pip install -U ninja packaging \"flash-linear-attention[cuda]\"\n"
        "    # causal-conv1d: nvcc must match torch.version.cuda (cu130 → 13.x)\n"
        "    pip install -U nvidia-cuda-nvcc   # or use system CUDA-13 toolkit\n"
        "    export CUDA_HOME=...   # see train/README.md §2b\n"
        "    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -U "
        "\"causal-conv1d>=1.4.0\" --no-build-isolation\n\n"
        "  Verify:\n"
        "    python -c \"import fla, causal_conv1d; print('ok')\"\n"
        "  Refusing to start without FLA stack.\n"
    )
