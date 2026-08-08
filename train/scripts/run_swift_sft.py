#!/usr/bin/env python3
"""Launch ms-swift SFT with BitsAndBytes patches for 2×~48GB Qwen3-Coder-Next.

1) llm_int8_enable_fp32_cpu_offload=True — accelerate may place MoE/custom layers on CPU.
2) Params4bit/Int8Params accept **kwargs — transformers≥5 passes `_is_hf_initialized`
   via accelerate; bitsandbytes<0.50 crashes without this (or upgrade bnb≥0.50).
"""
from __future__ import annotations

import inspect


def _patch_bnb_cpu_offload() -> None:
    from transformers import BitsAndBytesConfig

    orig = BitsAndBytesConfig.__init__

    def patched(self, *args, **kwargs):
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        return orig(self, *args, **kwargs)

    BitsAndBytesConfig.__init__ = patched  # type: ignore[method-assign]
    print("[biv] BitsAndBytesConfig.llm_int8_enable_fp32_cpu_offload=True", flush=True)


def _patch_bnb_params_kwargs() -> None:
    """Ignore unexpected kwargs (e.g. _is_hf_initialized) on Params4bit/Int8Params."""
    import bitsandbytes as bnb

    for cls_name in ("Params4bit", "Int8Params"):
        cls = getattr(bnb.nn, cls_name, None)
        if cls is None:
            continue
        orig_new = cls.__new__
        # Already accepts **kwargs (bnb≥0.50) — nothing to do.
        try:
            if any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(orig_new).parameters.values()
            ):
                print(f"[biv] {cls_name}.__new__ already accepts **kwargs", flush=True)
                continue
        except (TypeError, ValueError):
            pass

        def _make(orig):
            def patched_new(cls, *args, **kwargs):
                kwargs.pop("_is_hf_initialized", None)
                return orig(cls, *args, **kwargs)

            return patched_new

        cls.__new__ = staticmethod(_make(orig_new))  # type: ignore[method-assign]
        print(f"[biv] patched {cls_name}.__new__ to drop unknown kwargs", flush=True)


def main() -> None:
    _patch_bnb_cpu_offload()
    _patch_bnb_params_kwargs()
    # Mirror swift/cli/sft.py __main__
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()
    from swift.ray_utils import try_init_ray

    try_init_ray()
    from swift.pipelines import sft_main

    sft_main()


if __name__ == "__main__":
    main()
