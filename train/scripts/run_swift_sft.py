#!/usr/bin/env python3
"""Launch ms-swift SFT with BitsAndBytes patches for Qwen3-Coder-Next QLoRA.

Used under:
  - parallel=fsdp: accelerate launch … run_swift_sft.py (recommended on 2×~48GB)
  - parallel=device_map: single-process (legacy; bnb+CPU offload breaks at train step)

Patches:
1) bnb_4bit_use_double_quant=False — nested quant + meta/hooks crashes
2) Params4bit/Int8Params drop unknown kwargs — transformers≥5 `_is_hf_initialized`
3) optional llm_int8_enable_fp32_cpu_offload when BIV_BNB_CPU_OFFLOAD=1 (device_map only)
"""
from __future__ import annotations

import inspect
import os


def _patch_bnb_config() -> None:
    from transformers import BitsAndBytesConfig

    orig = BitsAndBytesConfig.__init__
    cpu_offload = os.environ.get("BIV_BNB_CPU_OFFLOAD", "0") in {"1", "true", "True"}

    def patched(self, *args, **kwargs):
        kwargs["bnb_4bit_use_double_quant"] = False
        if cpu_offload:
            kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        return orig(self, *args, **kwargs)

    BitsAndBytesConfig.__init__ = patched  # type: ignore[method-assign]
    print(
        "[biv] BitsAndBytesConfig: "
        f"bnb_4bit_use_double_quant=False, cpu_offload={cpu_offload}",
        flush=True,
    )


def _patch_bnb_params_kwargs() -> None:
    import bitsandbytes as bnb

    for cls_name in ("Params4bit", "Int8Params"):
        cls = getattr(bnb.nn, cls_name, None)
        if cls is None:
            continue
        orig_new = cls.__new__
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
    _patch_bnb_config()
    _patch_bnb_params_kwargs()
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()
    from swift.ray_utils import try_init_ray

    try_init_ray()
    from swift.pipelines import sft_main

    sft_main()


if __name__ == "__main__":
    main()
