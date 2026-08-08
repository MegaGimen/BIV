#!/usr/bin/env python3
"""Launch ms-swift SFT with BitsAndBytes CPU-offload enabled.

Qwen3-Coder-Next (80B MoE) + bnb 4bit + device_map=auto on 2×~48GB still plans
some modules onto CPU (custom/MoE layers often sized as bf16 in the planner).
bnb then raises ValueError unless llm_int8_enable_fp32_cpu_offload=True.

ms-swift does not expose that flag on the CLI, so we patch BitsAndBytesConfig
before importing the training pipeline.
"""
from __future__ import annotations

import sys


def _patch_bnb_cpu_offload() -> None:
    from transformers import BitsAndBytesConfig

    orig = BitsAndBytesConfig.__init__

    def patched(self, *args, **kwargs):
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        return orig(self, *args, **kwargs)

    BitsAndBytesConfig.__init__ = patched  # type: ignore[method-assign]
    print("[biv] BitsAndBytesConfig.llm_int8_enable_fp32_cpu_offload=True", flush=True)


def main() -> None:
    _patch_bnb_cpu_offload()
    # Mirror swift/cli/sft.py __main__
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()
    from swift.ray_utils import try_init_ray

    try_init_ray()
    from swift.pipelines import sft_main

    sft_main()


if __name__ == "__main__":
    main()
