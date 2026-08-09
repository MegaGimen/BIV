#!/usr/bin/env python3
"""Launch ms-swift SFT with BitsAndBytes patches for Qwen3-Coder-Next QLoRA.

Used under:
  - parallel=fsdp: accelerate launch … run_swift_sft.py (recommended on 2×~48GB)
  - parallel=device_map: single-process (legacy; bnb+CPU offload breaks at train step)

Patches:
1) bnb_4bit_use_double_quant=False — nested quant + meta/hooks crashes
2) Params4bit/Int8Params drop unknown kwargs — transformers≥5 `_is_hf_initialized`
3) optional llm_int8_enable_fp32_cpu_offload when BIV_BNB_CPU_OFFLOAD=1 (device_map only)
4) quiet ms-swift spam: skip raw dataset sample dump + full model architecture print
   (keeps Dataset Token Length stats and one-line model_parameter_info)
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


def _patch_quiet_swift_logs() -> None:
    """Stop ms-swift from dumping first-sample text and full nn.Module repr."""
    from swift.pipelines.train.sft import SwiftSft
    from swift.utils import get_logger, is_master

    logger = get_logger()

    def _show_dataset(self, train_dataset, val_dataset):
        # Upstream prints template.decode of sample[0] here — multi-MB chat spam.
        args = self.args
        predict_with_generate = getattr(args, "predict_with_generate", False)
        if not is_master() and hasattr(train_dataset, "__len__"):
            # Keep LazyLLMDataset RNG aligned with rank0's former sample[0] touch.
            _ = train_dataset[0]
        if val_dataset is not None and hasattr(val_dataset, "__len__") and len(val_dataset) == 0:
            val_dataset = None
        if not args.lazy_tokenize and not args.streaming:
            self.train_msg["train_dataset"] = self._stat_dataset(train_dataset)
            if val_dataset is not None and not predict_with_generate:
                self.train_msg["val_dataset"] = self._stat_dataset(val_dataset)

    SwiftSft._show_dataset = _show_dataset  # type: ignore[method-assign]

    # Upstream: logger.info(f'model: {self.model}') → multi-thousand-line Module tree.
    _orig_info = logger.info

    def _info(msg, *a, **kw):
        text = msg if isinstance(msg, str) else str(msg)
        if text.startswith("model: ") and ("\n" in text or "LoraModel" in text):
            return None
        return _orig_info(msg, *a, **kw)

    logger.info = _info  # type: ignore[method-assign]
    print("[biv] quiet logs: no dataset sample dump, no full model architecture", flush=True)


def main() -> None:
    _patch_bnb_config()
    _patch_bnb_params_kwargs()
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()
    from swift.ray_utils import try_init_ray

    try_init_ray()
    _patch_quiet_swift_logs()
    from swift.pipelines import sft_main

    sft_main()


if __name__ == "__main__":
    main()
