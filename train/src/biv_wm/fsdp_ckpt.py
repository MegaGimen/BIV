"""FSDP2 activation checkpointing that actually unshards on recompute.

HuggingFace ``gradient_checkpointing_enable()`` after ``fully_shard``
recomputes ``.forward`` and skips the FSDP2 unshard hook. On Qwen3.5
that is the GatedDeltaNet ``out_proj`` empty-storage crash, or 40
decoder layers stacked in activation memory. CAST Instruct solved
2-GPU 65536 by wrapping decoder layers with
``torch.distributed`` ``checkpoint_wrapper`` *after*
``accelerator.prepare``. JEPA uses the same wrap: CP mesh still
folds into the FSDP shard dim (weights split), the sequence is
**not** split unless something calls ``maybe_context_parallel``.
"""

from __future__ import annotations

import os
from typing import Any


def decoder_layers(model: Any) -> list[Any]:
    from biv_wm.arch import language_model

    lm = language_model(model)
    if lm is None:
        raise RuntimeError("no language_model.layers")
    layers = list(getattr(lm, "layers", []) or [])
    if not layers:
        raise RuntimeError("language_model.layers is empty")
    return layers


def disable_hf_gradient_checkpoint(model: Any) -> None:
    """Turn off HF layer checkpointing so FSDP2 unshard runs on recompute."""
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = False
        if hasattr(cfg, "gradient_checkpointing"):
            cfg.gradient_checkpointing = False
    try:
        layers = decoder_layers(model)
    except RuntimeError:
        layers = []
    for layer in layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = False


def wrap_fsdp_activation_checkpoint(model: Any) -> int:
    """Wrap Qwen3.5 decoder layers so backward recompute goes through FSDP.

    Must run after ``accelerator.prepare``. ``fully_shard`` rebinds the
    class to ``FSDP<Qwen3_5MoeDecoderLayer>``; ``endswith`` catches both.
    """
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except (ImportError, OSError, ValueError):
        return 0

    def _is_decoder(mod: Any) -> bool:
        return type(mod).__name__.endswith("Qwen3_5MoeDecoderLayer")

    n = sum(1 for m in model.modules() if _is_decoder(m))
    if os.environ.get("BIV_DEBUG_CKPT") == "1":
        names = sorted({type(m).__name__ for m in model.modules()})
        print(f"[jepa] decoder count={n} class names={names[:40]}", flush=True)
    if n == 0:
        return 0
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=_is_decoder,
    )
    return sum(
        1
        for m in model.modules()
        if "Checkpoint" in type(m).__name__ and "Wrapper" in type(m).__name__
    )
