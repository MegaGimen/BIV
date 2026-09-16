#!/usr/bin/env python3
"""CPU smoke for FSDP checkpoint helpers. No GPU, no 35B."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.fsdp_ckpt import (  # noqa: E402
    disable_hf_gradient_checkpoint,
    wrap_fsdp_activation_checkpoint,
)


class _Layer:
    gradient_checkpointing = True


class _LM:
    def __init__(self):
        self.layers = [_Layer(), _Layer()]


class _Fake:
    def __init__(self):
        self.disabled = False
        self.config = SimpleNamespace(use_cache=True, gradient_checkpointing=True)
        self.model = SimpleNamespace(language_model=_LM())

    def gradient_checkpointing_disable(self):
        self.disabled = True

    def get_base_model(self):
        return self


def test_disable_hf_clears_flags():
    fake = _Fake()
    disable_hf_gradient_checkpoint(fake)
    assert fake.disabled
    assert fake.config.use_cache is False
    assert fake.config.gradient_checkpointing is False
    assert fake.model.language_model.layers[0].gradient_checkpointing is False


def test_wrap_returns_zero_without_decoder_class():
    class Empty:
        def modules(self):
            return []

    assert wrap_fsdp_activation_checkpoint(Empty()) == 0


if __name__ == "__main__":
    test_disable_hf_clears_flags()
    test_wrap_returns_zero_without_decoder_class()
    print("ok")
