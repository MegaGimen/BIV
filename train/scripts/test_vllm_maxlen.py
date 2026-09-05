#!/usr/bin/env python3
"""CPU checks for vLLM /v1/models max_model_len parsing."""

from __future__ import annotations

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from eval.run_harbor import parse_vllm_models_max_model_len  # noqa: E402


def test_reads_named_card() -> None:
    payload = {
        "object": "list",
        "data": [
            {
                "id": "qwen-act",
                "max_model_len": 65536,
                "root": "/weights/act",
            }
        ],
    }
    assert parse_vllm_models_max_model_len(payload, model_id="qwen-act") == 65536
    assert parse_vllm_models_max_model_len(payload, model_id="openai/qwen-act") == 65536
    assert parse_vllm_models_max_model_len(payload, model_id=None) == 65536


def test_lora_null_falls_back_to_base_card() -> None:
    payload = {
        "data": [
            {"id": "Muse-Glimmer-30B", "max_model_len": 65536},
            {"id": "muse-lora", "max_model_len": None, "parent": "Muse-Glimmer-30B"},
        ]
    }
    assert parse_vllm_models_max_model_len(payload, model_id="muse-lora") == 65536


def test_missing_field_is_none() -> None:
    assert parse_vllm_models_max_model_len({"data": [{"id": "x"}]}, model_id="x") is None
    assert parse_vllm_models_max_model_len({}, model_id="x") is None


if __name__ == "__main__":
    test_reads_named_card()
    test_lora_null_falls_back_to_base_card()
    test_missing_field_is_none()
    print("ok")
