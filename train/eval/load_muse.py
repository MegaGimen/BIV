"""Load Muse Glimmer for local eval serving (base or base+PEFT ckpt)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_model_path(explicit: str | Path | None, *, train_root: Path) -> Path:
    """Default: yaml ``model_dir`` or ``outputs/models/Muse-Glimmer-30B``."""
    if explicit:
        p = Path(str(explicit))
        if not p.is_absolute():
            p = (train_root / p).resolve()
        if not p.exists():
            raise SystemExit(f"model path not found: {p}")
        return p

    cfg = train_root / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"
    if cfg.is_file():
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            raw = data.get("model_dir")
            if raw:
                p = Path(str(raw))
                if not p.is_absolute():
                    p = (train_root / p).resolve()
                if p.exists():
                    return p
        except Exception:
            pass

    fallback = (train_root / "outputs" / "models" / "Muse-Glimmer-30B").resolve()
    if fallback.exists():
        return fallback
    raise SystemExit(
        "Cannot find Muse Glimmer weights. Pass --model-path "
        "(e.g. outputs/models/Muse-Glimmer-30B) or run prepare_model.py first."
    )


def load_muse_for_infer(
    model_path: Path,
    *,
    ckpt: Path | None = None,
    dtype: str = "bfloat16",
):
    """Load generative Muse (+ optional PEFT adapter). Returns (model, tokenizer)."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    torch_dtype = (
        torch.bfloat16 if str(dtype).lower() in {"bf16", "bfloat16"} else torch.float16
    )
    print(f"[serve] tokenizer ← {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    print(f"[serve] loading Muse ← {model_path}", flush=True)
    model = None
    errors: list[str] = []
    try:
        from transformers import MuseGlimmerForConditionalGeneration

        model = MuseGlimmerForConditionalGeneration.from_pretrained(
            str(model_path), **load_kwargs
        )
    except Exception as e:
        errors.append(f"MuseGlimmerForConditionalGeneration: {e}")
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                str(model_path), **load_kwargs
            )
        except Exception as e2:
            errors.append(f"AutoModelForImageTextToText: {e2}")
            raise SystemExit("Failed to load Muse:\n  - " + "\n  - ".join(errors)) from e2

    print(f"[serve] loaded class={type(model).__name__}", flush=True)

    if ckpt is not None:
        ckpt = Path(ckpt)
        if not (ckpt / "adapter_config.json").is_file():
            raise SystemExit(
                f"--ckpt has no adapter_config.json under {ckpt}; "
                "need a PEFT/TRL LoRA checkpoint."
            )
        print(f"[serve] attaching PEFT adapter ← {ckpt}", flush=True)
        model = PeftModel.from_pretrained(model, str(ckpt))
        model.eval()
        print("[serve] PEFT adapter attached (weights are live)", flush=True)
    else:
        model.eval()
        print("[serve] base Muse-Glimmer (no --ckpt)", flush=True)

    return model, tokenizer


def pick_muse_python(train_root: Path) -> str:
    """Prefer .venv-muse (train stack) for loading Muse weights."""
    for c in (
        train_root / ".venv-muse" / "bin" / "python",
        train_root / ".venv" / "bin" / "python",
    ):
        if c.is_file():
            return str(c)
    return os.environ.get("MUSE_PYTHON") or sys_executable()


def sys_executable() -> str:
    import sys

    return sys.executable
