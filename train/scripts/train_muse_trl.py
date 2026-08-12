#!/usr/bin/env python3
"""Muse Glimmer-30B WM SFT via TRL + PEFT (LoRA / QLoRA).

Reads structure-right-truncated HF datasets (messages) from train_prep_mix,
applies Glimmer chat template, trains with assistant_only_loss.

Examples (prefer trainmodel.sh wrapper):
  python scripts/train_muse_trl.py \\
    --config configs/trl/muse_glimmer_30b_lora.yaml \\
    --max-length 8192 \\
    --cached-datasets outputs/trl_cache/.../wm_code ... \\
    --output-dir outputs/muse_glimmer_wm_mix_ml8192_c1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CONFIG = ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"

_VISION_NAME_HINTS = (
    "vision",
    "perception",
    "visual",
    "image_encoder",
    "vision_tower",
    "multi_modal_projector",
    "mm_projector",
)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML: {path}")
    return data


def _resolve(raw: str | Path) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _freeze_vision(model) -> int:
    n = 0
    for name, param in model.named_parameters():
        low = name.lower()
        if any(h in low for h in _VISION_NAME_HINTS):
            if param.requires_grad:
                param.requires_grad = False
                n += 1
    return n


def _filter_target_modules(model, modules: list[str]) -> list[str]:
    """Drop LoRA targets that do not exist on this checkpoint."""
    names = {n for n, _ in model.named_modules()}
    leaf = set()
    for n in names:
        leaf.add(n.rsplit(".", 1)[-1])
    kept = [m for m in modules if m in leaf]
    dropped = [m for m in modules if m not in leaf]
    if dropped:
        print(f"[muse] skip missing LoRA targets: {dropped}", flush=True)
    if not kept:
        raise SystemExit(f"No LoRA target_modules matched model; tried {modules}")
    return kept


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Align chat turns across sources (strip tool_calls / non-string content).

    wm_code / anti_forget Arrow schemas may include ``tool_calls`` while wm_os
    does not — concatenate_datasets then fails feature alignment.
    """
    import json

    out: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        # Fold tool_calls into text if present so schema stays role/content only.
        tc = m.get("tool_calls")
        if tc:
            try:
                tc_s = json.dumps(tc, ensure_ascii=False)
            except TypeError:
                tc_s = str(tc)
            content = (content + "\n" + tc_s).strip() if content else tc_s
        out.append({"role": role, "content": content})
    return out


def _load_concat_datasets(paths: list[Path]):
    from datasets import Features, Sequence, Value, concatenate_datasets, load_from_disk

    # Uniform schema so mix sources can concatenate.
    msg_features = Features(
        {
            "messages": [
                {"role": Value("string"), "content": Value("string")},
            ]
        }
    )

    parts = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing cached dataset: {p}")
        ds = load_from_disk(str(p))
        if "messages" not in ds.column_names:
            raise SystemExit(f"{p}: need 'messages' column, got {ds.column_names}")

        def _map(ex):
            return {"messages": _normalize_messages(ex["messages"])}

        ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "messages"])
        ds = ds.cast(msg_features)
        parts.append(ds)
        print(f"  loaded {p} ({len(ds):,} rows)", flush=True)
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def _build_model_and_tokenizer(
    *,
    model_path: str,
    qlora: bool,
    torch_dtype: str,
    freeze_vision: bool,
    target_modules: list[str],
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if str(torch_dtype).lower() in {"bf16", "bfloat16"} else torch.float16
    print(f"[muse] loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    distributed = world > 1 or os.environ.get("LOCAL_RANK") is not None
    if qlora:
        print("[muse] QLoRA: BitsAndBytes 4-bit NF4", flush=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb
        # device_map conflicts with accelerate DDP/FSDP — only for single process
        if not distributed:
            load_kwargs["device_map"] = "auto"
    else:
        if not distributed:
            load_kwargs["device_map"] = "auto"

    print(f"[muse] loading model from {model_path}", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        # Some multimodal checkpoints register under a different auto class.
        print(f"[muse] AutoModelForCausalLM failed ({e}); trying AutoModel…", flush=True)
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_path, **load_kwargs)

    if qlora:
        model = prepare_model_for_kbit_training(model)

    if freeze_vision:
        frozen = _freeze_vision(model)
        print(f"[muse] froze {frozen} vision/perception params", flush=True)

    targets = _filter_target_modules(model, target_modules)
    lora = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument(
        "--cached-datasets",
        nargs="+",
        required=True,
        help="HF dataset dirs from train_prep_mix (wm_code wm_os anti_forget)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=None, help="Override model path/id")
    p.add_argument("--qlora", action="store_true", help="Force 4-bit QLoRA")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-epochs", type=float, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    args = p.parse_args()

    cfg = _load_yaml(_resolve(args.config))
    train_cfg = cfg.get("train") or {}
    from biv_wm.model_store import resolve_model_for_train

    model_path = args.model or resolve_model_for_train(cfg, root=ROOT)
    qlora = bool(args.qlora or train_cfg.get("qlora") or os.environ.get("QLORA") in {"1", "true", "True"})
    max_length = int(args.max_length)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = [_resolve(x) for x in args.cached_datasets]
    print("=== Muse Glimmer TRL SFT ===", flush=True)
    print(f"  config:     {_resolve(args.config)}", flush=True)
    print(f"  model:      {model_path}", flush=True)
    print(f"  max_length: {max_length}", flush=True)
    print(f"  qlora:      {qlora}", flush=True)
    print(f"  output:     {out_dir}", flush=True)

    train_ds = _load_concat_datasets(cached)
    print(f"  train rows: {len(train_ds):,}", flush=True)

    target_modules = list(
        train_cfg.get("target_modules")
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model, tokenizer = _build_model_and_tokenizer(
        model_path=model_path,
        qlora=qlora,
        torch_dtype=str(train_cfg.get("torch_dtype", "bfloat16")),
        freeze_vision=bool(train_cfg.get("freeze_vision", True)),
        target_modules=target_modules,
        lora_rank=int(args.lora_rank or train_cfg.get("lora_rank", 16)),
        lora_alpha=int(args.lora_alpha or train_cfg.get("lora_alpha", 32)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
    )

    from trl import SFTConfig, SFTTrainer

    lr = float(args.learning_rate or train_cfg.get("learning_rate", 2e-4))
    epochs = float(args.num_epochs or train_cfg.get("num_epochs", 2))
    bs = int(args.per_device_train_batch_size or train_cfg.get("per_device_train_batch_size", 1))
    gas = int(
        args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps", 8)
    )
    seed = int(args.seed or train_cfg.get("seed", 42))
    logging_steps = int(args.logging_steps or train_cfg.get("logging_steps", 10))
    save_steps = int(args.save_steps or train_cfg.get("save_steps", 200))
    save_limit = int(args.save_total_limit or train_cfg.get("save_total_limit", 3))
    warmup = float(args.warmup_ratio or train_cfg.get("warmup_ratio", 0.03))
    packing = bool(train_cfg.get("packing", False))
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    assistant_only = bool(train_cfg.get("assistant_only_loss", True))

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=gas,
        learning_rate=lr,
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_limit,
        warmup_ratio=warmup,
        seed=seed,
        max_length=max_length,
        packing=packing,
        gradient_checkpointing=grad_ckpt,
        dataset_kwargs={"skip_prepare_dataset": False},
        report_to=os.environ.get("REPORT_TO", "none"),
    )
    # Prefer assistant-only CE when TRL supports it.
    if assistant_only:
        sft_kwargs["assistant_only_loss"] = True

    try:
        sft_args = SFTConfig(**sft_kwargs)
    except TypeError:
        # Older TRL: drop unsupported keys
        for k in ("assistant_only_loss", "dataset_kwargs", "max_length"):
            sft_kwargs.pop(k, None)
        if "max_seq_length" not in sft_kwargs:
            sft_kwargs["max_seq_length"] = max_length
        sft_args = SFTConfig(**sft_kwargs)
        print("[muse] older TRL SFTConfig; assistant_only_loss may be unavailable", flush=True)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
    )
    # TRL API drift: processing_class vs tokenizer
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[muse] saved adapter → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
