#!/usr/bin/env python3
"""CUDA LoRA SFT: fit real tool observations with Unsloth + Qwen3.5-9B.

Requires a CUDA GPU (designed for ~40GB; bf16 LoRA). Do not run on CPU-only hosts.

  cd train
  source .venv/bin/activate
  python scripts/train_sft.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for scripts/train_sft.py. "
            "This host has no GPU — run prepare_data.py here if needed, "
            "and launch training on a CUDA machine."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"CUDA OK: {torch.cuda.get_device_name(0)} "
        f"({props.total_memory / 1024**3:.1f} GiB), torch {torch.__version__}",
        flush=True,
    )


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_jsonl_messages(path: Path):
    from datasets import Dataset

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"row missing messages: {path}")
            rows.append({"messages": obj["messages"]})
    if not rows:
        raise SystemExit(f"empty dataset: {path}")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Dangerous: skip CUDA check (Unsloth training is unsupported here)",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)

    if not args.allow_cpu:
        _require_cuda()

    # Import Unsloth only after CUDA check so CPU servers fail fast with a clear message.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    mcfg = cfg["model"]
    tcfg = cfg["train"]
    dcfg = cfg["data"]

    train_path = ROOT / dcfg["train_file"]
    eval_path = ROOT / dcfg.get("eval_file", "")
    if not train_path.exists():
        raise SystemExit(
            f"Missing {train_path}. Run scripts/prepare_data.py first."
        )

    max_seq = int(mcfg["max_seq_length"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=max_seq,
        load_in_4bit=bool(mcfg.get("load_in_4bit", False)),
        load_in_16bit=bool(mcfg.get("load_in_16bit", True)),
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(mcfg.get("lora_r", 16)),
        target_modules=list(mcfg.get("target_modules")),
        lora_alpha=int(mcfg.get("lora_alpha", 16)),
        lora_dropout=float(mcfg.get("lora_dropout", 0.0)),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(tcfg.get("seed", 42)),
        max_seq_length=max_seq,
    )

    train_ds = _load_jsonl_messages(train_path)
    eval_ds = _load_jsonl_messages(eval_path) if eval_path and eval_path.exists() else None

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)
        return {"text": texts}

    train_ds = train_ds.map(
        formatting_func,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="format train",
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            formatting_func,
            batched=True,
            remove_columns=eval_ds.column_names,
            desc="format eval",
        )

    out_dir = ROOT / tcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        seed=int(tcfg.get("seed", 42)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(tcfg.get("eval_steps", 200)) if eval_ds is not None else None,
        save_steps=int(tcfg.get("save_steps", 200)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        optim=str(tcfg.get("optim", "adamw_8bit")),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=False,
        report_to=tcfg.get("report_to", "none"),
        max_seq_length=max_seq,
        packing=bool(dcfg.get("packing", False)),
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=str(tcfg.get("instruction_part", "<|im_start|>user\n")),
        response_part=str(tcfg.get("response_part", "<|im_start|>assistant\n")),
    )

    # Sanity-check masking: some labels must be supervised.
    sample = trainer.train_dataset[0]
    labels = sample.get("labels")
    if labels is not None:
        n_sup = sum(1 for x in labels if x != -100)
        print(f"Response-mask check: {n_sup}/{len(labels)} supervised tokens", flush=True)
        if n_sup == 0:
            raise SystemExit(
                "All labels are -100 — instruction/response_part markers do not "
                "match the chat template. Aborting."
            )

    trainer.train()
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out_dir / "train_config_snapshot.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Saved LoRA adapter -> {adapter_dir}", flush=True)


if __name__ == "__main__":
    # Avoid accidental CPU thrash on shared login nodes.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
