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
    """Memory-map JSONL via datasets Arrow — do not from_list the whole file."""
    from datasets import load_dataset

    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"empty or missing dataset: {path}")
    ds = load_dataset("json", data_files=str(path), split="train")
    if "messages" not in ds.column_names:
        raise SystemExit(f"{path} missing 'messages' column; columns={ds.column_names}")
    return ds


def _resolve_model_path(mcfg: dict) -> str:
    """Resolve checkpoint path; prefer ModelScope downloads in CN."""
    name = str(mcfg["name"])
    source = str(mcfg.get("source", "huggingface")).lower()
    if source in {"local", "path"}:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"Local model path not found: {path}")
        return str(path)

    if source in {"modelscope", "ms"}:
        from modelscope import snapshot_download

        print(f"Downloading/loading model from ModelScope: {name}", flush=True)
        local = snapshot_download(name)
        print(f"ModelScope cache path: {local}", flush=True)
        return str(local)

    # huggingface / hf-mirror via HF_ENDPOINT
    print(f"Loading model id from HuggingFace hub: {name}", flush=True)
    return name


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
    model_path = _resolve_model_path(mcfg)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
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

    print("Building SFTTrainer (may tokenize; can take a long time)...", flush=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    # Unsloth's train_on_responses_only ends with `_filter_fully_masked`, which
    # iterates the whole tokenized corpus via `to_pydict()` — on ~1e5–1e6 long
    # sequences this looks "stuck" for tens of minutes with 100% CPU and no bar.
    # Masking is what we need; dropping the rare all-masked rows is optional.
    print("Applying response-only loss mask (skipping full-corpus mask filter)...", flush=True)
    try:
        import unsloth_zoo.dataset_utils as _dataset_utils

        def _skip_full_mask_filter(dataset, name="dataset"):  # noqa: ANN001
            print(
                f"Skip _filter_fully_masked on {name} "
                f"(n≈{getattr(dataset, '__len__', lambda: '?')()})",
                flush=True,
            )
            return dataset

        _dataset_utils._filter_fully_masked = _skip_full_mask_filter
    except Exception as exc:  # noqa: BLE001
        print(f"Could not patch _filter_fully_masked ({exc}); may be slow.", flush=True)

    trainer = train_on_responses_only(
        trainer,
        instruction_part=str(tcfg.get("instruction_part", "<|im_start|>user\n")),
        response_part=str(tcfg.get("response_part", "<|im_start|>assistant\n")),
    )
    print("Response-only mask applied.", flush=True)
    _report_label_mask_stats(
        trainer.train_dataset,
        sample_size=int(tcfg.get("mask_stats_sample_size", 512)),
        seed=int(tcfg.get("seed", 42)),
    )

    trainer.train()
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out_dir / "train_config_snapshot.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Saved LoRA adapter -> {adapter_dir}", flush=True)


def _report_label_mask_stats(dataset, *, sample_size: int, seed: int) -> None:
    """Sample rows and print -100 / supervised token ratios (cheap; not full scan)."""
    import random

    n = len(dataset)
    if n == 0:
        raise SystemExit("train_dataset is empty after masking.")
    k = min(int(sample_size), n)
    rng = random.Random(seed)
    idxs = list(range(n)) if k == n else rng.sample(range(n), k)

    total_tok = 0
    masked_tok = 0
    supervised_tok = 0
    fully_masked_rows = 0
    empty_label_rows = 0

    for i in idxs:
        row = dataset[i]
        labels = row.get("labels")
        if labels is None:
            empty_label_rows += 1
            continue
        # HF may return list or numpy-like
        try:
            labels = list(labels)
        except TypeError:
            labels = [labels]
        if not labels:
            empty_label_rows += 1
            continue
        n_lab = len(labels)
        n_mask = sum(1 for x in labels if int(x) == -100)
        n_sup = n_lab - n_mask
        total_tok += n_lab
        masked_tok += n_mask
        supervised_tok += n_sup
        if n_sup == 0:
            fully_masked_rows += 1

    scanned = k - empty_label_rows
    if scanned <= 0 or total_tok == 0:
        raise SystemExit("No usable labels found while sampling mask stats.")

    mask_ratio = masked_tok / total_tok
    sup_ratio = supervised_tok / total_tok
    full_mask_row_ratio = fully_masked_rows / scanned

    print(
        "Label mask stats "
        f"(sample {scanned}/{n} rows, seed={seed}):\n"
        f"  tokens: -100={mask_ratio:.1%}  supervised={sup_ratio:.1%}  "
        f"(tok_total={total_tok})\n"
        f"  rows fully -100: {fully_masked_rows}/{scanned} = {full_mask_row_ratio:.1%}\n"
        f"  rows missing labels: {empty_label_rows}/{k}",
        flush=True,
    )

    if full_mask_row_ratio >= 0.30:
        raise SystemExit(
            f"Too many fully-masked rows in sample ({full_mask_row_ratio:.1%}). "
            "Check instruction_part/response_part vs chat template, or max_seq_length."
        )
    # Spot-check first row too
    labels0 = dataset[0].get("labels")
    if labels0 is not None and sum(1 for x in labels0 if int(x) != -100) == 0:
        raise SystemExit(
            "Row 0 is fully -100 after masking — aborting before train()."
        )


if __name__ == "__main__":
    # Avoid accidental CPU thrash on shared login nodes.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
