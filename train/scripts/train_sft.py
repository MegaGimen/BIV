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

    out_dir = ROOT / tcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    response_part = str(tcfg.get("response_part", "<|im_start|>assistant\n"))
    ds_cache_root = out_dir / "ds_cache"
    cache_meta = _build_ds_cache_meta(
        train_path=train_path,
        eval_path=eval_path if eval_path and eval_path.exists() else None,
        max_seq_length=max_seq,
        response_part=response_part,
        model_name=str(mcfg["name"]),
        packing=bool(dcfg.get("packing", False)),
    )

    cached = _try_load_ready_datasets(ds_cache_root, cache_meta)
    if cached is not None:
        train_ready, eval_ready = cached
        print(
            f"Using disk cache for masked+filtered datasets under {ds_cache_root}",
            flush=True,
        )
        sft_args = _make_sft_config(out_dir, tcfg, dcfg, max_seq, eval_ready is not None)
        print("Building SFTTrainer from cached tokenized datasets...", flush=True)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ready,
            eval_dataset=eval_ready,
            args=sft_args,
        )
    else:
        train_ds = _load_jsonl_messages(train_path)
        eval_ds = (
            _load_jsonl_messages(eval_path) if eval_path and eval_path.exists() else None
        )

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
            load_from_cache_file=True,
        )
        if eval_ds is not None:
            eval_ds = eval_ds.map(
                formatting_func,
                batched=True,
                remove_columns=eval_ds.column_names,
                desc="format eval",
                load_from_cache_file=True,
            )

        sft_args = _make_sft_config(out_dir, tcfg, dcfg, max_seq, eval_ds is not None)

        print("Building SFTTrainer (may tokenize; can take a long time)...", flush=True)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=sft_args,
        )

        print(
            f"Masking non-response tokens to -100 (response_part={response_part!r})...",
            flush=True,
        )
        trainer.train_dataset = _mask_dataset_responses_only(
            trainer.train_dataset,
            tokenizer,
            response_part=response_part,
            name="train",
            cache_dir=ds_cache_root / "map",
        )
        if trainer.eval_dataset is not None:
            trainer.eval_dataset = _mask_dataset_responses_only(
                trainer.eval_dataset,
                tokenizer,
                response_part=response_part,
                name="eval",
                cache_dir=ds_cache_root / "map",
            )

        print("Filtering fully -100 rows...", flush=True)
        trainer.train_dataset = _filter_fully_masked_with_progress(
            trainer.train_dataset, "train_dataset"
        )
        if trainer.eval_dataset is not None:
            trainer.eval_dataset = _filter_fully_masked_with_progress(
                trainer.eval_dataset, "eval_dataset"
            )
        print("Response-only mask + -100 filter done.", flush=True)
        _save_ready_datasets(
            ds_cache_root,
            cache_meta,
            trainer.train_dataset,
            trainer.eval_dataset,
        )

    sample = trainer.train_dataset[0]
    labels = sample.get("labels")
    if labels is not None:
        n_sup = sum(1 for x in labels if int(x) != -100)
        print(f"Row0 supervised tokens: {n_sup}/{len(list(labels))}", flush=True)
        if n_sup == 0:
            raise SystemExit("Row 0 is fully -100 after masking — aborting.")

    trainer.train()
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out_dir / "train_config_snapshot.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Saved LoRA adapter -> {adapter_dir}", flush=True)


DS_CACHE_VERSION = 1


def _make_sft_config(out_dir, tcfg, dcfg, max_seq, has_eval: bool):
    from trl import SFTConfig

    return SFTConfig(
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
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(tcfg.get("eval_steps", 200)) if has_eval else None,
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


def _file_sig(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _build_ds_cache_meta(
    *,
    train_path: Path,
    eval_path: Path | None,
    max_seq_length: int,
    response_part: str,
    model_name: str,
    packing: bool,
) -> dict:
    return {
        "version": DS_CACHE_VERSION,
        "train": _file_sig(train_path),
        "eval": _file_sig(eval_path) if eval_path is not None else None,
        "max_seq_length": max_seq_length,
        "response_part": response_part,
        "model_name": model_name,
        "packing": packing,
    }


def _try_load_ready_datasets(cache_root: Path, meta: dict):
    meta_path = cache_root / "meta.json"
    train_dir = cache_root / "train_ready"
    if not meta_path.exists() or not train_dir.exists():
        return None
    try:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if prev != meta:
        print("Dataset disk cache miss (inputs/config changed).", flush=True)
        return None
    from datasets import load_from_disk

    train_ds = load_from_disk(str(train_dir))
    eval_ds = None
    eval_dir = cache_root / "eval_ready"
    if meta.get("eval") is not None:
        if not eval_dir.exists():
            return None
        eval_ds = load_from_disk(str(eval_dir))
    print(
        f"Loaded cached datasets: train={len(train_ds)}"
        + (f" eval={len(eval_ds)}" if eval_ds is not None else ""),
        flush=True,
    )
    return train_ds, eval_ds


def _save_ready_datasets(cache_root: Path, meta: dict, train_ds, eval_ds) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    train_dir = cache_root / "train_ready"
    print(f"Saving masked+filtered train dataset -> {train_dir}", flush=True)
    train_ds.save_to_disk(str(train_dir))
    if eval_ds is not None:
        eval_dir = cache_root / "eval_ready"
        print(f"Saving masked+filtered eval dataset -> {eval_dir}", flush=True)
        eval_ds.save_to_disk(str(eval_dir))
    (cache_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Dataset disk cache ready under {cache_root}", flush=True)


def _encode_marker(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Empty token ids for marker {text!r}")
    return list(ids)


def _mask_labels_for_responses(
    input_ids: list[int],
    labels: list[int] | None,
    response_ids: list[int],
    im_end_ids: list[int],
) -> list[int]:
    """Supervise tokens inside assistant spans; everything else → -100."""
    src = list(labels) if labels is not None else list(input_ids)
    n = len(input_ids)
    out = [-100] * n
    rlen = len(response_ids)
    elen = len(im_end_ids)
    i = 0
    while i <= n - rlen:
        if input_ids[i : i + rlen] == response_ids:
            j = i + rlen
            while j < n:
                out[j] = int(src[j])
                j += 1
                if elen and j >= elen and input_ids[j - elen : j] == im_end_ids:
                    break
            i = j
        else:
            i += 1
    return out


def _mask_dataset_responses_only(
    dataset,
    tokenizer,
    *,
    response_part: str,
    name: str,
    cache_dir: Path | None = None,
):
    """Apply response-only -100 mask with HuggingFace map progress (+ disk cache)."""
    from datasets.utils.logging import enable_progress_bar

    enable_progress_bar()
    if "input_ids" not in dataset.column_names:
        raise SystemExit(
            f"{name} dataset missing input_ids after SFTTrainer init; columns={dataset.column_names}"
        )

    response_ids = _encode_marker(tokenizer, response_part)
    try:
        im_end_ids = _encode_marker(tokenizer, "<|im_end|>")
    except Exception:  # noqa: BLE001
        im_end_ids = []

    print(
        f"{name}: response_ids={response_ids[:12]}{'...' if len(response_ids) > 12 else ''} "
        f"len={len(response_ids)}; im_end_len={len(im_end_ids)}",
        flush=True,
    )

    def _batch_mask(batch):
        input_ids_batch = batch["input_ids"]
        labels_batch = batch.get("labels", input_ids_batch)
        new_labels = []
        for input_ids, labels in zip(input_ids_batch, labels_batch):
            new_labels.append(
                _mask_labels_for_responses(
                    list(input_ids),
                    list(labels) if labels is not None else None,
                    response_ids,
                    im_end_ids,
                )
            )
        return {"labels": new_labels}

    map_kwargs = {
        "batched": True,
        "batch_size": 256,
        "desc": f"mask responses ({name})",
        "load_from_cache_file": True,
    }
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        map_kwargs["cache_file_name"] = str(cache_dir / f"{name}_response_mask.arrow")

    return dataset.map(_batch_mask, **map_kwargs)


def _filter_fully_masked_with_progress(dataset, name="dataset"):  # noqa: ANN001
    """Drop rows whose labels are all -100; show tqdm + corpus -100 ratios."""
    from tqdm.auto import tqdm

    n = len(dataset)
    if n == 0:
        print(f"{name}: empty dataset", flush=True)
        return dataset

    batch_size = 1000
    keep: list[int] = []
    total_tok = 0
    masked_tok = 0
    fully_masked_rows = 0
    idx = 0

    view = dataset
    if "labels" in getattr(dataset, "column_names", []):
        view = dataset.select_columns(["labels"])

    pbar = tqdm(total=n, desc=f"scan -100 ({name})", unit="ex", dynamic_ncols=True)
    try:
        for batch in view.iter(batch_size=batch_size):
            labels_batch = batch["labels"]
            for labels in labels_batch:
                try:
                    labels_list = list(labels)
                except TypeError:
                    labels_list = [labels]
                n_lab = len(labels_list)
                if n_lab == 0:
                    fully_masked_rows += 1
                    idx += 1
                    pbar.update(1)
                    continue
                n_mask = sum(1 for x in labels_list if int(x) == -100)
                n_sup = n_lab - n_mask
                total_tok += n_lab
                masked_tok += n_mask
                if n_sup == 0:
                    fully_masked_rows += 1
                else:
                    keep.append(idx)
                idx += 1
                pbar.update(1)
                if idx % 5000 == 0 and total_tok:
                    pbar.set_postfix(
                        drop=f"{fully_masked_rows}/{idx}",
                        m100=f"{masked_tok / total_tok:.0%}",
                        refresh=False,
                    )
    finally:
        pbar.close()

    dropped = n - len(keep)
    mask_ratio = (masked_tok / total_tok) if total_tok else 0.0
    print(
        f"{name} -100 filter:\n"
        f"  rows kept={len(keep)}/{n}  dropped_fully_masked={dropped} ({dropped / n:.1%})\n"
        f"  tokens: -100={mask_ratio:.1%}  supervised={1.0 - mask_ratio:.1%}  "
        f"(tok_total={total_tok})",
        flush=True,
    )
    if not keep:
        raise ValueError(
            f"{name}: every row was fully -100 after response masking. "
            "Check instruction_part/response_part vs chat template / max_seq_length."
        )
    if dropped / n >= 0.30:
        print(
            f"WARNING: dropped {dropped / n:.1%} of {name} as fully -100. "
            "Mask markers or max_seq_length may be wrong.",
            flush=True,
        )
    return dataset.select(keep)


if __name__ == "__main__":
    # Avoid accidental CPU thrash on shared login nodes.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
