#!/usr/bin/env python3
"""CUDA LoRA SFT: fit real tool observations with Unsloth + Qwen3.5-9B.

Requires a CUDA GPU (designed for ~40GB; bf16 LoRA). Do not run on CPU-only hosts.

  cd train
  source .venv/bin/activate
  python scripts/train_sft.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint under train.output_dir",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from this checkpoint directory (overrides --resume)",
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
    max_train_samples = dcfg.get("max_train_samples")
    max_eval_samples = dcfg.get("max_eval_samples")
    max_train_samples = int(max_train_samples) if max_train_samples else None
    max_eval_samples = int(max_eval_samples) if max_eval_samples else None
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
    packing = bool(dcfg.get("packing", False))
    cache_meta = _build_ds_cache_meta(
        train_path=train_path,
        eval_path=eval_path if eval_path and eval_path.exists() else None,
        max_seq_length=max_seq,
        response_part=response_part,
        model_name=str(mcfg["name"]),
        packing=packing,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    shared_cache_root = _shared_ds_cache_root(dcfg)
    ds_cache_dir = _variant_cache_dir(shared_cache_root, cache_meta)
    print(f"Shared dataset cache root: {shared_cache_root}", flush=True)
    print(f"This-run cache dir: {ds_cache_dir}", flush=True)

    cached = _resolve_ready_datasets(
        shared_cache_root,
        cache_meta,
        seed=int(tcfg.get("seed", 42)),
        legacy_dirs=[
            out_dir / "ds_cache",
            ROOT / "outputs" / "wm_sft" / "ds_cache",
            ROOT / "outputs" / "wm_sft_pilot" / "ds_cache",
            ROOT / "outputs" / "wm_sft_pilot_shuffled" / "ds_cache",
        ],
    )
    # Pretokenized ready data: packing is a tokenize-time option only.
    if cached is not None:
        train_ready, eval_ready, cache_note = cached
        print(cache_note, flush=True)
        sft_args = _make_sft_config(
            out_dir, tcfg, dcfg, max_seq, eval_ready is not None, packing=False
        )
        print("Building SFTTrainer from cached tokenized datasets...", flush=True)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ready,
            eval_dataset=eval_ready,
            args=sft_args,
            callbacks=_make_epoch_save_eval_callbacks(eval_ready is not None),
        )
        if packing:
            print(
                "Note: packing=true ignored for pretokenized cache reuse "
                "(sequences already built). Truncate/subset reuse still applied.",
                flush=True,
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

        # Format BEFORE subset so HuggingFace map cache from full-corpus runs can hit.
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
        train_ds = _maybe_subset(
            train_ds,
            max_train_samples,
            name="train",
            seed=int(tcfg.get("seed", 42)),
        )
        if eval_ds is not None:
            eval_ds = _maybe_subset(
                eval_ds,
                max_eval_samples,
                name="eval",
                seed=int(tcfg.get("seed", 42)) + 1,
            )

        sft_args = _make_sft_config(
            out_dir, tcfg, dcfg, max_seq, eval_ds is not None, packing=packing
        )

        print("Building SFTTrainer (may tokenize; can take a long time)...", flush=True)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=sft_args,
            callbacks=_make_epoch_save_eval_callbacks(eval_ds is not None),
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
            cache_dir=ds_cache_dir / "map",
        )
        if trainer.eval_dataset is not None:
            trainer.eval_dataset = _mask_dataset_responses_only(
                trainer.eval_dataset,
                tokenizer,
                response_part=response_part,
                name="eval",
                cache_dir=ds_cache_dir / "map",
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
            ds_cache_dir,
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

    resume_ckpt = _resolve_resume_checkpoint(
        out_dir,
        resume=bool(args.resume or tcfg.get("resume", False)),
        resume_from=args.resume_from or tcfg.get("resume_from"),
    )
    if resume_ckpt:
        print(f"Resuming training from {resume_ckpt!r}", flush=True)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out_dir / "train_config_snapshot.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Saved LoRA adapter -> {adapter_dir}", flush=True)


DS_CACHE_VERSION = 3


def _resolve_resume_checkpoint(
    out_dir: Path,
    *,
    resume: bool,
    resume_from: Path | str | None,
):
    """Return a checkpoint path / True for Trainer, or None to start fresh."""
    if resume_from:
        path = Path(resume_from)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.exists():
            raise SystemExit(f"--resume-from not found: {path}")
        return str(path)
    if not resume:
        return None
    # Trainer interprets True as "latest checkpoint under output_dir"
    checkpoints = sorted(
        out_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    if not checkpoints:
        print(
            f"--resume set but no checkpoint-* under {out_dir}; starting fresh.",
            flush=True,
        )
        return None
    latest = checkpoints[-1]
    print(f"Auto-selected latest checkpoint: {latest}", flush=True)
    return str(latest)


def _make_epoch_save_eval_callbacks(has_eval: bool):
    """Force a checkpoint (+ eval when available) at every epoch boundary."""
    from transformers import TrainerCallback

    class _EpochSaveEvalCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001
            control.should_save = True
            if has_eval:
                control.should_evaluate = True
            print(
                f"Epoch {state.epoch}: requesting checkpoint save"
                + (" + eval" if has_eval else ""),
                flush=True,
            )
            return control

    return [_EpochSaveEvalCallback()]


def _make_sft_config(out_dir, tcfg, dcfg, max_seq, has_eval: bool, packing: bool | None = None):
    from trl import SFTConfig

    if packing is None:
        packing = bool(dcfg.get("packing", False))
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
        save_strategy="steps",
        save_steps=int(tcfg.get("save_steps", 35)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        save_on_each_node=True,
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        optim=str(tcfg.get("optim", "adamw_8bit")),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=False,
        report_to=tcfg.get("report_to", "none"),
        max_seq_length=max_seq,
        packing=packing,
        dataset_text_field="text",
        remove_unused_columns=False,
    )


def _file_sig(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _maybe_subset(dataset, max_samples: int | None, *, name: str, seed: int):
    """Deterministic head-after-shuffle subset for fast pilots."""
    if max_samples is None or max_samples <= 0:
        return dataset
    n = len(dataset)
    if max_samples >= n:
        print(f"{name}: using full {n} rows (max_samples={max_samples})", flush=True)
        return dataset
    print(f"{name}: subset {max_samples}/{n} (seed={seed})", flush=True)
    return dataset.shuffle(seed=seed).select(range(max_samples))


def _sample_cap(meta: dict, key: str) -> int | None:
    v = meta.get(key)
    if v is None or v == "" or int(v) <= 0:
        return None
    return int(v)


def _build_ds_cache_meta(
    *,
    train_path: Path,
    eval_path: Path | None,
    max_seq_length: int,
    response_part: str,
    model_name: str,
    packing: bool,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict:
    return {
        "version": DS_CACHE_VERSION,
        "train": _file_sig(train_path),
        "eval": _file_sig(eval_path) if eval_path is not None else None,
        "max_seq_length": max_seq_length,
        "response_part": response_part,
        "model_name": model_name,
        "packing": packing,
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
    }


def _meta_fingerprint(meta: dict) -> str:
    blob = json.dumps(meta, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _shared_ds_cache_root(dcfg: dict) -> Path:
    raw = dcfg.get("ds_cache_dir", "outputs/ds_cache")
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path)


def _variant_cache_dir(shared_root: Path, meta: dict) -> Path:
    return shared_root / f"v{meta.get('version', DS_CACHE_VERSION)}_{_meta_fingerprint(meta)}"


def _same_source_meta(a: dict, b: dict) -> bool:
    return (
        a.get("train") == b.get("train")
        and a.get("eval") == b.get("eval")
        and a.get("response_part") == b.get("response_part")
        and a.get("model_name") == b.get("model_name")
    )


def _is_exact_meta(prev: dict, meta: dict) -> bool:
    # Ignore version drift between 2↔3 if content fields match.
    keys = [
        "train",
        "eval",
        "max_seq_length",
        "response_part",
        "model_name",
        "packing",
        "max_train_samples",
        "max_eval_samples",
    ]
    for k in keys:
        if prev.get(k) != meta.get(k):
            return False
    return True


def _is_superset_meta(prev: dict, meta: dict) -> bool:
    """True if prev tokenized ready can be truncated/subsetted into meta."""
    if not _same_source_meta(prev, meta):
        return False
    if int(prev.get("max_seq_length", 0)) < int(meta["max_seq_length"]):
        return False
    # Cannot unpack packed sequences back into rows.
    if prev.get("packing") and not meta.get("packing"):
        return False
    pt, mt = _sample_cap(prev, "max_train_samples"), _sample_cap(meta, "max_train_samples")
    pe, me = _sample_cap(prev, "max_eval_samples"), _sample_cap(meta, "max_eval_samples")
    if mt is None and pt is not None:
        return False
    if mt is not None and pt is not None and pt < mt:
        return False
    if me is None and pe is not None:
        return False
    if me is not None and pe is not None and pe < me:
        return False
    return True


def _iter_cache_candidate_dirs(shared_root: Path, legacy_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    if shared_root.exists():
        for child in sorted(shared_root.iterdir()):
            if child.is_dir() and (child / "meta.json").exists():
                _add(child)
    for leg in legacy_dirs:
        if (leg / "meta.json").exists() and (leg / "train_ready").exists():
            _add(leg)
    return out


def _load_meta(cache_dir: Path) -> dict | None:
    path = cache_dir / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_ready_pair(cache_dir: Path, meta: dict):
    from datasets import load_from_disk

    train_dir = cache_dir / "train_ready"
    if not train_dir.exists():
        return None
    train_ds = load_from_disk(str(train_dir))
    eval_ds = None
    if meta.get("eval") is not None:
        eval_dir = cache_dir / "eval_ready"
        if not eval_dir.exists():
            return None
        eval_ds = load_from_disk(str(eval_dir))
    return train_ds, eval_ds


def _truncate_token_fields(dataset, max_len: int, desc: str):
    cols = [c for c in ("input_ids", "labels", "attention_mask") if c in dataset.column_names]
    if not cols:
        return dataset

    def _batch(batch):
        out = {}
        for k, vals in batch.items():
            if k in cols:
                out[k] = [row[:max_len] for row in vals]
            else:
                out[k] = vals
        return out

    return dataset.map(_batch, batched=True, desc=desc, load_from_cache_file=False)


def _derive_ready_from_superset(
    src_dir: Path,
    src_meta: dict,
    want_meta: dict,
    *,
    seed: int,
    dest_dir: Path,
):
    loaded = _load_ready_pair(src_dir, src_meta)
    if loaded is None:
        return None
    train_ds, eval_ds = loaded
    max_len = int(want_meta["max_seq_length"])
    if int(src_meta.get("max_seq_length", 0)) > max_len:
        print(
            f"Deriving cache: truncate seq {src_meta.get('max_seq_length')} → {max_len}",
            flush=True,
        )
        train_ds = _truncate_token_fields(train_ds, max_len, "truncate train")
        if eval_ds is not None:
            eval_ds = _truncate_token_fields(eval_ds, max_len, "truncate eval")

    train_ds = _maybe_subset(
        train_ds,
        _sample_cap(want_meta, "max_train_samples"),
        name="train",
        seed=seed,
    )
    if eval_ds is not None:
        eval_ds = _maybe_subset(
            eval_ds,
            _sample_cap(want_meta, "max_eval_samples"),
            name="eval",
            seed=seed + 1,
        )

    # Pretokenized reuse stores packing=false (cannot invent packing after the fact).
    save_meta = dict(want_meta)
    save_meta["packing"] = False
    save_meta["derived_from"] = {
        "dir": str(src_dir),
        "max_seq_length": src_meta.get("max_seq_length"),
        "packing": src_meta.get("packing"),
    }
    _save_ready_datasets(dest_dir, save_meta, train_ds, eval_ds)
    return train_ds, eval_ds


def _resolve_ready_datasets(
    shared_root: Path,
    meta: dict,
    *,
    seed: int,
    legacy_dirs: list[Path],
):
    """Exact hit, packing-relaxed hit, or derive via truncate/subset from a longer ready cache."""
    dest = _variant_cache_dir(shared_root, meta)
    candidates = _iter_cache_candidate_dirs(shared_root, legacy_dirs)

    # 1) Exact
    for cdir in [dest, *candidates]:
        prev = _load_meta(cdir)
        if prev is None:
            continue
        if _is_exact_meta(prev, meta):
            loaded = _load_ready_pair(cdir, prev)
            if loaded is not None:
                train_ds, eval_ds = loaded
                note = (
                    f"Using exact dataset cache under {cdir} "
                    f"(train={len(train_ds)}"
                    + (f" eval={len(eval_ds)}" if eval_ds is not None else "")
                    + ")"
                )
                if cdir != dest and not dest.exists():
                    _save_ready_datasets(dest, meta, train_ds, eval_ds)
                    note += f"; mirrored → {dest}"
                return train_ds, eval_ds, note

    # 2) Same rows/seq, only packing differs (want packed, have unpacked pretokenized)
    for cdir in candidates:
        prev = _load_meta(cdir)
        if prev is None:
            continue
        if (
            _same_source_meta(prev, meta)
            and int(prev.get("max_seq_length", -1)) == int(meta["max_seq_length"])
            and _sample_cap(prev, "max_train_samples") == _sample_cap(meta, "max_train_samples")
            and _sample_cap(prev, "max_eval_samples") == _sample_cap(meta, "max_eval_samples")
            and (not prev.get("packing"))
            and meta.get("packing")
        ):
            loaded = _load_ready_pair(cdir, prev)
            if loaded is not None:
                train_ds, eval_ds = loaded
                note = (
                    f"Reusing unpacked tokenized cache under {cdir} "
                    f"(packing request relaxed for pretokenized data)"
                )
                return train_ds, eval_ds, note

    # 3) Derive from longer / fuller unpacked (or compatible) ready cache
    supersets = []
    for cdir in candidates:
        prev = _load_meta(cdir)
        if prev is None:
            continue
        if _is_superset_meta(prev, meta):
            supersets.append((cdir, prev))
    # Prefer closest max_seq_length, then largest train coverage.
    supersets.sort(
        key=lambda x: (
            int(x[1].get("max_seq_length", 0)),
            _sample_cap(x[1], "max_train_samples") is None,
            _sample_cap(x[1], "max_train_samples") or 0,
        )
    )
    if supersets:
        cdir, prev = supersets[0]
        print(f"Dataset cache derive from {cdir}", flush=True)
        # Derived artifacts are always unpacked sequences.
        save_meta = dict(meta)
        save_meta["packing"] = False
        dest_eff = _variant_cache_dir(shared_root, save_meta)
        derived = _derive_ready_from_superset(
            cdir, prev, save_meta, seed=seed, dest_dir=dest_eff
        )
        if derived is not None:
            train_ds, eval_ds = derived
            note = (
                f"Derived dataset cache → {dest_eff} "
                f"(train={len(train_ds)}"
                + (f" eval={len(eval_ds)}" if eval_ds is not None else "")
                + "). packing ignored for pretokenized reuse."
            )
            return train_ds, eval_ds, note

    print("Dataset disk cache miss (no exact/compatible ready cache).", flush=True)
    return None


def _try_load_ready_datasets(cache_root: Path, meta: dict):
    """Backward-compatible helper (exact load only)."""
    prev = _load_meta(cache_root)
    if prev is None or not _is_exact_meta(prev, meta):
        return None
    return _load_ready_pair(cache_root, prev)


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


def _unwrap_tokenizer(tokenizer_or_processor):
    """Qwen3.5 Unsloth may return Qwen3VLProcessor; prefer the inner tokenizer."""
    if hasattr(tokenizer_or_processor, "encode") and callable(tokenizer_or_processor.encode):
        return tokenizer_or_processor
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode"):
        print(
            f"Using processor.tokenizer ({type(inner).__name__}) "
            f"instead of {type(tokenizer_or_processor).__name__}",
            flush=True,
        )
        return inner
    raise TypeError(
        f"Cannot encode with {type(tokenizer_or_processor)!r}; "
        "expected a tokenizer or processor with .tokenizer"
    )


def _encode_marker(tokenizer_or_processor, text: str) -> list[int]:
    tokenizer = _unwrap_tokenizer(tokenizer_or_processor)
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
