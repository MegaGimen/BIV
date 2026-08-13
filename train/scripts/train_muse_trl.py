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
    from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk

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

        # Rebuild with an explicit schema. map()+cast() keeps the old Arrow
        # struct (incl. tool_calls) and fails on anti_forget rows.
        rows = [{"messages": _normalize_messages(ex["messages"])} for ex in ds]
        part = Dataset.from_list(rows, features=msg_features)
        parts.append(part)
        print(f"  loaded {p} ({len(part):,} rows)", flush=True)
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def _is_main_process() -> bool:
    try:
        from accelerate import PartialState

        return bool(PartialState().is_main_process)
    except Exception:
        return int(os.environ.get("LOCAL_RANK", "0") or "0") == 0


def _barrier() -> None:
    try:
        from accelerate import PartialState

        PartialState().wait_for_everyone()
    except Exception:
        return


def _tokenized_cache_dir(
    cached: list[Path],
    *,
    max_length: int,
    assistant_only: bool,
    truncation_mode: str,
) -> Path:
    """Stable path beside prep run: …/train_runs/<run>/tokenized_mlN_…"""
    import hashlib

    run_root = cached[0].resolve().parent
    # v3 = offline single-process tokenize (avoids multi-GPU barrier during TRL prepare)
    parts = [f"v3|ml={max_length}|ao={int(assistant_only)}|trunc={truncation_mode}|gen=1"]
    for p in cached:
        rp = p.resolve()
        parts.append(str(rp))
        for name in ("state.json", "dataset_info.json"):
            meta = rp / name
            if meta.is_file():
                st = meta.stat()
                parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
                break
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return run_root / (
        f"tokenized_ml{max_length}_ao{int(assistant_only)}_{truncation_mode}_{digest}"
    )


def _tokenized_cache_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "state.json").is_file() or (path / "dataset_info.json").is_file()


def _try_load_tokenized_cache(path: Path):
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    if "input_ids" not in ds.column_names:
        raise SystemExit(f"Tokenized cache missing input_ids: {path} cols={ds.column_names}")
    if "labels" not in ds.column_names:
        raise SystemExit(f"Tokenized cache missing labels: {path} cols={ds.column_names}")
    return ds


def _save_tokenized_cache(dataset, path: Path) -> None:
    from datasets import Dataset

    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("input_ids", "labels", "attention_mask") if c in dataset.column_names]
    to_save = dataset.select_columns(keep) if keep and hasattr(dataset, "select_columns") else dataset
    if not isinstance(to_save, Dataset):
        # Iterable / custom — materialize via from_list if needed
        to_save = Dataset.from_list([{k: row[k] for k in keep} for row in to_save])
    to_save.save_to_disk(str(path))
    print(f"[muse] wrote tokenized cache → {path} ({len(to_save):,} rows)", flush=True)


def _load_tokenizer_only(model_path: str):
    from transformers import AutoTokenizer

    print(f"[muse] loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _ensure_training_chat_template(tokenizer)
    return tokenizer


def _tokenize_messages_offline(
    ds,
    tokenizer,
    *,
    max_length: int,
    truncation_mode: str,
    assistant_only: bool,
    num_proc: int,
):
    """Mirror TRL SFT prepare (tokenize → labels → truncate) without a distributed barrier."""

    def _as_msgs(raw):
        out = []
        for m in raw:
            if isinstance(m, dict):
                out.append({"role": str(m.get("role") or "user"), "content": m.get("content") or ""})
            else:
                out.append({"role": str(m["role"]), "content": m["content"] or ""})
        return out

    def _map(ex):
        msgs = _as_msgs(ex["messages"])
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "add_generation_prompt": False,
        }
        if assistant_only:
            kwargs["return_assistant_tokens_mask"] = True
        out = tokenizer.apply_chat_template(msgs, **kwargs)
        ids = out["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        ids = list(ids)
        if assistant_only:
            mask = out.get("assistant_masks")
            if mask is None:
                mask = out.get("assistant_mask")
            if mask is not None and mask and isinstance(mask[0], (list, tuple)):
                mask = mask[0]
            if mask is None or len(mask) != len(ids):
                labels = list(ids)
            else:
                labels = [tid if bool(mask[i]) else -100 for i, tid in enumerate(ids)]
        else:
            labels = list(ids)
        if truncation_mode == "keep_end":
            ids = ids[-max_length:]
            labels = labels[-max_length:]
        else:
            ids = ids[:max_length]
            labels = labels[:max_length]
        keep = 1
        if assistant_only and not any(x != -100 for x in labels):
            keep = 0
        return {"input_ids": ids, "labels": labels, "_keep": keep}

    nproc = max(1, int(num_proc))
    mapped = ds.map(
        _map,
        num_proc=nproc if nproc > 1 else None,
        remove_columns=ds.column_names,
        desc="Tokenizing train dataset (offline)",
    )
    before = len(mapped)
    mapped = mapped.filter(lambda r: int(r["_keep"]) == 1, num_proc=nproc if nproc > 1 else None)
    mapped = mapped.remove_columns(["_keep"])
    print(
        f"[muse] offline tokenize kept {len(mapped):,}/{before:,} "
        f"(dropped rows with no assistant labels)",
        flush=True,
    )
    return mapped


def _ensure_training_chat_template(tokenizer) -> None:
    """Wrap Muse assistant turns with TRL ``{% generation %}`` markers in-place.

    Glimmer's shipped Jinja is prefix-preserving and tool-capable, but lacks
    generation markers. TRL 1.8 ``SFTTrainer`` with ``assistant_only_loss=True``
    then raises (only known families like GPT-OSS/Qwen get auto-patched).
    Markers are whitespace-only for rendering; they drive assistant token masks.
    """
    import re

    ct = getattr(tokenizer, "chat_template", None)
    if not isinstance(ct, str) or not ct:
        raise SystemExit("Tokenizer has no chat_template; cannot enable assistant_only_loss")
    if re.search(r"\{%-?\s*generation\s*-?%\}", ct):
        print("[muse] chat_template already has {% generation %} markers", flush=True)
        return

    start = "{%- elif role == 'assistant' -%}"
    end_needle = "{%- endfor -%}{%- if add_generation_prompt"
    if start not in ct or end_needle not in ct:
        raise SystemExit(
            "Muse chat_template shape unexpected; cannot inject {% generation %} "
            "for assistant_only_loss. Update _ensure_training_chat_template or set "
            "train.assistant_only_loss: false."
        )
    idx = ct.find(start) + len(start)
    endfor_pos = ct.find(end_needle)
    last_endif = ct.rfind("{%- endif -%}", 0, endfor_pos)
    if last_endif < idx:
        raise SystemExit("Failed to locate assistant block end for generation markers")
    tokenizer.chat_template = (
        ct[:idx]
        + "{%- generation %}"
        + ct[idx:last_endif]
        + "{%- endgeneration %}"
        + ct[last_endif:]
    )
    print("[muse] injected {% generation %} markers into chat_template", flush=True)


def _load_muse_glimmer(model_path: str, load_kwargs: dict[str, Any]):
    """Load the generative Muse Glimmer head (not the bare MuseGlimmerModel).

    AutoModelForCausalLM does not map muse_glimmer; falling back to AutoModel
    yields MuseGlimmerModel without prepare_inputs_for_generation / lm_head,
    which breaks PEFT PeftModelForCausalLM.
    """
    errors: list[str] = []

    try:
        from transformers import MuseGlimmerForConditionalGeneration

        print("[muse] loading MuseGlimmerForConditionalGeneration …", flush=True)
        return MuseGlimmerForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        errors.append(f"MuseGlimmerForConditionalGeneration: {e}")

    try:
        from transformers import AutoModelForImageTextToText

        print("[muse] loading AutoModelForImageTextToText …", flush=True)
        return AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        errors.append(f"AutoModelForImageTextToText: {e}")

    try:
        from transformers import AutoModelForMultimodalLM

        print("[muse] loading AutoModelForMultimodalLM …", flush=True)
        return AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        errors.append(f"AutoModelForMultimodalLM: {e}")

    raise SystemExit(
        "Failed to load Muse Glimmer generative model. Tried:\n  - "
        + "\n  - ".join(errors)
        + "\nNeed transformers with muse_glimmer (vendor editable install)."
    )


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
    from transformers import AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if str(torch_dtype).lower() in {"bf16", "bfloat16"} else torch.float16
    print(f"[muse] loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _ensure_training_chat_template(tokenizer)

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
    model = _load_muse_glimmer(model_path, load_kwargs)
    print(f"[muse] loaded class={type(model).__name__}", flush=True)
    if not hasattr(model, "prepare_inputs_for_generation"):
        raise SystemExit(
            f"{type(model).__name__} lacks prepare_inputs_for_generation "
            "(need MuseGlimmerForConditionalGeneration, not MuseGlimmerModel)"
        )

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
    p.add_argument("--output-dir", type=Path, default=None)
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
    p.add_argument(
        "--force-retokenize",
        action="store_true",
        help="Ignore tokenized disk cache and re-run TRL prepare (Tokenizing/labels/trunc).",
    )
    p.add_argument(
        "--cp-size",
        type=int,
        default=None,
        help="Context-parallel size (Ring Attention). pad_to_multiple_of=cp_size*2. "
        "Also read from BIV_CP_SIZE.",
    )
    p.add_argument(
        "--prepare-tokenized-only",
        action="store_true",
        help="Single-process: build tokenized disk cache and exit (no model / no train). "
        "Run this before multi-GPU launch to avoid NCCL barrier timeouts during tokenize.",
    )
    args = p.parse_args()

    cfg = _load_yaml(_resolve(args.config))
    train_cfg = cfg.get("train") or {}
    from biv_wm.model_store import resolve_model_for_train

    model_path = args.model or resolve_model_for_train(cfg, root=ROOT)
    qlora = bool(args.qlora or train_cfg.get("qlora") or os.environ.get("QLORA") in {"1", "true", "True"})
    max_length = int(args.max_length)
    cached = [_resolve(x) for x in args.cached_datasets]
    truncation_mode = str(train_cfg.get("truncation_mode") or "keep_start")
    assistant_only = bool(train_cfg.get("assistant_only_loss", True))
    tok_cache = _tokenized_cache_dir(
        cached,
        max_length=max_length,
        assistant_only=assistant_only,
        truncation_mode=truncation_mode,
    )

    if args.prepare_tokenized_only:
        print("=== Muse prepare tokenized cache (single process) ===", flush=True)
        print(f"  model:      {model_path}", flush=True)
        print(f"  max_length: {max_length}", flush=True)
        print(f"  cache:      {tok_cache}", flush=True)
        if (not args.force_retokenize) and _tokenized_cache_ready(tok_cache):
            print(f"[muse] tokenized cache already ready → {tok_cache}", flush=True)
            return
        msgs_ds = _load_concat_datasets(cached)
        print(f"  messages rows: {len(msgs_ds):,}", flush=True)
        tokenizer = _load_tokenizer_only(model_path)
        num_proc = int(cfg.get("dataset_num_proc") or train_cfg.get("dataset_num_proc") or 8)
        tok_ds = _tokenize_messages_offline(
            msgs_ds,
            tokenizer,
            max_length=max_length,
            truncation_mode=truncation_mode,
            assistant_only=assistant_only,
            num_proc=num_proc,
        )
        _save_tokenized_cache(tok_ds, tok_cache)
        print("[muse] prepare-tokenized-only done", flush=True)
        return

    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --prepare-tokenized-only")
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Muse Glimmer TRL SFT ===", flush=True)
    print(f"  config:     {_resolve(args.config)}", flush=True)
    print(f"  model:      {model_path}", flush=True)
    print(f"  max_length: {max_length}", flush=True)
    print(f"  qlora:      {qlora}", flush=True)
    print(f"  output:     {out_dir}", flush=True)

    use_tok_cache = (not args.force_retokenize) and _tokenized_cache_ready(tok_cache)
    distributed = (
        int(os.environ.get("WORLD_SIZE", "1") or "1") > 1
        or os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("RANK") is not None
    )
    if use_tok_cache:
        print(f"[muse] tokenized cache HIT → {tok_cache}", flush=True)
        train_ds = _try_load_tokenized_cache(tok_cache)
        print(f"  train rows: {len(train_ds):,} (input_ids)", flush=True)
    else:
        if distributed:
            raise SystemExit(
                "[muse] tokenized cache MISS under multi-process launch.\n"
                "TRL prepare would hold a distributed barrier for 30+ min and NCCL times out.\n"
                "Build the cache first (single process), then re-launch train:\n"
                "  python scripts/train_muse_trl.py --prepare-tokenized-only "
                f"--config … --max-length {max_length} --cached-datasets … --model …\n"
                "Or use: bash scripts/trainmodel.sh …  (it pre-builds automatically)."
            )
        print(
            f"[muse] tokenized cache MISS → offline tokenize then train: {tok_cache}",
            flush=True,
        )
        msgs_ds = _load_concat_datasets(cached)
        tokenizer = _load_tokenizer_only(model_path)
        num_proc = int(cfg.get("dataset_num_proc") or train_cfg.get("dataset_num_proc") or 8)
        train_ds = _tokenize_messages_offline(
            msgs_ds,
            tokenizer,
            max_length=max_length,
            truncation_mode=truncation_mode,
            assistant_only=assistant_only,
            num_proc=num_proc,
        )
        _save_tokenized_cache(train_ds, tok_cache)
        use_tok_cache = True

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
    import inspect

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
    packing = bool(train_cfg.get("packing", False))
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    cp_size = int(
        args.cp_size
        if args.cp_size is not None
        else os.environ.get("BIV_CP_SIZE", "1")
        or "1"
    )
    if cp_size < 1:
        cp_size = 1
    # TRL Ring Attention (FSDP2+CP): sequences must be divisible by cp_size*2.
    pad_multiple = cp_size * 2 if cp_size > 1 else int(train_cfg.get("pad_to_multiple_of") or 0)
    # FSDP activation_checkpointing and Trainer gradient_checkpointing conflict.
    if cp_size > 1 and grad_ckpt:
        print(
            "[muse] CP enabled: disabling Trainer gradient_checkpointing "
            "(FSDP activation_checkpointing handles memory)",
            flush=True,
        )
        grad_ckpt = False

    # Meta docs use TrainingArguments without warmup_*; TRL 1.8 / current
    # transformers TrainingArguments dropped warmup_ratio — only warmup_steps.
    warmup_ratio = float(args.warmup_ratio or train_cfg.get("warmup_ratio", 0.03))
    steps_per_epoch = max(1, len(train_ds) // max(1, bs * gas))
    total_steps = max(1, int(epochs * steps_per_epoch))
    warmup_steps = int(train_cfg.get("warmup_steps") or max(1, int(total_steps * warmup_ratio)))

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
        warmup_steps=warmup_steps,
        seed=seed,
        max_length=max_length,
        packing=packing,
        gradient_checkpointing=grad_ckpt,
        report_to=os.environ.get("REPORT_TO", "none"),
        truncation_mode=truncation_mode,
    )
    if use_tok_cache:
        # Pretokenized input_ids+labels (assistant already masked with -100).
        # Do NOT set assistant_only_loss — TRL requires conversational messages for that.
        sft_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    elif assistant_only:
        sft_kwargs["assistant_only_loss"] = True
    if pad_multiple > 1:
        sft_kwargs["pad_to_multiple_of"] = pad_multiple
    if cp_size > 1:
        # Ring Attention CP currently requires SDPA (not FlashAttn).
        sft_kwargs["attn_implementation"] = "sdpa"
        if hasattr(model, "config"):
            try:
                model.config._attn_implementation = "sdpa"
            except Exception:
                pass
        print(
            f"[muse] CP size={cp_size}: pad_to_multiple_of={pad_multiple}, "
            f"attn=sdpa, ~{max_length // cp_size} tokens/GPU",
            flush=True,
        )

    # Keep only kwargs accepted by this TRL/transformers build (avoids API drift).
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    accepted.discard("self")
    dropped = sorted(k for k in sft_kwargs if k not in accepted)
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in accepted}
    if dropped:
        print(f"[muse] SFTConfig dropped unsupported kwargs: {dropped}", flush=True)
    # Older TRL used max_seq_length
    if "max_length" not in sft_kwargs and "max_seq_length" in accepted:
        sft_kwargs["max_seq_length"] = max_length

    print(
        f"[muse] SFTConfig warmup_steps={sft_kwargs.get('warmup_steps')} "
        f"(from ratio={warmup_ratio:g}, ~{total_steps} steps)",
        flush=True,
    )
    sft_args = SFTConfig(**sft_kwargs)

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
