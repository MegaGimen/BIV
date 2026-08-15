#!/usr/bin/env python3
"""Muse Glimmer-30B WM SFT via TRL + PEFT (LoRA / QLoRA).

Reads structure-right-truncated HF datasets (messages) from train_prep_mix,
applies Glimmer chat template, trains with assistant_only_loss.

Mid-run eval uses a fixed subsample of mix_dir/anti_forget/eval.jsonl
(``eval_anti_forget_loss``) as an anti-forgetting monitor — not an agent bench.

Examples (prefer trainmodel.sh wrapper):
  python scripts/train_muse_trl.py \\
    --config configs/trl/muse_glimmer_30b_lora.yaml \\
    --max-length 8192 \\
    --cached-datasets outputs/trl_cache/.../wm_code ... \\
    --output-dir outputs/muse_glimmer_wm_mix_ml8192_c1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
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


def _filter_target_modules(model, modules: list[str]) -> list[str]:
    """Drop LoRA targets that do not exist on this checkpoint (leaf-name match).

    Keep leaf matching (same as checkpoint-200) so adapter param names stay
    compatible on resume. Vision adapters are created then disabled via
    ``_freeze_vision`` *after* ``get_peft_model``.
    """
    names = {n for n, _ in model.named_modules()}
    leaf = {n.rsplit(".", 1)[-1] for n in names}
    kept = [m for m in modules if m in leaf]
    dropped = [m for m in modules if m not in leaf]
    if dropped:
        print(f"[muse] skip missing LoRA targets: {dropped}", flush=True)
    if not kept:
        raise SystemExit(f"No LoRA target_modules matched model; tried {modules}")
    return kept


def _freeze_vision(model) -> int:
    n = 0
    for name, param in model.named_parameters():
        low = name.lower()
        if any(h in low for h in _VISION_NAME_HINTS):
            if param.requires_grad:
                param.requires_grad = False
                n += 1
    return n


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Uniform train messages: keep Muse tool fields; repair anti_forget."""
    from biv_wm.adapters.normalize import messages_for_arrow, normalize_train_messages

    # Arrow-stable (arguments as JSON strings); chat_template path re-parses to dict.
    return messages_for_arrow(normalize_train_messages(messages))


def _load_concat_datasets(paths: list[Path]):
    from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk

    # Uniform schema so wm_* (no tools) and anti_forget (tools) can concatenate.
    # Empty tool_calls / blank name|tool_call_id for WM rows.
    # tool_calls.function.arguments stored as JSON string; parsed to dict at tokenize.
    msg_features = Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "name": Value("string"),
                    "tool_call_id": Value("string"),
                    "reasoning_content": Value("string"),
                    "tool_calls": [
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "function": {
                                "name": Value("string"),
                                "arguments": Value("string"),
                            },
                        }
                    ],
                }
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

        rows = [{"messages": _normalize_messages(ex["messages"])} for ex in ds]
        try:
            part = Dataset.from_list(rows, features=msg_features)
        except Exception as e:
            # Fallback without Features if Arrow rejects edge shapes.
            print(f"[muse] WARN features cast failed ({e!r}); from_list without Features", flush=True)
            part = Dataset.from_list(rows)
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
    # v5 = think→reasoning_content (Muse to=self) + vision LoRA freeze-after-PEFT
    parts = [f"v5|ml={max_length}|ao={int(assistant_only)}|trunc={truncation_mode}|gen=1"]
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
        from biv_wm.adapters.normalize import messages_for_chat_template

        return messages_for_chat_template(raw)

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
    attn_implementation: str | None = None,
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
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        # Must be set at load time — SFTConfig has no attn_implementation in TRL 1.8.
        # Muse is composite: text layers read text_config._attn_implementation (for CP/SDPA).
        load_kwargs["attn_implementation"] = attn_implementation
        print(f"[muse] attn_implementation={attn_implementation}", flush=True)
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

    if attn_implementation:
        _force_attn_implementation(model, attn_implementation)

    if qlora:
        model = prepare_model_for_kbit_training(model)

    if freeze_vision:
        frozen = _freeze_vision(model)
        print(f"[muse] froze {frozen} vision/perception base params (pre-LoRA)", flush=True)
        # Text-only WM SFT: park frozen vision on CPU to free VRAM (no pixel batch).
        if distributed or attn_implementation:
            moved = _offload_frozen_vision_to_cpu(model)
            if moved:
                print(f"[muse] offloaded {moved} frozen vision modules → CPU", flush=True)

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
    # Critical for resume: freeze AFTER PEFT so vision LoRA adapters are not
    # trainable. Previously freeze ran only pre-LoRA → vision *lora_* stayed
    # requires_grad=True → FSDP optimizer load failed vs older checkpoints.
    if freeze_vision:
        frozen_lora = _freeze_vision(model)
        print(
            f"[muse] froze {frozen_lora} vision params after LoRA "
            "(adapters excluded from optimizer)",
            flush=True,
        )
    model.print_trainable_parameters()
    return model, tokenizer


def _force_attn_implementation(model, impl: str) -> None:
    """Propagate attn impl into composite Muse configs (root + text_config + submodules)."""
    configs = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        configs.append(cfg)
        tc = getattr(cfg, "text_config", None)
        if tc is not None:
            configs.append(tc)
        vc = getattr(cfg, "vision_config", None)
        if vc is not None:
            configs.append(vc)
    # Peft / nested
    base = getattr(model, "get_base_model", lambda: None)()
    if base is not None and hasattr(base, "config"):
        configs.append(base.config)
        tc = getattr(base.config, "text_config", None)
        if tc is not None:
            configs.append(tc)
    for c in configs:
        try:
            c._attn_implementation = impl
        except Exception:
            pass
    # language_model.config (actual decoder)
    try:
        lm = model.get_base_model().model.language_model  # Peft → Muse → text
        if hasattr(lm, "config"):
            lm.config._attn_implementation = impl
    except Exception:
        try:
            lm = model.model.language_model
            if hasattr(lm, "config"):
                lm.config._attn_implementation = impl
        except Exception:
            pass
    print(f"[muse] forced attn_implementation={impl} on text/root configs", flush=True)


def _offload_frozen_vision_to_cpu(model) -> int:
    n = 0
    root = model
    try:
        root = model.get_base_model()
    except Exception:
        pass
    for attr in ("vision_tower", "vision_adapter", "vision_projection", "perception_emb_norm"):
        try:
            mod = getattr(root.model if hasattr(root, "model") else root, attr, None)
        except Exception:
            mod = None
        if mod is None:
            continue
        try:
            mod.to("cpu")
            n += 1
        except Exception:
            pass
    return n


# Rolling mid-run: checkpoint-e{epoch}-s{step}. Epoch-end permanent: checkpoint-epoch{N}-end-s{step}.
_ROLLING_CKPT_RE = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
_EPOCH_END_CKPT_RE = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
_HF_DIGIT_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def _find_latest_ckpt(out_dir: Path) -> Path | None:
    """Pick newest complete ckpt under ``out_dir`` (same key as ``train_daemon.sh``).

    Rank key ``(epoch, step, kind)`` — epoch dominates step:
      checkpoint-{step}                    → epoch 0, kind 0
      checkpoint-e{epoch}-s{step}          → kind 1
      checkpoint-epoch{epoch}-end-s{step}  → kind 2
    Requires ``trainer_state.json`` plus weights (adapter / FSDP / *.safetensors).
    """
    if not out_dir.is_dir():
        return None
    best: tuple[int, int, int, Path] | None = None
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        epoch = step = None
        kind = 0
        m = _EPOCH_END_CKPT_RE.match(name)
        if m:
            epoch, step, kind = int(m.group(1)), int(m.group(2)), 2
        else:
            m = _ROLLING_CKPT_RE.match(name)
            if m:
                epoch, step, kind = int(m.group(1)), int(m.group(2)), 1
            else:
                m = _HF_DIGIT_CKPT_RE.match(name)
                if m:
                    epoch, step, kind = 0, int(m.group(1)), 0
        if epoch is None or step is None:
            continue
        if not (p / "trainer_state.json").is_file():
            continue
        if not (
            (p / "adapter_model.safetensors").is_file()
            or (p / "pytorch_model_fsdp.bin").is_file()
            or any(p.glob("*.safetensors"))
        ):
            continue
        key = (epoch, step, kind)
        if best is None or key > (best[0], best[1], best[2]):
            best = (epoch, step, kind, p)
    return None if best is None else best[3]


def _resolve_resume(resume_from: Path | str | None, *, out_dir: Path) -> str | None:
    """Resolve ``resume_from``: path, or ``auto`` → latest ckpt under ``out_dir``."""
    if resume_from is None or str(resume_from).strip() in {"", "null", "None"}:
        return None
    raw = str(resume_from).strip()
    if raw.lower() == "auto":
        picked = _find_latest_ckpt(out_dir)
        if picked is None:
            print(
                f"[muse] resume_from=auto → no complete checkpoint under {out_dir}; "
                "starting fresh",
                flush=True,
            )
            return None
        print(f"[muse] resume_from=auto → {picked}", flush=True)
        return str(picked)
    path = _resolve(raw)
    if not path.is_dir():
        raise SystemExit(f"--resume-from must be an existing checkpoint directory: {path}")
    # Soft check: Trainer needs trainer_state.json for full opt/sched/RNG restore.
    state = path / "trainer_state.json"
    if not state.is_file():
        print(
            f"[muse] WARNING: {state.name} missing under {path}; "
            "weights may load but optimizer/LR scheduler/RNG will restart.",
            flush=True,
        )
    print(f"[muse] resume_from={path}", flush=True)
    return str(path)


def _load_peft_adapter_from_ckpt(model, ckpt: Path | str) -> None:
    """Load LoRA tensors from a Trainer/PEFT checkpoint without FSDP state restore."""
    import torch
    from peft import set_peft_model_state_dict

    ckpt_p = Path(ckpt)
    if not ckpt_p.is_dir():
        raise SystemExit(f"[muse] adapter ckpt not a directory: {ckpt_p}")

    sd = None
    source = None
    st_path = ckpt_p / "adapter_model.safetensors"
    bin_path = ckpt_p / "adapter_model.bin"
    if st_path.is_file():
        from safetensors.torch import load_file

        sd = load_file(str(st_path))
        source = st_path
    elif bin_path.is_file():
        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        source = bin_path
    else:
        for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model_fsdp.bin"):
            p = ckpt_p / name
            if not p.is_file():
                continue
            if p.suffix == ".safetensors":
                from safetensors.torch import load_file

                raw = load_file(str(p))
            else:
                raw = torch.load(p, map_location="cpu", weights_only=True)
                if isinstance(raw, dict) and "state_dict" in raw:
                    raw = raw["state_dict"]
            if not isinstance(raw, dict):
                continue
            filtered = {
                k: v for k, v in raw.items() if isinstance(k, str) and "lora_" in k
            }
            if filtered:
                sd = filtered
                source = p
                break

    if sd is None:
        raise SystemExit(
            f"[muse] no LoRA weights found under {ckpt_p} "
            "(need adapter_model.safetensors / adapter_model.bin / model*.safetensors)"
        )

    set_peft_model_state_dict(model, sd)
    print(f"[muse] loaded LoRA adapter weights from {source}", flush=True)


def _train_resume_adapter_only(trainer, model, resume_from: str) -> None:
    """Resume like full Trainer checkpoint, but skip FSDP *model* weight copy.

    LoRA tensors are loaded via PEFT first. Adam / LR scheduler / RNG / step /
    data-skip still go through ``trainer.train(resume_from_checkpoint=...)``.
    Only ``_load_from_checkpoint`` (FSDP DTensor copy → MeshLayout.axes crash)
    is bypassed.
    """
    import json

    _load_peft_adapter_from_ckpt(model, resume_from)
    state_path = Path(resume_from) / "trainer_state.json"
    if state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        print(
            f"[muse] trainer_state: global_step={st.get('global_step')} "
            f"epoch={st.get('epoch')} — will restore Adam/scheduler/RNG + "
            f"skip already-seen data; only FSDP model copy is skipped",
            flush=True,
        )
    else:
        print(
            f"[muse] WARNING: {state_path} missing — cannot restore step counter",
            flush=True,
        )

    def _skip_model_load(resume_from_checkpoint, model=None):
        print(
            f"[muse] skip FSDP model load from {resume_from_checkpoint} "
            "(adapter already applied; optimizer/scheduler/RNG still load)",
            flush=True,
        )
        return None

    trainer._load_from_checkpoint = _skip_model_load  # type: ignore[method-assign]
    # Do NOT patch _load_optimizer_and_scheduler / _load_rng_state — those should
    # restore from checkpoint exactly as a normal resume.
    trainer.train(resume_from_checkpoint=resume_from)


def _make_eval_slice(ds, *, max_samples: int, seed: int):
    """Deterministic subsample (train set unchanged when slicing a copy)."""
    if max_samples is None or max_samples <= 0:
        return None
    n = len(ds)
    if n <= 0:
        return None
    k = min(int(max_samples), n)
    return ds.shuffle(seed=seed).select(range(k))


def _anti_forget_eval_jsonl(mix_dir: Path) -> Path:
    return mix_dir / "anti_forget" / "eval.jsonl"


def _load_anti_forget_eval_messages(mix_dir: Path):
    """Load prepare_data held-out anti_forget/eval.jsonl as messages Dataset."""
    from datasets import Dataset, Features, Value

    path = _anti_forget_eval_jsonl(mix_dir)
    if not path.is_file():
        raise SystemExit(
            f"[muse] anti_forget held-out missing: {path}\n"
            "Run: python scripts/prepare_data.py --anti-forget "
            f"--out-dir {mix_dir}"
        )

    msg_features = Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "name": Value("string"),
                    "tool_call_id": Value("string"),
                    "reasoning_content": Value("string"),
                    "tool_calls": [
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "function": {
                                "name": Value("string"),
                                "arguments": Value("string"),
                            },
                        }
                    ],
                }
            ]
        }
    )

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs:
                continue
            rows.append({"messages": _normalize_messages(msgs)})

    if not rows:
        raise SystemExit(f"[muse] anti_forget held-out empty: {path}")

    try:
        ds = Dataset.from_list(rows, features=msg_features)
    except Exception as e:
        print(
            f"[muse] WARN anti_forget eval features cast failed ({e!r}); "
            "from_list without Features",
            flush=True,
        )
        ds = Dataset.from_list(rows)
    print(f"[muse] loaded anti_forget held-out {path} ({len(ds):,} rows)", flush=True)
    return ds


def _anti_forget_eval_cache_dir(
    tok_cache: Path,
    *,
    eval_jsonl: Path,
    eval_max_samples: int,
    seed: int,
) -> Path:
    """Side cache next to train tokenized dir (does not invalidate train cache)."""
    import hashlib

    parts = [
        f"eval_anti|v1|n={int(eval_max_samples)}|seed={int(seed)}",
        tok_cache.name,
        str(eval_jsonl.resolve()),
    ]
    if eval_jsonl.is_file():
        st = eval_jsonl.stat()
        parts.append(f"{st.st_mtime_ns}:{st.st_size}")
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return tok_cache.parent / (
        f"eval_anti_forget_n{int(eval_max_samples)}_{digest}"
    )


def _prepare_anti_forget_eval(
    *,
    mix_dir: Path,
    tokenizer,
    tok_cache: Path,
    max_length: int,
    truncation_mode: str,
    assistant_only: bool,
    num_proc: int,
    eval_max_samples: int,
    seed: int,
    force_retokenize: bool,
):
    """Held-out anti_forget panel for mid-run forgetting monitor (not agent bench)."""
    if eval_max_samples is None or int(eval_max_samples) <= 0:
        return None

    eval_jsonl = _anti_forget_eval_jsonl(mix_dir)
    cache = _anti_forget_eval_cache_dir(
        tok_cache,
        eval_jsonl=eval_jsonl,
        eval_max_samples=int(eval_max_samples),
        seed=int(seed),
    )
    if (not force_retokenize) and _tokenized_cache_ready(cache):
        ds = _try_load_tokenized_cache(cache)
        print(
            f"[muse] anti_forget eval cache HIT → {cache} ({len(ds):,} rows)",
            flush=True,
        )
        return ds

    msgs = _load_anti_forget_eval_messages(mix_dir)
    panel = _make_eval_slice(msgs, max_samples=int(eval_max_samples), seed=int(seed))
    if panel is None:
        return None
    print(
        f"[muse] anti_forget eval panel: {len(panel):,} / {len(msgs):,} rows "
        f"(eval_max_samples={eval_max_samples}, seed={seed})",
        flush=True,
    )
    tok_ds = _tokenize_messages_offline(
        panel,
        tokenizer,
        max_length=max_length,
        truncation_mode=truncation_mode,
        assistant_only=assistant_only,
        num_proc=num_proc,
    )
    if _is_main_process():
        _save_tokenized_cache(tok_ds, cache)
    _barrier()
    if (not _is_main_process()) and _tokenized_cache_ready(cache):
        tok_ds = _try_load_tokenized_cache(cache)
    return tok_ds


def _make_muse_sft_trainer_cls():
    """SFTTrainer that still logs mean_token_accuracy for Muse + use_liger_kernel.

    Root cause (TRL ≥0.26 / 1.8 + Muse Glimmer):
    - With ``use_liger_kernel=True``, ``SFTTrainer.compute_loss`` sets
      ``return_token_accuracy=True`` and **only** reads ``outputs.token_accuracy``
      (skips the logits argmax path; see huggingface/trl#4730 / PR#4302).
    - That field is filled only by Liger's **model-specific** forward patch
      (e.g. Qwen3 → ``LigerCausalLMOutputWithPast``).
    - ``muse_glimmer`` is not in Liger's ``MODEL_TYPE_TO_APPLY_LIGER_FN``;
      Muse always materializes ``logits`` via ``lm_head`` and returns
      ``MuseGlimmerCausalLMOutputWithPast`` **without** ``token_accuracy``
      (``accepts_loss_kwargs=False``).
    - Result: warning *liger-kernel did not return token_accuracy* and the
      train log loses ``mean_token_accuracy`` — loss/backprop unchanged.

    Fix: during ``compute_loss``, temporarily clear the liger flag so TRL uses
    the logits-based accuracy/entropy path Muse already provides. Init-time
    ``apply_liger_kernel`` (no-op for Muse) is unaffected.
    """
    from trl import SFTTrainer

    class MuseSFTTrainer(SFTTrainer):
        _muse_acc_note_printed = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            liger = bool(getattr(self.args, "use_liger_kernel", False))
            if liger and not MuseSFTTrainer._muse_acc_note_printed:
                print(
                    "[muse] Muse has no Liger FLCE/token_accuracy patch; "
                    "logging mean_token_accuracy from logits (same as non-liger path)",
                    flush=True,
                )
                MuseSFTTrainer._muse_acc_note_printed = True
            if liger:
                self.args.use_liger_kernel = False
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                if liger:
                    self.args.use_liger_kernel = True

    return MuseSFTTrainer


def _make_muse_checkpoint_callbacks(*, save_total_limit: int, has_eval: bool):
    """Rename HF digit ckpts, rotate rolling only, force epoch-end save (+ eval).

    Also re-apply logging/eval/save step intervals from *current* TrainingArguments
    after resume (HF loads stale ``state.save_steps`` etc. from trainer_state.json).
    """
    from transformers import TrainerCallback

    class MuseScheduleOverrideCallback(TrainerCallback):
        """Make resume honor current args for schedule intervals."""

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            # DefaultFlowCallback uses state.{logging,eval,save}_steps, not args.*.
            # Resume replaces state via trainer_state.json *after* compute_steps(args),
            # so stale intervals (e.g. save_steps=200) would stick without this.
            synced: list[str] = []
            for kind in ("logging", "eval", "save"):
                arg_v = getattr(args, f"{kind}_steps", None)
                if arg_v is None:
                    continue
                # Absolute step counts only (same as our SFTConfig).
                try:
                    new_v = int(arg_v)
                except (TypeError, ValueError):
                    continue
                if new_v <= 0:
                    continue
                old_v = getattr(state, f"{kind}_steps", None)
                if old_v != new_v:
                    setattr(state, f"{kind}_steps", new_v)
                    synced.append(f"{kind}_steps: {old_v} → {new_v}")
            if synced and state.is_world_process_zero:
                print(
                    "[muse] resume/schedule override from current args: "
                    + ", ".join(synced),
                    flush=True,
                )
            return control

    class MuseCheckpointCallback(TrainerCallback):
        def __init__(self) -> None:
            self._pending_epoch_end = False

        def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001
            self._pending_epoch_end = True
            control.should_save = True
            # eval_strategy=epoch already sets should_evaluate; keep force as belt.
            if has_eval:
                control.should_evaluate = True
            ep = int(round(float(state.epoch or 0)))
            print(
                f"[muse] epoch {ep} end: force checkpoint"
                + (" + evaluate" if has_eval else "")
                + " (permanent epoch ckpt)",
                flush=True,
            )
            return control

        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            # HF already wrote checkpoint-{global_step}; rename + custom rotate.
            if not state.is_world_process_zero:
                return control
            out = Path(args.output_dir)
            step = int(state.global_step)
            src = out / f"checkpoint-{step}"
            if not src.is_dir():
                # Already renamed (retry) or unexpected layout.
                self._rotate_rolling(out, save_total_limit)
                return control

            if self._pending_epoch_end:
                epoch_i = int(round(float(state.epoch or 0)))
                dst = out / f"checkpoint-epoch{epoch_i}-end-s{step}"
                self._pending_epoch_end = False
                kind = "epoch-end"
            else:
                epoch_i = int(math.floor(float(state.epoch or 0)))
                dst = out / f"checkpoint-e{epoch_i}-s{step}"
                kind = "rolling"

            if src.resolve() != dst.resolve():
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                src.rename(dst)
                print(f"[muse] checkpoint ({kind}) → {dst.name}", flush=True)

            self._rotate_rolling(out, save_total_limit)
            return control

        @staticmethod
        def _rotate_rolling(out: Path, limit: int) -> None:
            if limit is None or limit <= 0:
                return
            rolling: list[tuple[int, Path]] = []
            for p in out.iterdir():
                if not p.is_dir():
                    continue
                m = _ROLLING_CKPT_RE.match(p.name)
                if m:
                    rolling.append((int(m.group(2)), p))
                    continue
                # Leftover HF digit names (rename race / older runs).
                m2 = _HF_DIGIT_CKPT_RE.match(p.name)
                if m2:
                    rolling.append((int(m2.group(1)), p))
            rolling.sort(key=lambda t: t[0])
            while len(rolling) > limit:
                _, victim = rolling.pop(0)
                print(f"[muse] rotate: remove {victim.name}", flush=True)
                shutil.rmtree(victim, ignore_errors=True)

    # Schedule override first so intervals are fixed before any step-end save/log.
    return [MuseScheduleOverrideCallback(), MuseCheckpointCallback()]


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
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint dir, or 'auto' to pick latest complete ckpt under output_dir "
        "(same ranking as train_daemon.sh). Omit to start fresh.",
    )
    p.add_argument(
        "--resume-adapter-only",
        action="store_true",
        help="Load LoRA weights from checkpoint but skip Trainer FSDP full resume "
        "(optimizer/LR/RNG cold-start). Use when full resume hits "
        "'_MeshLayout' object has no attribute 'axes'.",
    )
    p.add_argument(
        "--resume-full",
        action="store_true",
        help="Force Trainer.resume_from_checkpoint (FSDP full state). "
        "Default under FSDP2+CP is adapter-only due to torch MeshLayout bugs.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Anti-forget held-out eval panel size (from mix_dir/anti_forget/eval.jsonl). "
        "0 disables eval.",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Anti-forget held-out eval every N steps. Default: same as save_steps.",
    )
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
        mix_dir = _resolve(cfg.get("mix_dir") or "data/processed/mix_v2")
        eval_max_prep = args.eval_max_samples
        if eval_max_prep is None:
            eval_max_prep = int(train_cfg.get("eval_max_samples", 128))
        seed_prep = int(args.seed or train_cfg.get("seed", 42))
        num_proc = int(cfg.get("dataset_num_proc") or train_cfg.get("dataset_num_proc") or 8)
        tokenizer = _load_tokenizer_only(model_path)
        if (not args.force_retokenize) and _tokenized_cache_ready(tok_cache):
            print(f"[muse] tokenized cache already ready → {tok_cache}", flush=True)
        else:
            msgs_ds = _load_concat_datasets(cached)
            print(f"  messages rows: {len(msgs_ds):,}", flush=True)
            tok_ds = _tokenize_messages_offline(
                msgs_ds,
                tokenizer,
                max_length=max_length,
                truncation_mode=truncation_mode,
                assistant_only=assistant_only,
                num_proc=num_proc,
            )
            _save_tokenized_cache(tok_ds, tok_cache)
        # Small side cache for mid-run anti_forget monitor (does not touch train cache).
        _prepare_anti_forget_eval(
            mix_dir=mix_dir,
            tokenizer=tokenizer,
            tok_cache=tok_cache,
            max_length=max_length,
            truncation_mode=truncation_mode,
            assistant_only=assistant_only,
            num_proc=num_proc,
            eval_max_samples=int(eval_max_prep),
            seed=seed_prep + 2,
            force_retokenize=bool(args.force_retokenize),
        )
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
    resume_from = _resolve_resume(
        args.resume_from if args.resume_from is not None else train_cfg.get("resume_from"),
        out_dir=out_dir,
    )

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
    cp_size = int(
        args.cp_size
        if args.cp_size is not None
        else os.environ.get("BIV_CP_SIZE", "1")
        or "1"
    )
    if cp_size < 1:
        cp_size = 1
    # Ring Attention CP requires SDPA; set at from_pretrained (SFTConfig lacks this kwarg).
    attn_impl = "sdpa" if cp_size > 1 else None
    model, tokenizer = _build_model_and_tokenizer(
        model_path=model_path,
        qlora=qlora,
        torch_dtype=str(train_cfg.get("torch_dtype", "bfloat16")),
        freeze_vision=bool(train_cfg.get("freeze_vision", True)),
        target_modules=target_modules,
        lora_rank=int(args.lora_rank or train_cfg.get("lora_rank", 16)),
        lora_alpha=int(args.lora_alpha or train_cfg.get("lora_alpha", 32)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        attn_implementation=attn_impl,
    )

    from trl import SFTConfig

    MuseSFTTrainer = _make_muse_sft_trainer_cls()
    import inspect

    lr = float(args.learning_rate or train_cfg.get("learning_rate", 2e-4))
    epochs = float(args.num_epochs or train_cfg.get("num_epochs", 2))
    bs = int(args.per_device_train_batch_size or train_cfg.get("per_device_train_batch_size", 1))
    gas = int(
        args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps", 8)
    )
    seed = int(args.seed or train_cfg.get("seed", 42))
    logging_steps = int(args.logging_steps or train_cfg.get("logging_steps", 10))
    save_steps = int(args.save_steps or train_cfg.get("save_steps", 25))
    save_limit = int(args.save_total_limit or train_cfg.get("save_total_limit", 3))
    packing = bool(train_cfg.get("packing", False))
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    eval_max = args.eval_max_samples
    if eval_max is None:
        eval_max = int(train_cfg.get("eval_max_samples", 128))
    eval_steps = args.eval_steps
    if eval_steps is None:
        eval_steps = train_cfg.get("eval_steps")
    # Default: same cadence as rolling checkpoints (save_steps).
    if eval_steps is None:
        eval_steps = save_steps
    else:
        eval_steps = int(eval_steps)
    eval_bs = int(train_cfg.get("per_device_eval_batch_size", 1))
    mix_dir = _resolve(cfg.get("mix_dir") or "data/processed/mix_v2")
    # resume_from already resolved at banner (supports path | auto).
    # TRL Ring Attention (FSDP2+CP): sequences must be divisible by cp_size*2.
    pad_multiple = cp_size * 2 if cp_size > 1 else int(train_cfg.get("pad_to_multiple_of") or 0)
    # Long-context: materializing logits [B,S,vocab≈202k] ≈ 12GB at S=32k.
    # Liger fused CE avoids that allocation; TRL marks it CP-compatible.
    use_liger = bool(train_cfg.get("use_liger_kernel", cp_size > 1 or max_length >= 16384))
    if use_liger:
        try:
            import liger_kernel  # noqa: F401
        except ImportError:
            print(
                "[muse] WARNING: use_liger_kernel requested but liger-kernel not installed; "
                "long max_length may OOM on full logits",
                flush=True,
            )
            use_liger = False
    # Prefer Trainer gradient checkpointing for Muse+PEFT. FSDP activation
    # checkpointing + PEFT has been flaky; cannot enable both (TRL).
    if cp_size > 1:
        grad_ckpt = True
        print(
            "[muse] CP: Trainer gradient_checkpointing=ON, "
            "expect FSDP activation_checkpointing=OFF (avoid dual ckpt)",
            flush=True,
        )

    # Meta docs use TrainingArguments without warmup_*; TRL 1.8 / current
    # transformers TrainingArguments dropped warmup_ratio — only warmup_steps.
    warmup_ratio = float(args.warmup_ratio or train_cfg.get("warmup_ratio", 0.03))
    steps_per_epoch = max(1, len(train_ds) // max(1, bs * gas))
    total_steps = max(1, int(epochs * steps_per_epoch))
    warmup_steps = int(train_cfg.get("warmup_steps") or max(1, int(total_steps * warmup_ratio)))

    num_proc = int(cfg.get("dataset_num_proc") or train_cfg.get("dataset_num_proc") or 8)
    eval_ds = _prepare_anti_forget_eval(
        mix_dir=mix_dir,
        tokenizer=tokenizer,
        tok_cache=tok_cache,
        max_length=max_length,
        truncation_mode=truncation_mode,
        assistant_only=assistant_only,
        num_proc=num_proc,
        eval_max_samples=int(eval_max),
        seed=seed + 2,
        force_retokenize=bool(args.force_retokenize),
    )
    has_eval = eval_ds is not None
    if has_eval and int(eval_steps) <= 0:
        raise SystemExit("[muse] eval_steps must be > 0 when anti_forget eval is enabled")
    if has_eval:
        print(
            f"[muse] anti_forget held-out eval: {len(eval_ds):,} rows "
            f"(every {eval_steps} steps [=save_steps={save_steps} when unset]; "
            f"eval_max_samples={eval_max})",
            flush=True,
        )
    else:
        print("[muse] eval_max_samples<=0 → evaluate disabled", flush=True)

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=gas,
        learning_rate=lr,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        # Disable HF rotate: with use_mtime it globs all checkpoint-* and would
        # delete permanent epoch-end dirs. Rolling limit enforced in callback.
        save_total_limit=None,
        save_only_model=False,  # keep optimizer/scheduler/RNG for resume
        warmup_steps=warmup_steps,
        seed=seed,
        max_length=max_length,
        packing=packing,
        gradient_checkpointing=grad_ckpt,
        report_to=os.environ.get("REPORT_TO", "none"),
        truncation_mode=truncation_mode,
        # Steps eval on anti_forget held-out; epoch-end callback also forces evaluate.
        eval_strategy="steps" if has_eval else "no",
        per_device_eval_batch_size=eval_bs,
    )
    # AutoDL / custom TB: LOGGING_DIR=/root/tf-logs (HF default is output_dir/runs/…)
    logging_dir = os.environ.get("LOGGING_DIR") or os.environ.get("TF_LOGS") or train_cfg.get(
        "logging_dir"
    )
    if logging_dir:
        sft_kwargs["logging_dir"] = str(logging_dir)
        print(f"[muse] tensorboard logging_dir={logging_dir}", flush=True)
    if has_eval:
        sft_kwargs["eval_steps"] = int(eval_steps)
    if use_liger:
        sft_kwargs["use_liger_kernel"] = True
        # Muse is not in Liger's supported model_type map → apply_liger is a no-op;
        # fused CE / outputs.token_accuracy are NOT active. Flag kept so TRL/init
        # paths stay consistent; MuseSFTTrainer restores accuracy from logits.
        print(
            "[muse] use_liger_kernel=True (note: muse_glimmer unsupported by Liger → "
            "no fused CE; mean_token_accuracy logged from logits via MuseSFTTrainer)",
            flush=True,
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
        # Belt-and-suspenders: Trainer→Accelerator only constructs ParallelismConfig
        # when this env is true; otherwise CP flags are ignored and each rank holds
        # the full sequence → OOM at long max_length.
        os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
        os.environ.setdefault("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_TP_SIZE", "1")
        os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cp_size)
        os.environ.setdefault("PARALLELISM_CONFIG_CP_BACKEND", "torch")
        print(
            f"[muse] CP size={cp_size}: pad_to_multiple_of={pad_multiple}, "
            f"attn=sdpa (load-time), env PARALLELISM_CONFIG_CP_SIZE="
            f"{os.environ.get('PARALLELISM_CONFIG_CP_SIZE')} "
            f"ACCELERATE_USE_PARALLELISM_CONFIG="
            f"{os.environ.get('ACCELERATE_USE_PARALLELISM_CONFIG')}",
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
        f"(from ratio={warmup_ratio:g}, ~{total_steps} steps); "
        f"save_steps={save_steps} rolling_limit={save_limit} "
        f"(epoch-end ckpts permanent)",
        flush=True,
    )
    sft_args = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        callbacks=_make_muse_checkpoint_callbacks(
            save_total_limit=save_limit,
            has_eval=has_eval,
        ),
    )
    if has_eval:
        trainer_kwargs["eval_dataset"] = {"anti_forget": eval_ds}
    # TRL API drift: processing_class vs tokenizer
    try:
        trainer = MuseSFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = MuseSFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    if cp_size > 1:
        pc = None
        try:
            pc = trainer.accelerator.parallelism_config
        except Exception:
            pc = None
        cp_ok = bool(pc is not None and getattr(pc, "cp_enabled", False))
        print(
            f"[muse] after MuseSFTTrainer: parallelism_config={pc!r} cp_enabled={cp_ok}",
            flush=True,
        )
        if not cp_ok:
            raise SystemExit(
                "[muse] Context Parallelism is NOT active (parallelism_config missing/"
                "cp_enabled=False). Long max_length will OOM as if CP_SIZE=1.\n"
                "Check: accelerate launch must set ACCELERATE_USE_PARALLELISM_CONFIG=true "
                "and PARALLELISM_CONFIG_CP_SIZE, with --use_fsdp --fsdp_version 2 "
                "--use_parallelism_config."
            )

    if resume_from:
        # Torch 2.13 + FSDP2/DTensor full Trainer resume often dies with:
        #   '_MeshLayout' object has no attribute 'axes'
        # (shapes match; mesh metadata copy fails). Default under CP: adapter-only.
        want_full = bool(args.resume_full) or (
            os.environ.get("BIV_RESUME_FULL", "").lower() in {"1", "true"}
        )
        want_adapter = bool(args.resume_adapter_only) or (
            os.environ.get("BIV_RESUME_ADAPTER_ONLY", "").lower() in {"1", "true"}
        )
        if want_full and want_adapter:
            raise SystemExit("[muse] pass only one of --resume-full / --resume-adapter-only")
        use_adapter_only = want_adapter or (cp_size > 1 and not want_full)
        if use_adapter_only:
            print(
                f"[muse] resume adapter-only from {resume_from} "
                "(LoRA via PEFT; Adam/scheduler/RNG/step via Trainer; "
                "skip broken FSDP model copy). Full path: --resume-full",
                flush=True,
            )
            _train_resume_adapter_only(trainer, model, resume_from)
        else:
            print(
                f"[muse] resume_from={resume_from} "
                "(restores model + optimizer + LR scheduler + RNG from Trainer ckpt)",
                flush=True,
            )
            trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[muse] saved adapter → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
