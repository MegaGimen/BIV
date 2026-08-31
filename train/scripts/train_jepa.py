#!/usr/bin/env python3
"""Stage 1: JEPA on AgentWorld's own, unmodified backbone. No fish-cut, no Instruct.

AgentWorld and Instruct are two separate backbones (see AGENTS.md "模型架构").
This script only ever touches AgentWorld: c_t, u*, z* are all encoded by
AgentWorld's own 40 layers. Instruct + draft/scorer/W (Stage 2) attach later
and call this trained, frozen JEPA like an advisor.

Encodes (h, a, o) from mix JSONL messages, predicts ẑ = JEPA(c_t, u*) vs stop-grad z*.
Inverse dynamics Inv(c, z*)→u sits on the same step (anti-collapse). No ranking NCE.
4-GPU FSDP2 + Context Parallel (Muse recipe), max_length=65536.

  cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa.sh
  bash scripts/train_jepa.sh --save-steps 1 --max-steps 2   # smoke the FSDP saver
  bash scripts/train_jepa.sh --resume                         # newest ckpt (epoch, then step)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
MERGE = ROOT / "merge"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MERGE) not in sys.path:
    sys.path.insert(0, str(MERGE))

from biv_wm.ckpt import (  # noqa: E402
    canonical_lora_key,
    epoch_end_name,
    find_latest_ckpt,
    rotate_rolling,
    rolling_name,
    write_trainer_state,
)
from biv_wm.hao import split_hao  # noqa: E402
from download import resolve_model  # noqa: E402

DEFAULT_CONFIG = TRAIN / "configs" / "jepa" / "stage1.yaml"


def log(msg: str) -> None:
    print(msg, flush=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"bad yaml {path}")
    return data


def _resolve(raw: str | Path, base: Path = TRAIN) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (base / p)


def resolve_mix(raw: str | Path, sources: list[str]) -> Path:
    wanted = _resolve(raw)
    candidates = [wanted]
    for name in ("mix_v2", "mix_v1"):
        alt = TRAIN / "data" / "processed" / name
        if alt not in candidates:
            candidates.append(alt)
    for path in candidates:
        if any((path / src / "train.jsonl").is_file() for src in sources):
            if path != wanted:
                log(f"mix {wanted} missing or empty, using {path}")
            return path
    raise SystemExit(
        f"no mix JSONL at {wanted} or mix_v2/mix_v1; "
        "run python train/scripts/prepare_data.py --wm-code --wm-os "
        "--out-dir train/data/processed/mix_v2"
    )


def two_d_lora_targets(model, suffixes: list[str]) -> list[str]:
    """PEFT LoRA needs 2D Linear. MoE expert gate/up/down are 3D — skip those."""
    want = set(suffixes)
    names: list[str] = []
    for n, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or getattr(w, "ndim", 0) != 2:
            continue
        leaf = n.rsplit(".", 1)[-1]
        if leaf in want:
            names.append(n)
    if not names:
        raise SystemExit(f"no 2D LoRA modules matching {suffixes}")
    return names


def _content(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def encode_texts(
    tokenizer,
    h_msgs: list,
    a_msg: dict,
    o_msg: dict,
    max_length: int,
) -> dict[str, Any]:
    try:
        h_text = tokenizer.apply_chat_template(
            h_msgs if h_msgs else [{"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        h_text = "\n".join(_content(m) for m in h_msgs)
    a_text = _content(a_msg)
    o_text = _content(o_msg)
    h = tokenizer(
        h_text, truncation=True, max_length=max_length, add_special_tokens=True
    )
    a = tokenizer(
        a_text, truncation=True, max_length=max_length, add_special_tokens=True
    )
    o = tokenizer(
        o_text, truncation=True, max_length=max_length, add_special_tokens=True
    )
    return {"h": dict(h), "a": dict(a), "o": dict(o)}


def load_rows(mix_dir: Path, sources: list[str], split: str, limit: int | None) -> list[list]:
    """Read (h, a, o) rows for each source.

    `limit`, if set, is a *global* row budget split evenly across `sources`
    and filled by reservoir sampling (Algorithm R) within each source. This
    keeps every source represented in a downsampled quick run instead of the
    old behavior, which just took the first `limit` rows in file order and
    could return zero rows from any source after the first once the budget
    was hit (e.g. wm_os would get nothing if wm_code alone exceeded limit).
    Reservoir sampling also avoids reading whole multi-GB files into memory
    just to shuffle, and avoids a positional bias if a file happens to be
    sorted/grouped (e.g. by task or time) rather than pre-shuffled.
    """
    per_source_limit = None if limit is None else max(1, math.ceil(limit / len(sources)))
    rng = random.Random(42)
    rows: list[list] = []
    for src in sources:
        path = mix_dir / src / f"{split}.jsonl"
        if not path.is_file():
            log(f"skip missing {path}")
            continue
        reservoir: list[list] = []
        seen = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                hao = split_hao(msgs) if isinstance(msgs, list) else None
                if hao is None:
                    continue
                item = list(hao)
                if per_source_limit is None:
                    reservoir.append(item)
                    continue
                seen += 1
                if len(reservoir) < per_source_limit:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < per_source_limit:
                        reservoir[j] = item
        suffix = f" (reservoir-sampled from {seen})" if per_source_limit is not None else ""
        log(f"{src}: {len(reservoir)} rows{suffix}")
        rows.extend(reservoir)
    return rows


class HaoDataset:
    def __init__(self, rows: list, tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        h, a, o = self.rows[idx]
        return encode_texts(self.tokenizer, h, a, o, self.max_length)


def _pad_len(n: int, multiple: int) -> int:
    if multiple <= 1:
        return n
    return ((n + multiple - 1) // multiple) * multiple


def collate(batch: list[dict[str, Any]], pad_id: int, pad_multiple: int = 1) -> dict[str, Any]:
    import torch

    def pad(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [ex[key]["input_ids"] for ex in batch]
        mlen = _pad_len(max(len(x) for x in ids), pad_multiple)
        out = torch.full((len(ids), mlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), mlen), dtype=torch.long)
        for i, row in enumerate(ids):
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return out, mask

    h_ids, h_mask = pad("h")
    a_ids, a_mask = pad("a")
    o_ids, o_mask = pad("o")
    return {
        "h_ids": h_ids,
        "h_mask": h_mask,
        "a_ids": a_ids,
        "a_mask": a_mask,
        "o_ids": o_ids,
        "o_mask": o_mask,
    }


def all_gather_seq(x, group=None):
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return x
    world = dist.get_world_size(group)
    if world <= 1:
        return x
    parts = [x.new_empty(x.shape) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=group)
    return __import__("torch").cat(parts, dim=1)


def last_hidden(model, input_ids, attention_mask, cp_size: int = 1):
    """Last non-pad hidden. CP shards seq across ranks — gather before indexing."""
    import torch

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hs = getattr(out, "hidden_states", None)
    if hs:
        h = hs[-1]
    else:
        h = getattr(out, "last_hidden_state", None)
    if h is None:
        raise RuntimeError(
            f"{type(out).__name__} has no hidden_states; "
            f"fields={getattr(out, '__dataclass_fields__', {})}"
        )
    mask = attention_mask
    if cp_size > 1 and h.size(1) < mask.size(1):
        h = all_gather_seq(h)
    elif cp_size > 1 and h.size(1) * cp_size == mask.size(1):
        h = all_gather_seq(h)
    idx = mask.long().sum(dim=1).clamp(min=1) - 1
    idx = idx.clamp(max=h.size(1) - 1)
    b = torch.arange(h.size(0), device=h.device)
    return h[b, idx]


def force_attn_implementation(model, impl: str) -> None:
    configs = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        configs.append(cfg)
        tc = getattr(cfg, "text_config", None)
        if tc is not None:
            configs.append(tc)
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
    for path in (
        lambda: model.get_base_model().model.language_model,
        lambda: model.model.language_model,
        lambda: model.language_model,
    ):
        try:
            lm = path()
            if hasattr(lm, "config"):
                lm.config._attn_implementation = impl
        except Exception:
            continue
    log(f"attn_implementation={impl}")


def load_backbone(model_dir: Path, dtype, checkpointing: bool, *, attn_implementation=None, distributed=False):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
        log(f"load attn_implementation={attn_implementation}")
    model = None
    err = None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = loader.from_pretrained(str(model_dir), **kwargs)
            break
        except TypeError:
            kwargs.pop("dtype", None)
            try:
                model = loader.from_pretrained(str(model_dir), **kwargs)
                break
            except Exception as e:
                err = e
        except Exception as e:
            err = e
    if model is None:
        raise SystemExit(f"failed to load {model_dir}: {err}")
    if attn_implementation:
        force_attn_implementation(model, attn_implementation)
    if checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tok


def apply_selective_checkpointing(model, rank_log_fn=None, max_uncheckpoint: int | None = None) -> int:
    """Disable gradient checkpointing on linear-attention layers only.

    Full-attention layers (O(L²) activation memory — ~8 GiB/layer at
    max_length=65536) MUST remain checkpointed. Linear-attention (Gated
    DeltaNet) layers are O(L), safe to uncheckpoint given the ~33 GiB/GPU
    headroom observed at average length. Uncheckpointing a layer means its
    intermediate activations are stored instead of recomputed during backward,
    saving one forward-equivalent recompute per layer per backward pass.

    GradientCheckpointingLayer (transformers/modeling_layers.py) stores the
    flag as an instance attribute set by gradient_checkpointing_enable(); we
    just override it per-layer with False for linear-attention layers.

    Discovery: match by class name "Qwen3_5MoeDecoderLayer" via
    model.named_modules(), pulling the layer index from the dotted module
    path suffix (e.g. "...layers.5" -> 5). This is robust to PEFT wrapping
    (named_modules() traverses the full tree regardless of wrapper depth)
    AND avoids a real collision found by probe.py --speed-advice: the
    decoder layer's own child mixer (Qwen3_5MoeGatedDeltaNet, for its
    recurrent DeltaNet state) also carries a `layer_idx` attribute, so a
    naive `layer_idx`-keyed modules() scan silently ends up holding the
    *child mixer* instead of the actual decoder layer (modules() visits
    parent-before-children, so a dict keyed by layer_idx gets overwritten
    by the later-visited child) — that mixer has no gradient_checkpointing
    attribute, so the previous version of this function found 0 layers
    and uncheckpointed nothing.

    max_uncheckpoint caps how many linear-attention layers get uncheckpointed
    (None = all of them). All linear-attn layers cost the same per-layer
    activation memory, so this just takes the first N by index — use it to
    dial back memory usage if 30/30 OOMs (e.g. try 15 for roughly half the
    activation memory and half the recompute savings).

    Returns the count of uncheckpointed layers.
    """
    import re

    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", cfg)
    layer_types: list[str] = getattr(tc, "layer_types", [])
    # fallback: every 4th layer is full attention in Qwen3.5-35B-A3B
    full_attn_fallback = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
    if layer_types:
        full_attn = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    else:
        full_attn = full_attn_fallback

    decoder_layers: dict[int, Any] = {}
    for name, m in model.named_modules():
        if type(m).__name__ == "Qwen3_5MoeDecoderLayer":
            match = re.search(r"\.(\d+)$", name)
            if match:
                decoder_layers[int(match.group(1))] = m

    if not decoder_layers:
        if rank_log_fn:
            rank_log_fn("selective_checkpointing: no Qwen3_5MoeDecoderLayer found by class name, skipping")
        return 0

    n_missing_attr = sum(1 for layer in decoder_layers.values() if not hasattr(layer, "gradient_checkpointing"))
    linear_idx_sorted = sorted(i for i in decoder_layers if i not in full_attn)
    if max_uncheckpoint is not None:
        linear_idx_sorted = linear_idx_sorted[: max(0, int(max_uncheckpoint))]
    to_uncheckpoint = set(linear_idx_sorted)

    n_uncheckpointed = 0
    for idx, layer in decoder_layers.items():
        if idx in to_uncheckpoint and hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = False
            n_uncheckpointed += 1

    if rank_log_fn:
        rank_log_fn(
            f"selective_checkpointing: {n_uncheckpointed}/{len(decoder_layers)} linear-attn "
            f"layers uncheckpointed (idx {sorted(to_uncheckpoint)})  "
            f"full-attn (checkpointed): {sorted(full_attn)}"
            + (f"  WARNING: {n_missing_attr} layers have no gradient_checkpointing attr" if n_missing_attr else "")
        )
    return n_uncheckpointed


def open_tb(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))
    log(f"tensorboard → {log_dir}  (tensorboard --logdir {log_dir.parent})")
    return writer


def resolve_tb_dir(tcfg: dict[str, Any], _out_dir: Path, cli: Path | None) -> Path:
    """AutoDL's TensorBoard panel watches ``/root/tf-logs`` (same as Muse / eval)."""
    raw = (
        cli
        or os.environ.get("LOGGING_DIR")
        or os.environ.get("TF_LOGS")
        or tcfg.get("logging_dir")
        or "/root/tf-logs"
    )
    root = Path(str(raw))
    if not root.is_absolute():
        root = _resolve(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return root / f"jepa-{stamp}"


def resolve_cp_size(cli: int | None) -> int:
    if cli is not None and int(cli) > 0:
        return int(cli)
    return int(os.environ.get("BIV_CP_SIZE") or os.environ.get("PARALLELISM_CONFIG_CP_SIZE") or "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model-dir", type=str, default=None, help="AgentWorld hub id or local dir")
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Override yaml save_steps (default 25).",
    )
    p.add_argument(
        "--log-steps",
        type=int,
        default=None,
        help="Loss + collapse log every N optimizer steps (default 5). Not save.",
    )
    p.add_argument(
        "--collapse-steps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume weights. Bare --resume picks the newest ckpt (epoch, then step).",
    )
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None, help="Context-parallel size (Ring Attention).")
    p.add_argument("--cache-dir", type=Path, default=None, help="Default: merge/output/cache")
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
        help="Where to download AgentWorld from if not already cached.",
    )
    p.add_argument(
        "--logging-dir",
        type=Path,
        default=None,
        help="TensorBoard root (default: $LOGGING_DIR or $TF_LOGS or /root/tf-logs)",
    )
    return p.parse_args()


def _full_cpu(param):
    """Materialize a (possibly DTensor) param on CPU. Collective if DTensor."""
    t = param.full_tensor() if hasattr(param, "full_tensor") else param
    t = t.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    return t.contiguous().clone()


def gather_lora_cpu(model, *, keep: bool) -> dict:
    """All ranks must call this: DTensor.full_tensor is a collective.

    Only LoRA tensors are gathered (full 35B gather OOMs). Cloned onto CPU so
    safetensors never sees an FSDP/DTensor storage pointer — that was the
    ``invalid python storage`` crash from ``PeftModel.save_pretrained``.
    """
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        full = _full_cpu(param)
        if keep:
            out[canonical_lora_key(name)] = full
        del full
    return out


def save_adapter(unwrapped, lora_cpu: dict, path: Path) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    if not lora_cpu:
        raise RuntimeError("no LoRA tensors gathered; refuse to write an empty adapter")
    save_file(lora_cpu, str(path / "adapter_model.safetensors"))
    adapter = getattr(unwrapped, "active_adapter", "default")
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0] if adapter else "default"
    cfg_map = getattr(unwrapped, "peft_config", None) or {}
    cfg = cfg_map.get(adapter) if isinstance(cfg_map, dict) else None
    if cfg is not None and hasattr(cfg, "save_pretrained"):
        cfg.save_pretrained(str(path))


def save_ckpt(
    accelerator,
    model,
    jepa,
    tokenizer,
    path: Path,
    extra: dict | None = None,
    *,
    epoch: int,
    step: int,
    inv=None,
) -> None:
    accelerator.wait_for_everyone()
    lora_cpu = gather_lora_cpu(model, keep=accelerator.is_main_process)
    jepa_cpu = {k: v.detach().cpu().contiguous().clone() for k, v in jepa.state_dict().items()}
    inv_cpu = None
    if inv is not None:
        inv_cpu = {k: v.detach().cpu().contiguous().clone() for k, v in inv.state_dict().items()}
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        save_adapter(unwrapped, lora_cpu, path)
        torch_mod = __import__("torch")
        torch_mod.save(jepa_cpu, path / "jepa.pt")
        if inv_cpu is not None:
            torch_mod.save(inv_cpu, path / "inv.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        meta = dict(extra or {})
        write_trainer_state(path, epoch=epoch, global_step=step, extra=meta)
        (path / "train_meta.json").write_text(
            json.dumps({"epoch": epoch, "global_step": step, **meta}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        log(f"saved {path}")
    accelerator.wait_for_everyone()


def load_resume_dir(out_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    if raw == "auto":
        found = find_latest_ckpt(out_dir)
        if found is None:
            raise SystemExit(f"--resume: no complete checkpoint under {out_dir}")
        return found
    p = Path(raw)
    if not p.is_absolute():
        p = _resolve(p)
    if p.is_dir() and (p / "trainer_state.json").is_file():
        return p
    raise SystemExit(f"--resume path is not a checkpoint: {p}")


def load_lora_into_model(model, sd: dict, log_fn) -> None:
    """Copy LoRA tensors by name. Skip PEFT's MoE WeightConverter (PEFT/transformers skew)."""
    live = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if not live:
        raise SystemExit("resume: model has no LoRA parameters")
    by_canon = {canonical_lora_key(n): n for n in live}
    mapped: dict[str, Any] = {}
    unmatched = 0
    shape_bad: list[str] = []
    for k, v in sd.items():
        if "lora_" not in k:
            continue
        dest = by_canon.get(canonical_lora_key(k))
        if dest is None:
            unmatched += 1
            continue
        if tuple(live[dest].shape) != tuple(v.shape):
            shape_bad.append(f"{dest} file={tuple(v.shape)} model={tuple(live[dest].shape)}")
            continue
        mapped[dest] = v
    if shape_bad:
        raise SystemExit("resume LoRA shape mismatch: " + "; ".join(shape_bad[:8]))
    if not mapped:
        raise SystemExit(
            f"resume: 0 LoRA tensors matched (file_keys={len(sd)} model_lora={len(live)})"
        )
    model.load_state_dict(mapped, strict=False)
    log_fn(
        f"resume adapter matched={len(mapped)}/{len(live)} "
        f"unmatched_saved={unmatched}"
    )


def load_adapter_and_heads(model, jepa, inv, path: Path, log_fn) -> tuple[int, int]:
    import torch
    from safetensors.torch import load_file

    adapter = path / "adapter_model.safetensors"
    if adapter.is_file():
        load_lora_into_model(model, load_file(str(adapter)), log_fn)
    else:
        raise SystemExit(f"resume: missing {adapter}")
    jepa_p = path / "jepa.pt"
    if jepa_p.is_file():
        jepa.load_state_dict(torch.load(jepa_p, map_location="cpu"))
        log_fn(f"resume jepa.pt from {path.name}")
    inv_p = path / "inv.pt"
    if inv is not None and inv_p.is_file():
        inv.load_state_dict(torch.load(inv_p, map_location="cpu"))
        log_fn(f"resume inv.pt from {path.name}")
    elif inv is not None:
        log_fn(f"no inv.pt in {path.name}; inverse head stays randomly initialized")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state.get("epoch") or 0), int(state.get("global_step") or 0)


def decode_ids_texts(tokenizer, ids, mask) -> list[str]:
    out: list[str] = []
    for i in range(ids.size(0)):
        tok = ids[i][mask[i].bool()].tolist()
        out.append(tokenizer.decode(tok, skip_special_tokens=True))
    return out


def decode_o_texts(tokenizer, batch) -> list[str]:
    return decode_ids_texts(tokenizer, batch["o_ids"], batch["o_mask"])


def decode_a_texts(tokenizer, batch) -> list[str]:
    return decode_ids_texts(tokenizer, batch["a_ids"], batch["a_mask"])


def _warmup_lambda(warmup: int):
    def fn(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup))

    return fn


def main() -> None:
    args = parse_args()

    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
    from biv_wm.jepa import (  # noqa: PLC0415
        InverseDyn,
        JEPAPred,
        bank_nce_loss,
        collapse_stats,
        cosine_align_loss,
        format_collapse_line,
    )

    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        alt = ROOT / args.config
        if alt.is_file():
            cfg_path = alt
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {args.config} (tried {TRAIN / args.config})")
    cfg = _load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    accum = int(tcfg.get("grad_accum") or 8)
    cp_size = resolve_cp_size(args.cp_size)
    if cp_size > 1:
        os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
        os.environ.setdefault("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_TP_SIZE", "1")
        os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cp_size)
        os.environ.setdefault("PARALLELISM_CONFIG_CP_BACKEND", "torch")

    accelerator = Accelerator(gradient_accumulation_steps=accum)
    is_main = accelerator.is_main_process

    def rank_log(msg: str) -> None:
        if is_main:
            log(msg)

    cache_dir = _resolve(args.cache_dir or "merge/output/cache", ROOT)
    model_dir = resolve_model(
        str(args.model_dir or cfg["model_dir"]),
        source=args.source,
        cache_dir=cache_dir,
        role="world",
    )
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    out_dir = _resolve(tcfg.get("output_dir") or "outputs/jepa_stage1")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)

    max_length = int(args.max_length or tcfg.get("max_length") or 65536)
    pad_multiple = cp_size * 2 if cp_size > 1 else int(tcfg.get("pad_to_multiple_of") or 0)
    attn_impl = "sdpa" if cp_size > 1 else None
    distributed = accelerator.num_processes > 1
    rank_log(f"model={model_dir}")
    rank_log(f"mix={mix_dir} sources={sources}")
    rank_log(
        f"max_length={max_length} cp_size={cp_size} pad_multiple={pad_multiple} "
        f"nproc={accelerator.num_processes} attn={attn_impl}"
    )

    model, tokenizer = load_backbone(
        model_dir,
        dtype,
        bool(tcfg.get("gradient_checkpointing", True)),
        attn_implementation=attn_impl,
        distributed=distributed,
    )
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False

    from biv_wm.arch import install_hidden_only_forward, log_world_architecture

    # No fish-cut, no Instruct tail: the whole AgentWorld backbone is "world".
    # get_peft_model() below freezes every base-model param except these LoRA
    # targets, so there is no separate freeze() call to make.
    suffixes = list(tcfg.get("target_modules") or [])
    targets = two_d_lora_targets(model, suffixes)
    if not targets:
        raise SystemExit(f"no LoRA targets matching {suffixes} in AgentWorld backbone")
    rank_log(f"lora 2D targets (full AgentWorld backbone, no cut)={len(targets)}")
    lora = LoraConfig(
        r=int(tcfg.get("lora_rank") or 16),
        lora_alpha=int(tcfg.get("lora_alpha") or 32),
        lora_dropout=float(tcfg.get("lora_dropout") or 0.05),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if is_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    install_hidden_only_forward(model)
    if bool(tcfg.get("selective_checkpointing", True)) and bool(tcfg.get("gradient_checkpointing", True)):
        max_uncheckpoint = tcfg.get("selective_checkpointing_max_layers")
        apply_selective_checkpointing(
            model,
            rank_log_fn=rank_log,
            max_uncheckpoint=int(max_uncheckpoint) if max_uncheckpoint is not None else None,
        )
    hidden = int(getattr(getattr(model.config, "text_config", model.config), "hidden_size", 2048))
    jepa_h = int(tcfg.get("jepa_hidden") or hidden * 2)
    jepa = JEPAPred(dim=hidden, hidden=jepa_h)
    inv = InverseDyn(dim=hidden, hidden=jepa_h)
    resume_dir = load_resume_dir(out_dir, args.resume)
    resume_epoch, resume_step = 0, 0
    if resume_dir is not None:
        resume_epoch, resume_step = load_adapter_and_heads(
            model, jepa, inv, resume_dir, rank_log
        )
        rank_log(
            f"resume {resume_dir.name} trainer_state epoch={resume_epoch} "
            f"global_step={resume_step} (newest by epoch, then step)"
        )
    if is_main:
        log_world_architecture(
            model=model,
            extra={"jepa": jepa, "inv": inv},
            model_dir=model_dir,
            log=log,
        )

    train_rows = load_rows(mix_dir, sources, "train", tcfg.get("max_train_samples"))
    if not train_rows:
        raise SystemExit(f"no (h,a,o) rows under {mix_dir}/{sources}/train.jsonl")
    rank_log(f"train_rows={len(train_rows)}")

    train_ds = HaoDataset(train_rows, tokenizer, max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch_size = int(tcfg.get("batch_size") or 1)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        collate_fn=lambda b: collate(b, pad_id, pad_multiple),
        num_workers=0,
    )

    # FSDP2: one model + its optimizer in the same prepare(). JEPA is a small
    # MLP on each rank, separate optimizer (not an FSDP module).
    # LRs are not Muse SFT: cosine-align a new predictor on an already-trained
    # AgentWorld, so LoRA is a small nudge and the MLP is a new head.
    backbone_lr = float(tcfg.get("lr") or 5e-5)
    jepa_lr = float(tcfg.get("jepa_lr") or 1e-3)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=backbone_lr,
        weight_decay=float(tcfg.get("weight_decay") or 0.01),
    )
    model, opt = accelerator.prepare(model, opt)
    jepa = jepa.to(device=accelerator.device, dtype=dtype)
    inv = inv.to(device=accelerator.device, dtype=dtype)
    opt_jepa = torch.optim.AdamW(
        list(jepa.parameters()) + list(inv.parameters()),
        lr=jepa_lr,
        weight_decay=float(tcfg.get("jepa_weight_decay") or 0.0),
    )

    max_norm = float(tcfg.get("max_grad_norm") or 1.0)
    log_every = int(
        args.log_steps
        if args.log_steps is not None
        else (
            args.collapse_steps
            if args.collapse_steps is not None
            else (
                tcfg.get("log_steps")
                or tcfg.get("logging_steps")
                or tcfg.get("collapse_steps")
                or 5
            )
        )
    )
    save_every = int(
        args.save_steps if args.save_steps is not None else (tcfg.get("save_steps") or 25)
    )
    if save_every < 1:
        raise SystemExit(f"save_steps must be >= 1, got {save_every}")
    if log_every < 1:
        raise SystemExit(f"log_steps must be >= 1, got {log_every}")
    save_limit = int(tcfg.get("save_total_limit") or 3)
    inv_w = float(tcfg.get("inv_weight") or 0.3)
    # Space-for-time: batch z and z1 into one forward when nce_w > 0.
    # Instead of two calls (z no_grad + z1 live), one call with batch_size×2 on
    # o_ids eliminates one full 40-layer FSDP2 all-gather round per micro-batch.
    batch_o_z1 = bool(tcfg.get("batch_o_z1", True))
    # Anti-collapse, part 2/3 (see AGENTS.md "JEPA 家族怎么防坍缩"): loss_simcse is
    # the only loss whose gradient reaches the observation-encoding forward pass
    # itself. It costs exactly ONE extra live forward per micro-batch (z1, same
    # o_ids as z but a second dropout draw) — folded into the same backward as
    # everything else, no separate group buffer needed: the positive is this
    # micro-batch's own no-grad z, negatives come from the shared bank below.
    nce_w = float(tcfg.get("nce_weight") or 0.1)
    nce_temp = float(tcfg.get("nce_temperature") or 0.05)
    # Anti-collapse, part 3/3: corrected queue. Small + fresh (not the 256-deep,
    # cross-epoch queue that blew up loss_align last time), softmax temperature
    # instead of a hand-picked "push cosine to 0" target.
    bank_w = float(tcfg.get("bank_weight") or 0.2)
    bank_temp = float(tcfg.get("bank_temperature") or 0.05)
    bank_size = max(1, int(tcfg.get("bank_size") or 64))
    epochs = int(tcfg.get("num_epochs") or 2)
    max_steps = args.max_steps
    steps_per_epoch = math.ceil(len(loader) / accum)
    planned = steps_per_epoch * epochs
    total_opt = planned if max_steps is None else min(planned, int(max_steps))
    warmup = int(tcfg.get("warmup_steps") or 50)
    warmup = max(0, min(warmup, max(total_opt - 1, 0)))
    sched = LambdaLR(opt, _warmup_lambda(warmup))
    sched_jepa = LambdaLR(opt_jepa, _warmup_lambda(warmup))
    for _ in range(resume_step):
        sched.step()
        sched_jepa.step()
    rank_log(
        f"epochs={epochs} steps_per_epoch≈{steps_per_epoch} accum={accum} "
        f"lr_lora={backbone_lr} lr_jepa={jepa_lr} warmup={warmup} "
        f"save_steps={save_every} log_steps={log_every} "
        f"inv_w={inv_w} save_total_limit={save_limit} resume_step={resume_step} "
        f"nce_w={nce_w} nce_temp={nce_temp} "
        f"bank_w={bank_w} bank_temp={bank_temp} bank_size={bank_size}"
    )

    ckpt_extra = {
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "sources": sources,
        "max_length": max_length,
        "cp_size": cp_size,
        "lm_head": "detached_at_runtime",
        "backbone": "AgentWorld only, no fish-cut, no Instruct tail",
    }
    seen_pred: deque = deque()
    seen_z: deque = deque()
    seen_o: deque = deque()
    # Inverse dynamics has the same blind spot JEPA had: nobody checked whether
    # inv_hat (guessed command encoding) collapses too. Same paired/mismatch
    # machinery, applied to (inv_hat, u*, a_text) instead of (pred, z*, o_text).
    seen_invhat: deque = deque()
    seen_u: deque = deque()
    seen_a: deque = deque()
    writer = None

    def emit(msg: str) -> None:
        if is_main:
            try:
                from tqdm.auto import tqdm as _tqdm

                _tqdm.write(msg)
            except Exception:
                log(msg)

    def check_collapse(step_i: int) -> None:
        def _clear() -> None:
            seen_pred.clear()
            seen_z.clear()
            seen_o.clear()
            seen_invhat.clear()
            seen_u.clear()
            seen_a.clear()

        if not is_main or not seen_pred:
            _clear()
            return
        stats = collapse_stats(
            torch.stack(list(seen_pred)),
            torch.stack(list(seen_z)),
            list(seen_o),
        )
        emit(f"[jepa] collapse step={step_i} {format_collapse_line(stats)}")
        inv_stats: dict[str, object] | None = None
        if seen_invhat:
            inv_stats = collapse_stats(
                torch.stack(list(seen_invhat)),
                torch.stack(list(seen_u)),
                list(seen_a),
            )
            emit(f"[jepa] collapse(inv) step={step_i} {format_collapse_line(inv_stats)}")
        payload_obj: dict[str, object] = dict(stats)
        if inv_stats is not None:
            payload_obj["inv"] = inv_stats
        payload = json.dumps(payload_obj, indent=2, ensure_ascii=False) + "\n"
        (out_dir / "collapse.json").write_text(payload, encoding="utf-8")
        (out_dir / f"collapse-s{step_i}.json").write_text(payload, encoding="utf-8")
        if writer is not None:
            def _sc(name: str, block: object, key: str) -> None:
                if isinstance(block, dict) and block.get(key) is not None:
                    writer.add_scalar(name, float(block[key]), step_i)

            paired = stats["paired"]
            mis = stats["mismatch"]
            z_self = stats["z_self"]
            pred_self = stats["pred_self"]
            writer.add_scalar("collapse/n", float(stats["n"]), step_i)
            writer.add_scalar("collapse/skipped_same_o", float(stats["skipped_same_o"]), step_i)
            _sc("collapse/paired_median", paired, "median")
            _sc("collapse/paired_mean", paired, "mean")
            _sc("collapse/mismatch_median", mis, "median")
            _sc("collapse/mismatch_p90", mis, "p90")
            _sc("collapse/z_self_median", z_self, "median")
            _sc("collapse/pred_self_median", pred_self, "median")
            writer.add_text("collapse/verdict", str(stats.get("verdict")), step_i)
            if inv_stats is not None:
                _sc("collapse/inv_paired_median", inv_stats["paired"], "median")
                _sc("collapse/inv_mismatch_median", inv_stats["mismatch"], "median")
            writer.flush()
        _clear()

    def dump_ckpt(kind: str, epoch_i: int, step_i: int) -> None:
        extra = {**ckpt_extra, "kind": kind}
        if kind == "epoch-end":
            dest = out_dir / epoch_end_name(epoch_i, step_i)
        else:
            dest = out_dir / rolling_name(epoch_i, step_i)
        save_ckpt(
            accelerator,
            model,
            jepa,
            tokenizer,
            dest,
            extra=extra,
            epoch=epoch_i,
            step=step_i,
            inv=inv,
        )
        if is_main:
            emit(f"[jepa] checkpoint ({kind}) → {dest.name}")
            rotate_rolling(out_dir, save_limit, log=lambda m: emit(f"[jepa] {m}"))

    tb_dir = resolve_tb_dir(tcfg, out_dir, args.logging_dir)
    ckpt_extra["tensorboard"] = str(tb_dir)
    if is_main:
        writer = open_tb(tb_dir)
        writer.add_text("data/mix_dir", str(mix_dir), 0)
        writer.add_text("data/sources", ", ".join(sources), 0)
        writer.add_text("train/max_length", str(max_length), 0)
        writer.add_text("train/cp_size", str(cp_size), 0)
        writer.add_text("train/epochs", str(epochs), 0)
        writer.add_text("train/save_steps", str(save_every), 0)
        writer.add_text("train/log_steps", str(log_every), 0)
        writer.add_text("train/inv_weight", str(inv_w), 0)
        writer.add_text("train/nce_weight", str(nce_w), 0)
        writer.add_text("train/nce_temperature", str(nce_temp), 0)
        writer.add_text("train/bank_weight", str(bank_w), 0)
        writer.add_text("train/bank_temperature", str(bank_temp), 0)
        writer.add_text("train/bank_size", str(bank_size), 0)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None  # type: ignore[assignment]

    pbar = None
    if is_main and tqdm is not None:
        pbar = tqdm(
            total=total_opt,
            initial=min(resume_step, total_opt),
            desc="jepa",
            unit="step",
            dynamic_ncols=True,
        )

    if resume_step >= total_opt:
        rank_log(f"resume_step={resume_step} already covers total_opt={total_opt}; nothing to do")
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.close()
        return

    model.train()
    jepa.train()
    inv.train()
    step = resume_step
    skip_micro = resume_step * accum
    micro_seen = 0
    opt.zero_grad(set_to_none=True)
    opt_jepa.zero_grad(set_to_none=True)
    running = 0.0
    run_align = run_inv = run_bank = 0.0
    run_simcse = 0.0
    n_loss = 0
    n_simcse = 0
    run_h = run_a = run_o = 0.0
    last_train_loss = None
    hit_max = False
    # Anti-collapse buffers, see AGENTS.md "JEPA 家族怎么防坍缩". Every rank keeps
    # its own copy (DP_REPLICATE=1: CP shards one example's sequence across
    # ranks, it does not partition distinct examples per rank, so every rank
    # sees the identical stream of examples and these stay in sync without an
    # all-gather).
    bank_z: deque = deque(maxlen=bank_size)
    bank_o: deque = deque(maxlen=bank_size)

    try:
        for epoch in range(epochs):
            epoch_base = epoch * steps_per_epoch
            epoch_opt = max(0, min(resume_step - epoch_base, steps_per_epoch))
            for batch in loader:
                if micro_seen < skip_micro:
                    micro_seen += 1
                    continue
                micro_seen += 1
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                h_len = float(batch["h_mask"].sum(dim=1).float().mean().item())
                a_len = float(batch["a_mask"].sum(dim=1).float().mean().item())
                o_len = float(batch["o_mask"].sum(dim=1).float().mean().item())
                with accelerator.accumulate(model):
                    # When batch_o_z1 and nce_w>0: z and z1 are both o_ids
                    # encodings (different dropout draws). Batching them into
                    # one call (batch_size×2) eliminates one full 40-layer
                    # FSDP2 all-gather round vs two separate calls. The live
                    # z1 row carries gradient; z row is stop-grad'd at every
                    # downstream use (cosine_align_loss detaches target
                    # internally; inv uses z.detach(); bank_nce_loss detaches
                    # pos_z internally), so only z1 contributes real gradient
                    # through this forward — the z row's recompute/store cost
                    # during backward is negligible (o_ids is short).
                    if nce_w > 0 and batch_o_z1:
                        o_ids2 = torch.cat([batch["o_ids"], batch["o_ids"]], dim=0)
                        o_mask2 = torch.cat([batch["o_mask"], batch["o_mask"]], dim=0)
                        z_both = last_hidden(model, o_ids2, o_mask2, cp_size)
                        bsz = batch["o_ids"].size(0)
                        z = z_both[:bsz]   # stop-grad'd at point of use below
                        z1 = z_both[bsz:]  # live, for loss_simcse
                    else:
                        with torch.no_grad():
                            z = last_hidden(model, batch["o_ids"], batch["o_mask"], cp_size)
                        z1 = None
                    c = last_hidden(model, batch["h_ids"], batch["h_mask"], cp_size)
                    u = last_hidden(model, batch["a_ids"], batch["a_mask"], cp_size)
                    pred = jepa(c, u)
                    align_loss = cosine_align_loss(pred, z)
                    inv_hat = inv(c, z.detach())
                    inv_loss = cosine_align_loss(inv_hat, u)
                    o_texts = decode_o_texts(tokenizer, batch)
                    bank_stack = torch.stack(list(bank_z)) if bank_z else None
                    bank_texts = list(bank_o)
                    # loss_bank: corrected queue, part 3/3 of anti-collapse (see
                    # AGENTS.md). Cheap — bank entries are detached, no extra
                    # forward pass, just a softmax over a small rolling buffer.
                    bank_loss = None
                    if bank_stack is not None:
                        bank_loss = bank_nce_loss(pred, z, o_texts, bank_stack, bank_texts, bank_temp)
                    loss = align_loss + inv_w * inv_loss
                    if bank_loss is not None:
                        loss = loss + bank_w * bank_loss
                    # loss_simcse: anti-collapse part 1/3 — the only term whose
                    # gradient reaches the observation encoder (see AGENTS.md).
                    # z1 is either the batch-merge second row (no extra forward)
                    # or a separate live forward when batch_o_z1 is off.
                    simcse_loss = None
                    if nce_w > 0 and bank_stack is not None:
                        if z1 is None:  # batch_o_z1 disabled
                            z1 = last_hidden(model, batch["o_ids"], batch["o_mask"], cp_size)
                        simcse_loss = bank_nce_loss(z1, z, o_texts, bank_stack, bank_texts, nce_temp)
                        if simcse_loss is not None:
                            loss = loss + nce_w * simcse_loss
                    if is_main:
                        a_texts = decode_a_texts(tokenizer, batch)
                        for i in range(pred.size(0)):
                            seen_pred.append(pred[i].detach().float().cpu())
                            seen_z.append(z[i].detach().float().cpu())
                            seen_o.append(o_texts[i])
                            seen_invhat.append(inv_hat[i].detach().float().cpu())
                            seen_u.append(u[i].detach().float().cpu())
                            seen_a.append(a_texts[i])
                    accelerator.backward(loss)
                    for i in range(z.size(0)):
                        bank_z.append(z[i].detach())
                        bank_o.append(o_texts[i])
                    running += float(loss.detach().float().item())
                    run_align += float(align_loss.detach().float().item())
                    run_inv += float(inv_loss.detach().float().item())
                    if bank_loss is not None:
                        run_bank += float(bank_loss.detach().float().item())
                    if simcse_loss is not None:
                        run_simcse += float(simcse_loss.detach().float().item())
                        n_simcse += 1
                    n_loss += 1
                    run_h += h_len
                    run_a += a_len
                    run_o += o_len
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm)
                        torch.nn.utils.clip_grad_norm_(jepa.parameters(), max_norm)
                        torch.nn.utils.clip_grad_norm_(inv.parameters(), max_norm)
                        opt.step()
                        opt_jepa.step()
                        sched.step()
                        sched_jepa.step()
                        opt.zero_grad(set_to_none=True)
                        opt_jepa.zero_grad(set_to_none=True)
                        step += 1
                        epoch_opt += 1
                        last_train_loss = running / max(n_loss, 1)
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(
                                epoch=f"{epoch + 1}/{epochs}",
                                loss=f"{last_train_loss:.4f}",
                                refresh=False,
                            )
                        is_last_in_epoch = epoch_opt >= steps_per_epoch
                        if step % log_every == 0:
                            if is_main:
                                denom = max(n_loss, 1)
                                simcse_denom = max(n_simcse, 1)
                                emit(
                                    f"epoch={epoch} step={step} loss={last_train_loss:.4f} "
                                    f"align={run_align / denom:.4f} inv={run_inv / denom:.4f} "
                                    f"bank={run_bank / denom:.4f} simcse={run_simcse / simcse_denom:.4f} "
                                    f"len(h/a/o)={run_h / denom:.0f}/{run_a / denom:.0f}/{run_o / denom:.0f}"
                                )
                                if writer is not None:
                                    writer.add_scalar("train/loss", last_train_loss, step)
                                    writer.add_scalar("train/loss_align", run_align / denom, step)
                                    writer.add_scalar("train/loss_inv", run_inv / denom, step)
                                    writer.add_scalar("train/loss_bank", run_bank / denom, step)
                                    if n_simcse > 0:
                                        writer.add_scalar("train/loss_simcse", run_simcse / simcse_denom, step)
                                    writer.add_scalar("train/lr_backbone", opt.param_groups[0]["lr"], step)
                                    writer.add_scalar("train/lr_jepa", opt_jepa.param_groups[0]["lr"], step)
                                    writer.add_scalar("train/len_h", run_h / denom, step)
                                    writer.add_scalar("train/len_a", run_a / denom, step)
                                    writer.add_scalar("train/len_o", run_o / denom, step)
                                running = 0.0
                                run_align = run_inv = run_bank = 0.0
                                run_simcse = 0.0
                                n_loss = 0
                                n_simcse = 0
                                run_h = run_a = run_o = 0.0
                            check_collapse(step)
                        if is_last_in_epoch:
                            emit(
                                f"[jepa] epoch {epoch + 1} end: force checkpoint "
                                "(permanent epoch ckpt)"
                            )
                            dump_ckpt("epoch-end", epoch + 1, step)
                        elif step % save_every == 0:
                            dump_ckpt("rolling", epoch, step)
                        if max_steps is not None and step >= max_steps:
                            hit_max = True
                            break
            if hit_max:
                break
    finally:
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.flush()
            writer.close()

    if step > 0 and not hit_max:
        rank_log(f"wrote last epoch-end under {out_dir}")
    elif step > 0 and hit_max:
        rank_log(f"max_steps={max_steps} stop at step={step} under {out_dir}")
    else:
        rank_log("no optimizer steps; nothing saved")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
