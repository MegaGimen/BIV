#!/usr/bin/env python3
"""Stage 1 LLM-JEPA on AgentWorld's own backbone. No fish-cut, no Instruct.

Two losses, same recipe as galilai-group/llm-jepa finetune.py:
  loss_ce   — write observation tokens given history+action (AgentWorld lm_head)
  loss_jepa — 1 - cosine(Enc(h+a) last hidden, Enc(o) last hidden); both live

No JEPA MLP, no inverse dynamics, no SimCSE/bank. 4-GPU FSDP2 + CP, max_length=65536.

  cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepallm.sh
  bash scripts/train_jepallm.sh --save-steps 1 --max-steps 2
  bash scripts/train_jepallm.sh --resume
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

DEFAULT_CONFIG = TRAIN / "configs" / "jepa" / "jepallm.yaml"


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


def pack_ids(h_ids: list[int], a_ids: list[int], o_ids: list[int], max_length: int):
    """Fit [h][a][o] into max_length: keep a, then o, then the tail of h."""
    a = list(a_ids[:max_length])
    room = max_length - len(a)
    o = list(o_ids[: max(0, room)])
    room = max_length - len(a) - len(o)
    h = list(h_ids[-room:] if room > 0 else [])
    if not h and not a:
        if h_ids:
            h = [h_ids[-1]]
        elif a_ids:
            a = [a_ids[0]]
        if len(h) + len(a) + len(o) > max_length:
            o = o[: max(0, max_length - len(h) - len(a))]
    return h, a, o


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
    h_ids, a_ids, o_ids = pack_ids(h["input_ids"], a["input_ids"], o["input_ids"], max_length)
    return {
        "h_ids": h_ids,
        "a_ids": a_ids,
        "o_ids": o_ids,
        "joint": h_ids + a_ids + o_ids,
        "left_len": len(h_ids) + len(a_ids),
    }


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

    def pad_rows(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        mlen = _pad_len(max(len(x) for x in rows), pad_multiple)
        out = torch.full((len(rows), mlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), mlen), dtype=torch.long)
        for i, row in enumerate(rows):
            if not row:
                continue
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return out, mask

    joint_ids, joint_mask = pad_rows([ex["joint"] for ex in batch])
    o_ids, o_mask = pad_rows([ex["o_ids"] for ex in batch])
    left_len = torch.tensor([ex["left_len"] for ex in batch], dtype=torch.long)
    h_len = torch.tensor([len(ex["h_ids"]) for ex in batch], dtype=torch.long)
    a_len = torch.tensor([len(ex["a_ids"]) for ex in batch], dtype=torch.long)
    o_len = torch.tensor([len(ex["o_ids"]) for ex in batch], dtype=torch.long)
    return {
        "joint_ids": joint_ids,
        "joint_mask": joint_mask,
        "o_ids": o_ids,
        "o_mask": o_mask,
        "left_len": left_len,
        "h_len": h_len,
        "a_len": a_len,
        "o_len": o_len,
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


def gather_hidden(h, attention_mask, cp_size: int = 1):
    """CP shards seq across ranks — gather to full length before indexing."""
    if cp_size > 1 and h.size(1) < attention_mask.size(1):
        return all_gather_seq(h)
    if cp_size > 1 and h.size(1) * cp_size == attention_mask.size(1):
        return all_gather_seq(h)
    return h


def full_hidden(model, input_ids, attention_mask, cp_size: int = 1):
    """Full last-layer hidden [B, S, D]. Forward is hidden-only (no full-seq logits)."""
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
    return gather_hidden(h, attention_mask, cp_size)


def last_from_hidden(h, attention_mask):
    import torch

    idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    idx = idx.clamp(max=h.size(1) - 1)
    b = torch.arange(h.size(0), device=h.device)
    return h[b, idx]


def cosine_live(left, right):
    """1 - cosine; neither side detached (LLM-JEPA Enc(Text) vs Enc(Code))."""
    import torch.nn.functional as F

    a = F.normalize(left.float(), dim=-1)
    b = F.normalize(right.float(), dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def observation_ce(hidden, left_len, o_ids, o_mask, lm_head):
    """CE only on observation tokens. hidden is the [h][a][o] joint last layer."""
    import torch
    import torch.nn.functional as F

    losses = []
    bsz = hidden.size(0)
    for i in range(bsz):
        n = int(o_mask[i].sum().item())
        left = int(left_len[i].item())
        if n <= 0 or left < 1:
            continue
        start = left - 1
        end = start + n
        if end > hidden.size(1):
            n = hidden.size(1) - start
            if n <= 0:
                continue
            end = start + n
        sl = hidden[i, start:end]
        logits = lm_head(sl)
        labels = o_ids[i, : sl.size(0)]
        losses.append(F.cross_entropy(logits.float(), labels))
    if not losses:
        return hidden.new_zeros(())
    return torch.stack(losses).mean()


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
    return root / f"jepallm-{stamp}"


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
    tokenizer,
    path: Path,
    extra: dict | None = None,
    *,
    epoch: int,
    step: int,
) -> None:
    accelerator.wait_for_everyone()
    lora_cpu = gather_lora_cpu(model, keep=accelerator.is_main_process)
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        save_adapter(unwrapped, lora_cpu, path)
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
    pth = Path(raw)
    if not pth.is_absolute():
        pth = _resolve(pth)
    if pth.is_dir() and (pth / "trainer_state.json").is_file():
        return pth
    raise SystemExit(f"--resume path is not a checkpoint: {pth}")


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


def load_adapter(model, path: Path, log_fn) -> tuple[int, int]:
    from safetensors.torch import load_file

    adapter = path / "adapter_model.safetensors"
    if adapter.is_file():
        load_lora_into_model(model, load_file(str(adapter)), log_fn)
    else:
        raise SystemExit(f"resume: missing {adapter}")
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
    from biv_wm.arch import install_hidden_only_forward, lm_head_module, log_world_architecture
    from biv_wm.jepa import collapse_stats, format_collapse_line

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
    out_dir = _resolve(tcfg.get("output_dir") or "outputs/jepallm_stage1")
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
    install_hidden_only_forward(model, detach_head=False)
    resume_dir = load_resume_dir(out_dir, args.resume)
    resume_epoch, resume_step = 0, 0
    if resume_dir is not None:
        resume_epoch, resume_step = load_adapter(model, resume_dir, rank_log)
        rank_log(
            f"resume {resume_dir.name} trainer_state epoch={resume_epoch} "
            f"global_step={resume_step} (newest by epoch, then step)"
        )
    if is_main:
        log_world_architecture(
            model=model,
            extra={},
            model_dir=model_dir,
            log=log,
            expect_lm_head="attached",
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

    backbone_lr = float(tcfg.get("lr") or 5e-5)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=backbone_lr,
        weight_decay=float(tcfg.get("weight_decay") or 0.01),
    )
    model, opt = accelerator.prepare(model, opt)
    lm_head = lm_head_module(accelerator.unwrap_model(model))
    if lm_head is None:
        raise SystemExit("LLM-JEPA Stage 1 needs AgentWorld lm_head attached")

    max_norm = float(tcfg.get("max_grad_norm") or 1.0)
    log_every = int(
        args.log_steps
        if args.log_steps is not None
        else (
            args.collapse_steps
            if args.collapse_steps is not None
            else (tcfg.get("log_steps") or tcfg.get("logging_steps") or 5)
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
    ce_w = float(tcfg.get("ce_weight") if tcfg.get("ce_weight") is not None else 1.0)
    jepa_w = float(tcfg.get("jepa_weight") if tcfg.get("jepa_weight") is not None else 0.1)
    epochs = int(tcfg.get("num_epochs") or 2)
    max_steps = args.max_steps
    steps_per_epoch = math.ceil(len(loader) / accum)
    planned = steps_per_epoch * epochs
    total_opt = planned if max_steps is None else min(planned, int(max_steps))
    warmup = int(tcfg.get("warmup_steps") or 50)
    warmup = max(0, min(warmup, max(total_opt - 1, 0)))
    sched = LambdaLR(opt, _warmup_lambda(warmup))
    for _ in range(resume_step):
        sched.step()
    rank_log(
        f"epochs={epochs} steps_per_epoch≈{steps_per_epoch} accum={accum} "
        f"lr_lora={backbone_lr} warmup={warmup} "
        f"save_steps={save_every} log_steps={log_every} "
        f"ce_w={ce_w} jepa_w={jepa_w} save_total_limit={save_limit} resume_step={resume_step}"
    )

    ckpt_extra = {
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "sources": sources,
        "max_length": max_length,
        "cp_size": cp_size,
        "lm_head": "attached_frozen_base",
        "recipe": "llm-jepa ce+cosine",
        "backbone": "AgentWorld only, no fish-cut, no Instruct tail",
    }
    seen_pred: deque = deque()
    seen_z: deque = deque()
    seen_o: deque = deque()
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

        if not seen_pred:
            _clear()
            return
        stats = collapse_stats(
            torch.stack(list(seen_pred)),
            torch.stack(list(seen_z)),
            list(seen_o),
        )
        if not is_main:
            _clear()
            return
        emit(f"[jepallm] collapse step={step_i} {format_collapse_line(stats)}")
        payload = json.dumps(dict(stats), indent=2, ensure_ascii=False) + "\n"
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
            tokenizer,
            dest,
            extra=extra,
            epoch=epoch_i,
            step=step_i,
        )
        if is_main:
            emit(f"[jepallm] checkpoint ({kind}) → {dest.name}")
            rotate_rolling(out_dir, save_limit, log=lambda m: emit(f"[jepallm] {m}"))

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
        writer.add_text("train/ce_weight", str(ce_w), 0)
        writer.add_text("train/jepa_weight", str(jepa_w), 0)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None  # type: ignore[assignment]

    pbar = None
    if is_main and tqdm is not None:
        pbar = tqdm(
            total=total_opt,
            initial=min(resume_step, total_opt),
            desc="jepallm",
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
    step = resume_step
    skip_micro = resume_step * accum
    micro_seen = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    run_ce = 0.0
    run_jepa = 0.0
    n_loss = 0
    run_h = run_a = run_o = 0.0
    last_train_loss = None
    hit_max = False

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
                h_len = float(batch["h_len"].float().mean().item())
                a_len = float(batch["a_len"].float().mean().item())
                o_len = float(batch["o_len"].float().mean().item())
                with accelerator.accumulate(model):
                    h_joint = full_hidden(
                        model, batch["joint_ids"], batch["joint_mask"], cp_size
                    )
                    left_idx = (batch["left_len"] - 1).clamp(min=0, max=h_joint.size(1) - 1)
                    b = torch.arange(h_joint.size(0), device=h_joint.device)
                    h_left = h_joint[b, left_idx]
                    ce_loss = observation_ce(
                        h_joint, batch["left_len"], batch["o_ids"], batch["o_mask"], lm_head
                    )
                    h_right_seq = full_hidden(
                        model, batch["o_ids"], batch["o_mask"], cp_size
                    )
                    h_right = last_from_hidden(h_right_seq, batch["o_mask"])
                    jepa_loss = cosine_live(h_left, h_right)
                    loss = ce_w * ce_loss + jepa_w * jepa_loss
                    o_texts = decode_o_texts(tokenizer, batch)
                    for i in range(h_left.size(0)):
                        seen_pred.append(h_left[i].detach().float().cpu())
                        seen_z.append(h_right[i].detach().float().cpu())
                        seen_o.append(o_texts[i])
                    accelerator.backward(loss)
                    running += float(loss.detach().float().item())
                    run_ce += float(ce_loss.detach().float().item())
                    run_jepa += float(jepa_loss.detach().float().item())
                    n_loss += 1
                    run_h += h_len
                    run_a += a_len
                    run_o += o_len
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm)
                        opt.step()
                        sched.step()
                        opt.zero_grad(set_to_none=True)
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
                                emit(
                                    f"epoch={epoch} step={step} loss={last_train_loss:.4f} "
                                    f"ce={run_ce / denom:.4f} jepa={run_jepa / denom:.4f} "
                                    f"len(h/a/o)={run_h / denom:.0f}/{run_a / denom:.0f}/{run_o / denom:.0f}"
                                )
                                if writer is not None:
                                    writer.add_scalar("train/loss", last_train_loss, step)
                                    writer.add_scalar("train/loss_ce", run_ce / denom, step)
                                    writer.add_scalar("train/loss_jepa", run_jepa / denom, step)
                                    writer.add_scalar("train/lr_backbone", opt.param_groups[0]["lr"], step)
                                    writer.add_scalar("train/len_h", run_h / denom, step)
                                    writer.add_scalar("train/len_a", run_a / denom, step)
                                    writer.add_scalar("train/len_o", run_o / denom, step)
                                running = 0.0
                                run_ce = 0.0
                                run_jepa = 0.0
                                n_loss = 0
                                run_h = run_a = run_o = 0.0
                            check_collapse(step)
                        if is_last_in_epoch:
                            emit(
                                f"[jepallm] epoch {epoch + 1} end: force checkpoint "
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
