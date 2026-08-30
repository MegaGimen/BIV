#!/usr/bin/env python3
"""Stage 1: JEPA on the fish-cut backbone. Observation tokens are not LM labels.

Encodes (h, a, o) from mix JSONL messages, predicts ẑ = JEPA(c_t, u*) vs stop-grad z*.
4-GPU FSDP2 + Context Parallel (Muse recipe), max_length=65536.

  cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.hao import split_hao  # noqa: E402

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
    rows: list[list] = []
    for src in sources:
        path = mix_dir / src / f"{split}.jsonl"
        if not path.is_file():
            log(f"skip missing {path}")
            continue
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
                rows.append(list(hao))
                if limit is not None and len(rows) >= limit:
                    return rows
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


def open_tb(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))
    log(f"tensorboard → {log_dir}  (tensorboard --logdir {log_dir.parent})")
    return writer


def resolve_tb_dir(tcfg: dict[str, Any], out_dir: Path, cli: Path | None) -> Path:
    raw = (
        cli
        or os.environ.get("LOGGING_DIR")
        or os.environ.get("TF_LOGS")
        or tcfg.get("logging_dir")
        or (out_dir / "tb")
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
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None, help="Context-parallel size (Ring Attention).")
    p.add_argument(
        "--logging-dir",
        type=Path,
        default=None,
        help="TensorBoard root (default: $LOGGING_DIR or $TF_LOGS or output_dir/tb)",
    )
    return p.parse_args()


def save_ckpt(accelerator, model, jepa, tokenizer, path: Path, extra: dict | None = None) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(path)
    torch_mod = __import__("torch")
    torch_mod.save(jepa.state_dict(), path / "jepa.pt")
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    if extra:
        (path / "train_meta.json").write_text(
            json.dumps(extra, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    log(f"saved {path}")


def main() -> None:
    args = parse_args()

    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from biv_wm.jepa import JEPAPred, cosine_align_loss

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

    model_dir = _resolve(args.model_dir or cfg["model_dir"])
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    if not model_dir.is_dir():
        raise SystemExit(f"missing cut model {model_dir}; run python train/scripts/cut_stage1.py")
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

    suffixes = list(tcfg.get("target_modules") or [])
    targets = two_d_lora_targets(model, suffixes)
    rank_log(f"lora 2D targets={len(targets)}")
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
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False
    hidden = int(getattr(getattr(model.config, "text_config", model.config), "hidden_size", 2048))
    jepa = JEPAPred(dim=hidden, hidden=int(tcfg.get("jepa_hidden") or hidden * 2))
    if is_main:
        from biv_wm.arch import log_train_architecture

        log_train_architecture(
            model=model,
            extra={"jepa": jepa},
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
    loader = DataLoader(
        train_ds,
        batch_size=int(tcfg.get("batch_size") or 1),
        shuffle=True,
        generator=gen,
        collate_fn=lambda b: collate(b, pad_id, pad_multiple),
        num_workers=0,
    )

    # FSDP2: only one nn.Module. JEPA is a small MLP, replicate it on each rank.
    model = accelerator.prepare(model)
    jepa = jepa.to(device=accelerator.device, dtype=dtype)
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": float(tcfg.get("lr") or 2e-4),
            },
            {"params": list(jepa.parameters()), "lr": float(tcfg.get("jepa_lr") or 1e-3)},
        ],
        weight_decay=0.01,
    )

    max_norm = float(tcfg.get("max_grad_norm") or 1.0)
    log_every = int(tcfg.get("logging_steps") or 10)
    save_every = int(tcfg.get("save_steps") or 100)
    epochs = int(tcfg.get("num_epochs") or 1)
    max_steps = args.max_steps
    steps_per_epoch = math.ceil(len(loader) / accum)
    rank_log(f"steps_per_epoch≈{steps_per_epoch} accum={accum}")

    writer = None
    tb_dir = resolve_tb_dir(tcfg, out_dir, args.logging_dir)
    if is_main:
        writer = open_tb(tb_dir)
        writer.add_text("data/mix_dir", str(mix_dir), 0)
        writer.add_text("data/sources", ", ".join(sources), 0)
        writer.add_text("train/max_length", str(max_length), 0)
        writer.add_text("train/cp_size", str(cp_size), 0)

    model.train()
    jepa.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    n_loss = 0
    run_h = run_a = run_o = 0.0

    try:
        for epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                h_len = float(batch["h_mask"].sum(dim=1).float().mean().item())
                a_len = float(batch["a_mask"].sum(dim=1).float().mean().item())
                o_len = float(batch["o_mask"].sum(dim=1).float().mean().item())
                with accelerator.accumulate(model):
                    with torch.no_grad():
                        z = last_hidden(model, batch["o_ids"], batch["o_mask"], cp_size)
                    c = last_hidden(model, batch["h_ids"], batch["h_mask"], cp_size)
                    u = last_hidden(model, batch["a_ids"], batch["a_mask"], cp_size)
                    pred = jepa(c, u)
                    loss = cosine_align_loss(pred, z)
                    accelerator.backward(loss)
                    running += float(loss.detach().float().item())
                    n_loss += 1
                    run_h += h_len
                    run_a += a_len
                    run_o += o_len
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm)
                        torch.nn.utils.clip_grad_norm_(jepa.parameters(), max_norm)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        step += 1
                        if step % log_every == 0 and is_main:
                            avg = running / max(n_loss, 1)
                            denom = max(n_loss, 1)
                            log(
                                f"epoch={epoch} step={step} loss={avg:.4f} "
                                f"len(h/a/o)={run_h/denom:.0f}/{run_a/denom:.0f}/{run_o/denom:.0f}"
                            )
                            if writer is not None:
                                writer.add_scalar("train/loss", avg, step)
                                writer.add_scalar("train/lr_backbone", opt.param_groups[0]["lr"], step)
                                writer.add_scalar("train/lr_jepa", opt.param_groups[1]["lr"], step)
                                writer.add_scalar("train/len_h", run_h / denom, step)
                                writer.add_scalar("train/len_a", run_a / denom, step)
                                writer.add_scalar("train/len_o", run_o / denom, step)
                                writer.flush()
                            running = 0.0
                            n_loss = 0
                            run_h = run_a = run_o = 0.0
                        if step % save_every == 0:
                            save_ckpt(
                                accelerator, model, jepa, None, out_dir / f"step-{step}"
                            )
                        if max_steps is not None and step >= max_steps:
                            break
            if max_steps is not None and step >= max_steps:
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    save_ckpt(
        accelerator,
        model,
        jepa,
        tokenizer,
        out_dir / "final",
        extra={
            "model_dir": str(model_dir),
            "mix_dir": str(mix_dir),
            "sources": sources,
            "steps": step,
            "max_length": max_length,
            "cp_size": cp_size,
            "tensorboard": str(tb_dir),
            "cut_meta": str(model_dir / "cut_meta.json"),
        },
    )
    rank_log(f"wrote {out_dir / 'final'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
