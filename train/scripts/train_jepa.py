#!/usr/bin/env python3
"""Stage 1: JEPA on the fish-cut backbone. Observation tokens are not LM labels.

Encodes (h, a, o) from mix JSONL messages, predicts ẑ = JEPA(c_t, u*) vs stop-grad z*.
Backbone LoRA + JEPA; lm_head frozen. Reuses data/processed/mix_v2/{wm_code,wm_os}.

  python train/scripts/cut_stage1.py
  python train/scripts/train_jepa.py --config train/configs/jepa/stage1.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    def pad(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [ex[key]["input_ids"] for ex in batch]
        mlen = max(len(x) for x in ids)
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


def last_hidden(model, input_ids, attention_mask):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.last_hidden_state
    if h is None:
        h = out.hidden_states[-1]
    idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    b = __import__("torch").arange(h.size(0), device=h.device)
    return h[b, idx]


def load_backbone(model_dir: Path, dtype, checkpointing: bool):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    model = None
    err = None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = loader.from_pretrained(str(model_dir), **kwargs)
            break
        except Exception as e:
            err = e
    if model is None:
        raise SystemExit(f"failed to load {model_dir}: {err}")
    if checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from biv_wm.jepa import JEPAPred, cosine_align_loss

    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    cfg = _load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    model_dir = _resolve(args.model_dir or cfg["model_dir"])
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    if not model_dir.is_dir():
        raise SystemExit(f"missing cut model {model_dir}; run python train/scripts/cut_stage1.py")
    out_dir = _resolve(tcfg.get("output_dir") or "outputs/jepa_stage1")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)

    log(f"model={model_dir}")
    log(f"mix={mix_dir} sources={sources}")
    model, tokenizer = load_backbone(
        model_dir, dtype, bool(tcfg.get("gradient_checkpointing", True))
    )
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False

    suffixes = list(tcfg.get("target_modules") or [])
    targets = two_d_lora_targets(model, suffixes)
    log(f"lora 2D targets={len(targets)}")
    lora = LoraConfig(
        r=int(tcfg.get("lora_rank") or 16),
        lora_alpha=int(tcfg.get("lora_alpha") or 32),
        lora_dropout=float(tcfg.get("lora_dropout") or 0.05),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False
    hidden = int(getattr(getattr(model.config, "text_config", model.config), "hidden_size", 2048))
    jepa = JEPAPred(dim=hidden, hidden=int(tcfg.get("jepa_hidden") or hidden * 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    jepa.to(device=device, dtype=dtype)
    log(f"device={device} class={type(model).__name__}")

    train_rows = load_rows(
        mix_dir, sources, "train", tcfg.get("max_train_samples")
    )
    eval_rows = load_rows(
        mix_dir, sources, "eval", tcfg.get("max_eval_samples")
    )
    if not train_rows:
        raise SystemExit(f"no (h,a,o) rows under {mix_dir}/{sources}/train.jsonl")
    log(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}")

    max_length = int(tcfg.get("max_length") or 8192)
    train_ds = HaoDataset(train_rows, tokenizer, max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(
        train_ds,
        batch_size=int(tcfg.get("batch_size") or 1),
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=0,
    )

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(tcfg.get("lr") or 2e-4)},
            {"params": jepa.parameters(), "lr": float(tcfg.get("jepa_lr") or 1e-3)},
        ],
        weight_decay=0.01,
    )
    accum = int(tcfg.get("grad_accum") or 8)
    max_norm = float(tcfg.get("max_grad_norm") or 1.0)
    log_every = int(tcfg.get("logging_steps") or 10)
    save_every = int(tcfg.get("save_steps") or 100)
    epochs = int(tcfg.get("num_epochs") or 1)
    max_steps = args.max_steps
    steps_per_epoch = math.ceil(len(loader) / accum)
    log(f"steps_per_epoch≈{steps_per_epoch} accum={accum}")

    model.train()
    jepa.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    n_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                z = last_hidden(model, batch["o_ids"], batch["o_mask"])
            c = last_hidden(model, batch["h_ids"], batch["h_mask"])
            u = last_hidden(model, batch["a_ids"], batch["a_mask"])
            pred = jepa(c, u)
            loss = cosine_align_loss(pred, z) / accum
            loss.backward()
            running += float(loss.item()) * accum
            n_loss += 1
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone_params) + list(jepa.parameters()), max_norm
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    log(f"epoch={epoch} step={step} loss={running / max(n_loss, 1):.4f}")
                    running = 0.0
                    n_loss = 0
                if step % save_every == 0:
                    ckpt = out_dir / f"step-{step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    torch.save(jepa.state_dict(), ckpt / "jepa.pt")
                    log(f"saved {ckpt}")
                if max_steps is not None and step >= max_steps:
                    break
        if max_steps is not None and step >= max_steps:
            break

    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    torch.save(jepa.state_dict(), final / "jepa.pt")
    tokenizer.save_pretrained(final)
    meta = {
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "sources": sources,
        "steps": step,
        "cut_meta": str(model_dir / "cut_meta.json"),
    }
    (final / "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"wrote {final}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
