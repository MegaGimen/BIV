#!/usr/bin/env python3
"""Forward-only collapse probe: same Enc(left)/Enc(right) as train_jepallm, no backward.

Collects last_token vectors on up to --max-rows (default = 25 optimizer steps
on one 2x2 replica group: 25 * grad_accum=8 = 200). Then runs collapse_stats
and writes close-pair observation/command texts.

  cd train
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/probe_jepallm_collapse.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
SCRIPTS = Path(__file__).resolve().parent
MERGE = ROOT / "merge"
for p in (str(SRC), str(MERGE), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_jepallm as tj  # noqa: E402
from biv_wm.arch import install_hidden_only_forward  # noqa: E402
from biv_wm.jepa import close_pair_records, collapse_stats, format_collapse_line  # noqa: E402
from download import resolve_model  # noqa: E402

# 32k/2x2 yaml: save_steps=25, batch=1, accum=8, dp_replicate=2.
# One optimizer step on one group = 8 rows. 25 steps → 200 rows on the
# logging rank (same size the training collapse window would cover if
# log_steps were 25). Both groups together = 400 unique rows; default
# cap is one group so the dump matches what TensorBoard collapse/* sees.
DEFAULT_MAX_ROWS = 25 * 8
HARD_CAP = 400
_PARALLEL_ENV = (
    "ACCELERATE_USE_PARALLELISM_CONFIG",
    "PARALLELISM_CONFIG_DP_REPLICATE_SIZE",
    "PARALLELISM_CONFIG_DP_SHARD_SIZE",
    "PARALLELISM_CONFIG_TP_SIZE",
    "PARALLELISM_CONFIG_CP_SIZE",
    "PARALLELISM_CONFIG_CP_BACKEND",
    "BIV_CP_SIZE",
    "BIV_PARALLEL",
)


def _visible_gpu_count() -> int:
    xs = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if xs:
        return len(xs)
    try:
        import torch

        if torch.cuda.is_available():
            return max(int(torch.cuda.device_count()), 1)
    except Exception:
        pass
    return 1


def _clear_parallelism_env() -> None:
    """Drop 4-GPU train leftovers so Accelerator does not build a 4-rank mesh."""
    for key in _PARALLEL_ENV:
        os.environ.pop(key, None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=TRAIN / "configs/jepa/jepallm_32k.yaml")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    p.add_argument("--close-threshold", type=float, default=0.7)
    p.add_argument("--max-pairs", type=int, default=80)
    p.add_argument("--snippet", type=int, default=800)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--source", choices=["modelscope", "huggingface"], default=os.environ.get("MERGE_SOURCE", "modelscope"))
    p.add_argument("--last-token", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = int(args.max_rows)
    if max_rows < 1:
        raise SystemExit("max-rows must be >= 1")
    if max_rows > HARD_CAP:
        raise SystemExit(f"max-rows cap is {HARD_CAP} (25 steps × 8 accum × 2 groups); got {max_rows}")

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = tj._load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    accum = int(tcfg.get("grad_accum") or 8)
    n_gpu = _visible_gpu_count()
    cp_size = tj.resolve_cp_size(args.cp_size)
    if n_gpu <= 1 or cp_size <= 1:
        cp_size = 1
        _clear_parallelism_env()
        accelerator = Accelerator(gradient_accumulation_steps=accum)
    else:
        os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
        os.environ.setdefault("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_TP_SIZE", "1")
        os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cp_size)
        os.environ.setdefault("PARALLELISM_CONFIG_CP_BACKEND", "torch")
        accelerator = Accelerator(
            gradient_accumulation_steps=accum, parallelism_config=tj.build_parallelism_config()
        )
    is_main = accelerator.is_main_process

    def rank_log(msg: str) -> None:
        if is_main:
            tj.log(msg)

    cache_dir = tj._resolve(args.cache_dir or "merge/output/cache", ROOT)
    model_dir = resolve_model(
        str(args.model_dir or cfg["model_dir"]),
        source=args.source,
        cache_dir=cache_dir,
        role="world",
    )
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = tj.resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    out_dir = args.out_dir or tj._resolve("outputs/jepallm32k_collapse_probe")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)
    max_length = int(args.max_length or tcfg.get("max_length") or 32768)
    last_token = int(args.last_token if args.last_token is not None else (tcfg.get("last_token") or -3))
    pad_multiple = cp_size * 2 if cp_size > 1 else 0
    attn_impl = "sdpa" if cp_size > 1 else None
    pad_id = 0

    model, tokenizer = tj.load_backbone(
        model_dir,
        dtype,
        bool(tcfg.get("gradient_checkpointing", True)),
        attn_implementation=attn_impl,
        distributed=accelerator.num_processes > 1,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    install_hidden_only_forward(model, detach_head=False)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)

    dp_rank, dp_size = tj.dp_replicate_info(accelerator, cp_size)
    load_budget = max_rows * max(dp_size, 1)
    rows = tj.load_rows(mix_dir, sources, "train", load_budget)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_ds = tj.HaoDataset(rows, tokenizer, max_length)
    if dp_size > 1:
        sampler = tj.ReplicaSampler(len(train_ds), dp_rank, dp_size, gen)
        loader = DataLoader(
            train_ds,
            batch_size=1,
            sampler=sampler,
            collate_fn=lambda b: tj.collate(b, pad_id, pad_multiple),
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda b: tj.collate(b, pad_id, pad_multiple),
        )
    model, loader = accelerator.prepare(model, loader)

    preds: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    o_texts: list[str] = []
    left_texts: list[str] = []
    n_done = 0
    rank_log(
        f"probe max_rows={max_rows} (25 opt steps × accum {accum} on one group; "
        f"hard cap {HARD_CAP}) last_token={last_token} cp={cp_size} dp={dp_size}"
    )
    with torch.no_grad():
        for batch in loader:
            if n_done >= max_rows:
                break
            batch_t = {
                k: v.to(accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            h_left_seq = tj.full_hidden(model, batch_t["left_ids"], batch_t["left_mask"], cp_size)
            h_right_seq = tj.full_hidden(model, batch_t["right_ids"], batch_t["right_mask"], cp_size)
            idx_l = tj.last_token_index(batch_t["left_ids"], batch_t["left_mask"], last_token)
            idx_r = tj.last_token_index(batch_t["right_ids"], batch_t["right_mask"], last_token)
            h_left = tj.gather_at(h_left_seq, idx_l)
            h_right = tj.gather_at(h_right_seq, idx_r)
            if is_main:
                for i in range(h_left.size(0)):
                    preds.append(h_left[i].detach().float().cpu())
                    zs.append(h_right[i].detach().float().cpu())
                    o_texts.append(batch["o_text"][i] if "o_text" in batch else "")
                    left_texts.append(batch["left_text"][i] if "left_text" in batch else "")
            n_done += h_left.size(0)
            if is_main and n_done % 10 == 0:
                rank_log(f"probe encoded {n_done}/{max_rows}")

    accelerator.wait_for_everyone()
    if not is_main:
        return
    if not preds:
        raise SystemExit("probe collected 0 rows")
    pred = torch.stack(preds)
    target = torch.stack(zs)
    stats = collapse_stats(pred, target, o_texts)
    pairs = close_pair_records(
        pred,
        target,
        o_texts,
        left_texts,
        threshold=float(args.close_threshold),
        max_pairs=int(args.max_pairs),
        snippet=int(args.snippet),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload: dict[str, Any] = {
        "stamp": stamp,
        "n_rows": len(o_texts),
        "max_rows": max_rows,
        "steps_equivalent": {"optimizer_steps_one_group": max_rows / accum, "accum": accum, "dp_size": dp_size},
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "last_token": last_token,
        "close_threshold": float(args.close_threshold),
        "collapse": json.loads(json.dumps(stats, default=str)),
        "close_pairs": pairs,
        "collapse_line": format_collapse_line(stats),
    }
    dest = out_dir / f"collapse_probe-{stamp}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "collapse_probe_latest.json").write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
    rank_log(f"[{format_collapse_line(stats)}]")
    rank_log(
        f"close pairs threshold={args.close_threshold} "
        f"z_self={pairs['n_z_self']} pred_self={pairs['n_pred_self']} mismatch={pairs['n_mismatch']}"
    )
    rank_log(f"wrote {dest}")


if __name__ == "__main__":
    main()
