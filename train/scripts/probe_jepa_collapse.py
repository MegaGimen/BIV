#!/usr/bin/env python3
"""Forward-only collapse probe for history-mediated JEPA. No backward, no LoRA.

Encodes \(z_t=\mathrm{Enc}(h)\) and \(z_{t+1}=\mathrm{Enc}(h,a,o)\) the same way
the next Stage 1 cut does. Close pairs keep clipped action/observation only —
never the history string.

  cd train
  CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepa_collapse.sh
"""

from __future__ import annotations

import argparse
import hashlib
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

import train_jepa as tj  # noqa: E402
from biv_wm.arch import install_hidden_only_forward  # noqa: E402
from biv_wm.jepa import close_pair_records, collapse_stats, format_close_preview, format_collapse_line  # noqa: E402
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


def _clear_parallelism_env() -> None:
    """Drop 4-GPU train leftovers so this 1-GPU probe cannot inherit a mesh."""
    for key in _PARALLEL_ENV:
        os.environ.pop(key, None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=TRAIN / "configs/jepa/stage1.yaml")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    p.add_argument("--close-threshold", type=float, default=0.7)
    p.add_argument("--max-pairs", type=int, default=24)
    p.add_argument("--snippet", type=int, default=120)
    p.add_argument("--print-pairs", type=int, default=8)
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
    from torch.utils.data import DataLoader

    _clear_parallelism_env()
    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = tj._load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    accum = int(tcfg.get("grad_accum") or 8)
    cp_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rank_log(msg: str) -> None:
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
    out_dir = args.out_dir or tj._resolve("outputs/jepa_collapse_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)
    max_length = int(args.max_length or tcfg.get("max_length") or 32768)
    last_token = int(args.last_token if args.last_token is not None else (tcfg.get("last_token") or -3))
    pad_multiple = 0
    pad_id = 0

    model, tokenizer = tj.load_backbone(
        model_dir,
        dtype,
        bool(tcfg.get("gradient_checkpointing", True)),
        attn_implementation=None,
        distributed=False,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    install_hidden_only_forward(model, detach_head=False)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)

    rows = tj.load_rows(mix_dir, sources, "train", max_rows)
    train_ds = tj.HaoDataset(rows, tokenizer, max_length, encoding="mediated")
    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: tj.collate_mediated(b, pad_id, pad_multiple),
    )

    preds: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    o_texts: list[str] = []
    a_texts: list[str] = []
    skip_keys: list[str] = []
    state_lens: list[int] = []
    next_lens: list[int] = []
    n_done = 0
    rank_log(
        f"probe mediated Enc(h) vs Enc(h,a,o) single-GPU device={device} "
        f"max_rows={max_rows} last_token={last_token} "
        f"snippet={args.snippet} print_pairs={args.print_pairs}"
    )
    with torch.no_grad():
        for batch in loader:
            if n_done >= max_rows:
                break
            batch_t = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            h_state_seq = tj.full_hidden(model, batch_t["state_ids"], batch_t["state_mask"], cp_size)
            h_next_seq = tj.full_hidden(model, batch_t["next_ids"], batch_t["next_mask"], cp_size)
            idx_s = tj.last_token_index(batch_t["state_ids"], batch_t["state_mask"], last_token)
            idx_n = tj.last_token_index(batch_t["next_ids"], batch_t["next_mask"], last_token)
            h_state = tj.gather_at(h_state_seq, idx_s)
            h_next = tj.gather_at(h_next_seq, idx_n)
            for i in range(h_state.size(0)):
                preds.append(h_state[i].detach().float().cpu())
                zs.append(h_next[i].detach().float().cpu())
                a_texts.append(batch["a_text"][i] if "a_text" in batch else "")
                o_texts.append(batch["o_text"][i] if "o_text" in batch else "")
                sl = int(batch["state_len"][i]) if "state_len" in batch else 0
                nl = int(batch["next_len"][i]) if "next_len" in batch else 0
                state_lens.append(sl)
                next_lens.append(nl)
                nxt = batch_t["next_ids"][i]
                nmask = batch_t["next_mask"][i].bool()
                token_key = hashlib.sha1(
                    repr(nxt[nmask].detach().cpu().tolist()).encode()
                ).hexdigest()[:16]
                skip_keys.append(token_key)
            n_done += h_state.size(0)
            if n_done % 25 == 0:
                rank_log(f"probe encoded {n_done}/{max_rows}")

    if not preds:
        raise SystemExit("probe collected 0 rows")
    pred = torch.stack(preds)
    target = torch.stack(zs)
    stats = collapse_stats(pred, target, o_texts, skip_texts=skip_keys)
    pairs = close_pair_records(
        pred,
        target,
        o_texts,
        a_texts=a_texts,
        skip_texts=skip_keys,
        state_lens=state_lens,
        next_lens=next_lens,
        threshold=float(args.close_threshold),
        max_pairs=int(args.max_pairs),
        snippet=int(args.snippet),
        compact=True,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload: dict[str, Any] = {
        "stamp": stamp,
        "encoding": "mediated",
        "z_t": "Enc(h)",
        "z_next": "Enc(h,a,o)",
        "n_rows": len(o_texts),
        "max_rows": max_rows,
        "steps_equivalent": {"optimizer_steps_one_group": max_rows / accum, "accum": accum, "dp_size": 1},
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "last_token": last_token,
        "close_threshold": float(args.close_threshold),
        "snippet": int(args.snippet),
        "collapse": json.loads(json.dumps(stats, default=str)),
        "close_pairs": pairs,
        "collapse_line": format_collapse_line(stats),
        "note": (
            "paired = cosine(z_t, z_next) on the same row (did a+o move the vector). "
            "z_self = z_next vs z_next across rows (same stdout, different h, should split). "
            "pred_self = z_t vs z_t. Close-pair JSON stores clipped a/o only, never h."
        ),
    }
    dest = out_dir / f"collapse_probe-{stamp}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "collapse_probe_latest.json").write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
    rank_log(f"[{format_collapse_line(stats)}]")
    rank_log(
        f"close pairs threshold={args.close_threshold} "
        f"z_self={pairs['n_z_self']} pred_self={pairs['n_pred_self']} mismatch={pairs['n_mismatch']}"
    )
    preview = format_close_preview(pairs, n_print=int(args.print_pairs), snippet=min(80, int(args.snippet)))
    if preview:
        rank_log(preview)
    rank_log(f"wrote {dest}")


if __name__ == "__main__":
    main()
