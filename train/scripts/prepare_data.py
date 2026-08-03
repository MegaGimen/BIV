#!/usr/bin/env python3
"""Prepare world-model SFT JSONL from local trajectories and/or ISETrace.

Examples (on a networked training machine):

  # Local example trajectories only
  python scripts/prepare_data.py \\
      --local data/examples/sample_trajectories.jsonl \\
      --out-dir data/processed

  # HuggingFace ISETrace + local mix
  python scripts/prepare_data.py \\
      --hf-isetrace --hf-max-rows 5000 \\
      --local data/examples/sample_trajectories.jsonl \\
      --out-dir data/processed --eval-ratio 0.05

Does not require a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biv_wm.data import (  # noqa: E402
    load_local_trajectories,
    records_to_sft_rows,
    try_load_isetrace,
    write_jsonl,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, nargs="*", default=[], help="Local json/jsonl trajectory files")
    p.add_argument("--hf-isetrace", action="store_true", help="Download valiere/ISETrace trajectories")
    p.add_argument(
        "--hf-config",
        default="trajectories",
        help="HF dataset config name (trajectories|intents)",
    )
    p.add_argument("--hf-split", default="train", help="HF split name")
    p.add_argument("--hf-max-rows", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--min-turns", type=int, default=1)
    p.add_argument("--max-prefix", type=int, default=None)
    p.add_argument("--every-k", type=int, default=1, help="Keep every k-th prefix length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also-shuffled-control", action="store_true", default=True)
    p.add_argument("--no-shuffled-control", action="store_false", dest="also_shuffled_control")
    args = p.parse_args()

    rng = random.Random(args.seed)
    records: list[dict] = []
    for path in args.local:
        records.extend(load_local_trajectories(path))
    if args.hf_isetrace:
        # ISETrace is HF-only today (not on ModelScope). Prefer CN mirror.
        if not os.environ.get("HF_ENDPOINT"):
            print(
                "Tip: ISETrace is not on ModelScope. "
                "For CN download set: export HF_ENDPOINT=https://hf-mirror.com",
                flush=True,
            )
        print(
            f"Loading ISETrace from HuggingFace "
            f"(name={args.hf_config!r}, split={args.hf_split!r}, "
            f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')})...",
            flush=True,
        )
        records.extend(
            try_load_isetrace(
                config=args.hf_config,
                split=args.hf_split,
                max_rows=args.hf_max_rows,
            )
        )

    if not records:
        raise SystemExit(
            "No records loaded. Pass --local ... and/or --hf-isetrace "
            "(network required for HF)."
        )

    rng.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_ratio)) if len(records) > 1 else 0
    eval_recs = records[:n_eval]
    train_recs = records[n_eval:] if n_eval else records

    def build(recs: list[dict], shuffle: bool) -> list[dict]:
        return records_to_sft_rows(
            recs,
            min_turns=args.min_turns,
            max_prefix=args.max_prefix,
            every_k=args.every_k,
            shuffle_obs=shuffle,
            rng=rng,
        )

    train_rows = build(train_recs, False)
    eval_rows = build(eval_recs, False) if eval_recs else []
    out = args.out_dir
    n_tr = write_jsonl(out / "train.jsonl", train_rows)
    n_ev = write_jsonl(out / "eval.jsonl", eval_rows)
    print(f"Wrote {n_tr} train / {n_ev} eval rows -> {out}")

    if args.also_shuffled_control:
        n_trs = write_jsonl(out / "train_shuffled.jsonl", build(train_recs, True))
        n_evs = write_jsonl(
            out / "eval_shuffled.jsonl", build(eval_recs, True) if eval_recs else []
        )
        print(f"Wrote control shuffled {n_trs} train / {n_evs} eval rows")

    meta = {
        "n_records": len(records),
        "n_train_records": len(train_recs),
        "n_eval_records": len(eval_recs),
        "n_train_rows": n_tr,
        "n_eval_rows": n_ev,
        "seed": args.seed,
        "sources": {
            "local": [str(x) for x in args.local],
            "hf_isetrace": bool(args.hf_isetrace),
            "hf_max_rows": args.hf_max_rows,
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
