#!/usr/bin/env python3
"""Prepare world-model SFT JSONL from local trajectories and/or ISETrace.

Streams one trajectory at a time (no full-corpus to_list) to avoid RAM blowups.

Examples:

  python scripts/prepare_data.py \\
      --local data/examples/sample_trajectories.jsonl \\
      --out-dir data/processed

  python scripts/prepare_data.py \\
      --hf-isetrace --hf-max-rows 2000 \\
      --out-dir data/processed --eval-ratio 0.05
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
    append_jsonl,
    extract_turns_from_record,
    iter_sft_rows_from_turns,
    load_local_trajectories,
    open_isetrace_dataset,
    reservoir_add,
)


def _reset_outputs(out: Path, also_shuffled: bool) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out / "train.jsonl",
        "eval": out / "eval.jsonl",
    }
    if also_shuffled:
        paths["train_shuffled"] = out / "train_shuffled.jsonl"
        paths["eval_shuffled"] = out / "eval_shuffled.jsonl"
    for p in paths.values():
        p.write_text("", encoding="utf-8")
    return paths


def _process_record(
    record: dict,
    *,
    is_eval: bool,
    paths: dict[str, Path],
    counts: dict[str, int],
    args: argparse.Namespace,
    obs_pool: list[str],
    rng: random.Random,
) -> int:
    turns = extract_turns_from_record(record)
    if not turns:
        return 0
    n = 0
    key = "eval" if is_eval else "train"
    for row in iter_sft_rows_from_turns(
        turns,
        min_turns=args.min_turns,
        max_prefix=args.max_prefix,
        every_k=args.every_k,
        shuffle_obs=False,
        rng=rng,
    ):
        append_jsonl(paths[key], row)
        counts[key] += 1
        n += 1
    if args.also_shuffled_control:
        sh_key = "eval_shuffled" if is_eval else "train_shuffled"
        for row in iter_sft_rows_from_turns(
            turns,
            min_turns=args.min_turns,
            max_prefix=args.max_prefix,
            every_k=args.every_k,
            shuffle_obs=True,
            obs_pool=obs_pool or ["<<empty_pool>>"],
            rng=rng,
        ):
            append_jsonl(paths[sh_key], row)
            counts[sh_key] += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, nargs="*", default=[], help="Local json/jsonl trajectory files")
    p.add_argument("--hf-isetrace", action="store_true", help="Download valiere/ISETrace trajectories")
    p.add_argument("--hf-config", default="trajectories")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-max-rows", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--min-turns", type=int, default=1)
    p.add_argument("--max-prefix", type=int, default=None)
    p.add_argument("--every-k", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also-shuffled-control", action="store_true", default=True)
    p.add_argument("--no-shuffled-control", action="store_false", dest="also_shuffled_control")
    p.add_argument(
        "--obs-pool-size",
        type=int,
        default=8192,
        help="Max observation strings kept for shuffled-control reservoir (RAM cap)",
    )
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    rng = random.Random(args.seed)
    paths = _reset_outputs(args.out_dir, args.also_shuffled_control)
    counts = {k: 0 for k in paths}
    n_records = 0
    n_train_records = 0
    n_eval_records = 0
    last_obs_pool_size = 0

    sources: list[tuple[str, object]] = []
    for path in args.local:
        sources.append(("local", load_local_trajectories(path)))

    if args.hf_isetrace:
        if not os.environ.get("HF_ENDPOINT"):
            print(
                "Tip: ISETrace is not on ModelScope. "
                "For CN download set: export HF_ENDPOINT=https://hf-mirror.com",
                flush=True,
            )
        print(
            f"Opening ISETrace (Arrow, streamed rows) "
            f"name={args.hf_config!r} split={args.hf_split!r} "
            f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}",
            flush=True,
        )
        ds = open_isetrace_dataset(
            config=args.hf_config,
            split=args.hf_split,
            max_rows=args.hf_max_rows,
        )
        sources.append(("hf", ds))

    if not sources:
        raise SystemExit("No records loaded. Pass --local and/or --hf-isetrace.")

    for kind, blob in sources:
        if kind == "local":
            records = list(blob)
            n = len(records)
            indices = list(range(n))
            rng.shuffle(indices)
            n_eval = max(1, int(n * args.eval_ratio)) if n > 1 else 0
            eval_set = set(indices[:n_eval])

            # Reservoir from all local turns first (tiny).
            obs_pool: list[str] = []
            seen = 0
            for rec in records:
                for t in extract_turns_from_record(rec):
                    seen = reservoir_add(
                        obs_pool, t["observation"], k=args.obs_pool_size, seen=seen, rng=rng
                    )

            for i, rec in enumerate(records):
                is_eval = i in eval_set
                wrote = _process_record(
                    rec,
                    is_eval=is_eval,
                    paths=paths,
                    counts=counts,
                    args=args,
                    obs_pool=obs_pool,
                    rng=rng,
                )
                if wrote:
                    n_records += 1
                    if is_eval:
                        n_eval_records += 1
                    else:
                        n_train_records += 1
            last_obs_pool_size = len(obs_pool)
            continue

        # HuggingFace Dataset: two lightweight passes (index + one row at a time).
        ds = blob
        n = len(ds)
        indices = list(range(n))
        rng.shuffle(indices)
        n_eval = max(1, int(n * args.eval_ratio)) if n > 1 else 0
        eval_set = set(indices[:n_eval])

        print(f"Pass 1/2: reservoir sampling observations over {n} trajectories...", flush=True)
        obs_pool = []
        seen = 0
        for i in range(n):
            turns = extract_turns_from_record(ds[i])
            for t in turns:
                seen = reservoir_add(
                    obs_pool, t["observation"], k=args.obs_pool_size, seen=seen, rng=rng
                )
            if args.log_every and (i + 1) % args.log_every == 0:
                print(f"  reservoir {i + 1}/{n} pool={len(obs_pool)}", flush=True)

        print(f"Pass 2/2: writing SFT jsonl ({n} trajectories)...", flush=True)
        for i in range(n):
            rec = ds[i]
            is_eval = i in eval_set
            wrote = _process_record(
                rec,
                is_eval=is_eval,
                paths=paths,
                counts=counts,
                args=args,
                obs_pool=obs_pool,
                rng=rng,
            )
            if wrote:
                n_records += 1
                if is_eval:
                    n_eval_records += 1
                else:
                    n_train_records += 1
            if args.log_every and (i + 1) % args.log_every == 0:
                print(
                    f"  wrote traj {i + 1}/{n} "
                    f"train_rows={counts['train']} eval_rows={counts['eval']}",
                    flush=True,
                )
            del rec
        last_obs_pool_size = len(obs_pool)

    meta = {
        "n_records": n_records,
        "n_train_records": n_train_records,
        "n_eval_records": n_eval_records,
        "n_train_rows": counts.get("train", 0),
        "n_eval_rows": counts.get("eval", 0),
        "n_train_shuffled_rows": counts.get("train_shuffled", 0),
        "n_eval_shuffled_rows": counts.get("eval_shuffled", 0),
        "obs_pool_size": last_obs_pool_size,
        "seed": args.seed,
        "streamed": True,
        "sources": {
            "local": [str(x) for x in args.local],
            "hf_isetrace": bool(args.hf_isetrace),
            "hf_max_rows": args.hf_max_rows,
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
