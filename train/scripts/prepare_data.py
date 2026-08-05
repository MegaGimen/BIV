#!/usr/bin/env python3
"""Prepare world-model SFT JSONL from local trajectories and/or SWE-Hero.

Streams one trajectory at a time (no full-corpus to_list) to avoid RAM blowups.

Examples:

  python scripts/prepare_data.py \\
      --local data/examples/sample_trajectories.jsonl \\
      --out-dir data/processed

  python scripts/prepare_data.py \\
      --swe-hero --swe-hero-max-rows 2000 \\
      --out-dir data/processed --eval-ratio 0.05

  # HuggingFace fallback (same schema as ModelScope mirror):
  python scripts/prepare_data.py --swe-hero --swe-hero-source huggingface
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
    open_swe_hero_dataset,
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
        expand_prefixes=args.expand_prefixes,
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
            expand_prefixes=args.expand_prefixes,
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
    p.add_argument(
        "--swe-hero",
        action="store_true",
        help="Load SWE-Hero OpenHands trajectories (default: ModelScope nv-community/...)",
    )
    p.add_argument(
        "--swe-hero-source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Hub for SWE-Hero (default: modelscope)",
    )
    p.add_argument(
        "--swe-hero-repo",
        default=None,
        help="Override repo id (default MS: nv-community/SWE-Hero-openhands-trajectories)",
    )
    p.add_argument("--swe-hero-split", default="train")
    p.add_argument("--swe-hero-max-rows", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--min-turns", type=int, default=1)
    p.add_argument(
        "--max-prefix",
        type=int,
        default=None,
        help="Optional cap on tool turns kept per trajectory (truncate long chains)",
    )
    p.add_argument(
        "--expand-prefixes",
        action="store_true",
        help="Legacy: emit turns[:1],[:2],… (over-samples early observations)",
    )
    p.add_argument(
        "--every-k",
        type=int,
        default=1,
        help="Only with --expand-prefixes: keep every k-th prefix length",
    )
    p.add_argument("--seed", type=int, default=42)    p.add_argument("--also-shuffled-control", action="store_true", default=True)
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

    if args.swe_hero:
        if args.swe_hero_source == "huggingface" and not os.environ.get("HF_ENDPOINT"):
            print(
                "Tip: for CN HF download set: export HF_ENDPOINT=https://hf-mirror.com",
                flush=True,
            )
        print(
            f"Opening SWE-Hero source={args.swe_hero_source!r} "
            f"repo={args.swe_hero_repo or '(default)'} split={args.swe_hero_split!r}",
            flush=True,
        )
        ds = open_swe_hero_dataset(
            split=args.swe_hero_split,
            max_rows=args.swe_hero_max_rows,
            source=args.swe_hero_source,
            repo_id=args.swe_hero_repo,
        )
        sources.append(("swe_hero", ds))

    if not sources:
        raise SystemExit("No records loaded. Pass --local and/or --swe-hero.")

    for kind, blob in sources:
        if kind == "local":
            records = list(blob)
            n = len(records)
            indices = list(range(n))
            rng.shuffle(indices)
            n_eval = max(1, int(n * args.eval_ratio)) if n > 1 else 0
            eval_set = set(indices[:n_eval])

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

        # HF Dataset: two lightweight passes (index + one row at a time).
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
        "corpus": "swe_hero_openhands",
        "expand_prefixes": bool(args.expand_prefixes),
        "sample_policy": "one_traj_one_row_all_obs_loss"
        if not args.expand_prefixes
        else "causal_prefixes",
        "sources": {
            "local": [str(x) for x in args.local],
            "swe_hero": bool(args.swe_hero),
            "swe_hero_source": args.swe_hero_source if args.swe_hero else None,
            "swe_hero_repo": args.swe_hero_repo,
            "swe_hero_max_rows": args.swe_hero_max_rows,
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
