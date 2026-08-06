#!/usr/bin/env python3
"""Prepare multi-source world-model + anti-forget JSONL for Qwen3-Coder-Next.

Sources (unified OpenAI-style ``messages`` + ``source`` field):
  - wm_code:      SWE-Hero OpenHands env I/O  → P(o|h,a)  [primary]
  - wm_os:        ISETrace OS agent tool I/O → P(o|h,a)  [primary]
  - anti_forget:  SWE-Zero full agent paths  → policy replay (deduped vs Hero)

Caching:
  - Hub: reuse HF / ModelScope snapshots via ``biv_wm.hub``
  - Processed: ``out_dir/fingerprint.json`` — skip rebuild when args match

Prints per-dataset raw hub rows and written JSONL line counts.

Examples:
  python scripts/prepare_data.py --all --out-dir data/processed/mix_v1
  python scripts/prepare_data.py --wm-code --wm-os --anti-forget \\
      --max-rows-per-source 500 --out-dir data/processed/pilot_mix
  python scripts/prepare_data.py --wm-code --hub-source huggingface
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biv_wm.adapters.normalize import (  # noqa: E402
    policy_row_from_openhands_record,
    record_ids,
    wm_row_from_isetrace_record,
    wm_row_from_openhands_record,
)
from biv_wm.data import (  # noqa: E402
    append_jsonl,
    extract_turns_from_openai_tool_messages,
    extract_turns_from_record,
    load_local_trajectories,
    reservoir_add,
)
from biv_wm.formatting import (  # noqa: E402
    SOURCE_ANTI_FORGET,
    SOURCE_WM_CODE,
    SOURCE_WM_OS,
)
from biv_wm.hub import open_dataset_with_cache  # noqa: E402


def _count_lines(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def _fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _reset_source_dir(out: Path, also_shuffled: bool) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"train": out / "train.jsonl", "eval": out / "eval.jsonl"}
    if also_shuffled:
        paths["train_shuffled"] = out / "train_shuffled.jsonl"
        paths["eval_shuffled"] = out / "eval_shuffled.jsonl"
    for p in paths.values():
        p.write_text("", encoding="utf-8")
    return paths


def _split_indices(n: int, eval_ratio: float, rng: random.Random) -> set[int]:
    if n <= 1 or eval_ratio <= 0:
        return set()
    indices = list(range(n))
    rng.shuffle(indices)
    n_eval = max(1, int(n * eval_ratio))
    return set(indices[:n_eval])


def _build_obs_pool_from_ds(
    ds,
    *,
    turn_extractor: Callable[[dict], list[dict]],
    pool_size: int,
    rng: random.Random,
    log_every: int,
) -> list[str]:
    pool: list[str] = []
    seen = 0
    n = len(ds)
    print(f"  Pass 1/2: obs reservoir over {n} records...", flush=True)
    for i in range(n):
        for t in turn_extractor(ds[i]):
            obs = t.get("observation")
            if isinstance(obs, str):
                seen = reservoir_add(pool, obs, k=pool_size, seen=seen, rng=rng)
        if log_every and (i + 1) % log_every == 0:
            print(f"    reservoir {i + 1}/{n} pool={len(pool)}", flush=True)
    return pool


def _prepare_wm_openhands(
    *,
    kind: str,
    source_tag: str,
    out_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    local = None
    repo = None
    if kind == "swe_hero":
        local = Path(args.swe_hero_local_dir) if args.swe_hero_local_dir else None
        repo = args.swe_hero_repo
    ds = open_dataset_with_cache(
        kind=kind,
        source=args.hub_source,
        repo_id=repo,
        split=args.split,
        max_rows=args.max_rows_per_source,
        local_dir=local,
    )
    raw_rows = len(ds)
    paths = _reset_source_dir(out_dir, args.also_shuffled_control)
    counts = {k: 0 for k in paths}
    eval_set = _split_indices(raw_rows, args.eval_ratio, rng)

    obs_pool = (
        _build_obs_pool_from_ds(
            ds,
            turn_extractor=extract_turns_from_record,
            pool_size=args.obs_pool_size,
            rng=rng,
            log_every=args.log_every,
        )
        if args.also_shuffled_control
        else []
    )

    instance_ids: set[str] = set()
    print(f"  Pass 2/2: writing WM jsonl ({raw_rows} records) → {out_dir}", flush=True)
    for i in range(raw_rows):
        rec = ds[i]
        iid, _ = record_ids(rec)
        if iid:
            instance_ids.add(iid)
        is_eval = i in eval_set
        rows = wm_row_from_openhands_record(
            rec,
            source=source_tag,
            min_turns=args.min_turns,
            max_prefix=args.max_prefix,
            expand_prefixes=args.expand_prefixes,
            every_k=args.every_k,
            shuffle_obs=False,
        )
        key = "eval" if is_eval else "train"
        for row in rows:
            append_jsonl(paths[key], row)
            counts[key] += 1
        if args.also_shuffled_control and rows:
            sh_rows = wm_row_from_openhands_record(
                rec,
                source=source_tag,
                min_turns=args.min_turns,
                max_prefix=args.max_prefix,
                expand_prefixes=args.expand_prefixes,
                every_k=args.every_k,
                shuffle_obs=True,
                shuffled_obs=rng.choice(obs_pool) if obs_pool else "<<empty_pool>>",
            )
            sh_key = "eval_shuffled" if is_eval else "train_shuffled"
            for row in sh_rows:
                append_jsonl(paths[sh_key], row)
                counts[sh_key] += 1
        if args.log_every and (i + 1) % args.log_every == 0:
            print(
                f"    {i + 1}/{raw_rows} train={counts['train']} eval={counts['eval']}",
                flush=True,
            )
        del rec

    line_counts = {name: _count_lines(path) for name, path in paths.items()}
    stats = {
        "source": source_tag,
        "kind": kind,
        "raw_hub_rows": raw_rows,
        "unique_instance_ids": len(instance_ids),
        "written": counts,
        "jsonl_line_counts": line_counts,
    }
    (out_dir / "instance_ids.json").write_text(
        json.dumps(sorted(instance_ids), indent=2), encoding="utf-8"
    )
    (out_dir / "counts.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({source_tag: stats}, indent=2), flush=True)
    return stats


def _prepare_wm_isetrace(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    ds = open_dataset_with_cache(
        kind="isetrace",
        source=args.hub_source,
        repo_id=args.isetrace_repo,
        split=args.split,
        max_rows=args.max_rows_per_source,
        local_dir=Path(args.isetrace_local_dir) if args.isetrace_local_dir else None,
        config_name="trajectories",
    )
    raw_rows = len(ds)
    paths = _reset_source_dir(out_dir, args.also_shuffled_control)
    counts = {k: 0 for k in paths}
    eval_set = _split_indices(raw_rows, args.eval_ratio, rng)

    def _turns(rec: dict) -> list[dict]:
        msgs = rec.get("messages") or []
        return extract_turns_from_openai_tool_messages(msgs) if msgs else []

    obs_pool = (
        _build_obs_pool_from_ds(
            ds,
            turn_extractor=_turns,
            pool_size=args.obs_pool_size,
            rng=rng,
            log_every=args.log_every,
        )
        if args.also_shuffled_control
        else []
    )

    print(f"  Pass 2/2: writing WM-OS jsonl ({raw_rows}) → {out_dir}", flush=True)
    for i in range(raw_rows):
        rec = ds[i]
        is_eval = i in eval_set
        rows = wm_row_from_isetrace_record(
            rec,
            min_turns=args.min_turns,
            max_prefix=args.max_prefix,
            expand_prefixes=args.expand_prefixes,
            every_k=args.every_k,
            shuffle_obs=False,
        )
        key = "eval" if is_eval else "train"
        for row in rows:
            append_jsonl(paths[key], row)
            counts[key] += 1
        if args.also_shuffled_control and rows:
            sh_rows = wm_row_from_isetrace_record(
                rec,
                min_turns=args.min_turns,
                max_prefix=args.max_prefix,
                expand_prefixes=args.expand_prefixes,
                every_k=args.every_k,
                shuffle_obs=True,
                shuffled_obs=rng.choice(obs_pool) if obs_pool else "<<empty_pool>>",
            )
            sh_key = "eval_shuffled" if is_eval else "train_shuffled"
            for row in sh_rows:
                append_jsonl(paths[sh_key], row)
                counts[sh_key] += 1
        if args.log_every and (i + 1) % args.log_every == 0:
            print(
                f"    {i + 1}/{raw_rows} train={counts['train']} eval={counts['eval']}",
                flush=True,
            )
        del rec

    line_counts = {name: _count_lines(path) for name, path in paths.items()}
    stats = {
        "source": SOURCE_WM_OS,
        "kind": "isetrace",
        "raw_hub_rows": raw_rows,
        "written": counts,
        "jsonl_line_counts": line_counts,
    }
    (out_dir / "counts.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"wm_os": stats}, indent=2), flush=True)
    return stats


def _prepare_anti_forget(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
    banned_instances: set[str],
) -> dict[str, Any]:
    ds = open_dataset_with_cache(
        kind="swe_zero",
        source=args.hub_source,
        repo_id=args.swe_zero_repo,
        split=args.split,
        max_rows=args.max_rows_per_source,
        local_dir=Path(args.swe_zero_local_dir) if args.swe_zero_local_dir else None,
    )
    raw_rows = len(ds)
    paths = _reset_source_dir(out_dir, also_shuffled=False)
    counts = {k: 0 for k in paths}
    eval_set = _split_indices(raw_rows, args.eval_ratio, rng)
    skipped_dup = 0
    skipped_bad = 0
    kept = 0
    target = args.anti_forget_max_rows

    print(
        f"  Writing anti_forget from SWE-Zero ({raw_rows} raw; "
        f"banned_instances={len(banned_instances)}; target_rows={target})",
        flush=True,
    )
    order = list(range(raw_rows))
    rng.shuffle(order)
    for n_seen, i in enumerate(order, 1):
        if target is not None and kept >= target:
            break
        rec = ds[i]
        iid, _ = record_ids(rec)
        if iid and iid in banned_instances:
            skipped_dup += 1
            continue
        row = policy_row_from_openhands_record(rec, max_tool_chars=args.max_tool_chars)
        if row is None:
            skipped_bad += 1
            continue
        is_eval = i in eval_set
        key = "eval" if is_eval else "train"
        append_jsonl(paths[key], row)
        counts[key] += 1
        kept += 1
        if args.log_every and n_seen % args.log_every == 0:
            print(
                f"    scanned={n_seen} kept={kept} "
                f"dup_skip={skipped_dup} bad_skip={skipped_bad}",
                flush=True,
            )
        del rec

    line_counts = {name: _count_lines(path) for name, path in paths.items()}
    stats = {
        "source": SOURCE_ANTI_FORGET,
        "kind": "swe_zero",
        "raw_hub_rows": raw_rows,
        "kept_rows": kept,
        "skipped_dup_instance": skipped_dup,
        "skipped_bad": skipped_bad,
        "written": counts,
        "jsonl_line_counts": line_counts,
        "banned_instance_count": len(banned_instances),
    }
    (out_dir / "counts.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"anti_forget": stats}, indent=2), flush=True)
    return stats


def _write_mix_manifest(
    out_root: Path, stats: dict[str, Any], args: argparse.Namespace
) -> None:
    datasets = []
    global_counts: dict[str, int] = {}
    weight_map = {
        "wm_code": args.weight_wm_code,
        "wm_os": args.weight_wm_os,
        "anti_forget": args.weight_anti_forget,
    }
    for name in stats:
        sub = out_root / name
        for split in ("train", "eval", "train_shuffled", "eval_shuffled"):
            p = sub / f"{split}.jsonl"
            if p.is_file():
                global_counts[f"{name}/{split}.jsonl"] = _count_lines(p)
        train_p = sub / "train.jsonl"
        n = global_counts.get(f"{name}/train.jsonl", 0)
        if n > 0:
            datasets.append(
                {
                    "path": str(train_p.relative_to(ROOT)),
                    "ds_type": "chat_template",
                    "field_messages": "messages",
                    "weight": weight_map.get(name, 1.0),
                }
            )

    manifest = {
        "out_dir": str(out_root),
        "datasets_for_axolotl": datasets,
        "jsonl_line_counts": global_counts,
        "per_source_stats": stats,
        "mix_weights": weight_map,
        "sample_policy": (
            "one_traj_one_row" if not args.expand_prefixes else "causal_prefixes"
        ),
    }
    (out_root / "counts.json").write_text(
        json.dumps({"jsonl_line_counts": global_counts, "raw_stats": stats}, indent=2),
        encoding="utf-8",
    )
    (out_root / "mix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Drop-in Axolotl dataset block matching this mix (paths exist only).
    gen = ROOT / "configs" / "axolotl" / "coder_next_qlora.datasets.yaml"
    gen.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated by prepare_data.py — merge into coder_next_qlora.yaml\n", "datasets:\n"]
    for d in datasets:
        lines.append(f"  - path: {d['path']}\n")
        lines.append("    type: chat_template\n")
        lines.append(f"    field_messages: {d['field_messages']}\n")
        lines.append(f"    weight: {d['weight']}\n")
    gen.write_text("".join(lines), encoding="utf-8")
    print("=== TOTAL JSONL LINE COUNTS ===", flush=True)
    for k, v in sorted(global_counts.items()):
        print(f"  {k}: {v:,}", flush=True)
    print(f"Wrote {out_root / 'counts.json'}", flush=True)
    print(f"Wrote {out_root / 'mix_manifest.json'}", flush=True)
    print(f"Wrote {gen}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="wm_code + wm_os + anti_forget")
    p.add_argument("--wm-code", action="store_true", help="SWE-Hero → WM code")
    p.add_argument("--wm-os", action="store_true", help="ISETrace → WM OS")
    p.add_argument("--anti-forget", action="store_true", help="SWE-Zero → policy replay")
    p.add_argument("--local", type=Path, nargs="*", default=[])
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "mix_v1")
    p.add_argument(
        "--hub-source",
        choices=["modelscope", "huggingface", "auto"],
        default="auto",
        help="auto: reuse HF snapshot if present else ModelScope",
    )
    p.add_argument("--swe-hero-repo", default=None)
    p.add_argument("--swe-hero-local-dir", default=None)
    p.add_argument("--swe-zero-repo", default=None)
    p.add_argument("--swe-zero-local-dir", default=None)
    p.add_argument("--isetrace-repo", default=None)
    p.add_argument("--isetrace-local-dir", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--max-rows-per-source", type=int, default=None)
    p.add_argument("--anti-forget-max-rows", type=int, default=None)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--min-turns", type=int, default=1)
    p.add_argument("--max-prefix", type=int, default=None)
    p.add_argument("--expand-prefixes", action="store_true")
    p.add_argument("--every-k", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also-shuffled-control", action="store_true", default=True)
    p.add_argument("--no-shuffled-control", action="store_false", dest="also_shuffled_control")
    p.add_argument("--obs-pool-size", type=int, default=8192)
    p.add_argument("--max-tool-chars", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--weight-wm-code", type=float, default=0.45)
    p.add_argument("--weight-wm-os", type=float, default=0.40)
    p.add_argument("--weight-anti-forget", type=float, default=0.15)
    p.add_argument("--force", action="store_true")
    # Back-compat aliases for old CLI
    p.add_argument("--swe-hero", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--swe-hero-source", default=None, help=argparse.SUPPRESS)
    p.add_argument("--swe-hero-max-rows", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.swe_hero:
        args.wm_code = True
    if args.swe_hero_source:
        args.hub_source = args.swe_hero_source
    if args.swe_hero_max_rows is not None and args.max_rows_per_source is None:
        args.max_rows_per_source = args.swe_hero_max_rows

    if args.all:
        args.wm_code = args.wm_os = args.anti_forget = True
    if args.local:
        args.wm_code = True
    if not (args.wm_code or args.wm_os or args.anti_forget):
        raise SystemExit("Enable --all or any of --wm-code / --wm-os / --anti-forget")

    if args.hub_source == "huggingface" and not os.environ.get("HF_ENDPOINT"):
        print("Tip: export HF_ENDPOINT=https://hf-mirror.com", flush=True)

    out_root = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fp_payload = {
        "wm_code": bool(args.wm_code),
        "wm_os": bool(args.wm_os),
        "anti_forget": bool(args.anti_forget),
        "hub_source": args.hub_source,
        "max_rows_per_source": args.max_rows_per_source,
        "anti_forget_max_rows": args.anti_forget_max_rows,
        "eval_ratio": args.eval_ratio,
        "expand_prefixes": bool(args.expand_prefixes),
        "seed": args.seed,
        "min_turns": args.min_turns,
        "max_prefix": args.max_prefix,
        "weights": [args.weight_wm_code, args.weight_wm_os, args.weight_anti_forget],
        "repos": {
            "hero": args.swe_hero_repo,
            "zero": args.swe_zero_repo,
            "ise": args.isetrace_repo,
        },
    }
    fp = _fingerprint(fp_payload)
    fp_path = out_root / "fingerprint.json"
    if not args.force and fp_path.is_file() and (out_root / "counts.json").is_file():
        prev = json.loads(fp_path.read_text(encoding="utf-8"))
        if prev.get("fingerprint") == fp:
            print(f"Cache hit fingerprint={fp}; reuse {out_root}", flush=True)
            counts = json.loads((out_root / "counts.json").read_text(encoding="utf-8"))
            print(json.dumps(counts, indent=2), flush=True)
            print("=== TOTAL JSONL LINE COUNTS (cached) ===", flush=True)
            for k, v in sorted((counts.get("jsonl_line_counts") or {}).items()):
                print(f"  {k}: {v:,}", flush=True)
            return

    rng = random.Random(args.seed)
    # Resolve hub_source for openers
    if args.hub_source == "auto":
        args.hub_source = "modelscope"

    stats: dict[str, Any] = {}
    banned: set[str] = set()

    if args.wm_code:
        print("=== Preparing wm_code (SWE-Hero) ===", flush=True)
        sub = out_root / "wm_code"
        stats["wm_code"] = _prepare_wm_openhands(
            kind="swe_hero",
            source_tag=SOURCE_WM_CODE,
            out_dir=sub,
            args=args,
            rng=rng,
        )
        ids_path = sub / "instance_ids.json"
        if ids_path.is_file():
            banned |= set(json.loads(ids_path.read_text(encoding="utf-8")))
        for path in args.local:
            print(f"=== Merging local fixture {path} ===", flush=True)
            for rec in load_local_trajectories(path):
                for row in wm_row_from_openhands_record(
                    rec, source=SOURCE_WM_CODE, min_turns=args.min_turns
                ):
                    append_jsonl(sub / "train.jsonl", row)
                    stats["wm_code"]["written"]["train"] = (
                        stats["wm_code"]["written"].get("train", 0) + 1
                    )

    if args.wm_os:
        print("=== Preparing wm_os (ISETrace) ===", flush=True)
        stats["wm_os"] = _prepare_wm_isetrace(
            out_dir=out_root / "wm_os", args=args, rng=rng
        )

    if args.anti_forget:
        print("=== Preparing anti_forget (SWE-Zero) ===", flush=True)
        stats["anti_forget"] = _prepare_anti_forget(
            out_dir=out_root / "anti_forget",
            args=args,
            rng=rng,
            banned_instances=banned,
        )

    _write_mix_manifest(out_root, stats, args)
    fp_path.write_text(
        json.dumps({"fingerprint": fp, "payload": fp_payload}, indent=2),
        encoding="utf-8",
    )
    print(f"Done → {out_root} fingerprint={fp}", flush=True)


if __name__ == "__main__":
    main()
