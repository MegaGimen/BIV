#!/usr/bin/env python3
"""Train-time length clean + optional 1:1:0.35 rebalance on ms-swift caches.

Called by ``scripts/train_coder_next.sh`` after the user passes ``--max-length``.

1) Drop rows with ``length > max_length`` (delete-style clean) from each source.
2) Print how many remain per source.
3) Interactive choice:
     1 = keep cleaned counts as-is (no code/os 1:1 flatten)
     2 = re-sample cleaned pools at code:os:anti = 1:1:0.35 (max_fill)
     3 = abort
4) Write filtered HF datasets + a run_manifest; emit shell exports via --write-env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "swift" / "coder_next_qlora.yaml"
SOURCE_KEYS = ("wm_code", "wm_os", "anti_forget")
MANIFEST_NAME = "tokenize_manifest.json"
REBALANCE_RATIOS = {"wm_code": 1.0, "wm_os": 1.0, "anti_forget": 0.35}


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable if iterable is not None else None
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    if iterable is None:
        return tqdm(**kwargs)
    return tqdm(iterable, **kwargs)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML: {path}")
    return data


def _resolve(raw: str | Path) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _find_manifest(cache_root: Path, tag: str | None) -> Path:
    if tag:
        p = cache_root / tag / MANIFEST_NAME
        if not p.is_file():
            raise SystemExit(f"Manifest not found: {p}")
        return p
    latest = cache_root / "LATEST"
    if latest.is_file():
        t = latest.read_text(encoding="utf-8").strip()
        p = cache_root / t / MANIFEST_NAME
        if p.is_file():
            return p
    cands = sorted(cache_root.glob(f"*/{MANIFEST_NAME}"), key=lambda x: x.stat().st_mtime)
    if not cands:
        raise SystemExit(
            f"No {MANIFEST_NAME} under {cache_root}. Run:\n  python scripts/tokenize_data.py"
        )
    return cands[-1]


def _load_dataset(path: Path):
    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise SystemExit(f"datasets required: pip install datasets\n({e})") from e
    return load_from_disk(str(path))


def _length_column(ds) -> str:
    for name in ("length", "lengths"):
        if name in ds.column_names:
            return name
    raise SystemExit(
        f"Dataset missing length column (got {ds.column_names}). "
        "Re-run tokenize_data.py with ms-swift>=3.11."
    )


def _count_kept(ds, col: str, max_length: int, *, desc: str) -> tuple[int, int]:
    """Return (kept, total) without materializing a filtered copy yet."""
    total = len(ds)
    kept = 0
    # batched map/filter counts via column scan
    lengths = ds[col]
    bar = _tqdm(total=total, unit="rows", desc=desc)
    if bar is None:
        for v in lengths:
            if int(v) <= max_length:
                kept += 1
    else:
        with bar:
            for v in lengths:
                if int(v) <= max_length:
                    kept += 1
                bar.update(1)
    return kept, total


def _filter_keep(ds, col: str, max_length: int, *, desc: str, num_proc: int):
    return ds.filter(
        lambda x: int(x[col]) <= max_length,
        num_proc=max(1, num_proc),
        desc=desc,
    )


def _sample_indices(n_keep: int, n_take: int, seed: int) -> list[int]:
    if n_take >= n_keep:
        return list(range(n_keep))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_keep), n_take))


def _max_fill_targets(kept: dict[str, int]) -> dict[str, int]:
    for k in SOURCE_KEYS:
        if kept[k] <= 0:
            raise SystemExit(
                f"After max_length clean, {k} has 0 rows — cannot rebalance. "
                "Raise --max-length or choose option 1 only if any source survives."
            )
    ratios = REBALANCE_RATIOS
    s = min(kept[k] / ratios[k] for k in SOURCE_KEYS)
    targets = {k: max(1, int(s * ratios[k])) for k in SOURCE_KEYS}
    n_wm = min(targets["wm_code"], targets["wm_os"], kept["wm_code"], kept["wm_os"])
    targets["wm_code"] = n_wm
    targets["wm_os"] = n_wm
    targets["anti_forget"] = min(
        kept["anti_forget"],
        max(1, int(round(n_wm * ratios["anti_forget"] / ratios["wm_code"]))),
    )
    for k in SOURCE_KEYS:
        if targets[k] > kept[k]:
            raise SystemExit(f"Internal error: target {k}={targets[k]} > kept={kept[k]}")
    if targets["wm_code"] != targets["wm_os"]:
        raise SystemExit("Internal error: code/os not equal after flatten")
    return targets


def _prompt_choice(preselected: int | None) -> int:
    if preselected is not None:
        if preselected not in {1, 2, 3}:
            raise SystemExit("--choice must be 1, 2, or 3")
        print(f"Choice (non-interactive): {preselected}", flush=True)
        return preselected
    print("", flush=True)
    print("选择下一步：", flush=True)
    print("  1 = 不按 1:1 拉平 code/os，清洗后原样开训", flush=True)
    print("  2 = 在清洗池上按 code:os:anti = 1:1:0.35 再 sample，然后开训", flush=True)
    print("  3 = 中断", flush=True)
    while True:
        raw = input("请输入 1 / 2 / 3: ").strip()
        if raw in {"1", "2", "3"}:
            return int(raw)
        print("无效输入，请重新输入 1、2 或 3。", flush=True)


def _write_env(path: Path, exports: dict[str, str]) -> None:
    lines = [f"export {k}={shlex.quote(v)}" for k, v in exports.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument(
        "--choice",
        type=int,
        default=None,
        help="Skip prompt: 1=as-is, 2=rebalance 1:1:0.35, 3=abort",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument(
        "--write-env",
        type=Path,
        required=True,
        help="Shell env file for train_coder_next.sh to source",
    )
    p.add_argument("--force", action="store_true", help="Rebuild run cache even if present")
    args = p.parse_args()

    if args.max_length <= 0:
        raise SystemExit("--max-length must be > 0")

    cfg = _load_yaml(_resolve(args.config))
    cache_root = _resolve(cfg.get("cache_root", "outputs/swift_cache/coder_next_mix_v1"))
    manifest_path = _find_manifest(cache_root, args.tag)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = manifest.get("cached_train") or {}
    tag = str(manifest.get("tag") or manifest_path.parent.name)
    model = manifest.get("model") or cfg.get("model")
    train_cfg = cfg.get("train") or {}

    print(f"Manifest:    {manifest_path}", flush=True)
    print(f"Tag:         {tag}", flush=True)
    print(f"Model:       {model}", flush=True)
    print(f"max_length:  {args.max_length}  (clean = drop length > max_length)", flush=True)
    print("", flush=True)
    print("=== retention after max_length clean ===", flush=True)

    source_paths: dict[str, Path] = {}
    kept_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    datasets_raw: dict[str, Any] = {}
    length_cols: dict[str, str] = {}

    for name in SOURCE_KEYS:
        rel = cached.get(name)
        if not rel:
            raise SystemExit(f"manifest missing cached_train.{name}")
        path = _resolve(rel)
        if not path.exists():
            raise SystemExit(f"Missing cached dataset: {path}")
        source_paths[name] = path
        ds = _load_dataset(path)
        col = _length_column(ds)
        kept, total = _count_kept(ds, col, args.max_length, desc=f"scan {name}")
        datasets_raw[name] = ds
        length_cols[name] = col
        kept_counts[name] = kept
        total_counts[name] = total
        pct = (100.0 * kept / total) if total else 0.0
        print(
            f"  {name:12s}: keep {kept:7,} / {total:,} ({pct:5.1f}%)",
            flush=True,
        )

    total_keep = sum(kept_counts.values())
    total_all = sum(total_counts.values())
    print(
        f"  {'TOTAL':12s}: keep {total_keep:7,} / {total_all:,}",
        flush=True,
    )

    if total_keep == 0:
        raise SystemExit("No rows survive this max_length. Raise --max-length.")

    # Preview option-2 targets (even if user picks 1 later)
    preview2: dict[str, int] | None = None
    if all(kept_counts[k] > 0 for k in SOURCE_KEYS):
        preview2 = _max_fill_targets(kept_counts)
        print("", flush=True)
        print("若选 2（1:1:0.35 max_fill）预计 sample：", flush=True)
        tsum = sum(preview2.values())
        for k in SOURCE_KEYS:
            share = 100.0 * preview2[k] / tsum if tsum else 0.0
            print(f"  {k:12s}: {preview2[k]:7,} ({share:5.1f}%)", flush=True)
        print(f"  {'TOTAL':12s}: {tsum:7,}", flush=True)
    else:
        print(
            "\nWARNING: 至少一个源清洗后为 0，选项 2 不可用；只能选 1 或 3。",
            flush=True,
        )

    choice = _prompt_choice(args.choice)
    if choice == 3:
        print("已中断。", flush=True)
        raise SystemExit(3)
    if choice == 2 and preview2 is None:
        raise SystemExit("选项 2 需要三个源清洗后都 >0。")

    if choice == 1:
        targets = dict(kept_counts)
        mode = "clean_as_is"
    else:
        targets = preview2  # type: ignore[assignment]
        mode = "clean_rebalance_1_1_0.35"

    if any(int(targets[k]) <= 0 for k in SOURCE_KEYS):
        raise SystemExit(
            "At least one source has 0 rows after clean/rebalance. "
            "Raise --max-length or adjust choice."
        )

    blob = (
        f"tag={tag}|ml={args.max_length}|mode={mode}|seed={args.seed}|"
        f"t={targets['wm_code']}:{targets['wm_os']}:{targets['anti_forget']}"
    )
    run_id = (
        f"ml{args.max_length}_{mode}"
        f"_n{targets['wm_code']}_{targets['wm_os']}_{targets['anti_forget']}_"
        f"{hashlib.sha1(blob.encode()).hexdigest()[:8]}"
    )
    run_root = manifest_path.parent / "train_runs" / run_id
    out_paths: dict[str, Path] = {k: run_root / k for k in SOURCE_KEYS}

    # HF save_to_disk writes state.json under each split dir.
    reuse = (
        not args.force
        and all((out_paths[k] / "state.json").is_file() for k in SOURCE_KEYS)
        and (run_root / "run_manifest.json").is_file()
    )

    if reuse:
        print(f"\nReusing prepared run cache: {run_root}", flush=True)
    else:
        if run_root.exists() and args.force:
            import shutil

            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"\nBuilding train mix under {run_root} …", flush=True)
        for i, name in enumerate(SOURCE_KEYS):
            ds = datasets_raw[name]
            col = length_cols[name]
            filtered = _filter_keep(
                ds,
                col,
                args.max_length,
                desc=f"filter {name}",
                num_proc=args.num_proc,
            )
            n_f = len(filtered)
            if n_f != kept_counts[name]:
                print(
                    f"WARNING: {name} filter size {n_f} != scanned keep {kept_counts[name]}",
                    flush=True,
                )
            n_take = targets[name]
            if n_take > n_f:
                raise SystemExit(f"{name}: need {n_take} but filtered only {n_f}")
            idxs = _sample_indices(n_f, n_take, seed=args.seed + i * 17)
            if n_take < n_f:
                print(f"  {name}: sample {n_take:,}/{n_f:,}", flush=True)
                final = filtered.select(idxs)
            else:
                print(f"  {name}: keep all {n_f:,}", flush=True)
                final = filtered
            out = out_paths[name]
            if out.exists():
                import shutil

                shutil.rmtree(out)
            final.save_to_disk(str(out))
            print(f"  wrote {out} ({len(final):,} rows)", flush=True)

        run_manifest = {
            "tag": tag,
            "parent_manifest": str(manifest_path.relative_to(ROOT)),
            "max_length": args.max_length,
            "mode": mode,
            "choice": choice,
            "seed": args.seed,
            "ratios": REBALANCE_RATIOS if choice == 2 else None,
            "kept_after_clean": kept_counts,
            "targets": targets,
            "cached_train": {k: str(out_paths[k].relative_to(ROOT)) for k in SOURCE_KEYS},
            "truncation_strategy": "delete",
            "note": "Rows already length-filtered; train uses delete for safety.",
        }
        (run_root / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2), encoding="utf-8"
        )

    cached_list = " ".join(str(out_paths[k]) for k in SOURCE_KEYS)
    out_dir = train_cfg.get("output_dir", "outputs/swift_coder_next_wm_mix")
    out_dir = f"{out_dir}_ml{args.max_length}_c{choice}"

    exports = {
        "TAG": tag,
        "RUN_ID": run_id,
        "RUN_ROOT": str(run_root),
        "MANIFEST": str(run_root / "run_manifest.json"),
        "MODEL": str(model),
        "MAX_LENGTH": str(args.max_length),
        "TRUNC": "delete",
        "CACHED_DATASETS": cached_list,
        "OUT_DIR": str(out_dir),
        "TRAIN_CHOICE": str(choice),
        "LR": str(train_cfg.get("learning_rate", 1e-4)),
        "EPOCHS": str(train_cfg.get("num_epochs", 2)),
        "LORA_RANK": str(train_cfg.get("lora_rank", 16)),
        "LORA_ALPHA": str(train_cfg.get("lora_alpha", 16)),
        "BS": str(train_cfg.get("per_device_train_batch_size", 1)),
        "GAS": str(train_cfg.get("gradient_accumulation_steps", 8)),
        "DEEPSPEED": str(train_cfg.get("deepspeed", "zero2")),
        "DTYPE": str(train_cfg.get("torch_dtype", "bfloat16")),
        "WARMUP": str(train_cfg.get("warmup_ratio", 0.03)),
        "LOG_STEPS": str(train_cfg.get("logging_steps", 10)),
        "SAVE_STEPS": str(train_cfg.get("save_steps", 200)),
        "SAVE_LIMIT": str(train_cfg.get("save_total_limit", 3)),
        "TARGET_MODULES": " ".join(
            train_cfg.get("target_modules")
            or ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    }
    env_path = Path(args.write_env)
    if not env_path.is_absolute():
        env_path = ROOT / env_path
    env_path.parent.mkdir(parents=True, exist_ok=True)
    _write_env(env_path, exports)

    print("", flush=True)
    print(f"Ready: choice={choice} mode={mode}", flush=True)
    print(f"  run: {run_root}", flush=True)
    print(
        "  targets: "
        + ", ".join(f"{k}={targets[k]:,}" for k in SOURCE_KEYS)
        + f"  total={sum(targets.values()):,}",
        flush=True,
    )
    print(f"  env → {env_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。", flush=True)
        sys.exit(3)
