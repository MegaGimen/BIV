#!/usr/bin/env python3
"""Step 2: ratio-sample mix JSONL, then Axolotl CPU tokenize.

Reads ``biv_mix`` from the Axolotl yaml (ignored by Axolotl itself after we
strip it into a generated ``*.run.yaml``):

  biv_mix:
    seed: 42
    mode: max_fill          # or total_rows
    total_rows: null        # used when mode=total_rows
    ratios:                 # relative shares; code:os should be 1:1
      wm_code: 1.0
      wm_os: 1.0
      anti_forget: 0.35     # ~15% of the merged train set

Sampling writes ``{mix_dir}/sampled/<tag>/{source}/train.jsonl``, then runs
``axolotl preprocess`` on a generated run config that points at those files.
Training must use the same ``*.run.yaml`` so train sees only the sampled rows.

Does **not** need a GPU — forces ``CUDA_VISIBLE_DEVICES=""``.

Examples:
  python scripts/tokenize.py
  python scripts/tokenize.py --force
  python scripts/tokenize.py --check
  python scripts/tokenize.py --sample-only   # write subsets, skip axolotl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "axolotl" / "coder_next_qlora.yaml"
DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v1"
SOURCE_KEYS = ("wm_code", "wm_os", "anti_forget")


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(f"PyYAML required: pip install pyyaml\n({e})") from e
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML (expected mapping): {path}")
    return data


def _dump_yaml(data: dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _prepared_path(cfg: dict, config_path: Path) -> Path:
    raw = cfg.get("dataset_prepared_path")
    if not raw:
        raise SystemExit(
            f"{config_path}: missing dataset_prepared_path "
            "(required so train can reuse the token cache)"
        )
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _resolve_full_sources(mix_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name in SOURCE_KEYS:
        p = mix_dir / name / "train.jsonl"
        if not p.is_file():
            raise SystemExit(
                f"Missing {p} — run prepare first:\n"
                f"  python scripts/prepare_data.py --all --out-dir {mix_dir.relative_to(ROOT)}"
            )
        out[name] = p
    return out


def _tqdm(**kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    return tqdm(**kwargs)


def _count_lines(path: Path, *, desc: str | None = None) -> int:
    """Count lines with a progress bar (byte-based when size is known)."""
    if not path.is_file():
        return 0
    size = path.stat().st_size
    if size == 0:
        return 0
    label = desc or path.name
    n = 0
    with path.open("rb") as f:
        bar = _tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"count {label}",
        )
        if bar is None:
            print(f"  counting lines: {label} ({size / 1e9:.2f} GB)…", flush=True)
            for line in f:
                n += 1
                if n % 50_000 == 0:
                    print(f"    … {n:,} lines", flush=True)
            return n
        with bar:
            for line in f:
                n += 1
                bar.update(len(line))
    return n


def _available_from_prepare_cache(
    mix_dir: Path, sources: dict[str, Path]
) -> dict[str, int] | None:
    """Reuse prepare_data line counts — do not rescan multi‑GB anti_forget JSONL."""
    available: dict[str, int] = {}

    root_counts = mix_dir / "counts.json"
    if root_counts.is_file():
        try:
            blob = json.loads(root_counts.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blob = {}
        jl = blob.get("jsonl_line_counts") or {}
        for name in SOURCE_KEYS:
            key = f"{name}/train.jsonl"
            if key in jl and jl[key] is not None:
                available[name] = int(jl[key])

    if len(available) == len(SOURCE_KEYS):
        print(f"Using line counts from {root_counts}", flush=True)
        return available

    # Per-source counts.json (written during prepare even before mix rollup)
    for name, path in sources.items():
        if name in available:
            continue
        cpath = mix_dir / name / "counts.json"
        if not cpath.is_file():
            continue
        try:
            st = json.loads(cpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        jl = st.get("jsonl_line_counts") or {}
        written = st.get("written") or {}
        n = jl.get("train")
        if n is None:
            n = written.get("train")
        if n is not None:
            available[name] = int(n)
            print(f"Using line count from {cpath} ({name}/train={n:,})", flush=True)

    if len(available) == len(SOURCE_KEYS):
        return available
    return None


def _resolve_available(mix_dir: Path, sources: dict[str, Path]) -> dict[str, int]:
    cached = _available_from_prepare_cache(mix_dir, sources)
    if cached is not None:
        for name, n in cached.items():
            print(f"  available {name}: {n:,} (cached)", flush=True)
        return cached

    print(
        "WARNING: prepare counts.json incomplete — scanning JSONL "
        "(anti_forget can take a long time). Prefer re-running prepare "
        "or ensure mix_v1/counts.json exists.",
        flush=True,
    )
    available: dict[str, int] = {}
    for name, path in sources.items():
        available[name] = _count_lines(path, desc=f"{name}/train.jsonl")
        print(f"  available {name}: {available[name]:,}", flush=True)
    return available


def _parse_biv_mix(cfg: dict) -> dict[str, Any]:
    raw = cfg.get("biv_mix")
    if not isinstance(raw, dict):
        raise SystemExit(
            "Config missing biv_mix: block. Example:\n"
            "biv_mix:\n"
            "  seed: 42\n"
            "  mode: max_fill\n"
            "  ratios:\n"
            "    wm_code: 1.0\n"
            "    wm_os: 1.0\n"
            "    anti_forget: 0.35\n"
        )
    ratios = raw.get("ratios") or {}
    if not isinstance(ratios, dict):
        raise SystemExit("biv_mix.ratios must be a mapping")
    for k in SOURCE_KEYS:
        if k not in ratios:
            raise SystemExit(f"biv_mix.ratios missing {k}")
        if float(ratios[k]) <= 0:
            raise SystemExit(f"biv_mix.ratios.{k} must be > 0")
    r_c, r_o = float(ratios["wm_code"]), float(ratios["wm_os"])
    if abs(r_c - r_o) > 1e-9:
        raise SystemExit(
            f"biv_mix requires wm_code:wm_os = 1:1, got {r_c}:{r_o}"
        )
    mode = str(raw.get("mode", "max_fill"))
    if mode not in {"max_fill", "total_rows"}:
        raise SystemExit("biv_mix.mode must be max_fill or total_rows")
    total_rows = raw.get("total_rows")
    if mode == "total_rows":
        if total_rows is None or int(total_rows) <= 0:
            raise SystemExit("biv_mix.mode=total_rows requires total_rows > 0")
        total_rows = int(total_rows)
    else:
        total_rows = int(total_rows) if total_rows else None
    return {
        "seed": int(raw.get("seed", 42)),
        "mode": mode,
        "total_rows": total_rows,
        "ratios": {k: float(ratios[k]) for k in SOURCE_KEYS},
    }


def _target_counts(
    available: dict[str, int], mix: dict[str, Any]
) -> dict[str, int]:
    ratios = mix["ratios"]
    rsum = sum(ratios[k] for k in SOURCE_KEYS)
    if mix["mode"] == "total_rows":
        T = int(mix["total_rows"])
        targets = {k: max(1, int(T * ratios[k] / rsum)) for k in SOURCE_KEYS}
        # Keep exact 1:1 for code/os after rounding
        n_wm = min(targets["wm_code"], targets["wm_os"])
        targets["wm_code"] = n_wm
        targets["wm_os"] = n_wm
        targets["anti_forget"] = max(
            1, int(round(n_wm * ratios["anti_forget"] / ratios["wm_code"]))
        )
    else:
        # Largest scale s such that floor(s*r_i) <= available[i]
        s = min(available[k] / ratios[k] for k in SOURCE_KEYS)
        targets = {k: max(1, int(s * ratios[k])) for k in SOURCE_KEYS}
        n_wm = min(targets["wm_code"], targets["wm_os"])
        targets["wm_code"] = n_wm
        targets["wm_os"] = n_wm
        targets["anti_forget"] = max(
            1, int(round(n_wm * ratios["anti_forget"] / ratios["wm_code"]))
        )

    for k in SOURCE_KEYS:
        if targets[k] > available[k]:
            raise SystemExit(
                f"Need {targets[k]} rows for {k} but only {available[k]} available. "
                "Lower anti ratio / total_rows, or prepare more data."
            )
    # Final 1:1 check
    if targets["wm_code"] != targets["wm_os"]:
        raise SystemExit(
            f"Internal error: code/os targets not equal "
            f"{targets['wm_code']} vs {targets['wm_os']}"
        )
    return targets


def _sample_tag(mix: dict[str, Any], targets: dict[str, int]) -> str:
    r = mix["ratios"]
    blob = (
        f"seed={mix['seed']}|mode={mix['mode']}|total={mix['total_rows']}|"
        f"r={r['wm_code']}:{r['wm_os']}:{r['anti_forget']}|"
        f"n={targets['wm_code']}:{targets['wm_os']}:{targets['anti_forget']}"
    )
    h = hashlib.sha1(blob.encode()).hexdigest()[:8]
    return (
        f"r{r['wm_code']:g}_{r['wm_os']:g}_{r['anti_forget']:g}"
        f"_n{targets['wm_code']}_{targets['anti_forget']}_{h}"
    )


def _sample_jsonl(
    src: Path, dst: Path, n_take: int, n_total: int, seed: int, name: str
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if n_take >= n_total:
        size = src.stat().st_size
        bar = _tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"copy {name}",
        )
        with src.open("rb") as fin, dst.open("wb") as fout:
            if bar is None:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            else:
                with bar:
                    while True:
                        chunk = fin.read(1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)
                        bar.update(len(chunk))
        print(f"  {name}: copy all {n_total:,} → {dst}", flush=True)
        return

    print(f"  {name}: drawing {n_take:,}/{n_total:,} indices (seed={seed})…", flush=True)
    rng = random.Random(seed)
    chosen = set(rng.sample(range(n_total), n_take))
    written = 0
    bar = _tqdm(total=n_total, unit="rows", desc=f"sample {name}")
    with src.open("rb") as fin, dst.open("wb") as fout:
        if bar is None:
            for i, line in enumerate(fin):
                if i in chosen:
                    fout.write(line)
                    written += 1
                if (i + 1) % 50_000 == 0:
                    print(
                        f"    {name}: scanned {i + 1:,}/{n_total:,} wrote {written:,}",
                        flush=True,
                    )
        else:
            with bar:
                for i, line in enumerate(fin):
                    if i in chosen:
                        fout.write(line)
                        written += 1
                    bar.update(1)
                    bar.set_postfix(wrote=written, refresh=False)
    if written != n_take:
        raise SystemExit(f"{name}: expected {n_take} lines, wrote {written}")
    print(f"  {name}: sampled {n_take:,}/{n_total:,} → {dst}", flush=True)


def _cache_ready(cache_dir: Path) -> bool:
    if not cache_dir.is_dir():
        return False
    for p in cache_dir.rglob("*"):
        if p.is_file() and (
            p.suffix in {".arrow", ".parquet"}
            or p.name in {"dataset_info.json", "state.json"}
            or p.stat().st_size > 1024
        ):
            return True
    return False


def _find_axolotl() -> list[str]:
    exe = shutil.which("axolotl")
    if exe:
        return [exe, "preprocess"]
    return [sys.executable, "-m", "axolotl.cli.preprocess"]


def _run_config_path(config_path: Path) -> Path:
    return config_path.with_suffix(".run.yaml")


def _build_run_config(
    cfg: dict,
    *,
    sampled_paths: dict[str, Path],
    mix: dict[str, Any],
    targets: dict[str, int],
    tag: str,
    cache_dir: Path,
) -> dict:
    run = {k: v for k, v in cfg.items() if k != "biv_mix"}
    # Point at sampled JSONL; sizes already enforce mix ratio → equal concat.
    run["datasets"] = [
        {
            "path": str(sampled_paths[name].relative_to(ROOT)),
            "type": "chat_template",
            "field_messages": "messages",
        }
        for name in SOURCE_KEYS
    ]
    # Namespace cache by sample tag so ratio changes do not reuse wrong tokens.
    base_cache = cache_dir
    # If yaml already ends with mix name, nest tag underneath
    run["dataset_prepared_path"] = str(
        (base_cache / tag).relative_to(ROOT)
        if base_cache.is_absolute()
        else Path(str(cfg["dataset_prepared_path"])) / tag
    )
    run["biv_mix_applied"] = {
        "tag": tag,
        "seed": mix["seed"],
        "mode": mix["mode"],
        "total_rows": mix["total_rows"],
        "ratios": mix["ratios"],
        "targets": targets,
        "note": "sizes enforce mix; datasets concatenated without weight re-sampling",
    }
    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ratio-sample mix JSONL then Axolotl CPU tokenize (step 2)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mix-dir", type=Path, default=DEFAULT_MIX)
    parser.add_argument("--force", action="store_true", help="Rebuild sample + token cache")
    parser.add_argument("--check", action="store_true", help="Verify sample+token cache ready")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only write sampled JSONL + run yaml; skip axolotl preprocess",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="Do not clear CUDA_VISIBLE_DEVICES (not recommended)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override biv_mix.seed",
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=None,
        help="Override: set mode=total_rows and this budget",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    mix_dir = args.mix_dir if args.mix_dir.is_absolute() else (ROOT / args.mix_dir)

    cfg = _load_yaml(config_path)
    mix = _parse_biv_mix(cfg)
    if args.seed is not None:
        mix["seed"] = int(args.seed)
    if args.total_rows is not None:
        mix["mode"] = "total_rows"
        mix["total_rows"] = int(args.total_rows)

    sources = _resolve_full_sources(mix_dir)
    available = _resolve_available(mix_dir, sources)
    targets = _target_counts(available, mix)
    tag = _sample_tag(mix, targets)
    sample_root = mix_dir / "sampled" / tag
    sampled_paths = {
        k: sample_root / k / "train.jsonl" for k in SOURCE_KEYS
    }
    meta_path = sample_root / "sample_manifest.json"
    run_path = _run_config_path(config_path)

    # Cache dir from base config (tag nested in run config)
    base_cache = _prepared_path(cfg, config_path)
    token_cache = base_cache / tag

    total = sum(targets.values())
    shares = {k: targets[k] / total for k in SOURCE_KEYS}
    print(f"Config:     {config_path}", flush=True)
    print(f"Mix dir:    {mix_dir}", flush=True)
    print(
        f"Ratios:     code:os:anti = "
        f"{mix['ratios']['wm_code']:g}:{mix['ratios']['wm_os']:g}:{mix['ratios']['anti_forget']:g}",
        flush=True,
    )
    print(f"Mode:       {mix['mode']} seed={mix['seed']}", flush=True)
    print("Available:  " + ", ".join(f"{k}={available[k]:,}" for k in SOURCE_KEYS), flush=True)
    print(
        "Targets:    "
        + ", ".join(f"{k}={targets[k]:,} ({shares[k]*100:.1f}%)" for k in SOURCE_KEYS)
        + f"  total={total:,}",
        flush=True,
    )
    print(f"Sample dir: {sample_root}", flush=True)
    print(f"Run yaml:   {run_path}", flush=True)
    print(f"Token cache:{token_cache}", flush=True)

    sample_ok = meta_path.is_file() and all(p.is_file() for p in sampled_paths.values())
    if sample_ok and not args.force:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        if prev.get("targets") != targets or prev.get("seed") != mix["seed"]:
            sample_ok = False

    if args.check:
        tok_ok = _cache_ready(token_cache) and run_path.is_file()
        print(f"Sample ready: {sample_ok}", flush=True)
        print(f"Token cache ready: {tok_ok}", flush=True)
        print(f"Run config present: {run_path.is_file()}", flush=True)
        raise SystemExit(0 if (sample_ok and tok_ok and run_path.is_file()) else 1)

    if args.force and sample_root.exists():
        print(f"--force: removing {sample_root}", flush=True)
        shutil.rmtree(sample_root)
        sample_ok = False

    if not sample_ok:
        print("Sampling full JSONL → ratio-matched subsets…", flush=True)
        src_bar = _tqdm(total=len(SOURCE_KEYS), unit="source", desc="sample sources")
        for i, name in enumerate(SOURCE_KEYS):
            _sample_jsonl(
                sources[name],
                sampled_paths[name],
                targets[name],
                available[name],
                seed=mix["seed"] + i * 17,
                name=name,
            )
            if src_bar is not None:
                src_bar.update(1)
                src_bar.set_postfix_str(name)
        if src_bar is not None:
            src_bar.close()
        meta = {
            "tag": tag,
            "seed": mix["seed"],
            "mode": mix["mode"],
            "total_rows": mix["total_rows"],
            "ratios": mix["ratios"],
            "available": available,
            "targets": targets,
            "shares": shares,
            "paths": {k: str(p.relative_to(ROOT)) for k, p in sampled_paths.items()},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote {meta_path}", flush=True)
    else:
        print(f"Reusing sampled JSONL under {sample_root}", flush=True)

    run_cfg = _build_run_config(
        cfg,
        sampled_paths=sampled_paths,
        mix=mix,
        targets=targets,
        tag=tag,
        cache_dir=base_cache,
    )
    # Drop keys Axolotl may not accept
    run_cfg.pop("biv_mix_applied", None)
    # Keep a sibling sidecar for humans
    sidecar = {
        "tag": tag,
        "seed": mix["seed"],
        "mode": mix["mode"],
        "ratios": mix["ratios"],
        "targets": targets,
        "shares": shares,
        "sample_dir": str(sample_root.relative_to(ROOT)),
        "dataset_prepared_path": run_cfg["dataset_prepared_path"],
        "run_config": str(run_path.relative_to(ROOT)),
    }
    _dump_yaml(run_cfg, run_path)
    (sample_root / "tokenize_sidecar.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    print(f"Wrote run config {run_path} (Axolotl trains on sampled paths only)", flush=True)

    if args.sample_only:
        print("--sample-only: skip axolotl preprocess.", flush=True)
        return

    tok_cache_resolved = (
        Path(run_cfg["dataset_prepared_path"])
        if Path(run_cfg["dataset_prepared_path"]).is_absolute()
        else ROOT / run_cfg["dataset_prepared_path"]
    )
    if _cache_ready(tok_cache_resolved) and not args.force:
        print(
            f"\nToken cache already present under {tok_cache_resolved}\n"
            "Skip preprocess. Use --force to rebuild.\n"
            "Train with:\n"
            f"  CUDA_VISIBLE_DEVICES=0,1 axolotl train {run_path.relative_to(ROOT)}\n"
            "  # or: bash scripts/train_coder_next.sh",
            flush=True,
        )
        return

    if args.force and tok_cache_resolved.exists():
        print(f"--force: removing {tok_cache_resolved}", flush=True)
        shutil.rmtree(tok_cache_resolved)

    if not args.allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print('CUDA_VISIBLE_DEVICES="" (CPU tokenize)', flush=True)

    # Prefer single-process map if stuck issues; user can override.
    os.environ.setdefault("AXOLOTL_DATASET_NUM_PROC", "1")

    cmd = _find_axolotl() + [str(run_path)]
    print(f"Running: {' '.join(cmd)}", flush=True)
    print(
        f"(Tokenizing only sampled rows: {total:,} total, not full anti_forget.)",
        flush=True,
    )
    print(
        "(Axolotl preprocess shows its own Tokenizing Prompts bars next.)",
        flush=True,
    )

    os.chdir(ROOT)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    if not _cache_ready(tok_cache_resolved):
        raise SystemExit(
            f"preprocess finished but cache still empty under {tok_cache_resolved}"
        )

    print(
        f"\nDone. Token cache ready: {tok_cache_resolved}\n"
        "Start training (sampled mix only):\n"
        f"  CUDA_VISIBLE_DEVICES=0,1 axolotl train {run_path.relative_to(ROOT)}\n"
        "  # or: bash scripts/train_coder_next.sh",
        flush=True,
    )


if __name__ == "__main__":
    main()
