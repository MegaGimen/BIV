#!/usr/bin/env python3
"""Step 3: ratio-sample mix JSONL → HF datasets with messages + approx lengths (CPU OK).

Note: named tokenize_data.py (not tokenize.py) so it does not shadow stdlib ``tokenize``.

Reads ``biv_mix`` from ``configs/trl/muse_glimmer_30b_lora.yaml`` (default).
Does **not** need ms-swift or a GPU — builds ``datasets`` Arrow caches for
``train_prep_mix.py`` / TRL. Lengths use a char-budget heuristic (no full
chat_template pass); train-time ``--max-length`` + struct-right trim apply later.

Examples:
  python scripts/tokenize_data.py
  python scripts/tokenize_data.py --force
  python scripts/tokenize_data.py --check
  python scripts/tokenize_data.py --sample-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CONFIG = ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"
DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v2"
SOURCE_KEYS = ("wm_code", "wm_os", "anti_forget")
MANIFEST_NAME = "tokenize_manifest.json"
_CHARS_PER_TOKEN = 3.0
_MSG_OVERHEAD_CHARS = 64


def _tqdm(**kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    return tqdm(**kwargs)


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


def _resolve_path(raw: str | Path, *, base: Path = ROOT) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (base / p)


def _count_lines(path: Path, *, desc: str | None = None) -> int:
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
            for _ in f:
                n += 1
            return n
        with bar:
            for line in f:
                n += 1
                bar.update(len(line))
    return n


def _available_from_prepare_cache(
    mix_dir: Path, sources: dict[str, Path]
) -> dict[str, int] | None:
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

    for name in SOURCE_KEYS:
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
    print("WARNING: prepare counts.json incomplete — scanning JSONL…", flush=True)
    available: dict[str, int] = {}
    for name, path in sources.items():
        available[name] = _count_lines(path, desc=f"{name}/train.jsonl")
        print(f"  available {name}: {available[name]:,}", flush=True)
    return available


def _parse_biv_mix(cfg: dict) -> dict[str, Any]:
    raw = cfg.get("biv_mix")
    if not isinstance(raw, dict):
        raise SystemExit("Config missing biv_mix: block")
    mode = str(raw.get("mode", "full_wm"))
    if mode not in {"full_wm", "max_fill", "total_rows"}:
        raise SystemExit("biv_mix.mode must be full_wm, max_fill, or total_rows")

    if mode == "full_wm":
        anti_to_os = raw.get("anti_to_os")
        if anti_to_os is None:
            ratios = raw.get("ratios") or {}
            anti_to_os = ratios.get("anti_forget", 0.35)
        anti_to_os = float(anti_to_os)
        if anti_to_os <= 0:
            raise SystemExit("biv_mix.anti_to_os must be > 0")
        return {
            "seed": int(raw.get("seed", 42)),
            "mode": mode,
            "total_rows": None,
            "anti_to_os": anti_to_os,
            "ratios": {
                "wm_code": 1.0,
                "wm_os": 1.0,
                "anti_forget": anti_to_os,
            },
        }

    ratios = raw.get("ratios") or {}
    for k in SOURCE_KEYS:
        if k not in ratios or float(ratios[k]) <= 0:
            raise SystemExit(f"biv_mix.ratios.{k} must be > 0")
    r_c, r_o = float(ratios["wm_code"]), float(ratios["wm_os"])
    if abs(r_c - r_o) > 1e-9:
        raise SystemExit(f"biv_mix max_fill/total_rows require wm_code:wm_os = 1:1, got {r_c}:{r_o}")
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
        "anti_to_os": float(ratios["anti_forget"]) / float(ratios["wm_os"]),
        "ratios": {k: float(ratios[k]) for k in SOURCE_KEYS},
    }


def _target_counts(available: dict[str, int], mix: dict[str, Any]) -> dict[str, int]:
    if mix["mode"] == "full_wm":
        n_os = int(available["wm_os"])
        targets = {
            "wm_code": int(available["wm_code"]),
            "wm_os": n_os,
            "anti_forget": max(1, int(round(n_os * float(mix["anti_to_os"])))),
        }
    else:
        ratios = mix["ratios"]
        rsum = sum(ratios[k] for k in SOURCE_KEYS)
        if mix["mode"] == "total_rows":
            T = int(mix["total_rows"])
            targets = {k: max(1, int(T * ratios[k] / rsum)) for k in SOURCE_KEYS}
            n_wm = min(targets["wm_code"], targets["wm_os"])
            targets["wm_code"] = n_wm
            targets["wm_os"] = n_wm
            targets["anti_forget"] = max(
                1, int(round(n_wm * ratios["anti_forget"] / ratios["wm_code"]))
            )
        else:
            s = min(available[k] / ratios[k] for k in SOURCE_KEYS)
            targets = {k: max(1, int(s * ratios[k])) for k in SOURCE_KEYS}
            n_wm = min(targets["wm_code"], targets["wm_os"])
            targets["wm_code"] = n_wm
            targets["wm_os"] = n_wm
            targets["anti_forget"] = max(
                1, int(round(n_wm * ratios["anti_forget"] / ratios["wm_code"]))
            )
        if targets["wm_code"] != targets["wm_os"]:
            raise SystemExit("Internal error: code/os targets not equal")
    for k in SOURCE_KEYS:
        if targets[k] > available[k]:
            raise SystemExit(
                f"Need {targets[k]} rows for {k} but only {available[k]} available."
            )
    return targets


def _sample_tag(mix: dict[str, Any], targets: dict[str, int]) -> str:
    a2o = float(mix["anti_to_os"])
    blob = (
        f"seed={mix['seed']}|mode={mix['mode']}|total={mix['total_rows']}|"
        f"anti_to_os={a2o:g}|"
        f"n={targets['wm_code']}:{targets['wm_os']}:{targets['anti_forget']}"
    )
    h = hashlib.sha1(blob.encode()).hexdigest()[:8]
    return (
        f"{mix['mode']}_a2o{a2o:g}"
        f"_n{targets['wm_code']}_{targets['wm_os']}_{targets['anti_forget']}_{h}"
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


def _msg_char_cost(m: dict) -> int:
    n = _MSG_OVERHEAD_CHARS
    c = m.get("content")
    if isinstance(c, str):
        n += len(c)
    elif c is not None:
        n += len(json.dumps(c, ensure_ascii=False))
    tc = m.get("tool_calls")
    if tc is not None:
        n += len(json.dumps(tc, ensure_ascii=False))
    return n


def approx_token_len_messages(messages: list) -> int:
    if not messages:
        return 0
    chars = 0
    for m in messages:
        if isinstance(m, dict):
            chars += _msg_char_cost(m)
    return max(1, int(chars / _CHARS_PER_TOKEN))


def _cache_ready(source_cache: Path) -> bool:
    train = source_cache / "train"
    if (train / "state.json").is_file() or (train / "dataset_info.json").is_file():
        return True
    if (source_cache / "state.json").is_file() or (
        source_cache / "dataset_info.json"
    ).is_file():
        return True
    return False


def _cache_train_dir(source_cache: Path) -> Path:
    train = source_cache / "train"
    if train.is_dir() and (
        (train / "state.json").is_file() or (train / "dataset_info.json").is_file()
    ):
        return train
    return source_cache


def _export_hf_dataset(jsonl: Path, source_cache: Path, name: str, num_proc: int) -> Path:
    """Build HF Dataset from JSONL messages; write under source_cache/train."""
    from datasets import Dataset

    out = source_cache / "train"
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    bad = 0
    size = jsonl.stat().st_size
    bar = _tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"hf map {name}",
    )
    with jsonl.open("rb") as f:
        for line in f:
            if bar is not None:
                bar.update(len(line))
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                bad += 1
                continue
            L = approx_token_len_messages(messages)
            row: dict[str, Any] = {
                "messages": messages,
                "lengths": [L],
                "length": L,
            }
            for meta in ("source", "instance_id", "trajectory_id", "n_turns"):
                if meta in obj:
                    row[meta] = obj[meta]
            if "source" not in row:
                row["source"] = name
            rows.append(row)
    if bar is not None:
        bar.close()
    if not rows:
        raise SystemExit(f"{name}: no valid messages rows in {jsonl}")
    print(
        f"  {name}: {len(rows):,} rows (skipped {bad:,}) → HF dataset …",
        flush=True,
    )
    ds = Dataset.from_list(rows)
    if num_proc and num_proc > 1:
        # Touch map to materialize; lengths already set.
        pass
    ds.save_to_disk(str(out))
    print(f"  {name}: wrote {out} ({len(ds):,} rows)", flush=True)
    return out


def main() -> None:
    # Avoid accidental GPU init during CPU tokenize.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    parser = argparse.ArgumentParser(
        description="Ratio-sample + HF dataset cache export (Muse / TRL)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mix-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-rows", type=int, default=None)
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=None,
        help="Reserved for HF map workers (default from config).",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    cfg = _load_yaml(config_path)
    mix = _parse_biv_mix(cfg)
    if args.seed is not None:
        mix["seed"] = int(args.seed)
    if args.total_rows is not None:
        mix["mode"] = "total_rows"
        mix["total_rows"] = int(args.total_rows)

    mix_dir = (
        _resolve_path(args.mix_dir)
        if args.mix_dir
        else _resolve_path(cfg.get("mix_dir", DEFAULT_MIX))
    )
    cache_root = _resolve_path(
        cfg.get("cache_root", "outputs/trl_cache/muse_glimmer_mix_v2")
    )
    from biv_wm.model_store import model_dir_ready, resolve_model_dir, resolve_model_for_train

    model = resolve_model_for_train(cfg, root=ROOT)
    prepared = resolve_model_dir(cfg, root=ROOT)
    if model_dir_ready(prepared):
        print(f"Using prepared local model: {model}", flush=True)
    else:
        print(
            "NOTE: model_dir not ready yet (OK for char-budget tokenize). Prefer:\n"
            "  python scripts/prepare_model.py\n"
            f"  before train. Hub id fallback: {model!r}",
            flush=True,
        )
    dataset_num_proc = int(
        args.dataset_num_proc
        if args.dataset_num_proc is not None
        else cfg.get("dataset_num_proc", 8)
    )

    sources = {k: mix_dir / k / "train.jsonl" for k in SOURCE_KEYS}
    for name, p in sources.items():
        if not p.is_file():
            raise SystemExit(
                f"Missing {p} — run prepare first:\n"
                f"  python scripts/prepare_data.py --all --out-dir {mix_dir}"
            )

    available = _resolve_available(mix_dir, sources)
    targets = _target_counts(available, mix)
    tag = _sample_tag(mix, targets)
    sample_root = mix_dir / "sampled" / tag
    sampled_paths = {k: sample_root / k / "train.jsonl" for k in SOURCE_KEYS}
    tag_cache = cache_root / tag
    source_caches = {k: tag_cache / k for k in SOURCE_KEYS}
    manifest_path = tag_cache / MANIFEST_NAME

    total = sum(targets.values())
    shares = {k: targets[k] / total for k in SOURCE_KEYS}
    print(f"Config:     {config_path}", flush=True)
    print(f"Model:      {model}", flush=True)
    print(f"Mix dir:    {mix_dir}", flush=True)
    print(
        f"Mix mode:   {mix['mode']}  anti_to_os={mix['anti_to_os']:g}",
        flush=True,
    )
    print(
        "Targets:    "
        + ", ".join(f"{k}={targets[k]:,} ({shares[k]*100:.1f}%)" for k in SOURCE_KEYS)
        + f"  total={total:,}",
        flush=True,
    )
    print(f"Sample dir: {sample_root}", flush=True)
    print(f"Cache root: {tag_cache}", flush=True)

    sample_ok = all(p.is_file() for p in sampled_paths.values())
    if sample_ok and not args.force:
        meta_p = sample_root / "sample_manifest.json"
        if meta_p.is_file():
            prev = json.loads(meta_p.read_text(encoding="utf-8"))
            if prev.get("targets") != targets or prev.get("seed") != mix["seed"]:
                sample_ok = False
        else:
            sample_ok = False

    caches_ok = all(_cache_ready(source_caches[k]) for k in SOURCE_KEYS)

    if args.check:
        print(f"Sample ready: {sample_ok}", flush=True)
        print(f"HF caches ready: {caches_ok}", flush=True)
        print(f"Manifest: {manifest_path.is_file()}", flush=True)
        raise SystemExit(0 if (sample_ok and caches_ok and manifest_path.is_file()) else 1)

    if args.force and sample_root.exists():
        print(f"--force: removing {sample_root}", flush=True)
        shutil.rmtree(sample_root)
        sample_ok = False

    if not sample_ok:
        print("Sampling full JSONL → ratio-matched subsets…", flush=True)
        for i, name in enumerate(SOURCE_KEYS):
            _sample_jsonl(
                sources[name],
                sampled_paths[name],
                targets[name],
                available[name],
                seed=mix["seed"] + i * 17,
                name=name,
            )
        (sample_root / "sample_manifest.json").write_text(
            json.dumps(
                {
                    "tag": tag,
                    "seed": mix["seed"],
                    "mode": mix["mode"],
                    "ratios": mix["ratios"],
                    "available": available,
                    "targets": targets,
                    "shares": shares,
                    "paths": {
                        k: str(p.relative_to(ROOT)) for k, p in sampled_paths.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        print(f"Reusing sampled JSONL under {sample_root}", flush=True)

    if args.sample_only:
        print("--sample-only: skip HF dataset export.", flush=True)
        return

    if args.force and tag_cache.exists():
        print(f"--force: removing {tag_cache}", flush=True)
        shutil.rmtree(tag_cache)
        caches_ok = False

    train_dirs: dict[str, str] = {}
    if caches_ok and not args.force:
        print(f"Reusing HF caches under {tag_cache}", flush=True)
        for k in SOURCE_KEYS:
            train_dirs[k] = str(_cache_train_dir(source_caches[k]).relative_to(ROOT))
    else:
        print(
            "Building per-source HF datasets (char-budget lengths; "
            f"chars/token≈{_CHARS_PER_TOKEN})…",
            flush=True,
        )
        tag_cache.mkdir(parents=True, exist_ok=True)
        for name in SOURCE_KEYS:
            train_path = _export_hf_dataset(
                sampled_paths[name],
                source_caches[name],
                name,
                dataset_num_proc,
            )
            train_dirs[name] = str(train_path.relative_to(ROOT))

    manifest = {
        "tag": tag,
        "model": model,
        "config": str(config_path.relative_to(ROOT)),
        "mix_dir": str(mix_dir.relative_to(ROOT)),
        "seed": mix["seed"],
        "mode": mix["mode"],
        "anti_to_os": mix["anti_to_os"],
        "targets": targets,
        "cached_train": train_dirs,
        "length_backend": "char_budget",
        "chars_per_token": _CHARS_PER_TOKEN,
        "note": (
            "messages + approx lengths; train_prep_mix does struct-right trim; "
            "TRL tokenizes with Muse Glimmer chat_template at train time."
        ),
    }
    tag_cache.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (cache_root / "LATEST").write_text(tag + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    print(f"LATEST → {tag}", flush=True)
    print(
        "Next:\n"
        "  python scripts/stat.py --max-length 8192\n"
        "  CUDA_VISIBLE_DEVICES=0 bash scripts/trainmodel.sh --max-length 8192 --choice 1\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
