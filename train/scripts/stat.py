#!/usr/bin/env python3
"""Step 3 (optional): token-length stats per source from ms-swift cached_dataset.

Reads ``tokenize_manifest.json`` written by ``scripts/tokenize_data.py`` and loads each
source's HF dataset (``length`` field). Prints distribution + how many rows
survive common ``max_length`` cutoffs (delete-style), so you can pick train
``--max_length`` against GPU VRAM without re-exporting.

Examples:
  python scripts/stat.py
  python scripts/stat.py --tag r1_1_0.35_n21961_7686_xxxxxxxx
  python scripts/stat.py --cutoffs 8192,16384,32768
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "swift" / "coder_next_qlora.yaml"
SOURCE_KEYS = ("wm_code", "wm_os", "anti_forget")
MANIFEST_NAME = "tokenize_manifest.json"


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
    # fallback: newest manifest under cache_root
    cands = sorted(cache_root.glob(f"*/{MANIFEST_NAME}"), key=lambda x: x.stat().st_mtime)
    if not cands:
        raise SystemExit(
            f"No {MANIFEST_NAME} under {cache_root}. Run:\n  python scripts/tokenize_data.py"
        )
    return cands[-1]


def _as_int_length(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, list):
        # some versions nest; take scalar if possible
        if len(v) == 1 and isinstance(v[0], (int, float)):
            return int(v[0])
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _load_lengths(ds_path: Path, *, desc: str) -> list[int]:
    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise SystemExit(f"datasets required: pip install datasets\n({e})") from e

    ds = load_from_disk(str(ds_path))
    # prefer column 'length' (ms-swift cached_dataset)
    col = None
    for name in ("length", "lengths"):
        if name in ds.column_names:
            col = name
            break
    if col is None:
        raise SystemExit(
            f"{ds_path}: no 'length' column (got {ds.column_names}). "
            "Re-run tokenize with ms-swift>=3.11 cached_dataset export."
        )

    lengths: list[int] = []
    n = len(ds)
    bar = _tqdm(total=n, unit="rows", desc=desc)
    # batched for speed
    batch = 1024
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = ds[start:end][col]
        for v in chunk:
            iv = _as_int_length(v)
            if iv is not None and iv >= 0:
                lengths.append(iv)
        if bar is not None:
            bar.update(end - start)
    if bar is not None:
        bar.close()
    if not lengths:
        raise SystemExit(f"{ds_path}: no valid length values")
    return lengths


def _pct(xs: list[int], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _hist(xs: list[int], edges: list[int]) -> list[tuple[str, int]]:
    counts = [0] * (len(edges) + 1)
    for x in xs:
        placed = False
        for i, e in enumerate(edges):
            if x < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = []
    prev = 0
    for e in edges:
        labels.append(f"[{prev}, {e})")
        prev = e
    labels.append(f"[{prev}, +inf)")
    return list(zip(labels, counts))


def _print_source(name: str, lengths: list[int], cutoffs: list[int]) -> dict[str, Any]:
    n = len(lengths)
    print(f"\n=== {name} (n={n:,}) ===", flush=True)
    print(
        f"  min={min(lengths):,}  max={max(lengths):,}  "
        f"mean={statistics.mean(lengths):.1f}  median={_pct(lengths, 50):.1f}",
        flush=True,
    )
    print(
        f"  p90={_pct(lengths, 90):.1f}  p95={_pct(lengths, 95):.1f}  "
        f"p99={_pct(lengths, 99):.1f}",
        flush=True,
    )
    edges = [2048, 4096, 8192, 16384, 32768, 65536]
    print("  histogram:", flush=True)
    for label, c in _hist(lengths, edges):
        bar = "#" * min(40, int(40 * c / max(n, 1)))
        print(f"    {label:>16}: {c:7,} ({100 * c / n:5.1f}%) {bar}", flush=True)

    retention = {}
    print("  retention if train uses truncation_strategy=delete at max_length:", flush=True)
    for m in cutoffs:
        keep = sum(1 for x in lengths if x <= m)
        retention[str(m)] = {"keep": keep, "drop": n - keep, "keep_pct": keep / n}
        print(
            f"    max_length={m:>6}: keep {keep:7,} / {n:,} "
            f"({100 * keep / n:5.1f}%)  drop {n - keep:,}",
            flush=True,
        )
    return {
        "n": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "p50": _pct(lengths, 50),
        "p90": _pct(lengths, 90),
        "p95": _pct(lengths, 95),
        "p99": _pct(lengths, 99),
        "retention_delete": retention,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-source token length stats from ms-swift tokenize cache."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="8192,16384,32768",
        help="Comma-separated max_length cutoffs for delete-style retention",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full stats JSON",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    cfg = _load_yaml(config_path) if config_path.is_file() else {}
    cache_root = _resolve(cfg.get("cache_root", "outputs/swift_cache/coder_next_mix_v1"))
    manifest_path = _find_manifest(cache_root, args.tag)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cutoffs = [int(x.strip()) for x in args.cutoffs.split(",") if x.strip()]
    cached = manifest.get("cached_train") or {}
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Tag:      {manifest.get('tag')}", flush=True)
    print(f"Model:    {manifest.get('model')}", flush=True)
    print(f"Targets:  {manifest.get('targets')}", flush=True)

    report: dict[str, Any] = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "tag": manifest.get("tag"),
        "sources": {},
    }

    src_bar = _tqdm(total=len(SOURCE_KEYS), unit="source", desc="stat sources")
    for name in SOURCE_KEYS:
        rel = cached.get(name)
        if not rel:
            print(f"WARNING: no cached_train.{name} in manifest — skip", flush=True)
            if src_bar is not None:
                src_bar.update(1)
            continue
        path = _resolve(rel)
        if not path.is_dir():
            raise SystemExit(f"Missing cache dir {path}")
        lengths = _load_lengths(path, desc=f"lengths {name}")
        report["sources"][name] = _print_source(name, lengths, cutoffs)
        if src_bar is not None:
            src_bar.update(1)
            src_bar.set_postfix_str(name)
    if src_bar is not None:
        src_bar.close()

    # mix retention under delete at each cutoff
    print("\n=== mix retention (delete @ max_length) ===", flush=True)
    print(
        "(Assumes each source contributes independently; ratios after delete may skew.)",
        flush=True,
    )
    for m in cutoffs:
        parts = []
        total_keep = 0
        total_n = 0
        for name in SOURCE_KEYS:
            st = report["sources"].get(name)
            if not st:
                continue
            r = st["retention_delete"][str(m)]
            parts.append(f"{name}={r['keep']:,}")
            total_keep += r["keep"]
            total_n += st["n"]
        print(
            f"  max_length={m}: "
            + ", ".join(parts)
            + f" | total keep={total_keep:,}/{total_n:,}",
            flush=True,
        )

    out = args.json_out
    if out is None:
        out = manifest_path.parent / "length_stats.json"
    else:
        out = out if out.is_absolute() else (ROOT / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
