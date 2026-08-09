#!/usr/bin/env python3
"""Step 3 (optional): length + hard-truncate retention stats from cached_dataset.

For each of the 3 sources (wm_code / wm_os / anti_forget) prints **2**
distributions (6 total):

  1) token-length histogram
  2) hard-truncate retention-ratio histogram at ``--max-length``
     ratio_i = min(L_i, max_length) / L_i
     (estimate only — no structure-preserving assistant cut)

Also reports delete-style keep counts at the same max_length (scalar, not a dist).

Examples:
  python scripts/stat.py --max-length 8192
  python scripts/stat.py --max-length 16384 --tag full_wm_a2o0.35_...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "swift" / "coder_30b_a3b_qlora.yaml"
SOURCE_KEYS = ("wm_code", "wm_os", "anti_forget")
MANIFEST_NAME = "tokenize_manifest.json"
LENGTH_EDGES = [2048, 4096, 8192, 16384, 32768, 65536]
# retention ratio buckets; exact 1.0 (no trunc) counted separately at the end
RATIO_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def _pct(xs: list[float] | list[int], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _hist_int(xs: list[int], edges: list[int]) -> list[tuple[str, int]]:
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


def _hist_ratio(ratios: list[float], edges: list[float]) -> list[tuple[str, int]]:
    """Histogram on (0, 1]; put exact 1.0 in a final '=1.0 (no trunc)' bucket."""
    n_partial = len(edges)  # bins for r < 1.0 ending at each edge
    counts = [0] * (n_partial + 1)
    for r in ratios:
        if r >= 1.0 - 1e-12:
            counts[-1] += 1
            continue
        placed = False
        for i, e in enumerate(edges):
            if r < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            # r in [last_edge, 1.0) — should be rare if last edge is 1.0
            counts[-2] += 1
    labels: list[str] = []
    prev = 0.0
    for e in edges:
        labels.append(f"[{prev:.1f}, {e:.1f})")
        prev = e
    labels.append("=1.0 (no trunc)")
    return list(zip(labels, counts))


def _print_bar(label: str, c: int, n: int) -> None:
    bar = "#" * min(40, int(40 * c / max(n, 1)))
    print(f"    {label:>22}: {c:7,} ({100 * c / n:5.1f}%) {bar}", flush=True)


def _trunc_retention_ratios(lengths: list[int], max_length: int) -> list[float]:
    out: list[float] = []
    for x in lengths:
        if x <= 0:
            out.append(1.0)
        else:
            out.append(min(1.0, max_length / float(x)))
    return out


def _print_source(
    name: str,
    lengths: list[int],
    max_length: int,
    *,
    dist_idx: int,
) -> dict[str, Any]:
    """Print 2 distributions for one source; dist_idx is 1-based index of first dist."""
    n = len(lengths)
    ratios = _trunc_retention_ratios(lengths, max_length)
    keep_delete = sum(1 for x in lengths if x <= max_length)

    print(f"\n=== {name} (n={n:,}) ===", flush=True)

    # --- dist A: token length ---
    print(
        f"--- distribution {dist_idx}/6: {name} token length ---",
        flush=True,
    )
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
    print("  histogram:", flush=True)
    length_hist = _hist_int(lengths, LENGTH_EDGES)
    for label, c in length_hist:
        _print_bar(label, c, n)

    # --- dist B: hard-truncate retention ratio ---
    print(
        f"--- distribution {dist_idx + 1}/6: {name} hard-trunc retention "
        f"@ max_length={max_length} ---",
        flush=True,
    )
    print(
        "  estimate: ratio = min(L, max_length) / L  "
        "(no structure-preserving assistant cut)",
        flush=True,
    )
    print(
        f"  ratio min={min(ratios):.4f}  max={max(ratios):.4f}  "
        f"mean={statistics.mean(ratios):.4f}  median={_pct(ratios, 50):.4f}",
        flush=True,
    )
    print(
        f"  p10={_pct(ratios, 10):.4f}  p25={_pct(ratios, 25):.4f}  "
        f"p75={_pct(ratios, 75):.4f}  p90={_pct(ratios, 90):.4f}",
        flush=True,
    )
    # aggregate token mass kept under hard trunc
    kept_tokens = sum(min(x, max_length) for x in lengths)
    total_tokens = sum(lengths)
    print(
        f"  token-mass kept: {kept_tokens:,} / {total_tokens:,} "
        f"({100 * kept_tokens / max(total_tokens, 1):.1f}%)",
        flush=True,
    )
    print(
        f"  delete-style keep (L<=max_length): {keep_delete:,} / {n:,} "
        f"({100 * keep_delete / n:5.1f}%)  — scalar contrast, not a dist",
        flush=True,
    )
    print("  histogram (per-row retention ratio):", flush=True)
    ratio_hist = _hist_ratio(ratios, RATIO_EDGES)
    for label, c in ratio_hist:
        _print_bar(label, c, n)

    return {
        "n": n,
        "max_length": max_length,
        "length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.mean(lengths),
            "p50": _pct(lengths, 50),
            "p90": _pct(lengths, 90),
            "p95": _pct(lengths, 95),
            "p99": _pct(lengths, 99),
            "histogram": {lab: c for lab, c in length_hist},
        },
        "hard_trunc_retention": {
            "ratio_min": min(ratios),
            "ratio_max": max(ratios),
            "ratio_mean": statistics.mean(ratios),
            "ratio_p50": _pct(ratios, 50),
            "ratio_p10": _pct(ratios, 10),
            "ratio_p25": _pct(ratios, 25),
            "ratio_p75": _pct(ratios, 75),
            "ratio_p90": _pct(ratios, 90),
            "token_mass_kept": kept_tokens,
            "token_mass_total": total_tokens,
            "token_mass_kept_pct": kept_tokens / max(total_tokens, 1),
            "histogram": {lab: c for lab, c in ratio_hist},
        },
        "delete_keep": {
            "keep": keep_delete,
            "drop": n - keep_delete,
            "keep_pct": keep_delete / n,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-source token length + hard-truncate retention-ratio distributions "
            "(2 × 3 sources = 6)."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--max-length",
        "--max_length",
        type=int,
        required=True,
        dest="max_length",
        help="Hard-truncate estimate length; also used for delete-style keep scalar",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full stats JSON",
    )
    args = parser.parse_args()

    if args.max_length <= 0:
        raise SystemExit("--max-length must be a positive integer")

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    cfg = _load_yaml(config_path) if config_path.is_file() else {}
    cache_root = _resolve(cfg.get("cache_root", "outputs/swift_cache/coder_next_mix_v1"))
    manifest_path = _find_manifest(cache_root, args.tag)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cached = manifest.get("cached_train") or {}
    print(f"Manifest:   {manifest_path}", flush=True)
    print(f"Tag:        {manifest.get('tag')}", flush=True)
    print(f"Model:      {manifest.get('model')}", flush=True)
    print(f"Targets:    {manifest.get('targets')}", flush=True)
    print(f"max_length: {args.max_length}  (hard-trunc estimate; 2 dists × 3 sources)", flush=True)

    report: dict[str, Any] = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "tag": manifest.get("tag"),
        "max_length": args.max_length,
        "note": (
            "hard_trunc ratio = min(L, max_length)/L; "
            "not structure-preserving assistant cut"
        ),
        "sources": {},
    }

    src_bar = _tqdm(total=len(SOURCE_KEYS), unit="source", desc="stat sources")
    dist_idx = 1
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
        report["sources"][name] = _print_source(
            name, lengths, args.max_length, dist_idx=dist_idx
        )
        dist_idx += 2
        if src_bar is not None:
            src_bar.update(1)
            src_bar.set_postfix_str(name)
    if src_bar is not None:
        src_bar.close()

    print("\n=== mix summary @ max_length (scalars) ===", flush=True)
    parts_del = []
    parts_mass = []
    total_keep = 0
    total_n = 0
    mass_kept = 0
    mass_tot = 0
    for name in SOURCE_KEYS:
        st = report["sources"].get(name)
        if not st:
            continue
        d = st["delete_keep"]
        h = st["hard_trunc_retention"]
        parts_del.append(f"{name}={d['keep']:,}")
        parts_mass.append(f"{name}={100 * h['token_mass_kept_pct']:.1f}%")
        total_keep += d["keep"]
        total_n += st["n"]
        mass_kept += h["token_mass_kept"]
        mass_tot += h["token_mass_total"]
    print(
        "  delete keep rows: "
        + ", ".join(parts_del)
        + f" | total {total_keep:,}/{total_n:,}",
        flush=True,
    )
    print(
        "  hard-trunc token-mass kept: "
        + ", ".join(parts_mass)
        + f" | mix {100 * mass_kept / max(mass_tot, 1):.1f}%",
        flush=True,
    )

    out = args.json_out
    if out is None:
        out = manifest_path.parent / f"length_stats_ml{args.max_length}.json"
    else:
        out = out if out.is_absolute() else (ROOT / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
