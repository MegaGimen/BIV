#!/usr/bin/env python3
"""Truncation stats for Stage 1 LLM-JEPA (AgentWorld tokenizer, mix JSONL).

Same three sequences as train_jepallm.py: full = chat(h+a+o), left = chat(h+a),
right = chat(o). Lengths are counted *before* the 65k fit. Retention is
min(L, seqlen)/L; full/left keep the tail, right keeps the head.

  cd train
  python scripts/stat.py
  python scripts/stat.py --max-length 65536
  python scripts/stat.py --max-length 65536 --max-samples 2000

Lengths are cached under train/outputs/stat_cache/jepallm/ (untruncated
full/left/right counts, keyed by row content + tokenizer). Rerun hits disk.
--recompute ignores the cache. --no-cache does not read or write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SCRIPTS = Path(__file__).resolve().parent
SRC = TRAIN / "src"
MERGE = ROOT / "merge"
for p in (str(SCRIPTS), str(SRC), str(MERGE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_jepallm as tj  # noqa: E402
from download import resolve_model  # noqa: E402

DEFAULT_CONFIG = TRAIN / "configs" / "jepa" / "jepallm.yaml"
DEFAULT_STAT_CACHE = TRAIN / "outputs" / "stat_cache" / "jepallm"
RECIPE = "jepallm-seqlen-v1"
SEQS = ("full", "left", "right")
LENGTH_EDGES = [2048, 4096, 8192, 16384, 32768, 65536]
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
    counts = [0] * (len(edges) + 1)
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


def _retention(lengths: list[int], seqlen: int) -> list[float]:
    out: list[float] = []
    for x in lengths:
        out.append(1.0 if x <= 0 else min(1.0, seqlen / float(x)))
    return out


def _print_seq(src: str, seq: str, lengths: list[int], seqlen: int) -> dict[str, Any]:
    n = len(lengths)
    ratios = _retention(lengths, seqlen)
    keep = sum(1 for x in lengths if x <= seqlen)
    kept_tokens = sum(min(x, seqlen) for x in lengths)
    total_tokens = sum(lengths)
    keep_how = "suffix (tail)" if seq in ("full", "left") else "prefix (head)"

    print(f"\n=== {src} / {seq}  n={n:,}  keep={keep_how} ===", flush=True)
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
    print("  length histogram:", flush=True)
    length_hist = _hist_int(lengths, LENGTH_EDGES)
    for label, c in length_hist:
        _print_bar(label, c, n)

    print(f"  hard-trunc retention @ seqlen={seqlen}:", flush=True)
    print(
        f"  ratio min={min(ratios):.4f}  max={max(ratios):.4f}  "
        f"mean={statistics.mean(ratios):.4f}  median={_pct(ratios, 50):.4f}",
        flush=True,
    )
    print(
        f"  token-mass kept: {kept_tokens:,} / {total_tokens:,} "
        f"({100 * kept_tokens / max(total_tokens, 1):.1f}%)",
        flush=True,
    )
    print(
        f"  rows fully in window (L<=seqlen): {keep:,} / {n:,} "
        f"({100 * keep / n:5.1f}%)",
        flush=True,
    )
    print("  retention-ratio histogram:", flush=True)
    ratio_hist = _hist_ratio(ratios, RATIO_EDGES)
    for label, c in ratio_hist:
        _print_bar(label, c, n)

    return {
        "n": n,
        "keep": keep_how,
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
        "retention": {
            "ratio_mean": statistics.mean(ratios),
            "ratio_p50": _pct(ratios, 50),
            "token_mass_kept": kept_tokens,
            "token_mass_total": total_tokens,
            "token_mass_kept_pct": kept_tokens / max(total_tokens, 1),
            "rows_in_window": keep,
            "rows_in_window_pct": keep / n,
            "histogram": {lab: c for lab, c in ratio_hist},
        },
    }


def _row_key(h, a, o) -> str:
    blob = json.dumps(
        {"h": h, "a": a, "o": o},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tok_slug(model_dir: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(model_dir))[-120:]


class LengthCache:
    """Append-only jsonl of {k, full, left, right} for one mix source."""

    def __init__(self, path: Path, *, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.store: dict[str, dict[str, int]] = {}
        if enabled and path.is_file():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    k = obj.get("k")
                    if not k:
                        continue
                    self.store[str(k)] = {
                        "full": int(obj["full"]),
                        "left": int(obj["left"]),
                        "right": int(obj["right"]),
                    }

    def get(self, key: str) -> dict[str, int] | None:
        if not self.enabled:
            return None
        return self.store.get(key)

    def put(self, key: str, lens: dict[str, int]) -> None:
        rec = {s: int(lens[s]) for s in SEQS}
        self.store[key] = rec
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, **rec}, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--max-length",
        "--seqlen",
        type=int,
        default=65536,
        dest="max_length",
        help="Training window (default 65536).",
    )
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap rows across sources (reservoir). Default: all train rows.",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Model download cache. Default: merge/output/cache",
    )
    p.add_argument(
        "--stat-cache",
        type=Path,
        default=None,
        help=f"Length jsonl cache. Default: {DEFAULT_STAT_CACHE.relative_to(TRAIN)}",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the length cache.",
    )
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore existing length cache and overwrite it.",
    )
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=None,
        help="Tokenizer download if AgentWorld is not cached.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise SystemExit("--max-length must be a positive integer")

    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = tj._load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = tj.resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    cache_dir = tj._resolve(args.cache_dir or "merge/output/cache", ROOT)
    model_dir = resolve_model(
        str(args.model_dir or cfg["model_dir"]),
        source=args.source or "modelscope",
        cache_dir=cache_dir,
        role="world",
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = int(1e12)

    seqlen = int(args.max_length)
    limit = args.max_samples
    use_cache = not args.no_cache
    stat_root = args.stat_cache
    if stat_root is None:
        stat_root = DEFAULT_STAT_CACHE
    elif not stat_root.is_absolute():
        stat_root = TRAIN / stat_root
    cache_ns = stat_root / _tok_slug(model_dir)
    if use_cache:
        cache_ns.mkdir(parents=True, exist_ok=True)
        man = cache_ns / "manifest.json"
        meta = {"recipe": RECIPE, "model_dir": str(model_dir), "mix_dir": str(mix_dir)}
        if man.is_file():
            old = json.loads(man.read_text(encoding="utf-8"))
            if old.get("recipe") != RECIPE or old.get("model_dir") != str(model_dir):
                print(
                    f"stat cache recipe/tokenizer mismatch at {cache_ns}; using it as a new namespace",
                    flush=True,
                )
        man.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.recompute:
            for src in sources:
                pth = cache_ns / f"{src}.jsonl"
                if pth.is_file():
                    pth.unlink()
            print(f"stat cache: recompute, cleared jsonl under {cache_ns}", flush=True)

    print(f"tokenizer: {model_dir}", flush=True)
    print(f"mix:       {mix_dir}  sources={sources}", flush=True)
    print(f"seqlen:    {seqlen}", flush=True)
    if use_cache:
        print(f"stat cache: {cache_ns}  recipe={RECIPE}", flush=True)
    else:
        print("stat cache: off", flush=True)
    if limit is not None:
        print(f"rows:      max-samples={limit} (reservoir per source)", flush=True)

    report: dict[str, Any] = {
        "seqlen": seqlen,
        "mix_dir": str(mix_dir),
        "model_dir": str(model_dir),
        "sources": {},
        "note": (
            "lengths are untruncated AgentWorld chat-template token counts; "
            "full/left keep suffix, right keeps prefix when L>seqlen"
        ),
    }

    for src in sources:
        rows = tj.load_rows(mix_dir, [src], "train", limit)
        if not rows:
            print(f"WARNING: no rows for {src}", flush=True)
            continue
        buckets = {k: [] for k in SEQS}
        cache = LengthCache(
            cache_ns / f"{src}.jsonl",
            enabled=use_cache,
        )
        hits = misses = 0
        bar = _tqdm(rows, desc=f"tokenize {src}", unit="row")
        for h, a, o in bar:
            key = _row_key(h, a, o)
            lens = cache.get(key)
            if lens is None:
                lens = tj.sequence_lengths(tok, h, a, o)
                cache.put(key, lens)
                misses += 1
            else:
                hits += 1
            for k in SEQS:
                buckets[k].append(lens[k])
            if bar is not None and hasattr(bar, "set_postfix"):
                bar.set_postfix(hit=hits, miss=misses, refresh=False)
        print(f"  {src} cache hit={hits:,} miss={misses:,} stored={len(cache.store):,}", flush=True)
        report["sources"][src] = {}
        for seq in SEQS:
            report["sources"][src][seq] = _print_seq(src, seq, buckets[seq], seqlen)

    print("\n=== mix summary @ seqlen (rows fully in window) ===", flush=True)
    for seq in SEQS:
        parts = []
        keep = n = 0
        mass_k = mass_t = 0
        for src in sources:
            st = (report["sources"].get(src) or {}).get(seq)
            if not st:
                continue
            r = st["retention"]
            parts.append(f"{src}={r['rows_in_window']:,}/{st['n']:,}")
            keep += r["rows_in_window"]
            n += st["n"]
            mass_k += r["token_mass_kept"]
            mass_t += r["token_mass_total"]
        if n:
            print(
                f"  {seq}: {', '.join(parts)} | "
                f"{keep:,}/{n:,} rows  token-mass {100 * mass_k / max(mass_t, 1):.1f}%",
                flush=True,
            )

    out = args.json_out
    if out is None:
        out_dir = tj._resolve(tcfg.get("output_dir") or "outputs/jepallm_stage1")
        out = out_dir / f"length_stats_ml{seqlen}.json"
    else:
        out = out if out.is_absolute() else (TRAIN / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
