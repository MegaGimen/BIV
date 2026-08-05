#!/usr/bin/env python3
"""Count tokens from train_sft ds_cache (pretokenized train_ready).

Uses the same local ChatML token ids as training — no transformers download.
Prefers `outputs/ds_cache/*/train_ready` (HF Dataset with input_ids).

Fallback: `--from-jsonl` re-tokenizes processed JSONL (needs transformers).

Examples:
  cd train
  python scripts/token.py
  python scripts/token.py --cache-dir outputs/ds_cache/v3_xxxxxxxxxxxx
  python scripts/token.py --epochs 2 --price-per-k 0.02
  python scripts/token.py --from-jsonl --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _iter_ready_dirs(cache_root: Path) -> list[Path]:
    out: list[Path] = []
    if not cache_root.exists():
        return out
    for meta in sorted(cache_root.rglob("meta.json")):
        ready = meta.parent / "train_ready"
        if ready.is_dir():
            out.append(meta.parent)
    return out


def _pick_cache_dir(explicit: Path | None, cache_root: Path) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else (ROOT / explicit)
        if not (p / "train_ready").is_dir():
            raise SystemExit(f"No train_ready under {p}")
        return p
    cands = _iter_ready_dirs(cache_root)
    if not cands:
        raise SystemExit(
            f"No ds_cache under {cache_root}.\n"
            "Build with: python scripts/train_sft.py --config configs/default.yaml\n"
            "Or: python scripts/token.py --from-jsonl\n"
            "Or: --cache-dir path/to/v3_<fingerprint>"
        )

    def _size(d: Path) -> int:
        tr = d / "train_ready"
        return sum(f.stat().st_size for f in tr.rglob("*") if f.is_file())

    best = max(cands, key=_size)
    rel = best.relative_to(ROOT) if best.is_relative_to(ROOT) else best
    print(f"Auto-picked cache: {rel}", flush=True)
    return best


def _count_train_ready(
    train_ready: Path,
    *,
    max_length: int | None,
    scan_rows: int | None,
) -> dict:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Missing datasets. pip install datasets") from exc

    ds = load_from_disk(str(train_ready))
    if "input_ids" not in ds.column_names:
        raise SystemExit(f"{train_ready} missing input_ids; columns={ds.column_names}")

    n = len(ds)
    if scan_rows is not None and scan_rows > 0 and scan_rows < n:
        ds = ds.select(range(scan_rows))
        print(f"Scanning first {scan_rows} / {n} rows (--scan-rows)", flush=True)
        n = len(ds)

    has_mask = "attention_mask" in ds.column_names
    has_labels = "labels" in ds.column_names

    total = 0
    billable = 0
    supervised = 0
    dropped_rows = 0
    max_seen = 0
    batch_size = 1024

    for start in range(0, n, batch_size):
        batch = ds[start : start + batch_size]
        ids_batch = batch["input_ids"]
        mask_batch = batch["attention_mask"] if has_mask else None
        labels_batch = batch["labels"] if has_labels else None
        for i, ids in enumerate(ids_batch):
            if mask_batch is not None:
                tok = int(sum(1 for x in mask_batch[i] if x))
            else:
                tok = len(ids)
            total += tok
            max_seen = max(max_seen, tok)
            if max_length is not None and tok > max_length:
                dropped_rows += 1
                continue
            billable += tok
            if labels_batch is not None:
                supervised += sum(1 for t in labels_batch[i] if t != -100)
        if (start // batch_size) % 20 == 0:
            print(f"  … {min(start + batch_size, n):,}/{n:,} rows", flush=True)

    kept = n - dropped_rows
    return {
        "n_rows": n,
        "total_tokens": total,
        "billable_tokens": billable if max_length else total,
        "supervised_tokens": supervised,
        "dropped_rows": dropped_rows,
        "max_seen": max_seen,
        "mean_tokens": (billable / kept) if kept else 0.0,
        "max_length": max_length,
        "columns": list(ds.column_names),
    }


def _count_jsonl(
    processed: Path,
    splits: list[str],
    model_id: str,
    *,
    max_length: int | None,
) -> dict:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001 — show real import failure
        raise SystemExit(
            f"Cannot import transformers AutoTokenizer ({exc!r}).\n"
            "Prefer ds_cache instead: python scripts/token.py\n"
            "Or fix venv / use same env as train_sft."
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(tok, "encode") and hasattr(tok, "tokenizer"):
        tok = tok.tokenizer

    grand_total = 0
    grand_billable = 0
    grand_rows = 0
    max_seen = 0
    dropped = 0
    per: list[dict] = []

    for name in splits:
        path = processed / f"{name}.jsonl"
        if not path.is_file():
            print(f"SKIP missing {path}", flush=True)
            continue
        rows = 0
        total = 0
        billable = 0
        drop = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row.get("text"), str) and row["text"].strip():
                    text = row["text"]
                else:
                    text = tok.apply_chat_template(
                        row["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                n = len(tok.encode(text, add_special_tokens=False))
                rows += 1
                total += n
                max_seen = max(max_seen, n)
                if max_length is not None and n > max_length:
                    drop += 1
                else:
                    billable += n
        per.append({"split": name, "rows": rows, "tokens": total, "billable": billable})
        grand_rows += rows
        grand_total += total
        grand_billable += billable
        dropped += drop
        print(f"  {name}: rows={rows:,} tokens={total:,}", flush=True)

    return {
        "n_rows": grand_rows,
        "total_tokens": grand_total,
        "billable_tokens": grand_billable,
        "supervised_tokens": 0,
        "dropped_rows": dropped,
        "max_seen": max_seen,
        "mean_tokens": (grand_billable / (grand_rows - dropped)) if (grand_rows - dropped) else 0.0,
        "max_length": max_length,
        "splits": per,
        "source": "jsonl",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=ROOT / "outputs" / "ds_cache")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=8192, help="0 = no drop filter")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--price-per-k", type=float, default=None, help="¥ per 1k tokens")
    p.add_argument("--scan-rows", type=int, default=None)
    p.add_argument(
        "--from-jsonl",
        action="store_true",
        help="Re-tokenize processed JSONL (needs transformers); default is ds_cache",
    )
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = p.parse_args()

    max_length = None if not args.max_length else int(args.max_length)

    if args.from_jsonl:
        processed = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (ROOT / args.processed_dir)
        )
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        print(f"Source: JSONL under {processed}", flush=True)
        stats = _count_jsonl(processed, splits, args.model, max_length=max_length)
    else:
        cache_root = (
            args.cache_root if args.cache_root.is_absolute() else (ROOT / args.cache_root)
        )
        cache_dir = _pick_cache_dir(args.cache_dir, cache_root)
        meta_path = cache_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        print(f"Source: {cache_dir / 'train_ready'}", flush=True)
        if meta:
            print(
                f"meta: model={meta.get('model_name')} max_seq={meta.get('max_seq_length')} "
                f"packing={meta.get('packing')}",
                flush=True,
            )
        stats = _count_train_ready(
            cache_dir / "train_ready",
            max_length=max_length,
            scan_rows=args.scan_rows,
        )
        stats["cache_dir"] = str(cache_dir)
        stats["meta"] = meta

    token_epochs = stats["billable_tokens"] * args.epochs
    out = {
        **stats,
        "epochs": args.epochs,
        "token_epochs": token_epochs,
    }
    if args.price_per_k is not None:
        fee = (stats["billable_tokens"] / 1000.0) * args.price_per_k * args.epochs
        out["price_per_k_cny"] = args.price_per_k
        out["est_fee_cny"] = round(fee, 4)

    print(json.dumps(out, indent=2, ensure_ascii=False, default=str), flush=True)
    print(
        f"\nSummary: {stats['n_rows']:,} rows | total={stats['total_tokens']:,} | "
        f"billable={stats['billable_tokens']:,}"
        + (f" (drop>{max_length}: {stats['dropped_rows']:,})" if max_length else "")
        + f" | ×{args.epochs:g} → {token_epochs:,.0f} token·epoch"
        + (
            f" | supervised(labels≠-100)={stats['supervised_tokens']:,}"
            if stats.get("supervised_tokens")
            else ""
        ),
        flush=True,
    )
    if args.price_per_k is not None:
        print(f"Est. fee: ¥{out['est_fee_cny']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
