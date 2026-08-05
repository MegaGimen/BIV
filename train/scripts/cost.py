#!/usr/bin/env python3
"""Estimate Bailian (DashScope) SFT training cost from a local ds_cache.

Bailian formula (text SFT, Qwen3.5-9B list price):
  费用 = (训练 Token + 混合训练 Token) × n_epochs × 单价
  单价 default: ¥0.02 / 千 Token  for model code qwen3.5-9b
  Docs: https://help.aliyun.com/zh/model-studio/model-training-and-deployment-billing

Token counting (from pretokenized HF Dataset under */train_ready):
  - billable ≈ sum(attention_mask) if present, else sum(len(input_ids))
  - Bailian SFT drops rows whose token length > max_length; those are excluded
    from the billable sum and reported separately.
  - Also print supervised label tokens (labels != -100) for reference; Bailian
    bills on 训练数据 Token 总数 (sequence tokens), not response-only loss tokens.

Mixed training (data_augmentation on Bailian):
  Platform mixes in preset corpora (对话/通用/代码/NLP…) at ratios you set
  (augmentation_ratio, 0~2.0 per type). Those extra tokens ARE billed.
  mix_tokens ≈ train_tokens * sum(ratios). Offline estimate only.

Examples:
  cd train
  python scripts/cost.py
  python scripts/cost.py --cache-dir outputs/ds_cache/v3_xxxxxxxxxxxx
  python scripts/cost.py --epochs 1 --max-length 8192 --price-per-k 0.02
  python scripts/cost.py --mix-ratios 0.1,0.05,0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TokenStats:
    n_rows: int
    billable_tokens: int
    supervised_tokens: int
    dropped_rows: int
    dropped_tokens: int
    max_seen: int
    mean_billable: float


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
            f"No ds_cache found under {cache_root}.\n"
            "Build one with: python scripts/train_sft.py --config configs/pilot.yaml\n"
            "Or pass: --cache-dir path/to/.../v3_<fingerprint>"
        )
    # Prefer largest train_ready (bytes) as "main" corpus cache.
    def _size(d: Path) -> int:
        tr = d / "train_ready"
        return sum(f.stat().st_size for f in tr.rglob("*") if f.is_file())

    best = max(cands, key=_size)
    print(f"Auto-picked cache: {best.relative_to(ROOT) if best.is_relative_to(ROOT) else best}")
    return best


def _load_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _count_tokens(train_ready: Path, *, max_length: int, scan_rows: int | None) -> TokenStats:
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

    billable = 0
    supervised = 0
    dropped_rows = 0
    dropped_tokens = 0
    max_seen = 0

    # Batched iteration is much faster than per-row Python loops on Arrow.
    batch_size = 1024
    for start in range(0, n, batch_size):
        batch = ds[start : start + batch_size]
        ids_batch = batch["input_ids"]
        mask_batch = batch["attention_mask"] if has_mask else None
        labels_batch = batch["labels"] if has_labels else None
        for i, ids in enumerate(ids_batch):
            if mask_batch is not None:
                m = mask_batch[i]
                tok = int(sum(1 for x in m if x))
            else:
                tok = len(ids)
            max_seen = max(max_seen, tok)
            if tok > max_length:
                dropped_rows += 1
                dropped_tokens += tok
                continue
            billable += tok
            if labels_batch is not None:
                lab = labels_batch[i]
                supervised += sum(1 for t in lab if t != -100)

    kept = n - dropped_rows
    mean = (billable / kept) if kept else 0.0
    return TokenStats(
        n_rows=n,
        billable_tokens=billable,
        supervised_tokens=supervised,
        dropped_rows=dropped_rows,
        dropped_tokens=dropped_tokens,
        max_seen=max_seen,
        mean_billable=mean,
    )


def _parse_mix_ratios(raw: str | None) -> list[float]:
    if not raw:
        return []
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v < 0 or v > 2.0:
            raise SystemExit(f"augmentation ratio {v} out of Bailian range [0, 2.0]")
        out.append(v)
    return out


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n:,} ({n / 1e9:.3f}B)"
    if n >= 1_000_000:
        return f"{n:,} ({n / 1e6:.2f}M)"
    return f"{n:,}"


def _estimate_cny(
    *,
    train_tokens: int,
    mix_ratios: list[float],
    epochs: float,
    price_per_k: float,
) -> dict:
    ratio_sum = sum(mix_ratios)
    mix_tokens = int(round(train_tokens * ratio_sum))
    total_per_epoch = train_tokens + mix_tokens
    billed = total_per_epoch * epochs
    cost = (billed / 1000.0) * price_per_k
    return {
        "train_tokens": train_tokens,
        "mix_ratio_sum": ratio_sum,
        "mix_tokens": mix_tokens,
        "tokens_per_epoch": total_per_epoch,
        "epochs": epochs,
        "billed_tokens": billed,
        "price_per_k_cny": price_per_k,
        "cost_cny": cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs" / "ds_cache",
        help="Shared ds_cache root (default outputs/ds_cache)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Specific variant dir containing train_ready/ + meta.json",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Bailian hyper_parameters.max_length; longer rows are dropped (SFT)",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Bailian hyper_parameters.n_epochs",
    )
    parser.add_argument(
        "--price-per-k",
        type=float,
        default=0.02,
        help="CNY per 1000 tokens (qwen3.5-9b list: 0.02)",
    )
    parser.add_argument(
        "--model-code",
        type=str,
        default="qwen3.5-9b",
        help="Bailian model code (for printing only)",
    )
    parser.add_argument(
        "--mix-ratios",
        type=str,
        default="",
        help=(
            "Bailian data_augmentation ratios, comma-separated, matching "
            "augmentation_types order. Example: 0.1,0.05,0.15 → mix ~30%% extra"
        ),
    )
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=None,
        help="Optional cap on rows scanned (smoke / speed)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write estimate JSON to this path",
    )
    args = parser.parse_args()

    cache_root = args.cache_root if args.cache_root.is_absolute() else (ROOT / args.cache_root)
    cache_dir = _pick_cache_dir(args.cache_dir, cache_root)
    meta = _load_meta(cache_dir)
    mix_ratios = _parse_mix_ratios(args.mix_ratios or None)

    print("=" * 60)
    print("Bailian cost estimate (local ds_cache)")
    print("=" * 60)
    print(f"cache_dir:     {cache_dir}")
    if meta:
        print(f"meta.packing:  {meta.get('packing')}")
        print(f"meta.max_seq:  {meta.get('max_seq_length')}")
        print(f"meta.max_train:{meta.get('max_train_samples')}")
        print(f"meta.model:    {meta.get('model_name')}")
    print(f"bailian.model: {args.model_code}")
    print(f"max_length:    {args.max_length}")
    print(f"n_epochs:      {args.epochs}")
    print(f"price:         ¥{args.price_per_k}/千Token")
    if mix_ratios:
        print(f"mix ratios:    {mix_ratios} (sum={sum(mix_ratios):.4f})")
    else:
        print("mix ratios:    (off) — set --mix-ratios if enabling data_augmentation")

    stats = _count_tokens(
        cache_dir / "train_ready",
        max_length=args.max_length,
        scan_rows=args.scan_rows,
    )
    est = _estimate_cny(
        train_tokens=stats.billable_tokens,
        mix_ratios=mix_ratios,
        epochs=args.epochs,
        price_per_k=args.price_per_k,
    )

    print("-" * 60)
    print(f"rows scanned:       {stats.n_rows:,}")
    print(f"rows dropped (>L):  {stats.dropped_rows:,}  (excluded from billable)")
    print(f"max seq tokens:     {stats.max_seen:,}")
    print(f"billable tokens:    {_fmt_tokens(stats.billable_tokens)}")
    print(f"mean tokens/row:    {stats.mean_billable:.1f}")
    if stats.supervised_tokens:
        print(
            f"supervised tokens:  {_fmt_tokens(stats.supervised_tokens)}  "
            "(labels≠-100; NOT the Bailian bill base)"
        )
    print("-" * 60)
    print(f"mix tokens:         {_fmt_tokens(est['mix_tokens'])}")
    print(f"tokens × epochs:    {_fmt_tokens(int(est['billed_tokens']))}")
    print(f"ESTIMATED COST:     ¥{est['cost_cny']:,.2f}")
    print("=" * 60)
    print(
        "Caveats: Bailian may tokenize slightly differently from local Qwen "
        "ChatML cache; console「预估训练费用」is authoritative. "
        "Price list can change — verify billing docs before budget sign-off.",
        flush=True,
    )

    payload = {
        "cache_dir": str(cache_dir),
        "meta": meta,
        "stats": stats.__dict__,
        "estimate": est,
        "model_code": args.model_code,
        "max_length": args.max_length,
        "mix_ratios": mix_ratios,
        "note": (
            "cost = (train_tokens + train_tokens*sum(mix_ratios)) * epochs "
            f"* ({args.price_per_k}/1000) CNY"
        ),
    }
    if args.json_out is not None:
        out = args.json_out if args.json_out.is_absolute() else (ROOT / args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
