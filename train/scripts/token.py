#!/usr/bin/env python3
"""Count tokens in prepared SWE-Hero WM JSONL (for cost estimates).

Uses the same ChatML rendering as training (`apply_chat_template` on `messages`).
Does **not** require GPU / Unsloth — only `transformers` + a tokenizer.

Examples:
  cd train
  python scripts/token.py
  python scripts/token.py --processed-dir data/processed --splits train,eval
  python scripts/token.py --model Qwen/Qwen3.5-9B --max-length 8192 --epochs 2
  python scripts/token.py --price-per-k 0.02   # Bailian-style ¥ / 1k tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("pip install transformers") from exc

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Qwen VL processor wrappers sometimes appear; unwrap if needed.
    if not hasattr(tok, "encode") and hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    return tok


def _row_text(row: dict, tokenizer) -> str:
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("row missing messages/text")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _count_file(
    path: Path,
    tokenizer,
    *,
    max_length: int | None,
    log_every: int,
) -> dict:
    n_rows = 0
    total = 0
    supervised_hint = 0  # assistant spans not tokenized separately; skip
    truncated_rows = 0
    truncated_overflow = 0
    max_seen = 0
    lengths: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = _row_text(row, tokenizer)
            ids = tokenizer.encode(text, add_special_tokens=False)
            n = len(ids)
            n_rows += 1
            total += n
            max_seen = max(max_seen, n)
            lengths.append(n)
            if max_length is not None and n > max_length:
                truncated_rows += 1
                truncated_overflow += n - max_length
            if log_every and n_rows % log_every == 0:
                print(
                    f"  {path.name}: {n_rows} rows  tokens={total:,}  "
                    f"last_len={n}  max={max_seen}",
                    flush=True,
                )

    billable = total
    if max_length is not None:
        # Bailian-style: drop rows longer than max_length from training bill
        billable = 0
        kept = 0
        for n in lengths:
            if n <= max_length:
                billable += n
                kept += 1
        dropped = n_rows - kept
    else:
        dropped = 0
        kept = n_rows

    mean = (total / n_rows) if n_rows else 0.0
    return {
        "path": str(path),
        "n_rows": n_rows,
        "token_sum": total,
        "token_mean": mean,
        "token_max": max_seen,
        "max_length": max_length,
        "rows_kept_under_max": kept,
        "rows_over_max": dropped if max_length else truncated_rows,
        "billable_tokens_if_drop_over_max": billable if max_length else total,
        "supervised_hint": supervised_hint,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory with train.jsonl / eval.jsonl",
    )
    p.add_argument(
        "--splits",
        default="train,eval",
        help="Comma-separated split names (files: <name>.jsonl)",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Tokenizer source (same ChatML as training)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Report truncation stats; billable excludes rows longer than this "
        "(set 0 to disable)",
    )
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument(
        "--price-per-k",
        type=float,
        default=None,
        help="Optional ¥ per 1k tokens × epochs (e.g. 0.02 Bailian text SFT)",
    )
    p.add_argument("--log-every", type=int, default=2000)
    args = p.parse_args()

    processed = args.processed_dir if args.processed_dir.is_absolute() else (ROOT / args.processed_dir)
    max_length = None if not args.max_length else int(args.max_length)

    print(f"Tokenizer: {args.model}", flush=True)
    tokenizer = _load_tokenizer(args.model)

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    per_split: list[dict] = []
    grand_total = 0
    grand_billable = 0
    grand_rows = 0

    for name in split_names:
        path = processed / f"{name}.jsonl"
        if not path.is_file():
            print(f"SKIP missing {path}", flush=True)
            continue
        print(f"Counting {path} ...", flush=True)
        stats = _count_file(
            path, tokenizer, max_length=max_length, log_every=args.log_every
        )
        per_split.append(stats)
        grand_total += stats["token_sum"]
        grand_billable += stats["billable_tokens_if_drop_over_max"]
        grand_rows += stats["n_rows"]
        print(
            f"  -> rows={stats['n_rows']:,}  tokens={stats['token_sum']:,}  "
            f"mean={stats['token_mean']:.1f}  max={stats['token_max']:,}",
            flush=True,
        )

    if not per_split:
        raise SystemExit(f"No jsonl found under {processed} for splits={split_names}")

    summary = {
        "processed_dir": str(processed),
        "model_tokenizer": args.model,
        "splits": per_split,
        "total_rows": grand_rows,
        "total_tokens": grand_total,
        "billable_tokens_drop_over_max": grand_billable,
        "max_length": max_length,
        "epochs": args.epochs,
        "token_epochs": grand_billable * args.epochs,
    }
    if args.price_per_k is not None:
        # ¥ = (tokens/1000) * price * epochs
        fee = (grand_billable / 1000.0) * args.price_per_k * args.epochs
        summary["price_per_k_cny"] = args.price_per_k
        summary["est_fee_cny"] = round(fee, 4)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(
        f"\nSummary: {grand_rows:,} rows | {grand_total:,} tokens"
        + (
            f" | billable≤{max_length}: {grand_billable:,} "
            f"| ×{args.epochs:g} epoch → {grand_billable * args.epochs:,.0f} token·epoch"
            if max_length
            else ""
        ),
        flush=True,
    )
    if args.price_per_k is not None:
        print(
            f"Est. fee @ ¥{args.price_per_k}/kTok × {args.epochs:g} epoch: "
            f"¥{summary['est_fee_cny']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
