#!/usr/bin/env python3
"""Print the first (or N-th) chat sample from mix JSONL in readable turn layout.

Examples:
  python scripts/debug.py
  python scripts/debug.py --mix-dir data/processed/mix_v2 --source wm_code
  python scripts/debug.py --source wm_os --index 0
  python scripts/debug.py --source anti_forget --json   # also dump raw JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v2"
FALLBACK_MIX = ROOT / "data" / "processed" / "mix_v1"
SOURCES = ("wm_code", "wm_os", "anti_forget")


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p)


def _default_mix() -> Path:
    if DEFAULT_MIX.is_dir():
        return DEFAULT_MIX
    return FALLBACK_MIX


def _load_nth_row(path: Path, index: int) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as f:
        n = -1
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            n += 1
            if n < index:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON at row {n} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Row {n} is not an object")
            return obj
    raise SystemExit(f"Only found {n + 1} rows in {path}; --index {index} out of range")


def _fmt_block(role: str, content: str | None, extra: str | None = None) -> str:
    bar = "=" * 72
    head = f"{bar}\n[{role}]"
    if extra:
        head += f"  {extra}"
    body = content if content is not None else ""
    # Pretty-print if content is JSON object/array
    if isinstance(body, str) and body.strip()[:1] in "{[":
        try:
            parsed = json.loads(body)
            body = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return f"{head}\n{body}\n"


def _print_messages(messages: list, *, show_tools: bool) -> None:
    if not messages:
        print("(empty messages)", flush=True)
        return
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            print(f"--- msg[{i}] (non-dict): {m!r}\n", flush=True)
            continue
        role = str(m.get("role", "?"))
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        extra_bits: list[str] = []
        tool_calls = m.get("tool_calls")
        if show_tools and tool_calls:
            extra_bits.append(f"tool_calls×{len(tool_calls)}")
        tid = m.get("tool_call_id")
        if tid:
            extra_bits.append(f"tool_call_id={tid}")
        name = m.get("name")
        if name and role == "tool":
            extra_bits.append(f"name={name}")

        print(_fmt_block(role, content, " | ".join(extra_bits) if extra_bits else None), flush=True)

        if show_tools and tool_calls:
            print("- tool_calls -", flush=True)
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2), flush=True)
            print("", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix-dir", type=Path, default=None, help="default: mix_v2 if exists else mix_v1")
    p.add_argument("--source", choices=SOURCES, default="wm_code")
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument("--index", type=int, default=0, help="0-based row index")
    p.add_argument("--json", action="store_true", help="also print full raw JSON object")
    p.add_argument(
        "--no-tool-details",
        action="store_true",
        help="for anti_forget: hide expanded tool_calls dump",
    )
    p.add_argument(
        "--path",
        type=Path,
        default=None,
        help="explicit JSONL path (overrides mix-dir/source/split)",
    )
    args = p.parse_args()

    if args.path:
        path = _resolve(args.path)
    else:
        mix = _resolve(args.mix_dir) if args.mix_dir else _default_mix()
        path = mix / args.source / f"{args.split}.jsonl"

    row = _load_nth_row(path, args.index)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise SystemExit(f"Row has no messages list (keys={list(row.keys())})")

    print(f"file:    {path}", flush=True)
    print(f"index:   {args.index}", flush=True)
    meta = {k: row[k] for k in ("source", "n_turns", "instance_id", "trajectory_id") if k in row}
    if meta:
        print(f"meta:    {json.dumps(meta, ensure_ascii=False)}", flush=True)
    print(f"turns:   {len(messages)} messages", flush=True)
    print("", flush=True)

    _print_messages(messages, show_tools=not args.no_tool_details)

    if args.json:
        print("=" * 72, flush=True)
        print("[raw JSON]", flush=True)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        sys.exit(130)
