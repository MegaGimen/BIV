#!/usr/bin/env python3
"""Patch prepared WM JSONL in-place: drop chat wrappers without re-prepare.

Removes from ``role=user`` contents (wm_code / wm_os only):
  - ``Previous tool steps in this session: <n>.\\n``
  - ``Latest tool call:\\n``

Does **not** touch anti_forget (native agent/tool format).

After ``--apply``, re-export tokenize cache so tokens match the JSONL::

  python scripts/delta.py --mix-dir data/processed/mix_v1 --apply
  python scripts/tokenize_data.py --force

Examples:
  python scripts/delta.py --dry-run
  python scripts/delta.py --apply
  python scripts/delta.py --apply --also-sampled
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v1"
WM_SOURCES = ("wm_code", "wm_os")
SPLITS = ("train", "eval")

# Order matters: strip step-count first, then Latest tool call.
_RE_PREV_STEPS = re.compile(
    r"^Previous tool steps in this session: \d+\.\n",
)
_RE_LATEST = re.compile(r"^Latest tool call:\n")


def clean_user_content(text: str) -> tuple[str, bool]:
    if not isinstance(text, str) or not text:
        return text, False
    new = text
    new2 = _RE_PREV_STEPS.sub("", new)
    new3 = _RE_LATEST.sub("", new2)
    # Defensive: wrappers sometimes mid-string after odd joins
    if "Latest tool call:\n" in new3:
        new3 = new3.replace("Latest tool call:\n", "")
    if new3 != text:
        return new3, True
    return text, False


def clean_row(obj: dict) -> tuple[dict, int]:
    """Return (row, n_user_msgs_changed)."""
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return obj, 0
    changed = 0
    new_msgs = []
    for m in msgs:
        if not isinstance(m, dict):
            new_msgs.append(m)
            continue
        if m.get("role") != "user":
            new_msgs.append(m)
            continue
        content = m.get("content")
        cleaned, ok = clean_user_content(content if isinstance(content, str) else content)
        if ok:
            m = dict(m)
            m["content"] = cleaned
            changed += 1
        new_msgs.append(m)
    if changed:
        out = dict(obj)
        out["messages"] = new_msgs
        return out, changed
    return obj, 0


def _iter_targets(mix_dir: Path, *, also_sampled: bool) -> list[Path]:
    paths: list[Path] = []
    for src in WM_SOURCES:
        for split in SPLITS:
            p = mix_dir / src / f"{split}.jsonl"
            if p.is_file():
                paths.append(p)
    if also_sampled:
        sampled = mix_dir / "sampled"
        if sampled.is_dir():
            for src in WM_SOURCES:
                for p in sorted(sampled.glob(f"*/{src}/train.jsonl")):
                    paths.append(p)
    return paths


def _patch_file(path: Path, *, apply: bool, backup: bool) -> dict:
    stats = {
        "path": str(path),
        "rows": 0,
        "rows_changed": 0,
        "user_msgs_changed": 0,
        "applied": False,
    }
    if not apply:
        # Stream count only
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["rows"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _, n = clean_row(obj)
                if n:
                    stats["rows_changed"] += 1
                    stats["user_msgs_changed"] += n
        return stats

    tmp = path.with_suffix(path.suffix + ".delta_tmp")
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.is_file():
            shutil.copy2(path, bak)

    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            stats["rows"] += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                fout.write(raw + "\n")
                continue
            new_obj, n = clean_row(obj)
            if n:
                stats["rows_changed"] += 1
                stats["user_msgs_changed"] += n
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            else:
                fout.write(raw + "\n")

    tmp.replace(path)
    stats["applied"] = True
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix-dir", type=Path, default=DEFAULT_MIX)
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default is dry-run)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report (default if --apply not set)",
    )
    p.add_argument(
        "--also-sampled",
        action="store_true",
        help="Also patch mix_dir/sampled/*/wm_*/train.jsonl",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .jsonl.bak on first apply",
    )
    args = p.parse_args()
    apply = bool(args.apply) and not bool(args.dry_run)

    mix_dir = args.mix_dir if args.mix_dir.is_absolute() else (ROOT / args.mix_dir)
    if not mix_dir.is_dir():
        raise SystemExit(f"mix-dir not found: {mix_dir}")

    targets = _iter_targets(mix_dir, also_sampled=args.also_sampled)
    if not targets:
        raise SystemExit(
            f"No wm_code/wm_os JSONL under {mix_dir}. "
            "Run prepare_data first (once), then delta."
        )

    print(f"mix-dir: {mix_dir}", flush=True)
    print(f"mode:    {'APPLY' if apply else 'DRY-RUN'}", flush=True)
    print(f"files:   {len(targets)}", flush=True)

    total_rows = total_changed = total_msgs = 0
    for path in targets:
        st = _patch_file(path, apply=apply, backup=not args.no_backup)
        total_rows += st["rows"]
        total_changed += st["rows_changed"]
        total_msgs += st["user_msgs_changed"]
        flag = "wrote" if st["applied"] else "would"
        print(
            f"  {flag} {path.relative_to(mix_dir)}: "
            f"rows {st['rows_changed']:,}/{st['rows']:,} "
            f"(user msgs touched {st['user_msgs_changed']:,})",
            flush=True,
        )

    print(
        f"\nTOTAL: rows changed {total_changed:,}/{total_rows:,}, "
        f"user messages {total_msgs:,}",
        flush=True,
    )
    if not apply:
        print("\nDry-run only. Re-run with --apply to write.", flush=True)
        return

    print(
        "\nJSONL patched. Re-export tokenize so the swift cache matches:\n"
        "  python scripts/tokenize_data.py --force\n"
        "(no need to re-run prepare_data)",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted", flush=True)
        sys.exit(130)
