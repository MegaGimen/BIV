#!/usr/bin/env python3
"""Patch prepared WM JSONL in-place: drop chat wrappers without re-prepare.

Removes from ``role=user`` contents (wm_code / wm_os only):
  - ``Previous tool steps in this session: <n>.\\n``
  - ``Latest tool call:\\n``

Does **not** touch anti_forget (native agent/tool format).

**dry-run (默认)**: 只扫文件、报告会改多少行，**不写盘**。
**--apply**: 真正改写 JSONL（首次默认留 ``.bak``）。

After ``--apply``, re-export tokenize cache so tokens match the JSONL::

  python scripts/delta.py --mix-dir data/processed/mix_v1 --apply
  python scripts/tokenize_data.py --force

Examples:
  python scripts/delta.py                 # dry-run
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

_RE_PREV_STEPS = re.compile(r"^Previous tool steps in this session: \d+\.\n")
_RE_LATEST = re.compile(r"^Latest tool call:\n")
# Fast prefilter: skip json.loads when neither marker appears in the raw line.
_MARKERS = ("Latest tool call:", "Previous tool steps in this session:")


def _tqdm(**kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    return tqdm(**kwargs)


def clean_user_content(text: str) -> tuple[str, bool]:
    if not isinstance(text, str) or not text:
        return text, False
    new = _RE_PREV_STEPS.sub("", text)
    new = _RE_LATEST.sub("", new)
    if "Latest tool call:\n" in new:
        new = new.replace("Latest tool call:\n", "")
    if new != text:
        return new, True
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


def _line_may_need_patch(raw: str) -> bool:
    return any(m in raw for m in _MARKERS)


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


def _patch_file(path: Path, *, apply: bool, backup: bool, rel: str) -> dict:
    stats = {
        "path": str(path),
        "rows": 0,
        "rows_changed": 0,
        "user_msgs_changed": 0,
        "applied": False,
        "bytes": path.stat().st_size,
    }
    size = stats["bytes"]
    print(
        f"\n→ {'patching' if apply else 'scanning'} {rel} "
        f"({size / (1024**2):.1f} MiB) …",
        flush=True,
    )

    if not apply:
        bar = _tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=rel)
        with path.open("r", encoding="utf-8") as f:
            if bar is None:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    stats["rows"] += 1
                    if not _line_may_need_patch(raw):
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    _, n = clean_row(obj)
                    if n:
                        stats["rows_changed"] += 1
                        stats["user_msgs_changed"] += n
            else:
                with bar:
                    for line in f:
                        bar.update(len(line.encode("utf-8", errors="ignore")))
                        raw = line.strip()
                        if not raw:
                            continue
                        stats["rows"] += 1
                        if not _line_may_need_patch(raw):
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        _, n = clean_row(obj)
                        if n:
                            stats["rows_changed"] += 1
                            stats["user_msgs_changed"] += n
                            bar.set_postfix(changed=stats["rows_changed"], refresh=False)
        return stats

    # apply: optional full-file backup can be huge — copy with progress
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.is_file():
            print(f"  writing backup {bak.name} …", flush=True)
            _copy_with_progress(path, bak, desc=f"bak {rel}")
        else:
            print(f"  backup exists, skip: {bak.name}", flush=True)

    tmp = path.with_suffix(path.suffix + ".delta_tmp")
    bar = _tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"write {rel}")
    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        if bar is None:
            for line in fin:
                raw = line.rstrip("\n")
                if not raw.strip():
                    fout.write(line if line.endswith("\n") else line + "\n")
                    continue
                stats["rows"] += 1
                if not _line_may_need_patch(raw):
                    fout.write(raw + "\n")
                    continue
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
        else:
            with bar:
                for line in fin:
                    bar.update(len(line.encode("utf-8", errors="ignore")))
                    raw = line.rstrip("\n")
                    if not raw.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        continue
                    stats["rows"] += 1
                    if not _line_may_need_patch(raw):
                        fout.write(raw + "\n")
                        continue
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
                        bar.set_postfix(changed=stats["rows_changed"], refresh=False)
                    else:
                        fout.write(raw + "\n")

    tmp.replace(path)
    stats["applied"] = True
    return stats


def _copy_with_progress(src: Path, dst: Path, *, desc: str) -> None:
    size = src.stat().st_size
    bar = _tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
    with src.open("rb") as fin, dst.open("wb") as fout:
        if bar is None:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
            return
        with bar:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix-dir", type=Path, default=DEFAULT_MIX)
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default is dry-run = report only, no writes)",
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
        help="Do not write .jsonl.bak on first apply (faster, less disk)",
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
    print(f"mode:    {'APPLY (will write)' if apply else 'DRY-RUN (report only, no writes)'}", flush=True)
    print(f"files:   {len(targets)}", flush=True)
    for t in targets:
        print(f"         - {t.relative_to(mix_dir)} ({t.stat().st_size / (1024**2):.1f} MiB)", flush=True)

    total_rows = total_changed = total_msgs = 0
    for path in targets:
        rel = str(path.relative_to(mix_dir))
        st = _patch_file(path, apply=apply, backup=not args.no_backup, rel=rel)
        total_rows += st["rows"]
        total_changed += st["rows_changed"]
        total_msgs += st["user_msgs_changed"]
        flag = "wrote" if st["applied"] else "would change"
        print(
            f"  {flag} {rel}: "
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
        print(
            "\n这是 DRY-RUN：只统计，没有改任何文件。\n"
            "确认无误后执行写入：\n"
            "  python scripts/delta.py --mix-dir data/processed/mix_v1 --apply --no-backup\n"
            "（大文件建议 --no-backup，避免再复制一整份 .bak）",
            flush=True,
        )
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
