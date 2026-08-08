#!/usr/bin/env python3
"""Build mix_v2 from mix_v1 by stripping WM chat wrappers (no in-place overwrite).

Reads ``wm_code`` / ``wm_os`` JSONL from ``--src-dir`` (default mix_v1), removes:
  - ``Previous tool steps in this session: <n>.\\n``
  - ``Latest tool call:\\n``
from ``role=user`` contents, and writes results under ``--out-dir`` (default mix_v2).

Also copies ``anti_forget`` + root/per-source manifests & counts so the out dir
is tokenize-ready. Original mix_v1 is never modified.

**dry-run (默认)**: 只统计，不写盘。
**--apply**: 写出 mix_v2。

Examples:
  python scripts/delta.py
  python scripts/delta.py --apply
  python scripts/delta.py --src-dir data/processed/mix_v1 --out-dir data/processed/mix_v2 --apply

Then:
  python scripts/tokenize_data.py --mix-dir data/processed/mix_v2 --force
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "processed" / "mix_v1"
DEFAULT_OUT = ROOT / "data" / "processed" / "mix_v2"
WM_SOURCES = ("wm_code", "wm_os")
COPY_SOURCES = ("anti_forget",)
ALL_SOURCES = WM_SOURCES + COPY_SOURCES
SPLITS = ("train", "eval")
ROOT_META = (
    "counts.json",
    "mix_manifest.json",
    "fingerprint.json",
    "GENERATOR.md",
)

_RE_PREV_STEPS = re.compile(r"^Previous tool steps in this session: \d+\.\n")
_RE_LATEST = re.compile(r"^Latest tool call:\n")
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


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p)


def _wm_jsonl_targets(src_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name in WM_SOURCES:
        for split in SPLITS:
            p = src_dir / name / f"{split}.jsonl"
            if p.is_file():
                paths.append(p)
    return paths


def _copy_with_progress(src: Path, dst: Path, *, desc: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
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


def _scan_or_write_wm(
    src: Path,
    dst: Path | None,
    *,
    apply: bool,
    rel: str,
) -> dict:
    stats = {
        "rows": 0,
        "rows_changed": 0,
        "user_msgs_changed": 0,
        "bytes": src.stat().st_size,
    }
    size = stats["bytes"]
    mode = "writing" if apply else "scanning"
    print(f"\n→ {mode} {rel} ({size / (1024**2):.1f} MiB) …", flush=True)

    if apply:
        assert dst is not None
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_f = dst.open("w", encoding="utf-8")
    else:
        out_f = None

    bar = _tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=rel,
    )
    try:
        with src.open("r", encoding="utf-8") as fin:
            ctx = bar if bar is not None else None

            def handle(raw_line: str) -> None:
                raw = raw_line.rstrip("\n")
                if not raw.strip():
                    if out_f is not None:
                        out_f.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                    return
                stats["rows"] += 1
                if not _line_may_need_patch(raw):
                    if out_f is not None:
                        out_f.write(raw + "\n")
                    return
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    if out_f is not None:
                        out_f.write(raw + "\n")
                    return
                new_obj, n = clean_row(obj)
                if n:
                    stats["rows_changed"] += 1
                    stats["user_msgs_changed"] += n
                    if out_f is not None:
                        out_f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                else:
                    if out_f is not None:
                        out_f.write(raw + "\n")

            if ctx is None:
                for line in fin:
                    handle(line)
            else:
                with ctx:
                    for line in fin:
                        ctx.update(len(line.encode("utf-8", errors="ignore")))
                        handle(line)
                        if stats["rows_changed"]:
                            ctx.set_postfix(changed=stats["rows_changed"], refresh=False)
    finally:
        if out_f is not None:
            out_f.close()
    return stats


def _copy_tree_file(src: Path, dst: Path, *, desc: str) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        print(f"→ copy dir {desc} …", flush=True)
        shutil.copytree(src, dst)
        return
    _copy_with_progress(src, dst, desc=desc)


def _copy_sidecar_and_anti(src_dir: Path, out_dir: Path) -> None:
    """Copy anti_forget + counts/manifests so out_dir is tokenize-ready."""
    for name in ROOT_META:
        sp = src_dir / name
        if sp.is_file():
            print(f"→ copy {name}", flush=True)
            shutil.copy2(sp, out_dir / name)

    for name in ALL_SOURCES:
        src_src = src_dir / name
        if not src_src.is_dir():
            continue
        out_src = out_dir / name
        out_src.mkdir(parents=True, exist_ok=True)
        # per-source meta
        for meta in ("counts.json", "instance_ids.json"):
            sp = src_src / meta
            if sp.is_file():
                shutil.copy2(sp, out_src / meta)
                print(f"→ copy {name}/{meta}", flush=True)

    for name in COPY_SOURCES:
        src_src = src_dir / name
        if not src_src.is_dir():
            print(f"WARNING: missing {src_src} — skip", flush=True)
            continue
        for split in SPLITS:
            sp = src_src / f"{split}.jsonl"
            if not sp.is_file():
                continue
            dp = out_dir / name / f"{split}.jsonl"
            _copy_with_progress(sp, dp, desc=f"copy {name}/{split}.jsonl")


def _stamp_out_manifest(src_dir: Path, out_dir: Path) -> None:
    note = {
        "delta_from": str(src_dir),
        "delta_to": str(out_dir),
        "operation": "strip_wm_user_wrappers",
        "stripped": [
            "Previous tool steps in this session: <n>.",
            "Latest tool call:",
        ],
        "sources_patched": list(WM_SOURCES),
        "sources_copied": list(COPY_SOURCES),
    }
    (out_dir / "delta_manifest.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    # Annotate mix_manifest if present
    mm = out_dir / "mix_manifest.json"
    if mm.is_file():
        try:
            blob = json.loads(mm.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if isinstance(blob, dict):
            blob["delta"] = note
            mm.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", "--mix-dir", type=Path, default=DEFAULT_SRC, dest="src_dir")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--apply", action="store_true", help="Write mix_v2 (default: dry-run)")
    p.add_argument("--dry-run", action="store_true", help="Report only (default)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace existing out-dir contents for patched/copied files",
    )
    args = p.parse_args()
    apply = bool(args.apply) and not bool(args.dry_run)

    src_dir = _resolve(args.src_dir)
    out_dir = _resolve(args.out_dir)
    if not src_dir.is_dir():
        raise SystemExit(f"src-dir not found: {src_dir}")
    if src_dir.resolve() == out_dir.resolve():
        raise SystemExit("src-dir and out-dir must differ (refusing in-place overwrite)")

    targets = _wm_jsonl_targets(src_dir)
    if not targets:
        raise SystemExit(f"No wm_code/wm_os JSONL under {src_dir}")

    print(f"src-dir: {src_dir}", flush=True)
    print(f"out-dir: {out_dir}", flush=True)
    print(
        f"mode:    {'APPLY (write out-dir, leave src untouched)' if apply else 'DRY-RUN (report only)'}",
        flush=True,
    )
    print(f"wm jsonl files: {len(targets)}", flush=True)
    for t in targets:
        print(f"  - {t.relative_to(src_dir)} ({t.stat().st_size / (1024**2):.1f} MiB)", flush=True)

    if apply:
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            raise SystemExit(
                f"out-dir not empty: {out_dir}\n"
                "Pass --force to overwrite, or choose another --out-dir."
            )
        out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = total_changed = total_msgs = 0
    for src_path in targets:
        rel = str(src_path.relative_to(src_dir))
        dst_path = out_dir / rel if apply else None
        if apply and dst_path is not None and dst_path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite {dst_path} (use --force)")
        st = _scan_or_write_wm(src_path, dst_path, apply=apply, rel=rel)
        total_rows += st["rows"]
        total_changed += st["rows_changed"]
        total_msgs += st["user_msgs_changed"]
        verb = "wrote" if apply else "would change"
        print(
            f"  {verb} {rel}: rows {st['rows_changed']:,}/{st['rows']:,} "
            f"(user msgs {st['user_msgs_changed']:,})",
            flush=True,
        )

    print(
        f"\nWM TOTAL: rows changed {total_changed:,}/{total_rows:,}, "
        f"user messages {total_msgs:,}",
        flush=True,
    )

    if not apply:
        print(
            "\n这是 DRY-RUN：不写盘，源 mix_v1 也不会改。\n"
            "确认后写出 mix_v2：\n"
            "  python scripts/delta.py --apply --force\n"
            "然后：\n"
            "  python scripts/tokenize_data.py --mix-dir data/processed/mix_v2 --force",
            flush=True,
        )
        return

    print("\nCopying anti_forget + manifests into out-dir …", flush=True)
    _copy_sidecar_and_anti(src_dir, out_dir)
    _stamp_out_manifest(src_dir, out_dir)

    print(
        f"\nDone. Wrote {out_dir} (src {src_dir} unchanged).\n"
        "Next:\n"
        "  python scripts/tokenize_data.py --mix-dir data/processed/mix_v2 --force\n"
        "Or set mix_dir/cache_root in configs/swift/coder_next_qlora.yaml to mix_v2.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted", flush=True)
        sys.exit(130)
