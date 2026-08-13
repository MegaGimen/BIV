#!/usr/bin/env python3
"""Export one anti_forget sample for offline inspection / normalize redesign.

Prefer local mix JSONL (post SWE-Zero → policy_row). Optional: hub raw trajectory.

Examples:
  python scripts/tmpexport.py
  python scripts/tmpexport.py --index 0 --out /tmp/anti_forget_raw.json
  python scripts/tmpexport.py --mix-dir data/processed/mix_v2 --index 0
  python scripts/tmpexport.py --from-prep --run-root outputs/trl_cache/.../train_runs/ml65536_...
  python scripts/tmpexport.py --hub-raw   # one SWE-Zero hub record (needs net/cache)

Prints JSON to stdout (and optional --out file). Paste into ~/…/tmp for review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v2"
FALLBACK_MIX = ROOT / "data" / "processed" / "mix_v1"
DEFAULT_CONFIG = ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"


def _resolve(p: Path | str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def _default_mix() -> Path:
    if DEFAULT_MIX.is_dir():
        return DEFAULT_MIX
    return FALLBACK_MIX


def _load_nth_jsonl(path: Path, index: int) -> dict[str, Any]:
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


def _find_train_runs(cache_root: Path) -> list[Path]:
    runs: list[Path] = []
    if not cache_root.is_dir():
        return runs
    for p in cache_root.rglob("run_manifest.json"):
        root = p.parent
        if (root / "anti_forget").is_dir():
            runs.append(root)
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs


def _load_prep_row(run_root: Path, index: int) -> tuple[dict[str, Any], Path]:
    from datasets import load_from_disk

    path = run_root / "anti_forget"
    if not path.is_dir():
        raise SystemExit(f"Missing prep dataset: {path}")
    ds = load_from_disk(str(path))
    if index < 0 or index >= len(ds):
        raise SystemExit(f"anti_forget prep: index {index} out of range (n={len(ds)})")
    row = ds[index]
    if not isinstance(row, dict):
        row = dict(row)
    # HF Arrow → plain Python for JSON dump
    out: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "tolist"):
            try:
                out[k] = v.tolist()
                continue
            except Exception:
                pass
        out[k] = v
    return out, path


def _load_hub_raw(index: int, config: dict) -> tuple[dict[str, Any], str]:
    from biv_wm.hub import open_dataset_with_cache

    hub_source = str(
        (config.get("hub") or {}).get("source")
        or config.get("model_source")
        or "modelscope"
    )
    # prepare_data defaults; keep light
    ds = open_dataset_with_cache(
        kind="swe_zero",
        source=hub_source if hub_source in {"modelscope", "ms", "huggingface", "hf"} else "modelscope",
        repo_id=None,
        split="train",
        max_rows=None,
        local_dir=None,
    )
    if index < 0 or index >= len(ds):
        raise SystemExit(f"SWE-Zero hub: index {index} out of range (n={len(ds)})")
    rec = ds[index]
    if not isinstance(rec, dict):
        rec = dict(rec)
    # Drop huge non-JSON bits if any
    clean: dict[str, Any] = {}
    for k, v in rec.items():
        try:
            json.dumps(v, ensure_ascii=False)
            clean[k] = v
        except TypeError:
            clean[k] = repr(v)
    note = f"swe_zero hub row[{index}] via open_dataset_with_cache(source={hub_source})"
    return clean, note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write JSON here (default: stdout only)",
    )
    p.add_argument(
        "--from-prep",
        action="store_true",
        help="load from train_prep_mix HF cache (messages after struct-right)",
    )
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument(
        "--hub-raw",
        action="store_true",
        help="export one unprocessed SWE-Zero hub record (trajectory field)",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--compact",
        action="store_true",
        help="single-line JSON (default: indented)",
    )
    args = p.parse_args()

    cfg: dict = {}
    cfg_path = _resolve(args.config)
    if cfg_path.is_file():
        import yaml

        with cfg_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            cfg = loaded

    meta: dict[str, Any] = {
        "source": "anti_forget",
        "index": args.index,
        "note": "",
        "path": "",
    }

    if args.hub_raw:
        row, note = _load_hub_raw(args.index, cfg)
        meta["note"] = note
        meta["path"] = "hub:swe_zero"
        payload = {"_export_meta": meta, "record": row}
    elif args.from_prep:
        run_root = args.run_root
        if run_root is None:
            roots = []
            if args.cache_root:
                roots.append(_resolve(args.cache_root))
            cr = cfg.get("cache_root")
            if cr:
                roots.append(_resolve(cr))
            roots.append(ROOT / "outputs" / "trl_cache")
            found = None
            for r in roots:
                hits = _find_train_runs(r)
                if hits:
                    found = hits[0]
                    break
            if found is None:
                raise SystemExit(
                    "No train_runs with anti_forget found; pass --run-root …/train_runs/<id>"
                )
            run_root = found
            print(f"[tmpexport] auto run-root → {run_root}", file=sys.stderr, flush=True)
        else:
            run_root = _resolve(run_root)
        row, path = _load_prep_row(run_root, args.index)
        meta["note"] = "train_prep_mix HF anti_forget row (pre Muse _normalize_messages)"
        meta["path"] = str(path)
        payload = {"_export_meta": meta, "row": row}
    else:
        mix = _resolve(args.mix_dir) if args.mix_dir else _default_mix()
        path = mix / "anti_forget" / f"{args.split}.jsonl"
        row = _load_nth_jsonl(path, args.index)
        meta["note"] = (
            "mix JSONL after policy_row_from_openhands_record "
            "(SWE-Zero → messages; think/finish ideally dropped)"
        )
        meta["path"] = str(path)
        payload = {"_export_meta": meta, "row": row}

    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=None if args.compact else 2,
    )
    print(text, flush=True)
    if args.out is not None:
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(f"[tmpexport] wrote {out}", file=sys.stderr, flush=True)

    # Reminder for checkpoint location (stderr so stdout stays pure JSON)
    train_cfg = cfg.get("train") or {}
    base_out = train_cfg.get("output_dir", "outputs/muse_glimmer_wm_mix")
    print(
        "[tmpexport] Muse checkpoints live under train output_dir, e.g.\n"
        f"  {ROOT / base_out}_ml<MAX>_c<choice>/\n"
        "    checkpoint-e{{epoch}}-s{{step}}/          # rolling (keep 3)\n"
        "    checkpoint-epoch{{N}}-end-s{{step}}/      # permanent epoch-end\n"
        "  (older runs may still have checkpoint-{{step}}/)",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        sys.exit(130)
