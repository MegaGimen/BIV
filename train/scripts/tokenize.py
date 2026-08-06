#!/usr/bin/env python3
"""Step 2: Axolotl tokenize / preprocess for Coder-Next (CPU-only).

After ``prepare_data.py`` writes JSONL under ``data/processed/mix_v1``, run this
so chat-template formatting + tokenization land in ``dataset_prepared_path``.
Then ``axolotl train`` / ``train_coder_next.sh`` can load the cache immediately
without re-tokenizing during training (avoids the Axolotl VRAM warning).

Does **not** need a GPU — forces ``CUDA_VISIBLE_DEVICES=""``.

Examples:
  python scripts/tokenize.py
  python scripts/tokenize.py --config configs/axolotl/coder_next_qlora.yaml
  python scripts/tokenize.py --force   # rebuild cache
  python scripts/tokenize.py --check   # only verify cache exists
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "axolotl" / "coder_next_qlora.yaml"
DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v1"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML required: pip install pyyaml\n" f"(failed: {e})"
        ) from e
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML (expected mapping): {path}")
    return data


def _prepared_path(cfg: dict, config_path: Path) -> Path:
    raw = cfg.get("dataset_prepared_path")
    if not raw:
        raise SystemExit(
            f"{config_path}: missing dataset_prepared_path "
            "(required so train can reuse the token cache)"
        )
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _dataset_jsonls(cfg: dict, config_path: Path) -> list[Path]:
    rows = cfg.get("datasets") or []
    if not rows:
        raise SystemExit(f"{config_path}: no datasets: entries")
    out: list[Path] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("path"):
            raise SystemExit(f"{config_path}: datasets[{i}] missing path")
        p = Path(str(row["path"]))
        out.append(p if p.is_absolute() else (ROOT / p))
    return out


def _cache_ready(cache_dir: Path) -> bool:
    """Heuristic: Axolotl wrote a non-empty prepared tree under dataset_prepared_path."""
    if not cache_dir.is_dir():
        return False
    # Axolotl nests a hash subdir; accept any arrow/parquet/dataset_info marker.
    markers = (
        "dataset_info.json",
        "state.json",
        "data-00000-of-00001.arrow",
    )
    for p in cache_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in markers or p.suffix in {".arrow", ".parquet"}:
            return True
        # HF datasets sometimes only has *.arrow with other names
        if p.suffix == ".arrow" or p.name.endswith(".arrow"):
            return True
    # Fallback: any non-trivial file > 1KB
    for p in cache_dir.rglob("*"):
        if p.is_file() and p.stat().st_size > 1024:
            return True
    return False


def _require_prepare(jsonls: list[Path], mix_dir: Path) -> None:
    missing = [p for p in jsonls if not p.is_file()]
    if missing:
        print("Missing train JSONL (run prepare first):", flush=True)
        for p in missing:
            print(f"  {p}", flush=True)
        raise SystemExit(
            f"\n  python scripts/prepare_data.py --all --out-dir {mix_dir.relative_to(ROOT)}"
        )
    manifest = mix_dir / "mix_manifest.json"
    if not manifest.is_file():
        print(
            f"Warning: {manifest} missing — JSONL present; continuing.",
            flush=True,
        )


def _find_axolotl() -> list[str]:
    exe = shutil.which("axolotl")
    if exe:
        return [exe, "preprocess"]
    # Fallback: module entry (same venv)
    return [sys.executable, "-m", "axolotl.cli.preprocess"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU tokenize/preprocess for Axolotl Coder-Next mix (step 2 after prepare_data)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Axolotl yaml (default: configs/axolotl/coder_next_qlora.yaml)",
    )
    parser.add_argument(
        "--mix-dir",
        type=Path,
        default=DEFAULT_MIX,
        help="prepare_data out-dir (for hint messages)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing dataset_prepared_path and rebuild",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify cache is ready; exit 0/1",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="Do not clear CUDA_VISIBLE_DEVICES (not recommended)",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    mix_dir = args.mix_dir if args.mix_dir.is_absolute() else (ROOT / args.mix_dir)
    cfg = _load_yaml(config_path)
    cache_dir = _prepared_path(cfg, config_path)
    jsonls = _dataset_jsonls(cfg, config_path)

    print(f"Config:  {config_path}", flush=True)
    print(f"Cache:   {cache_dir}", flush=True)
    print("JSONL:", flush=True)
    for p in jsonls:
        ok = "ok" if p.is_file() else "MISSING"
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p.is_file() else "-"
        print(f"  [{ok}] {p} ({size})", flush=True)

    if args.check:
        ready = _cache_ready(cache_dir)
        print(f"Cache ready: {ready}", flush=True)
        raise SystemExit(0 if ready else 1)

    _require_prepare(jsonls, mix_dir)

    if _cache_ready(cache_dir) and not args.force:
        print(
            f"\nToken cache already present under {cache_dir}\n"
            "Skip preprocess. Use --force to rebuild.\n"
            "Train with:\n"
            "  CUDA_VISIBLE_DEVICES=0,1 axolotl train "
            f"{config_path.relative_to(ROOT)}\n"
            "  # or: bash scripts/train_coder_next.sh",
            flush=True,
        )
        return

    if args.force and cache_dir.exists():
        print(f"--force: removing {cache_dir}", flush=True)
        shutil.rmtree(cache_dir)

    if not args.allow_gpu:
        # Strict CPU tokenize — matches Axolotl recommended preprocess path.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CUDA_VISIBLE_DEVICES=\"\" (CPU tokenize)", flush=True)

    cmd = _find_axolotl() + [str(config_path)]
    print(f"Running: {' '.join(cmd)}", flush=True)
    print("(This can take a long time for anti_forget ~270k rows.)", flush=True)

    os.chdir(ROOT)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    if not _cache_ready(cache_dir):
        raise SystemExit(
            f"preprocess finished but cache still empty under {cache_dir}\n"
            "Check axolotl logs / dataset_prepared_path."
        )

    print(
        f"\nDone. Token cache ready: {cache_dir}\n"
        "Start training (loads cache, no re-tokenize):\n"
        "  CUDA_VISIBLE_DEVICES=0,1 axolotl train "
        f"{config_path.relative_to(ROOT)}\n"
        "  # or: bash scripts/train_coder_next.sh",
        flush=True,
    )


if __name__ == "__main__":
    main()
