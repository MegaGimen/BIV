#!/usr/bin/env python3
"""Step 2: download base LLM for tokenize + train (CPU OK).

Workflow:
  prepare_data → prepare_model → tokenize_data → train_coder_next

Reads ``model`` / ``model_source`` / ``model_dir`` from
``configs/swift/glm47_flash_qlora.yaml`` (default on this branch).

Needs ms-swift>=4.0 and transformers>=5.0 to *train* this arch; download itself
only needs hub access (ModelScope ``ZhipuAI/GLM-4.7-Flash`` or HF ``zai-org/…``).

Examples:
  python scripts/prepare_model.py
  python scripts/prepare_model.py --source huggingface
  python scripts/prepare_model.py --check
  python scripts/prepare_model.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CONFIG = ROOT / "configs" / "swift" / "glm47_flash_qlora.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(f"PyYAML required: pip install pyyaml\n({e})") from e
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML (expected mapping): {path}")
    return data


def main() -> None:
    from biv_wm.model_store import (
        download_model,
        load_manifest,
        model_dir_ready,
        resolve_model_dir,
        resolve_model_id,
        resolve_model_source,
        write_manifest,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface", "local"],
        default=None,
        help="Override config model_source",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override hub id / local path (default from config.model)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override local destination (default from config.model_dir)",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if ready")
    p.add_argument("--check", action="store_true", help="Exit 0 iff model_dir is ready")
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    cfg = _load_yaml(config_path)

    model_id = args.model or resolve_model_id(cfg)
    source = args.source or resolve_model_source(cfg)
    if args.model_dir is not None:
        dest = args.model_dir if args.model_dir.is_absolute() else (ROOT / args.model_dir)
    else:
        dest = resolve_model_dir(cfg, root=ROOT)

    print(f"Config:  {config_path}", flush=True)
    print(f"Model:   {model_id}", flush=True)
    print(f"Source:  {source}", flush=True)
    print(f"Dest:    {dest}", flush=True)

    if args.check:
        ok = model_dir_ready(dest)
        man = load_manifest(dest)
        print(f"Ready:   {ok}", flush=True)
        if man:
            print(f"Manifest local_path: {man.get('local_path')}", flush=True)
        raise SystemExit(0 if ok else 1)

    # Discourage filling tiny system disks (e.g. AutoDL / 30GB).
    try:
        import shutil

        free_gb = shutil.disk_usage(str(dest.parent)).free / (1024**3)
        if free_gb < 80:
            print(
                f"WARNING: only {free_gb:.1f} GiB free under {dest.parent}. "
                "GLM-4.7-Flash needs ~60GB+ for weights plus cache/ckpt; prefer a "
                "large data disk (e.g. AutoDL /root/autodl-tmp).",
                flush=True,
            )
    except OSError:
        pass

    local = download_model(
        model_id=model_id,
        source=source,
        dest=dest,
        force=args.force,
    )
    man_path = write_manifest(
        model_dir=dest,
        model_id=model_id,
        source=source,
        local_path=local,
        root=ROOT,
    )
    print(f"Ready model: {local}", flush=True)
    print(f"Manifest:    {man_path}", flush=True)
    print(
        "Next: python scripts/tokenize_data.py\n"
        "  (tokenize/train will use this local path automatically)",
        flush=True,
    )


if __name__ == "__main__":
    main()
