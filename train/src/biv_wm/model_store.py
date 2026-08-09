"""Local checkpoint store for train base models (prepare_model → tokenize → train)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_MODEL_SOURCE = "modelscope"
DEFAULT_MODEL_DIR = "outputs/models/Qwen3-Coder-30B-A3B-Instruct"
MANIFEST_NAME = "model_manifest.json"

_CONFIG_MARKERS = ("config.json", "configuration.json")


def model_dir_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in _CONFIG_MARKERS)


def resolve_model_dir(cfg: dict[str, Any], *, root: Path) -> Path:
    raw = cfg.get("model_dir") or DEFAULT_MODEL_DIR
    p = Path(str(raw))
    return p if p.is_absolute() else (root / p)


def resolve_model_id(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model") or DEFAULT_MODEL_ID)


def resolve_model_source(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model_source") or DEFAULT_MODEL_SOURCE).strip().lower()


def manifest_path(model_dir: Path) -> Path:
    return model_dir / MANIFEST_NAME


def load_manifest(model_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(model_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def resolve_model_for_train(cfg: dict[str, Any], *, root: Path) -> str:
    """Prefer prepared local ``model_dir``; else return hub id from config."""
    model_dir = resolve_model_dir(cfg, root=root)
    if model_dir_ready(model_dir):
        return str(model_dir.resolve())
    man = load_manifest(model_dir)
    if man:
        local = man.get("local_path") or man.get("path")
        if local:
            p = Path(str(local))
            if not p.is_absolute():
                p = root / p
            if model_dir_ready(p):
                return str(p.resolve())
    return resolve_model_id(cfg)


def download_model(
    *,
    model_id: str,
    source: str,
    dest: Path,
    force: bool = False,
) -> Path:
    """Download hub weights into ``dest`` (CPU OK). Returns local path."""
    src = source.strip().lower()
    if dest.exists() and model_dir_ready(dest) and not force:
        print(f"Reusing prepared model at {dest}", flush=True)
        return dest.resolve()

    if force and dest.exists():
        print(f"--force: removing {dest}", flush=True)
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if src in {"local", "path"}:
        src_path = Path(model_id)
        if not model_dir_ready(src_path):
            raise SystemExit(f"Local model not ready: {src_path}")
        if src_path.resolve() != dest.resolve():
            if dest.exists():
                shutil.rmtree(dest)
            print(f"Copying local model {src_path} → {dest}", flush=True)
            shutil.copytree(src_path, dest)
        return dest.resolve()

    if src in {"modelscope", "ms"}:
        try:
            from modelscope import snapshot_download
        except ImportError as e:
            raise SystemExit(
                "modelscope required for model_source=modelscope: pip install modelscope\n"
                f"({e})"
            ) from e
        print(f"Downloading from ModelScope: {model_id}", flush=True)
        print(f"  → {dest}", flush=True)
        # Prefer writing directly into dest when supported.
        try:
            local = snapshot_download(model_id, local_dir=str(dest))
        except TypeError:
            local = snapshot_download(model_id)
            local_p = Path(local)
            if local_p.resolve() != dest.resolve():
                if dest.exists():
                    shutil.rmtree(dest)
                # Prefer symlink to hub cache to save disk; fall back to copy.
                try:
                    dest.symlink_to(local_p, target_is_directory=True)
                    print(f"Symlinked {dest} → {local_p}", flush=True)
                except OSError:
                    print(f"Copying ModelScope cache {local_p} → {dest}", flush=True)
                    shutil.copytree(local_p, dest)
                return dest.resolve()
        dest_final = Path(local)
        if not model_dir_ready(dest_final):
            raise SystemExit(f"Download finished but config.json missing under {dest_final}")
        print(f"ModelScope path: {dest_final}", flush=True)
        return dest_final.resolve()

    if src in {"huggingface", "hf"}:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
        except ImportError as e:
            raise SystemExit(
                "huggingface_hub required: pip install huggingface_hub\n"
                f"({e})"
            ) from e
        if not os.environ.get("HF_ENDPOINT"):
            print(
                "Tip (CN): export HF_ENDPOINT=https://hf-mirror.com",
                flush=True,
            )
        print(f"Downloading from HuggingFace: {model_id}", flush=True)
        print(f"  → {dest}", flush=True)
        local = hf_snapshot_download(repo_id=model_id, local_dir=str(dest))
        dest_final = Path(local)
        if not model_dir_ready(dest_final):
            raise SystemExit(f"Download finished but config.json missing under {dest_final}")
        return dest_final.resolve()

    raise SystemExit(
        f"Unknown model_source={source!r} (use modelscope | huggingface | local)"
    )


def write_manifest(
    *,
    model_dir: Path,
    model_id: str,
    source: str,
    local_path: Path,
    root: Path,
) -> Path:
    try:
        rel = str(local_path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(local_path.resolve())
    payload = {
        "model_id": model_id,
        "model_source": source,
        "local_path": rel,
        "absolute_path": str(local_path.resolve()),
        "ready": model_dir_ready(local_path),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(model_dir if model_dir_ready(model_dir) else local_path)
    # Prefer writing next to the resolved weights.
    out = local_path / MANIFEST_NAME if model_dir_ready(local_path) else path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Also mirror under configured model_dir when different.
    if model_dir.resolve() != local_path.resolve():
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return out
