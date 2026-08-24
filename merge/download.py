#!/usr/bin/env python3
"""Download AgentWorld and Qwen3.5 Instruct (plus Base for Chat Vector).

The two checkpoints you merge conceptually::

    Qwen/Qwen-AgentWorld-35B-A3B
    Qwen/Qwen3.5-35B-A3B

Chat Vector still subtracts Instruct from the shared ancestor, so this
script also pulls ``Qwen/Qwen3.5-35B-A3B-Base`` unless you pass ``--no-base``.

Writes under ``merge/output/cache/<id>``. After it finishes, merge.py with
the default hub ids reuses that cache and does not hit the network again.

GPU / disk host::

    python merge/download.py
    python merge/download.py --force
    python merge/download.py --no-base
    python merge/download.py --source huggingface
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "merge" / "output" / "cache"

DEFAULT_WORLD = "Qwen/Qwen-AgentWorld-35B-A3B"
DEFAULT_AGENT = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_BASE = "Qwen/Qwen3.5-35B-A3B-Base"
DEFAULT_SOURCE = os.environ.get("MERGE_SOURCE", "modelscope")


def log(msg: str) -> None:
    print(msg, flush=True)


def has_config(path: Path) -> bool:
    return path.is_dir() and any(
        (path / n).is_file() for n in ("config.json", "configuration.json")
    )


def sanitize(model_id: str) -> str:
    return model_id.strip().replace("/", "--").replace(" ", "_")


def cache_path(spec: str, cache_dir: Path) -> Path:
    p = Path(spec).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return cache_dir / sanitize(spec)


def existing_or_spec(spec: str, cache_dir: Path) -> str:
    """Local dir if already downloaded; otherwise the hub id / given spec."""
    p = Path(spec).expanduser()
    if has_config(p):
        return str(p.resolve())
    dest = cache_dir / sanitize(spec)
    if has_config(dest):
        return str(dest.resolve())
    return spec


def resolve_model(
    spec: str,
    *,
    source: str,
    cache_dir: Path,
    role: str,
    force: bool = False,
) -> Path:
    """Return a local dir with config.json; download into cache_dir if needed."""
    p = Path(spec).expanduser()
    if has_config(p):
        if force:
            log(f"[{role}] --force ignored for local path {p.resolve()}")
        log(f"[{role}] local checkpoint: {p.resolve()}")
        return p.resolve()

    dest = cache_dir / sanitize(spec)
    if force and dest.exists():
        log(f"[{role}] --force: removing {dest}")
        shutil.rmtree(dest)
    elif has_config(dest):
        log(f"[{role}] reusing cache: {dest}")
        return dest.resolve()

    dest.parent.mkdir(parents=True, exist_ok=True)
    src = source.strip().lower()
    log(f"[{role}] downloading {spec} via {src} → {dest}")

    if src in {"modelscope", "ms"}:
        try:
            from modelscope import snapshot_download
        except ImportError as e:
            raise SystemExit(
                "modelscope is required: pip install modelscope\n"
                f"({e})"
            ) from e
        try:
            local = snapshot_download(spec, local_dir=str(dest))
        except TypeError:
            local = snapshot_download(spec)
        out = Path(local)
        if not has_config(out) and has_config(dest):
            out = dest
        if not has_config(out):
            raise SystemExit(f"[{role}] download finished but no config.json under {out}")
        return out.resolve()

    if src in {"huggingface", "hf"}:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
        except ImportError as e:
            raise SystemExit(
                "huggingface_hub is required: pip install huggingface_hub\n"
                f"({e})"
            ) from e
        if not os.environ.get("HF_ENDPOINT"):
            log("Tip (CN): export HF_ENDPOINT=https://hf-mirror.com")
        local = hf_snapshot_download(repo_id=spec, local_dir=str(dest))
        out = Path(local)
        if not has_config(out):
            raise SystemExit(f"[{role}] download finished but no config.json under {out}")
        return out.resolve()

    raise SystemExit(f"Unknown --source {source!r} (modelscope | huggingface)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world", default=DEFAULT_WORLD, help="AgentWorld hub id or local dir")
    p.add_argument("--agent", default=DEFAULT_AGENT, help="Qwen3.5 Instruct hub id or local dir")
    p.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help="Shared ancestor hub id (used unless --no-base)",
    )
    p.add_argument(
        "--no-base",
        action="store_true",
        help="Skip Qwen3.5-35B-A3B-Base (merge.py will still need it later)",
    )
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete cached copies under --cache-dir and download again",
    )
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=DEFAULT_SOURCE,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)

    jobs: list[tuple[str, str]] = [
        ("world", args.world),
        ("agent", args.agent),
    ]
    if not args.no_base:
        jobs.append(("base", args.base_model))

    log(f"source={args.source}  cache={cache_dir}  force={args.force}")
    paths: dict[str, Path] = {}
    for role, spec in jobs:
        paths[role] = resolve_model(
            spec,
            source=args.source,
            cache_dir=cache_dir,
            role=role,
            force=args.force,
        )

    log("done:")
    for role, path in paths.items():
        log(f"  {role}: {path}")
    if args.no_base:
        log("skipped Base; merge.py will download Qwen3.5-35B-A3B-Base on its own")
    else:
        log("next: python merge/merge.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
