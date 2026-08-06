"""Hub / local-cache loaders for SWE-Hero, SWE-Zero, ISETrace.

Prefer already-downloaded snapshots (HF hub cache, ModelScope, explicit --local-dir).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from biv_wm.data import (
    DEFAULT_SWE_HERO_HF,
    DEFAULT_SWE_HERO_MS,
    open_swe_hero_dataset,
)

DEFAULT_SWE_ZERO_HF = "nvidia/SWE-Zero-openhands-trajectories"
DEFAULT_SWE_ZERO_MS = "nv-community/SWE-Zero-openhands-trajectories"
DEFAULT_ISETRACE_HF = "valiere/ISETrace"
DEFAULT_ISETRACE_MS = "valiere/ISETrace"  # override if a ModelScope mirror exists


def _hf_hub_snapshot_dirs(repo_id: str) -> list[Path]:
    """Candidate snapshot dirs for datasets--org--name under HF_HOME."""
    safe = f"datasets--{repo_id.replace('/', '--')}"
    roots: list[Path] = []
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    roots.append(hf_home / "hub" / safe / "snapshots")
    # Common alternate layouts
    roots.append(Path.home() / ".cache" / "huggingface" / "hub" / safe / "snapshots")
    out: list[Path] = []
    for snap_root in roots:
        if not snap_root.is_dir():
            continue
        for d in sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir():
                out.append(d)
    return out


def _try_load_local_dataset(path: Path, *, split: str, name: str | None = None):
    from datasets import DatasetDict, load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if name:
        # config name for multi-config datasets (ISETrace trajectories/intents)
        try:
            return load_dataset(str(path), name, **kwargs)
        except TypeError:
            return load_dataset(str(path), name=name, **kwargs)
    ds = load_dataset(str(path), **kwargs)
    if isinstance(ds, DatasetDict):
        key = split if split in ds else next(iter(ds.keys()))
        return ds[key]
    return ds


def open_dataset_with_cache(
    *,
    kind: str,
    source: str = "modelscope",
    repo_id: str | None = None,
    split: str = "train",
    max_rows: int | None = None,
    local_dir: Path | None = None,
    config_name: str | None = None,
):
    """Load HF Dataset, reusing local snapshots when present.

    kind: swe_hero | swe_zero | isetrace
    """
    from datasets import DatasetDict, load_dataset

    kind = kind.strip().lower()
    src = (source or "modelscope").strip().lower()

    if kind == "swe_hero":
        if local_dir is not None:
            ds = _try_load_local_dataset(Path(local_dir), split=split)
        else:
            # Always try a complete HF hub snapshot first (reuse prior SWE-Hero downloads).
            rid_hf = DEFAULT_SWE_HERO_HF if not repo_id or "SWE-Hero" in repo_id else repo_id
            local_hit = None
            for snap in _hf_hub_snapshot_dirs(rid_hf):
                files = [
                    p for p in snap.rglob("*") if p.is_file() and p.stat().st_size > 10_000
                ]
                if files:
                    local_hit = snap
                    break
            if local_hit is not None:
                print(f"Reusing HF hub snapshot: {local_hit}", flush=True)
                try:
                    ds = _try_load_local_dataset(local_hit, split=split)
                except Exception as exc:  # noqa: BLE001
                    print(f"Local HF snap load failed ({exc!r}); falling back", flush=True)
                    local_hit = None
            if local_hit is None:
                ds = open_swe_hero_dataset(
                    split=split,
                    max_rows=None,
                    source=(
                        "huggingface" if src in {"huggingface", "hf"} else "modelscope"
                    ),
                    repo_id=repo_id,
                )
    elif kind == "swe_zero":
        rid = repo_id or (
            DEFAULT_SWE_ZERO_MS if src in {"modelscope", "ms"} else DEFAULT_SWE_ZERO_HF
        )
        if local_dir is not None:
            ds = _try_load_local_dataset(Path(local_dir), split=split)
        elif src in {"modelscope", "ms"}:
            ds = _load_modelscope_dataset(rid, split=split)
        else:
            # try local HF cache first
            for snap in _hf_hub_snapshot_dirs(DEFAULT_SWE_ZERO_HF):
                files = [p for p in snap.rglob("*") if p.is_file() and p.stat().st_size > 10_000]
                if files:
                    print(f"Reusing HF hub snapshot: {snap}", flush=True)
                    ds = _try_load_local_dataset(snap, split=split)
                    break
            else:
                print(f"load_dataset({DEFAULT_SWE_ZERO_HF!r})", flush=True)
                ds = load_dataset(DEFAULT_SWE_ZERO_HF, split=split)
    elif kind == "isetrace":
        cfg = config_name or "trajectories"
        rid = repo_id or (
            DEFAULT_ISETRACE_MS if src in {"modelscope", "ms"} else DEFAULT_ISETRACE_HF
        )
        if local_dir is not None:
            ds = _try_load_local_dataset(Path(local_dir), split=split, name=cfg)
        elif src in {"modelscope", "ms"}:
            try:
                ds = _load_modelscope_dataset(rid, split=split, config_name=cfg)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"ModelScope ISETrace failed ({exc!r}); falling back to HuggingFace",
                    flush=True,
                )
                ds = load_dataset(DEFAULT_ISETRACE_HF, cfg, split=split)
        else:
            for snap in _hf_hub_snapshot_dirs(DEFAULT_ISETRACE_HF):
                files = [p for p in snap.rglob("*") if p.is_file() and p.stat().st_size > 10_000]
                if files:
                    print(f"Reusing HF hub snapshot: {snap}", flush=True)
                    try:
                        ds = _try_load_local_dataset(snap, split=split, name=cfg)
                        break
                    except Exception:
                        continue
            else:
                print(f"load_dataset({DEFAULT_ISETRACE_HF!r}, {cfg!r})", flush=True)
                ds = load_dataset(DEFAULT_ISETRACE_HF, cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset kind={kind!r}")

    if isinstance(ds, DatasetDict):
        key = split if split in ds else next(iter(ds.keys()))
        ds = ds[key]
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    print(f"Opened {kind}: rows={len(ds)} columns={ds.column_names}", flush=True)
    return ds


def _load_modelscope_dataset(repo_id: str, *, split: str, config_name: str | None = None):
    from datasets import Dataset, DatasetDict, load_dataset

    local_dir = None
    try:
        try:
            from modelscope.hub.snapshot_download import dataset_snapshot_download
        except ImportError:
            from modelscope import dataset_snapshot_download  # type: ignore

        print(f"ModelScope dataset_snapshot_download: {repo_id}", flush=True)
        try:
            local_dir = dataset_snapshot_download(repo_id, repo_type="dataset")
        except TypeError:
            local_dir = dataset_snapshot_download(repo_id)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        print(f"snapshot_download failed ({exc!r})", flush=True)
        raise

    if config_name:
        return _try_load_local_dataset(Path(local_dir), split=split, name=config_name)
    try:
        return load_dataset(str(local_dir), split=split)
    except Exception:
        from modelscope.msdatasets import MsDataset

        raw = MsDataset.load(repo_id, split=split)
        if hasattr(raw, "to_hf_dataset"):
            out = raw.to_hf_dataset()
        elif isinstance(raw, (Dataset, DatasetDict)):
            out = raw
        else:
            raise TypeError(f"Unsupported MsDataset type {type(raw)}")
        if isinstance(out, DatasetDict):
            key = split if split in out else next(iter(out.keys()))
            return out[key]
        return out
