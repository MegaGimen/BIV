#!/usr/bin/env python3
"""Chat Vector merge: AgentWorld as the base, Qwen3.5 Instruct as the overlay.

    θ = θ_AgentWorld + λ (θ_Instruct_lang − θ_Base)

Language tensors only. Visual keys from Instruct are dropped. AgentWorld is
never trimmed; optional DARE applies only to the agent task vector.

config.json stays AgentWorld (language-only). tokenizer / chat_template /
generation_config come from Instruct so Harbor Terminus tool format matches.

35B MoE is streamed shard-by-shard; three full copies are never loaded.

GPU (AutoDL)::

    python merge/download.py
    python merge/merge.py
    python merge/merge.py --lambda 0.7 --dare-density 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_MERGE_DIR = Path(__file__).resolve().parent
if str(_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_DIR))

from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_BASE,
    DEFAULT_CACHE,
    DEFAULT_WORLD,
    ROOT,
    resolve_model,
)

DEFAULT_OUT = ROOT / "merge" / "output" / "chatvector"

_VISUAL_PARTS = {
    "visual",
    "vision",
    "vision_tower",
    "vision_model",
    "mm_projector",
    "multi_modal_projector",
    "audio",
    "audio_tower",
}

_AGENT_TEXT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)

_WORLD_TEXT_FILES = (
    "config.json",
    "configuration.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def is_visual_key(name: str) -> bool:
    return any(part in _VISUAL_PARTS for part in name.split("."))


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SystemExit(f"Empty weight_map in {index}")
        return {str(k): str(v) for k, v in weight_map.items()}
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors found in {model_dir}")
    from safetensors import safe_open

    mapping: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                mapping[k] = shard.name
    return mapping


class TensorStore:
    """Memory-map safetensor shards; fetch one tensor at a time."""

    def __init__(self, model_dir: Path, weight_map: dict[str, str]) -> None:
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._handles: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        fname = self.weight_map.get(key)
        if fname is None:
            return None
        handle = self._handles.get(fname)
        if handle is None:
            path = self.model_dir / fname
            if not path.is_file():
                raise SystemExit(f"Missing shard {path}")
            from safetensors import safe_open

            handle = safe_open(str(path), framework="pt", device="cpu")
            self._handles[fname] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            exit_fn = getattr(handle, "__exit__", None)
            if callable(exit_fn):
                exit_fn(None, None, None)
                continue
            closer = getattr(handle, "close", None)
            if callable(closer):
                closer()
        self._handles.clear()


def apply_dare(tau: Any, density: float, generator: Any) -> Any:
    import torch

    if density >= 1.0:
        return tau
    if density <= 0.0:
        return torch.zeros_like(tau)
    mask = torch.rand(tau.shape, generator=generator, dtype=torch.float32) < density
    return tau * mask.to(tau.dtype) / density


def copy_existing(src_dir: Path, dest_dir: Path, names: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    for name in names:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)
            copied.append(name)
    return copied


def copy_custom_code(src_dir: Path, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    for src in src_dir.glob("*.py"):
        shutil.copy2(src, dest_dir / src.name)
        copied.append(src.name)
    return copied


def shard_groups(
    weight_map: dict[str, str], keys: list[str]
) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(weight_map[key], []).append(key)
    return [(fname, grouped[fname]) for fname in sorted(grouped)]


def merge(
    *,
    world_dir: Path,
    agent_dir: Path,
    base_dir: Path,
    out_dir: Path,
    lam: float,
    dare_density: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    world_map = load_weight_map(world_dir)
    agent_map = load_weight_map(agent_dir)
    base_map = load_weight_map(base_dir)

    world_keys = [k for k in world_map if not is_visual_key(k)]
    skipped_visual_world = sum(1 for k in world_map if is_visual_key(k))
    skipped_visual_agent = sum(1 for k in agent_map if is_visual_key(k))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_store = TensorStore(world_dir, world_map)
    agent_store = TensorStore(agent_dir, agent_map)
    base_store = TensorStore(base_dir, base_map)
    generator = torch.Generator()
    generator.manual_seed(seed)

    new_weight_map: dict[str, str] = {}
    total_size = 0
    n_merged = 0
    n_copied = 0
    n_missing = 0
    n_shape_mismatch = 0

    try:
        groups = shard_groups(world_map, world_keys)
        n_shards = len(groups)
        for i, (fname, keys) in enumerate(groups, start=1):
            log(f"shard {i}/{n_shards}: {fname} ({len(keys)} tensors)")
            tensors: dict[str, Any] = {}
            for key in keys:
                world_t = world_store.get(key)
                if world_t is None:
                    raise SystemExit(f"World is missing tensor {key}")
                agent_t = agent_store.get(key)
                base_t = base_store.get(key)
                if agent_t is None or base_t is None:
                    tensors[key] = world_t.contiguous()
                    n_copied += 1
                    n_missing += 1
                    continue
                if agent_t.shape != world_t.shape or base_t.shape != world_t.shape:
                    log(
                        f"  shape mismatch on {key}: "
                        f"world{tuple(world_t.shape)} agent{tuple(agent_t.shape)} "
                        f"base{tuple(base_t.shape)} → keep world"
                    )
                    tensors[key] = world_t.contiguous()
                    n_copied += 1
                    n_shape_mismatch += 1
                    continue
                w = world_t.to(torch.float32)
                tau = agent_t.to(torch.float32) - base_t.to(torch.float32)
                tau = apply_dare(tau, dare_density, generator)
                merged_t = (w + lam * tau).to(world_t.dtype).contiguous()
                tensors[key] = merged_t
                n_merged += 1
                del w, tau, merged_t, agent_t, base_t, world_t

            save_file(tensors, str(out_dir / fname), metadata={"format": "pt"})
            for key, tensor in tensors.items():
                new_weight_map[key] = fname
                total_size += int(tensor.nbytes)
            del tensors
    finally:
        world_store.close()
        agent_store.close()
        base_store.close()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )

    copied_world = copy_existing(world_dir, out_dir, _WORLD_TEXT_FILES)
    copied_world += copy_custom_code(world_dir, out_dir)
    copied_agent = copy_existing(agent_dir, out_dir, _AGENT_TEXT_FILES)
    for extra in sorted(agent_dir.glob("chat_template*")):
        if extra.is_file() and extra.name not in copied_agent:
            shutil.copy2(extra, out_dir / extra.name)
            copied_agent.append(extra.name)

    meta = {
        "method": "chat_vector",
        "formula": "world + lambda * (agent_lang - base)",
        "lambda": lam,
        "dare_density": dare_density,
        "dare_seed": seed,
        "world": str(world_dir),
        "agent": str(agent_dir),
        "base": str(base_dir),
        "n_language_keys": len(world_keys),
        "n_merged": n_merged,
        "n_copied_world_only": n_copied,
        "n_missing_agent_or_base": n_missing,
        "n_shape_mismatch": n_shape_mismatch,
        "n_visual_skipped_world": skipped_visual_world,
        "n_visual_skipped_agent": skipped_visual_agent,
        "copied_from_world": copied_world,
        "copied_from_agent": copied_agent,
        "total_size": total_size,
    }
    (out_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta


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
        help="Shared ancestor (Qwen3.5-35B-A3B-Base) hub id or local dir",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
    )
    p.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=1.0,
        help="Scale on the agent task vector (default 1.0)",
    )
    p.add_argument(
        "--dare-density",
        type=float,
        default=1.0,
        help="Keep this fraction of τ_agent (DARE). 1.0 = off.",
    )
    p.add_argument("--seed", type=int, default=0, help="DARE RNG seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.lam < 0:
        raise SystemExit("--lambda must be >= 0")
    if not (0.0 <= args.dare_density <= 1.0):
        raise SystemExit("--dare-density must be in [0, 1]")

    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)

    world_dir = resolve_model(args.world, source=args.source, cache_dir=cache_dir, role="world")
    agent_dir = resolve_model(args.agent, source=args.source, cache_dir=cache_dir, role="agent")
    base_dir = resolve_model(
        args.base_model, source=args.source, cache_dir=cache_dir, role="base"
    )

    log(
        f"Chat Vector: θ = AgentWorld + {args.lam} * (Instruct_lang - Base)"
        + (f", DARE density={args.dare_density}" if args.dare_density < 1.0 else "")
    )
    meta = merge(
        world_dir=world_dir,
        agent_dir=agent_dir,
        base_dir=base_dir,
        out_dir=out_dir,
        lam=args.lam,
        dare_density=args.dare_density,
        seed=args.seed,
    )
    log(f"wrote {out_dir}")
    log(
        f"merged={meta['n_merged']} copied={meta['n_copied_world_only']} "
        f"visual_skipped_agent={meta['n_visual_skipped_agent']}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
