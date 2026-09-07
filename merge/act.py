#!/usr/bin/env python3
"""ACT masked task-vector merge: write AgentWorld channels into Instruct.

Ability model = AgentWorld. Target = Instruct (the agent). Same tokens, same
slots: only the top-p% ACT mask rows move.

    θ_i^{merged} = θ_i^{Instruct} + λ (θ_i^{AgentWorld} − θ_i^{Instruct})

Unmasked rows stay Instruct. Tokenizer / config stay Instruct so Harbor
Terminus still talks to the command mouth. Default λ=0.4 (ACT transfer-only).

Mask comes from ``python train/scripts/compare_act.py`` →
``train/outputs/act/mask.json``.

    python merge/act.py
    python merge/act.py --lambda 0.4 --no-lm-head
    python merge/act.py --no-lm-head --workers 2
    python merge/eval.py --act --max-model-len 32768
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_MERGE_DIR = Path(__file__).resolve().parent
ROOT = _MERGE_DIR.parent
_SRC = ROOT / "train" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_DIR))

from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_CACHE,
    DEFAULT_WORLD,
    resolve_model,
)
from merge import (  # noqa: E402
    TensorStore,
    is_visual_key,
    load_weight_map,
    shard_groups,
)

from biv_wm.act import (  # noqa: E402
    blend_masked_channels,
    channel_axis,
    group_mask_channels,
    load_mask_rows,
    param_canonical_key,
    world_param_candidates,
)

DEFAULT_MASK = ROOT / "train" / "outputs" / "act" / "mask.json"
DEFAULT_OUT = ROOT / "merge" / "output" / "act"
DEFAULT_LAMBDA = 0.4


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_world_key(instruct_key: str, world_map: dict[str, str]) -> str | None:
    for cand in world_param_candidates(instruct_key):
        if cand in world_map:
            return cand
    return None


def copy_instruct_sidecars(agent_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    skip_names = {"model.safetensors.index.json"}
    for src in agent_dir.iterdir():
        if src.name in skip_names or src.suffix == ".safetensors":
            continue
        dest = out_dir / src.name
        if src.is_file():
            shutil.copy2(src, dest)
            copied.append(src.name)
        elif src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            copied.append(src.name + "/")
    return copied


def _process_instruct_shard(
    *,
    fname: str,
    keys: list[str],
    patch_keys: dict[str, list[int]],
    world_dir: Path,
    agent_dir: Path,
    out_dir: Path,
    world_map: dict[str, str],
    lam: float,
) -> dict[str, Any]:
    """Copy or rewrite one Instruct shard. Own TensorStore handles (thread-safe)."""
    from safetensors.torch import save_file

    need = [k for k in keys if k in patch_keys]
    if not need:
        shutil.copy2(agent_dir / fname, out_dir / fname)
        return {
            "fname": fname,
            "copied": True,
            "n_keys": len(keys),
            "n_patched": 0,
            "n_rows": 0,
            "n_missing_world": 0,
            "n_shape_mismatch": 0,
            "n_oob_skipped": 0,
            "nbytes": 0,
            "examples": [],
            "missing_examples": [],
            "weight_keys": list(keys),
        }

    world_store = TensorStore(world_dir, world_map)
    agent_store = TensorStore(agent_dir, agent_map)
    n_patched = 0
    n_rows = 0
    n_missing_world = 0
    n_shape_mismatch = 0
    n_oob_skipped = 0
    examples: list[str] = []
    missing_examples: list[str] = []
    tensors: dict[str, Any] = {}
    try:
        for key in keys:
            agent_t = agent_store.get(key)
            if agent_t is None:
                raise SystemExit(f"Instruct is missing tensor {key}")
            chans = patch_keys.get(key)
            if not chans or is_visual_key(key):
                tensors[key] = agent_t.contiguous()
                continue
            world_name = resolve_world_key(key, world_map)
            if world_name is None:
                tensors[key] = agent_t.contiguous()
                n_missing_world += 1
                if len(missing_examples) < 12:
                    missing_examples.append(key)
                continue
            world_t = world_store.get(world_name)
            if world_t is None:
                tensors[key] = agent_t.contiguous()
                n_missing_world += 1
                continue
            if tuple(world_t.shape) != tuple(agent_t.shape):
                tensors[key] = agent_t.contiguous()
                n_shape_mismatch += 1
                continue
            axis = channel_axis(key, int(agent_t.ndim))
            n_axis = int(agent_t.shape[axis])
            valid = [c for c in chans if 0 <= c < n_axis]
            n_oob_skipped += len(chans) - len(valid)
            if not valid:
                tensors[key] = agent_t.contiguous()
                continue
            merged = blend_masked_channels(
                agent_t, world_t, valid, lam, axis=axis
            )
            tensors[key] = merged.contiguous()
            n_patched += 1
            n_rows += len(valid)
            if len(examples) < 16:
                examples.append(f"{key} axis={axis} n={len(valid)}")
            del merged, world_t, agent_t
        save_file(tensors, str(out_dir / fname), metadata={"format": "pt"})
        nbytes = sum(int(t.nbytes) for t in tensors.values())
    finally:
        world_store.close()
        agent_store.close()
        del tensors
    return {
        "fname": fname,
        "copied": False,
        "n_keys": len(keys),
        "n_patched": n_patched,
        "n_rows": n_rows,
        "n_missing_world": n_missing_world,
        "n_shape_mismatch": n_shape_mismatch,
        "n_oob_skipped": n_oob_skipped,
        "nbytes": nbytes,
        "examples": examples,
        "missing_examples": missing_examples,
        "weight_keys": list(keys),
    }


def merge_act(
    *,
    world_dir: Path,
    agent_dir: Path,
    out_dir: Path,
    mask_path: Path,
    lam: float,
    no_lm_head: bool,
    workers: int = 1,
) -> dict[str, Any]:

    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    rows = load_mask_rows(payload, no_lm_head=no_lm_head)
    by_module = group_mask_channels(rows)
    n_mask_rows = sum(len(v) for v in by_module.values())
    n_lm_head = sum(
        len(v) for k, v in by_module.items() if k == "lm_head" or k.endswith(".lm_head")
    )

    agent_map = load_weight_map(agent_dir)
    world_map = load_weight_map(world_dir)

    patch_keys: dict[str, list[int]] = {}
    unmatched_modules = set(by_module)
    for key in agent_map:
        canon = param_canonical_key(key)
        chans = by_module.get(canon)
        if not chans:
            continue
        unmatched_modules.discard(canon)
        patch_keys[key] = chans

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied_sidecars = copy_instruct_sidecars(agent_dir, out_dir)

    n_copied_shards = 0
    n_rewritten_shards = 0
    n_tensors_patched = 0
    n_rows_written = 0
    n_missing_world = 0
    n_shape_mismatch = 0
    n_oob_skipped = 0
    patched_examples: list[str] = []
    missing_examples: list[str] = []

    new_weight_map: dict[str, str] = {}
    total_size = 0
    workers = max(1, int(workers))

    groups = shard_groups(agent_map, list(agent_map))
    n_shards = len(groups)
    log(f"shards={n_shards} workers={workers}")

    def _run_one(item: tuple[int, str, list[str]]) -> tuple[int, dict[str, Any]]:
        i, fname, keys = item
        result = _process_instruct_shard(
            fname=fname,
            keys=keys,
            patch_keys=patch_keys,
            world_dir=world_dir,
            agent_dir=agent_dir,
            out_dir=out_dir,
            world_map=world_map,
            lam=lam,
        )
        return i, result

    jobs = [(i, fname, keys) for i, (fname, keys) in enumerate(groups, start=1)]
    if workers == 1:
        ordered = [_run_one(job) for job in jobs]
    else:
        ordered = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run_one, job): job[0] for job in jobs}
            for fut in as_completed(futs):
                ordered.append(fut.result())
        ordered.sort(key=lambda x: x[0])

    for i, result in ordered:
        fname = result["fname"]
        for key in result["weight_keys"]:
            new_weight_map[key] = fname
        n_missing_world += int(result["n_missing_world"])
        n_shape_mismatch += int(result["n_shape_mismatch"])
        n_oob_skipped += int(result["n_oob_skipped"])
        if len(patched_examples) < 16:
            patched_examples.extend(result["examples"][: 16 - len(patched_examples)])
        if len(missing_examples) < 12:
            missing_examples.extend(
                result["missing_examples"][: 12 - len(missing_examples)]
            )
        if result["copied"]:
            n_copied_shards += 1
            log(f"shard {i}/{n_shards}: {fname} copy ({result['n_keys']} tensors)")
            continue
        n_rewritten_shards += 1
        n_tensors_patched += int(result["n_patched"])
        n_rows_written += int(result["n_rows"])
        total_size += int(result["nbytes"])
        log(
            f"shard {i}/{n_shards}: {fname} patch "
            f"{result['n_patched']}/{result['n_keys']} tensors"
        )

    if len(new_weight_map) != len(agent_map):
        raise SystemExit(
            f"weight_map size {len(new_weight_map)} != instruct {len(agent_map)}"
        )

    src_index = agent_dir / "model.safetensors.index.json"
    orig_size = None
    if src_index.is_file():
        try:
            orig_size = json.loads(src_index.read_text(encoding="utf-8")).get(
                "metadata", {}
            ).get("total_size")
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            orig_size = None
    if orig_size is None:
        orig_size = 0
        for name in set(new_weight_map.values()):
            shard_path = out_dir / name
            if shard_path.is_file():
                orig_size += shard_path.stat().st_size
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": orig_size},
                "weight_map": new_weight_map,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lm_frac = (n_lm_head / n_mask_rows) if n_mask_rows else 0.0
    if (not no_lm_head) and lm_frac >= 0.5:
        log(
            f"NOTE: lm_head is {lm_frac:.0%} of the mask ({n_lm_head}/{n_mask_rows}). "
            "ACT includes it. Pass --no-lm-head to keep Instruct's vocab rows."
        )

    meta = {
        "method": "act_masked_task_vector",
        "formula": "instruct + lambda * (world - instruct) on mask rows only",
        "lambda": lam,
        "no_lm_head": no_lm_head,
        "mask": str(mask_path.resolve()),
        "world": str(world_dir),
        "agent": str(agent_dir),
        "n_mask_rows": n_mask_rows,
        "n_mask_modules": len(by_module),
        "n_lm_head_rows": n_lm_head,
        "n_tensors_patched": n_tensors_patched,
        "n_rows_written": n_rows_written,
        "n_copied_shards": n_copied_shards,
        "n_rewritten_bytes": total_size,
        "n_rewritten_shards": n_rewritten_shards,
        "n_missing_world": n_missing_world,
        "n_shape_mismatch": n_shape_mismatch,
        "n_oob_skipped": n_oob_skipped,
        "n_unmatched_modules": len(unmatched_modules),
        "unmatched_modules": sorted(unmatched_modules)[:32],
        "patched_examples": patched_examples,
        "missing_world_examples": missing_examples,
        "copied_sidecars": copied_sidecars,
        "workers": workers,
        "p": payload.get("p"),
        "mask_threshold": payload.get("mask_threshold"),
    }
    (out_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(mask_path, out_dir / "mask.json")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument(
        "--mask",
        type=Path,
        default=DEFAULT_MASK,
        help="compare_act mask.json (default: train/outputs/act/mask.json)",
    )
    p.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=DEFAULT_LAMBDA,
        help="ACT transfer scale (paper transfer-only ≈ 0.4)",
    )
    p.add_argument(
        "--no-lm-head",
        action="store_true",
        help="Use mask_no_lm_head (keep Instruct's vocab rows)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel shard rewrite (default 2; matches two GPUs' disk bandwidth)",
    )
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.lam < 0:
        raise SystemExit("--lambda must be >= 0")
    mask_path = args.mask if args.mask.is_absolute() else (ROOT / args.mask)
    if not mask_path.is_file():
        raise SystemExit(
            f"No ACT mask at {mask_path}\n"
            "Run: python train/scripts/compare_act.py"
        )
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)

    world_dir = resolve_model(args.world, source=args.source, cache_dir=cache_dir, role="world")
    agent_dir = resolve_model(args.agent, source=args.source, cache_dir=cache_dir, role="agent")

    log(
        f"ACT merge → Instruct  λ={args.lam}"
        + ("  (no lm_head)" if args.no_lm_head else "")
    )
    log(f"  mask:  {mask_path}")
    log(f"  world: {world_dir}")
    log(f"  agent: {agent_dir}")
    log(f"  out:   {out_dir}")
    meta = merge_act(
        world_dir=Path(world_dir),
        agent_dir=Path(agent_dir),
        out_dir=out_dir,
        mask_path=mask_path,
        lam=args.lam,
        no_lm_head=bool(args.no_lm_head),
        workers=max(1, args.workers),
    )
    log(f"wrote {out_dir}")
    log(
        f"patched_tensors={meta['n_tensors_patched']} "
        f"rows={meta['n_rows_written']} "
        f"rewritten_shards={meta['n_rewritten_shards']} "
        f"copied_shards={meta['n_copied_shards']}"
    )
    log("Serve: python merge/eval.py --act --max-model-len 32768")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
