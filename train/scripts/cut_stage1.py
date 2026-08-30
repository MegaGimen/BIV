#!/usr/bin/env python3
"""Write the Stage 1 backbone: AgentWorld[:ℓ] + Instruct[ℓ:] + Instruct lm_head.

Drops ViT and official MTP. Reads merge/output/cache. ℓ comes from
train/outputs/probe/report.json (group-boundary rule) unless --cut is set.

CPU streaming; no GPU. Then train JEPA with train/scripts/train_jepa.py.

  python train/scripts/cut_stage1.py
  python train/scripts/cut_stage1.py --cut 12
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
MERGE = ROOT / "merge"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(MERGE))

from biv_wm.cut import pick_cut, tensor_source  # noqa: E402
from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_BASE,
    DEFAULT_CACHE,
    DEFAULT_WORLD,
    resolve_model,
)
from merge import (  # noqa: E402
    TensorStore,
    _AGENT_TEXT_FILES,
    _WORLD_TEXT_FILES,
    copy_custom_code,
    copy_existing,
    load_weight_map,
    log,
    shard_groups,
)

DEFAULT_PROBE = TRAIN / "outputs" / "probe" / "report.json"
DEFAULT_OUT = TRAIN / "outputs" / "stage1_cut"
DEFAULT_MIX = TRAIN / "data" / "processed" / "mix_v2"


def load_ell(report_path: Path, override: int | None) -> tuple[int, dict[str, Any]]:
    if override is not None:
        return override, {"ell": override, "rule": "cli --cut"}
    if not report_path.is_file():
        raise SystemExit(
            f"No {report_path}; run python train/scripts/probe.py first, or pass --cut"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    layers = (report.get("weights") or {}).get("per_layer") or []
    if layers:
        picked = pick_cut(layers)
        return int(picked["ell"]), picked
    rec = report.get("recommended_cut") or {}
    if rec.get("ell") is not None and not rec.get("error"):
        return int(rec["ell"]), rec
    raise SystemExit(
        f"{report_path} has no per_layer stats; re-run probe.py without --skip-weights, or pass --cut"
    )


def write_cut(
    *,
    world_dir: Path,
    agent_dir: Path,
    out_dir: Path,
    ell: int,
    picked: dict[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    world_map = load_weight_map(world_dir)
    agent_map = load_weight_map(agent_dir)
    world_store = TensorStore(world_dir, world_map)
    agent_store = TensorStore(agent_dir, agent_map)

    keep = [k for k in world_map if tensor_source(k, ell) != "drop"]
    n_world = n_inst = 0
    new_weight_map: dict[str, str] = {}
    total_size = 0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        groups = shard_groups(world_map, keep)
        n_shards = len(groups)
        for i, (fname, keys) in enumerate(groups, start=1):
            log(f"shard {i}/{n_shards}: {fname} ({len(keys)} tensors)")
            tensors: dict[str, Any] = {}
            for key in keys:
                src = tensor_source(key, ell)
                if src == "instruct":
                    t = agent_store.get(key)
                    if t is None:
                        raise SystemExit(f"Instruct missing {key}")
                    n_inst += 1
                else:
                    t = world_store.get(key)
                    if t is None:
                        raise SystemExit(f"AgentWorld missing {key}")
                    n_world += 1
                tensors[key] = t.contiguous()
            save_file(tensors, str(out_dir / fname), metadata={"format": "pt"})
            for key, tensor in tensors.items():
                new_weight_map[key] = fname
                total_size += int(tensor.nbytes)
            del tensors
    finally:
        world_store.close()
        agent_store.close()

    index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    copied_world = copy_existing(world_dir, out_dir, _WORLD_TEXT_FILES)
    copied_world += copy_custom_code(world_dir, out_dir)
    copied_agent = copy_existing(agent_dir, out_dir, _AGENT_TEXT_FILES)
    for extra in sorted(agent_dir.glob("chat_template*")):
        if extra.is_file() and extra.name not in copied_agent:
            shutil.copy2(extra, out_dir / extra.name)
            copied_agent.append(extra.name)

    meta = {
        "stage": 1,
        "ell": ell,
        "pick": picked,
        "formula": f"language_model.layers[0:{ell}] AgentWorld; [{ell}:40] Instruct; lm_head+final_norm Instruct; drop visual+mtp",
        "n_tensors": len(keep),
        "n_from_world": n_world,
        "n_from_instruct": n_inst,
        "world": str(world_dir),
        "instruct": str(agent_dir),
        "copied_from_world": copied_world,
        "copied_from_agent": copied_agent,
        "total_size": total_size,
        "mix_dir": str(DEFAULT_MIX),
        "next": "python train/scripts/train_jepa.py --config train/configs/jepa/stage1.yaml",
    }
    (out_dir / "cut_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--probe-report", type=Path, default=DEFAULT_PROBE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cut", type=int, default=None, help="Override ℓ (0..39). Default: pick from probe report.")
    p.add_argument(
        "--print-cut",
        action="store_true",
        help="Print ℓ from the probe report and exit (no 70GB write).",
    )
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)
    report_path = args.probe_report if args.probe_report.is_absolute() else (ROOT / args.probe_report)
    if args.cut is not None and not (0 <= args.cut <= 39):
        raise SystemExit("--cut must be in 0..39")

    ell, picked = load_ell(report_path, args.cut)
    log(f"ℓ={ell}  {picked.get('rule')}")
    if args.print_cut:
        print(json.dumps({"ell": ell, "pick": picked}, indent=2, ensure_ascii=False))
        return
    world_dir = resolve_model(args.world, source=args.source, cache_dir=cache_dir, role="world")
    agent_dir = resolve_model(args.agent, source=args.source, cache_dir=cache_dir, role="agent")
    meta = write_cut(
        world_dir=world_dir,
        agent_dir=agent_dir,
        out_dir=out_dir,
        ell=ell,
        picked=picked,
    )
    log(f"wrote {out_dir}")
    log(f"from_world={meta['n_from_world']} from_instruct={meta['n_from_instruct']}")
    log(meta["next"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
