#!/usr/bin/env python3
"""Compare AgentWorld / Instruct vs Base with Layer-Swapping row-MAV (text, not a heatmap).

Same delta as Bandarkar et al. 2410.01335: W_Δ = W_ft − W_pre, mean-abs per row, then
mean of rows. Reads ``merge/output/cache``. If Base (or any of the three) is missing
or incomplete, downloads it.

    python train/scripts/compare.py
    python train/scripts/compare.py --source huggingface

Writes ``train/outputs/compare/summary.txt`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_MERGE_DIR = ROOT / "merge"
_SRC = ROOT / "train" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_DIR))

from biv_wm.cut import decoder_layer_index  # noqa: E402
from biv_wm.mav import row_mav  # noqa: E402
from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_BASE,
    DEFAULT_CACHE,
    DEFAULT_SOURCE,
    DEFAULT_WORLD,
    resolve_model,
)
from merge import TensorStore, is_visual_key, load_weight_map  # noqa: E402

DEFAULT_OUT = ROOT / "train" / "outputs" / "compare"
BAR_W = 24
N_LAYERS = 40

ATTN_MARK = (
    "self_attn",
    "linear_attn",
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".o_proj.",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "output_gate_proj",
)
FFN_MARK = (".mlp.", "shared_expert", "experts.", "moe")


def log(msg: str) -> None:
    print(msg, flush=True)


def skip_key(key: str) -> bool:
    if is_visual_key(key):
        return True
    parts = set(key.split("."))
    if "mtp" in parts:
        return True
    return False


def leaf_group(key: str) -> str:
    if any(m in key for m in FFN_MARK):
        return "ffn"
    if any(m in key for m in ATTN_MARK):
        return "attn"
    if decoder_layer_index(key) is not None:
        return "other"
    return "nonlayer"


def bar(value: float, peak: float, width: int = BAR_W) -> str:
    if peak <= 0:
        return "." * width
    n = int(round(width * min(1.0, value / peak)))
    n = max(0, min(width, n))
    return "#" * n + "." * (width - n)


def load_layer_types(model_dir: Path) -> list[str]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return []
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    types = tc.get("layer_types") if isinstance(tc, dict) else None
    if isinstance(types, list) and types:
        return [str(x) for x in types]
    return []


def acc_mav(bucket: dict[str, float], mav: float, n_rows: int) -> None:
    bucket["sum"] += mav * n_rows
    bucket["n_rows"] += n_rows
    bucket["n_tensors"] += 1
    bucket["sum_unweighted"] += mav


def finalize(bucket: dict[str, float]) -> dict[str, float]:
    n = bucket["n_rows"]
    nt = bucket["n_tensors"]
    return {
        "n_tensors": int(nt),
        "n_rows": int(n),
        "mav": (bucket["sum"] / n) if n else 0.0,
        "mav_unweighted": (bucket["sum_unweighted"] / nt) if nt else 0.0,
    }


def empty_bucket() -> dict[str, float]:
    return {"sum": 0.0, "n_rows": 0.0, "n_tensors": 0.0, "sum_unweighted": 0.0}


def n_rows_of(t: Any) -> int:
    if t.ndim <= 1:
        return 1
    return int(t.reshape(-1, t.shape[-1]).shape[0])


def run_compare(
    world_dir: Path,
    agent_dir: Path,
    base_dir: Path,
) -> dict[str, Any]:
    world_map = load_weight_map(world_dir)
    agent_map = load_weight_map(agent_dir)
    base_map = load_weight_map(base_dir)

    keys = sorted(
        set(world_map) & set(agent_map) & set(base_map)
    )
    keys = [k for k in keys if not skip_key(k)]

    world_store = TensorStore(world_dir, world_map)
    agent_store = TensorStore(agent_dir, agent_map)
    base_store = TensorStore(base_dir, base_map)

    per_layer: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: {"aw": empty_bucket(), "inst": empty_bucket()}
    )
    per_layer_group: dict[tuple[int, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"aw": empty_bucket(), "inst": empty_bucket()}
    )
    special: dict[str, dict[str, float]] = {}
    skipped = {"shape": 0, "missing": 0, "visual_mtp": 0}

    types = load_layer_types(world_dir) or load_layer_types(agent_dir)

    try:
        n = len(keys)
        for i, key in enumerate(keys, start=1):
            if i == 1 or i % 80 == 0 or i == n:
                log(f"  tensors {i}/{n}: {key}")
            wt = world_store.get(key)
            at = agent_store.get(key)
            bt = base_store.get(key)
            if wt is None or at is None or bt is None:
                skipped["missing"] += 1
                continue
            if tuple(wt.shape) != tuple(bt.shape) or tuple(at.shape) != tuple(bt.shape):
                skipped["shape"] += 1
                continue
            daw = wt.float() - bt.float()
            dag = at.float() - bt.float()
            mav_aw = row_mav(daw)
            mav_inst = row_mav(dag)
            rows = n_rows_of(wt)
            li = decoder_layer_index(key)
            grp = leaf_group(key)
            if li is not None:
                acc_mav(per_layer[li]["aw"], mav_aw, rows)
                acc_mav(per_layer[li]["inst"], mav_inst, rows)
                acc_mav(per_layer_group[(li, grp)]["aw"], mav_aw, rows)
                acc_mav(per_layer_group[(li, grp)]["inst"], mav_inst, rows)
            else:
                special[key] = {"aw": mav_aw, "instruct": mav_inst, "n_rows": rows}
            del wt, at, bt, daw, dag
    finally:
        world_store.close()
        agent_store.close()
        base_store.close()

    layers = []
    for i in range(N_LAYERS):
        if i not in per_layer:
            continue
        aw = finalize(per_layer[i]["aw"])
        inst = finalize(per_layer[i]["inst"])
        groups = {}
        for g in ("attn", "ffn", "other"):
            b = per_layer_group.get((i, g))
            if not b:
                continue
            groups[g] = {"aw": finalize(b["aw"]), "instruct": finalize(b["inst"])}
        kind = types[i] if i < len(types) else "?"
        layers.append(
            {
                "layer": i,
                "kind": kind,
                "aw": aw,
                "instruct": inst,
                "groups": groups,
            }
        )

    return {
        "method": (
            "row-level MAV of (W_ft - W_base), Bandarkar et al. arXiv:2410.01335; "
            "text bars instead of a color plot"
        ),
        "n_keys": len(keys),
        "skipped": skipped,
        "layers": layers,
        "special": special,
        "paths": {
            "world": str(world_dir),
            "instruct": str(agent_dir),
            "base": str(base_dir),
        },
    }


def format_summary(report: dict[str, Any]) -> str:
    layers: list[dict[str, Any]] = report.get("layers") or []
    peak_aw = max((float(r["aw"]["mav"]) for r in layers), default=0.0)
    peak_inst = max((float(r["instruct"]["mav"]) for r in layers), default=0.0)
    peak = max(peak_aw, peak_inst, 1e-12)

    lines: list[str] = []
    lines.append("compare.py — 改得狠度 = 相对 Base 的行均值绝对差 (Layer Swapping MAV)")
    lines.append(report["method"])
    lines.append(f"world     {report['paths']['world']}")
    lines.append(f"instruct  {report['paths']['instruct']}")
    lines.append(f"base      {report['paths']['base']}")
    lines.append(f"keys={report['n_keys']} skipped={report['skipped']}")
    lines.append("")
    lines.append(
        f"{'L':>3} {'kind':<16} {'AW_MAV':>10} {'Inst_MAV':>10}  "
        f"{'AW':<{BAR_W}}  {'Instruct':<{BAR_W}}"
    )
    lines.append("-" * (3 + 1 + 16 + 1 + 10 + 1 + 10 + 2 + BAR_W + 2 + BAR_W))
    for r in layers:
        i = int(r["layer"])
        aw = float(r["aw"]["mav"])
        inst = float(r["instruct"]["mav"])
        lines.append(
            f"{i:3d} {str(r['kind'])[:16]:<16} {aw:10.4e} {inst:10.4e}  "
            f"{bar(aw, peak)}  {bar(inst, peak)}"
        )

    lines.append("")
    lines.append("per layer split attn vs ffn (same MAV):")
    lines.append(
        f"{'L':>3} {'g':<5} {'AW_MAV':>10} {'Inst_MAV':>10}  "
        f"{'AW':<{BAR_W}}  {'Instruct':<{BAR_W}}"
    )
    for r in layers:
        i = int(r["layer"])
        for g in ("attn", "ffn"):
            gg = (r.get("groups") or {}).get(g)
            if not gg:
                continue
            aw = float(gg["aw"]["mav"])
            inst = float(gg["instruct"]["mav"])
            lines.append(
                f"{i:3d} {g:<5} {aw:10.4e} {inst:10.4e}  "
                f"{bar(aw, peak)}  {bar(inst, peak)}"
            )

    def topn(role: str, n: int = 8) -> list[str]:
        scored = [(float(r[role]["mav"]), int(r["layer"])) for r in layers]
        scored.sort(reverse=True)
        return [f"L{i}={v:.4e}" for v, i in scored[:n]]

    lines.append("")
    lines.append("AgentWorld 相对 Base 最狠: " + ", ".join(topn("aw")))
    lines.append("Instruct  相对 Base 最狠: " + ", ".join(topn("instruct")))

    spec = report.get("special") or {}
    if spec:
        lines.append("")
        lines.append("non-layer tensors (embed / lm_head / norm):")
        for k, v in sorted(spec.items()):
            lines.append(
                f"  {k}: AW={v['aw']:.4e}  Instruct={v['instruct']:.4e}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--agent", default=DEFAULT_AGENT, help="Instruct hub id or local dir")
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=DEFAULT_SOURCE,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"source={args.source} cache={cache_dir}")
    world_dir = resolve_model(
        args.world, source=args.source, cache_dir=cache_dir, role="world"
    )
    agent_dir = resolve_model(
        args.agent, source=args.source, cache_dir=cache_dir, role="instruct"
    )
    base_dir = resolve_model(
        args.base_model, source=args.source, cache_dir=cache_dir, role="base"
    )

    log("streaming row-MAV vs Base (one tensor at a time)")
    report = run_compare(world_dir, agent_dir, base_dir)
    text = format_summary(report)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(text)
    log(f"wrote {out_dir / 'summary.txt'}")
    log(f"wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
