#!/usr/bin/env python3
"""CPU probe: which MoE channels compare_act actually covers.

Does **not** load 35B onto a GPU and does not run a forward. It reads
safetensors headers + config, then walks ``named_modules`` on meta if
transformers is around.

The 20-row smoke already showed the gap: 272 hooked modules, kinds =
attn / ln / embed, **no ffn**. This script dumps why, on the machine that
has ``merge/output/cache``.

  cd train
  bash scripts/probe_act_moe.sh
  python scripts/probe_act_moe.py --skip-modules   # headers only, seconds

Writes ``train/outputs/act_moe_probe/report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
MERGE = ROOT / "merge"
SCRIPTS = Path(__file__).resolve().parent
for p in (str(SRC), str(MERGE), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_CACHE,
    DEFAULT_WORLD,
    cache_path,
    checkpoint_ready,
)
from merge import load_weight_map  # noqa: E402

from biv_wm.act import (  # noqa: E402
    canonical_module_key,
    channel_axis,
    hook_kind,
    param_canonical_key,
)

DEFAULT_OUT = TRAIN / "outputs" / "act_moe_probe"

# Smoke 20-row dual-GPU run: captured modules / kinds. Used as a checksum
# against the header walk, not as a substitute for it.
SMOKE_CAPTURED_MODULES = 272
SMOKE_KINDS = ("attn", "ln", "embed")


def log(msg: str) -> None:
    print(msg, flush=True)


def classify_weight_key(key: str) -> str:
    """Bucket a safetensors key for MoE / ACT coverage.

    Packed routed experts are ``nn.Parameter`` 3-D tensors on
    ``mlp.experts``, not ``nn.Linear`` children named ``experts.0.down_proj``.
    """
    n = key
    if any(s in n for s in ("visual", "vision", "mtp.", "rotary")):
        return "skip"
    if ".mlp.experts.gate_up_proj" in n or n.endswith(".mlp.experts.gate_up_proj"):
        return "packed_expert_gate_up"
    if ".mlp.experts.down_proj" in n or n.endswith(".mlp.experts.down_proj"):
        return "packed_expert_down"
    if ".mlp.experts." in n:
        return "packed_expert_other"
    if n.endswith(".mlp.experts") or ".mlp.experts.weight" in n:
        return "packed_expert_other"
    if ".shared_expert_gate" in n:
        return "shared_expert_gate"
    if ".shared_expert." in n:
        leaf = n.rsplit(".", 1)[-1]
        if leaf in {"weight", "bias"}:
            leaf = n.rsplit(".", 2)[-2]
        if leaf in {"gate_proj", "up_proj", "down_proj"}:
            return "shared_expert_ffn"
        return "shared_expert_other"
    if ".mlp.gate." in n or n.endswith(".mlp.gate.weight"):
        return "router"
    module = param_canonical_key(key)
    kind = hook_kind(module)
    if kind is not None:
        return f"hooked_{kind}"
    return "other"


def module_path_for_param(key: str) -> str:
    """Activation-side name ``hook_kind`` would see for this weight."""
    return param_canonical_key(key)


def would_register_hook(key: str, *, include_experts: bool = True) -> bool:
    """True only if ActCapture would register this as a *module* hook.

    Packed ``experts.down_proj`` is an ``nn.Parameter``, not an ``nn.Linear``.
    ``hook_kind`` would still return ``ffn`` because the leaf is ``down_proj``;
    compare_act walks ``named_modules``, so that path never appears.
    """
    if classify_weight_key(key).startswith("packed_expert"):
        return False
    return hook_kind(module_path_for_param(key), include_experts=include_experts) is not None


def merge_axis_note(key: str, shape: list[int]) -> dict[str, Any]:
    axis = channel_axis(key, len(shape))
    n_axis = int(shape[axis]) if shape else 0
    packed = classify_weight_key(key).startswith("packed_expert")
    return {
        "axis": axis,
        "n_along_axis": n_axis,
        "shape": shape,
        "packed_3d": packed,
        "axis_meaning": (
            "expert index (NOT an FFN output channel)"
            if packed and len(shape) == 3 and axis == 0
            else "Linear out_features / RMSNorm dim"
        ),
    }


def load_text_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if not isinstance(tc, dict):
        return {}
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "intermediate_size",
        "layer_types",
    )
    out = {k: tc.get(k) for k in keys if k in tc}
    out["architectures"] = cfg.get("architectures")
    return out


def iter_tensor_meta(model_dir: Path) -> list[dict[str, Any]]:
    from safetensors import safe_open

    mapping = load_weight_map(model_dir)
    by_file: dict[str, list[str]] = {}
    for key, fname in mapping.items():
        by_file.setdefault(fname, []).append(key)
    rows: list[dict[str, Any]] = []
    for fname, keys in sorted(by_file.items()):
        path = model_dir / fname
        if not path.is_file():
            raise SystemExit(f"missing shard {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in keys:
                sl = handle.get_slice(key)
                shape = [int(x) for x in sl.get_shape()]
                bucket = classify_weight_key(key)
                module = module_path_for_param(key)
                kind = hook_kind(module)
                if bucket.startswith("packed_expert"):
                    kind = None
                rows.append(
                    {
                        "key": key,
                        "module": module,
                        "bucket": bucket,
                        "hook_kind": kind,
                        "would_hook": would_register_hook(key),
                        "shape": shape,
                        "ndim": len(shape),
                        "merge": merge_axis_note(key, shape),
                        "shard": fname,
                    }
                )
    return rows


def summarize_tensors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = Counter(r["bucket"] for r in rows)
    hooked = Counter(r["hook_kind"] for r in rows if r["would_hook"])
    packed = [r for r in rows if str(r["bucket"]).startswith("packed_expert")]
    shared = [r for r in rows if r["bucket"] == "shared_expert_ffn"]
    examples = {}
    for bucket in (
        "packed_expert_gate_up",
        "packed_expert_down",
        "shared_expert_ffn",
        "shared_expert_gate",
        "router",
        "hooked_attn",
        "hooked_ln",
        "hooked_embed",
        "hooked_ffn",
        "other",
    ):
        hit = next((r for r in rows if r["bucket"] == bucket), None)
        if hit is not None:
            examples[bucket] = {
                "key": hit["key"],
                "shape": hit["shape"],
                "hook_kind": hit["hook_kind"],
                "merge": hit["merge"],
            }
    return {
        "n_tensors": len(rows),
        "buckets": dict(buckets),
        "hooked_kinds_from_weights": dict(hooked),
        "n_packed_expert_tensors": len(packed),
        "n_shared_expert_ffn_tensors": len(shared),
        "packed_shapes": [list(s) for s in sorted({tuple(r["shape"]) for r in packed})],
        "shared_shapes": [list(s) for s in sorted({tuple(r["shape"]) for r in shared})],
        "examples": examples,
    }


def walk_named_modules(model_dir: Path) -> dict[str, Any]:
    """Meta init: what ActCapture would register. No GPU, no weights."""
    from contextlib import nullcontext

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    model = None
    err = None
    try:
        from accelerate import init_empty_weights
    except ImportError:
        init_empty_weights = None  # type: ignore[assignment]

    ctx = init_empty_weights() if init_empty_weights is not None else nullcontext()
    try:
        with ctx:
            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            except Exception as e1:
                err = e1
                from transformers import AutoModelForImageTextToText

                model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
    except Exception as e2:
        return {"ok": False, "error": f"{type(err or e2).__name__}: {err or e2}"}

    registered: list[dict[str, Any]] = []
    skipped_moe: list[str] = []
    seen: set[str] = set()
    skipped_seen: set[str] = set()
    for name, _mod in model.named_modules():
        kind = hook_kind(name, include_experts=True)
        canon = canonical_module_key(name)
        moeish = any(
            s in name
            for s in (
                ".mlp.experts",
                ".shared_expert",
                ".mlp.gate",
                "shared_expert_gate",
            )
        )
        if kind is None:
            if moeish and canon not in skipped_seen:
                skipped_seen.add(canon)
                skipped_moe.append(canon)
            continue
        if canon in seen:
            continue
        seen.add(canon)
        registered.append({"name": canon, "kind": kind, "raw": name})

    kinds = Counter(r["kind"] for r in registered)
    moe_reg = [r for r in registered if any(s in r["name"] for s in ("mlp.", "shared_expert", "experts"))]
    return {
        "ok": True,
        "n_registered": len(registered),
        "kinds": dict(kinds),
        "n_moe_registered": len(moe_reg),
        "moe_registered_examples": moe_reg[:24],
        "n_moe_skipped": len(skipped_moe),
        "moe_skipped_examples": skipped_moe[:40],
        "class_name": type(model).__name__,
    }


def as_btc_verdict() -> dict[str, Any]:
    """Document the 2-D drop without a 35B forward.

    Qwen3_5MoeSparseMoeBlock does ``hidden.view(-1, hidden_dim)`` then
    ``shared_expert`` / ``experts``. Linear then emits ``[S, C]``. compare_act
    ``_as_btc`` keeps only ndim==3, so those hooks fire and discard.
    """
    try:
        import torch
        from compare_act import _as_btc
    except Exception as e:
        return {"ok": False, "error": str(e)}

    kept_3d = _as_btc(torch.zeros(1, 4, 8))
    dropped_2d = _as_btc(torch.zeros(4, 8))
    return {
        "ok": True,
        "keeps_btc_3d": kept_3d is not None,
        "keeps_sc_2d": dropped_2d is not None,
        "note": (
            "MoE block reshapes to [S, H] before shared_expert / experts. "
            "A Linear hook then sees [S, C]; _as_btc returns None."
        ),
    }


def verdict_text(world_sum: dict[str, Any], modules: dict[str, Any] | None) -> list[str]:
    lines = []
    n_pack = int(world_sum.get("n_packed_expert_tensors") or 0)
    n_shared = int(world_sum.get("n_shared_expert_ffn_tensors") or 0)
    hooked = world_sum.get("hooked_kinds_from_weights") or {}
    lines.append(
        "结论：当前 compare_act 没有把 routed MoE 专家通道编进 ACT 掩码。"
    )
    lines.append(
        f"  权重里的 packed expert 张量 = {n_pack}（3D Parameter，不是 experts.0.down_proj）。"
    )
    lines.append(
        f"  hook_kind 能对上的 FFN 权重 = {hooked.get('ffn', 0)} "
        f"（这些是 shared_expert.gate/up/down_proj，不是 256 路 routed 专家）。"
    )
    lines.append(
        f"  shared_expert 的 Linear 权重 = {n_shared}；"
        "模块树里叶子名对得上，但 MoE 前向先 view 成 2D，_as_btc 丢掉，所以 smoke 的 kind 表里没有 ffn。"
    )
    lines.append(
        "  路由 mlp.gate 和 shared_expert_gate 的叶子名不在 FFN_PROJ_LEAVES 里，也不会 hook。"
    )
    if modules and modules.get("ok"):
        lines.append(
            f"  meta named_modules 将注册 {modules['n_registered']} 个模块，"
            f"kinds={modules.get('kinds')}；其中 MoE 相关 {modules.get('n_moe_registered')}。"
        )
        if modules.get("n_registered") == SMOKE_CAPTURED_MODULES:
            lines.append(
                f"  这和 smoke 的 captured {SMOKE_CAPTURED_MODULES} 一致："
                "就是注意力 + LN + embed，没有专家。"
            )
    lines.append(
        f"  smoke checksum：captured={SMOKE_CAPTURED_MODULES} kinds={list(SMOKE_KINDS)} ffn=0。"
    )
    lines.append(
        "  merge/act.py 即便拿到专家激活也接不上：packed 张量没有 .weight 后缀，"
        "channel_axis 会把 3D 的第 0 轴当成专家编号，不是输出通道。"
    )
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--skip-modules",
        action="store_true",
        help="only read safetensors headers (no transformers meta init)",
    )
    p.add_argument(
        "--world-only",
        action="store_true",
        help="skip Instruct (same language_model boxes; faster)",
    )
    return p.parse_args()


def require_local(spec: str, cache_dir: Path, role: str) -> Path:
    p = Path(spec).expanduser()
    if checkpoint_ready(p):
        return p.resolve()
    dest = cache_path(spec, cache_dir)
    if checkpoint_ready(dest):
        return dest.resolve()
    raise SystemExit(
        f"[{role}] 本地没有权重：{dest}\n"
        "在已经 download 过的 GPU 机上跑，不要在这台 CPU 机上下载。"
    )


def dump_one(label: str, model_dir: Path, *, skip_modules: bool) -> dict[str, Any]:
    log(f"[{label}] {model_dir}")
    cfg = load_text_config(model_dir)
    log(f"[{label}] config { {k: cfg[k] for k in cfg if k != 'layer_types'} }")
    rows = iter_tensor_meta(model_dir)
    summary = summarize_tensors(rows)
    log(f"[{label}] tensors={summary['n_tensors']} buckets={summary['buckets']}")
    log(
        f"[{label}] packed_experts={summary['n_packed_expert_tensors']} "
        f"shapes={summary['packed_shapes']}"
    )
    log(
        f"[{label}] shared_expert_ffn={summary['n_shared_expert_ffn_tensors']} "
        f"shapes={summary['shared_shapes']}"
    )
    log(f"[{label}] hook_kind from weights: {summary['hooked_kinds_from_weights']}")
    for bucket, ex in summary["examples"].items():
        log(
            f"[{label}]   e.g. {bucket}: {ex['key']} {ex['shape']} "
            f"hook={ex['hook_kind']} merge={ex['merge']['axis_meaning']}"
        )
    modules = None
    if not skip_modules:
        log(f"[{label}] walking named_modules on meta (no GPU)…")
        modules = walk_named_modules(model_dir)
        if modules.get("ok"):
            log(
                f"[{label}] registered={modules['n_registered']} "
                f"kinds={modules['kinds']} moe_hooked={modules['n_moe_registered']} "
                f"moe_skipped={modules['n_moe_skipped']}"
            )
            for ex in modules.get("moe_skipped_examples") or []:
                if "experts" in ex or "shared_expert" in ex or ex.endswith(".gate"):
                    log(f"[{label}]   skip {ex}")
        else:
            log(f"[{label}] named_modules failed: {modules.get('error')}")
    moe_rows = [
        r
        for r in rows
        if str(r["bucket"]).startswith(("packed_expert", "shared_expert", "router"))
    ]
    return {
        "dir": str(model_dir),
        "config": cfg,
        "summary": summary,
        "modules": modules,
        "moe_tensors": moe_rows[:80],
    }


def main() -> None:
    args = parse_args()
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (Path.cwd() / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_dir = require_local(args.world, cache_dir, "world")
    world = dump_one("world", world_dir, skip_modules=args.skip_modules)
    agent = None
    if not args.world_only:
        agent_dir = require_local(args.agent, cache_dir, "instruct")
        agent = dump_one("instruct", agent_dir, skip_modules=args.skip_modules)

    btc = as_btc_verdict()
    log(f"[_as_btc] {btc}")

    lines = verdict_text(world["summary"], world.get("modules"))
    for line in lines:
        log(line)

    report = {
        "method": (
            "CPU header walk of AgentWorld/Instruct safetensors + optional meta "
            "named_modules. No 35B forward. Checks whether compare_act hook_kind "
            "and merge channel_axis can see Qwen3.5 packed MoE experts."
        ),
        "smoke_checksum": {
            "captured_modules": SMOKE_CAPTURED_MODULES,
            "kinds": list(SMOKE_KINDS),
            "ffn": 0,
        },
        "as_btc": btc,
        "verdict": lines,
        "world": world,
        "instruct": agent,
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
