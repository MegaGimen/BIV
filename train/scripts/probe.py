#!/usr/bin/env python3
"""Probe Qwen3.5-35B AgentWorld/Instruct for Transformers-based fish-cut.

Training library (this branch): PyTorch + HuggingFace Transformers + PEFT
+ Accelerate FSDP2. The loop is custom (JEPA / draft / W). Not Unsloth,
not Axolotl, not ms-swift, not TRL SFTTrainer (Muse uses that last one).

Reads weights from merge/output/cache (same as merge/download.py). Writes
under train/outputs/probe/.

GPU host::

    python train/scripts/probe.py
    python train/scripts/probe.py --skip-weights   # skip 3-way cosine; still forward
    python train/scripts/probe.py --skip-forward   # disk stats only
    python train/scripts/probe.py --cut 36         # copy Instruct L36..L39 + lm_head onto AgentWorld

Writes train/outputs/probe/report.json (includes recommended_cut ℓ) and summary.txt.

Then cut the Stage 1 checkpoint and train JEPA::

    python train/scripts/cut_stage1.py             # CPU; reads recommended_cut
    python train/scripts/train_jepa.py --config train/configs/jepa/stage1.yaml

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from biv_wm.cut import pick_cut  # noqa: E402
from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_BASE,
    DEFAULT_CACHE,
    DEFAULT_WORLD,
    resolve_model,
)
from merge import TensorStore, is_visual_key, load_weight_map  # noqa: E402

DEFAULT_OUT = ROOT / "train" / "outputs" / "probe"
PROBE_PROMPT = "cd /tmp && pwd\n"

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Expected from HF config / modeling_qwen3_5_moe.py — probe verifies on disk.
FULL_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
LORA_CANDIDATES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "output_gate_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "shared_expert_gate",
    "in_proj",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_cfg(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_cfg(v, key))
        return out
    if isinstance(obj, list):
        if prefix.endswith("layer_types") or (
            len(obj) <= 8 and all(not isinstance(x, (dict, list)) for x in obj)
        ):
            out[prefix] = obj
        else:
            out[prefix] = f"<list n={len(obj)}>"
        return out
    out[prefix] = obj
    return out


def cfg_diff(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[dict[str, Any]]:
    fa = flatten_cfg(a or {})
    fb = flatten_cfg(b or {})
    keys = sorted(set(fa) | set(fb))
    rows = []
    for k in keys:
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if va != vb:
            rows.append({"key": k, "left": va, "right": vb})
    return rows


def bucket_key(name: str) -> str:
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    parts = name.split(".")
    if is_visual_key(name) or "visual" in parts:
        return "visual"
    if name == "lm_head.weight" or name.endswith(".lm_head.weight") or name == "lm_head.bias":
        return "lm_head"
    if "embed_tokens" in name:
        return "embed"
    m = LAYER_RE.search(name)
    if m:
        return f"layer_{int(m.group(1))}"
    if name.endswith("norm.weight") or name.endswith(".norm.weight"):
        return "final_norm"
    return "other"


def layer_index(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def list_sidecar_files(model_dir: Path) -> dict[str, Any]:
    names = []
    for p in sorted(model_dir.iterdir()):
        if p.is_file():
            names.append(
                {
                    "name": p.name,
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p) if p.suffix in {".json", ".py", ".jinja"} else None,
                }
            )
    return {
        "dir": str(model_dir),
        "n_files": len(names),
        "py": [x["name"] for x in names if x["name"].endswith(".py")],
        "files": names,
    }


def tokenizer_snapshot(model_dir: Path) -> dict[str, Any]:
    cfg = load_json(model_dir / "tokenizer_config.json") or {}
    special = load_json(model_dir / "special_tokens_map.json")
    added = load_json(model_dir / "added_tokens.json")
    tmpl = None
    for cand in ("chat_template.jinja", "chat_template.json"):
        p = model_dir / cand
        if p.is_file():
            tmpl = {"file": cand, "sha256": sha256_file(p), "bytes": p.stat().st_size}
            break
    if cfg.get("chat_template") and tmpl is None:
        raw = cfg["chat_template"]
        if isinstance(raw, str):
            tmpl = {"file": "tokenizer_config.json:chat_template", "sha256": hashlib.sha256(raw.encode()).hexdigest(), "bytes": len(raw)}
    return {
        "model_max_length": cfg.get("model_max_length"),
        "tokenizer_class": cfg.get("tokenizer_class"),
        "bos_token": cfg.get("bos_token"),
        "eos_token": cfg.get("eos_token"),
        "pad_token": cfg.get("pad_token"),
        "added_tokens_decoder_n": len(cfg.get("added_tokens_decoder") or {}),
        "special_tokens_map": special,
        "added_tokens": added,
        "chat_template": tmpl,
        "tokenizer_json_sha256": sha256_file(model_dir / "tokenizer.json"),
    }


def text_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not cfg:
        return {}
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def key_inventory(weight_map: dict[str, str]) -> dict[str, Any]:
    buckets: dict[str, int] = defaultdict(int)
    leaves: dict[str, int] = defaultdict(int)
    sample_by_bucket: dict[str, list[str]] = defaultdict(list)
    for k in weight_map:
        b = bucket_key(k)
        buckets[b] += 1
        leaves[leaf_name(k)] += 1
        if len(sample_by_bucket[b]) < 8:
            sample_by_bucket[b].append(k)
    return {
        "n_keys": len(weight_map),
        "buckets": dict(sorted(buckets.items(), key=lambda kv: kv[0])),
        "leaf_counts": dict(sorted(leaves.items(), key=lambda kv: (-kv[1], kv[0]))),
        "samples": {b: sample_by_bucket[b] for b in sorted(sample_by_bucket)},
        "n_visual": sum(1 for k in weight_map if bucket_key(k) == "visual" or is_visual_key(k)),
        "n_mtp": sum(1 for k in weight_map if bucket_key(k) == "mtp"),
    }


def _stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "dot_aw_ag": 0.0,
        "sq_aw": 0.0,
        "sq_ag": 0.0,
        "sq_ba": 0.0,
        "dot_daw_dag": 0.0,
        "sq_daw": 0.0,
        "sq_dag": 0.0,
        "abs_aw_ag": 0.0,
        "abs_aw_ba": 0.0,
        "abs_ag_ba": 0.0,
        "max_abs_aw_ag": 0.0,
    }


def _acc(st: dict[str, float], aw, ag, ba) -> None:
    import torch

    x = aw.reshape(-1).to(torch.float32)
    y = ag.reshape(-1).to(torch.float32)
    z = ba.reshape(-1).to(torch.float32) if ba is not None else None
    n = int(x.numel())
    st["n"] += n
    st["dot_aw_ag"] += float(torch.dot(x, y).item())
    st["sq_aw"] += float(torch.dot(x, x).item())
    st["sq_ag"] += float(torch.dot(y, y).item())
    diff = (x - y).abs()
    st["abs_aw_ag"] += float(diff.sum().item())
    st["max_abs_aw_ag"] = max(st["max_abs_aw_ag"], float(diff.max().item()))
    if z is not None:
        st["sq_ba"] += float(torch.dot(z, z).item())
        daw = x - z
        dag = y - z
        st["dot_daw_dag"] += float(torch.dot(daw, dag).item())
        st["sq_daw"] += float(torch.dot(daw, daw).item())
        st["sq_dag"] += float(torch.dot(dag, dag).item())
        st["abs_aw_ba"] += float((x - z).abs().sum().item())
        st["abs_ag_ba"] += float((y - z).abs().sum().item())
        del daw, dag, z
    del x, y, diff


def _finalize(st: dict[str, float]) -> dict[str, Any]:
    n = st["n"]
    if n <= 0:
        return {"n": 0}
    def safe_cos(dot: float, sa: float, sb: float) -> float | None:
        den = (sa * sb) ** 0.5
        if den <= 0:
            return None
        return float(dot / den)

    rms = lambda s: float((s / n) ** 0.5)
    out = {
        "n": int(n),
        "cos_aw_instruct": safe_cos(st["dot_aw_ag"], st["sq_aw"], st["sq_ag"]),
        "mean_abs_aw_instruct": st["abs_aw_ag"] / n,
        "max_abs_aw_instruct": st["max_abs_aw_ag"],
        "rms_aw": rms(st["sq_aw"]),
        "rms_instruct": rms(st["sq_ag"]),
    }
    if st["sq_ba"] > 0:
        out["cos_delta"] = safe_cos(st["dot_daw_dag"], st["sq_daw"], st["sq_dag"])
        out["rms_delta_aw"] = rms(st["sq_daw"])
        out["rms_delta_instruct"] = rms(st["sq_dag"])
        out["mean_abs_delta_aw"] = st["abs_aw_ba"] / n
        out["mean_abs_delta_instruct"] = st["abs_ag_ba"] / n
        if out["rms_delta_aw"] and out["rms_delta_aw"] > 0:
            out["delta_ratio_instruct_over_aw"] = out["rms_delta_instruct"] / out["rms_delta_aw"]
    return out


def compare_weights(
    *,
    world_dir: Path,
    agent_dir: Path,
    base_dir: Path | None,
    world_map: dict[str, str],
    agent_map: dict[str, str],
    base_map: dict[str, str] | None,
    layer_types: list[str] | None = None,
) -> dict[str, Any]:
    import torch

    world_keys = set(world_map)
    agent_keys = set(agent_map)
    base_keys = set(base_map or {})

    only_world = sorted(world_keys - agent_keys)
    only_agent = sorted(agent_keys - world_keys)
    both = sorted(world_keys & agent_keys)

    shape_mismatch: list[dict[str, Any]] = []
    dtype_mismatch: list[dict[str, Any]] = []
    missing_base: list[str] = []
    per_bucket: dict[str, dict[str, float]] = defaultdict(_stats)
    per_layer: dict[int, dict[str, float]] = defaultdict(_stats)
    per_leaf: dict[str, dict[str, float]] = defaultdict(_stats)
    special_shapes: dict[str, dict[str, Any]] = {}
    moe_3d: list[dict[str, Any]] = []

    world_store = TensorStore(world_dir, world_map)
    agent_store = TensorStore(agent_dir, agent_map)
    base_store = TensorStore(base_dir, base_map) if base_dir is not None and base_map else None

    n_ok = 0
    try:
        n_both = len(both)
        for i, key in enumerate(both, start=1):
            if i == 1 or i % 80 == 0 or i == n_both:
                log(f"  tensors {i}/{n_both}: {key}")
            wt = world_store.get(key)
            at = agent_store.get(key)
            if wt is None or at is None:
                continue
            if tuple(wt.shape) != tuple(at.shape):
                shape_mismatch.append(
                    {"key": key, "world": list(wt.shape), "instruct": list(at.shape)}
                )
                continue
            if wt.dtype != at.dtype:
                dtype_mismatch.append(
                    {"key": key, "world": str(wt.dtype), "instruct": str(at.dtype)}
                )
            bt = None
            if base_store is not None:
                bt = base_store.get(key)
                if bt is None:
                    missing_base.append(key)
                elif tuple(bt.shape) != tuple(wt.shape):
                    missing_base.append(key)
                    bt = None

            bkt = bucket_key(key)
            if key in {
                "lm_head.weight",
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            } or key.endswith("embed_tokens.weight"):
                special_shapes[key] = {
                    "shape": list(wt.shape),
                    "dtype": str(wt.dtype).replace("torch.", ""),
                    "nbytes": int(wt.nbytes),
                }
            if wt.ndim >= 3:
                moe_3d.append(
                    {
                        "key": key,
                        "shape": list(wt.shape),
                        "dtype": str(wt.dtype).replace("torch.", ""),
                    }
                )

            _acc(per_bucket[bkt], wt, at, bt)
            li = layer_index(key)
            if li is not None:
                _acc(per_layer[li], wt, at, bt)
            _acc(per_leaf[leaf_name(key)], wt, at, bt)
            n_ok += 1
            del wt, at, bt
            if i % 40 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        world_store.close()
        agent_store.close()
        if base_store is not None:
            base_store.close()

    layers_out = []
    types = layer_types or []
    for i in sorted(per_layer):
        row = _finalize(per_layer[i])
        row["layer"] = i
        if i < len(types):
            row["layer_type"] = types[i]
        else:
            row["layer_type"] = "full_attention" if i in FULL_ATTN_LAYERS else "linear_attention"
        layers_out.append(row)

    return {
        "n_compared": n_ok,
        "only_world_n": len(only_world),
        "only_instruct_n": len(only_agent),
        "only_world_head": only_world[:80],
        "only_instruct_head": only_agent[:80],
        "only_world_buckets": _bucket_list(only_world),
        "only_instruct_buckets": _bucket_list(only_agent),
        "keys_in_world_not_base_n": len(world_keys - base_keys) if base_keys else None,
        "keys_in_instruct_not_base_n": len(agent_keys - base_keys) if base_keys else None,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch_n": len(dtype_mismatch),
        "dtype_mismatch_head": dtype_mismatch[:40],
        "missing_or_shape_mismatch_base_n": len(missing_base),
        "missing_base_head": missing_base[:40],
        "special_shapes": special_shapes,
        "moe_3d_n": len(moe_3d),
        "moe_3d_head": moe_3d[:20],
        "per_bucket": {k: _finalize(v) for k, v in sorted(per_bucket.items())},
        "per_layer": layers_out,
        "per_leaf": {
            k: fin
            for k, v in sorted(per_leaf.items(), key=lambda kv: -kv[1]["n"])
            for fin in [_finalize(v)]
            if fin.get("n", 0) > 0
        },
        "swap_ok_layers": _swap_ok_from_maps(world_map, agent_map, shape_mismatch),
    }


def _bucket_list(keys: list[str]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for k in keys:
        c[bucket_key(k)] += 1
    return dict(sorted(c.items()))


def _swap_ok_from_maps(
    world_map: dict[str, str],
    agent_map: dict[str, str],
    shape_mismatch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Whether Instruct layer i can be copied onto AgentWorld (names + shapes)."""
    bad = {row["key"] for row in shape_mismatch}
    rows = []
    for i in range(40):
        token = f"layers.{i}."
        w = {k for k in world_map if token in k}
        a = {k for k in agent_map if token in k}
        missing_in_world = sorted(a - w)
        missing_in_instruct = sorted(w - a)
        mismatch = sorted(k for k in (w & a) if k in bad)
        rows.append(
            {
                "layer": i,
                "ok": not missing_in_world and not mismatch,
                "n_world": len(w),
                "n_instruct": len(a),
                "missing_in_world": missing_in_world[:20],
                "missing_in_instruct": missing_in_instruct[:20],
                "shape_mismatch_keys": mismatch[:10],
            }
        )
    return rows


def try_meta_tree(model_dir: Path) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    except Exception as e:
        return {"ok": False, "error": f"import: {e!r}"}

    try:
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        return {"ok": False, "error": f"AutoConfig: {e!r}"}

    model = None
    err = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            with torch.device("meta"):
                model = loader.from_config(config, trust_remote_code=True)
            break
        except Exception as e:
            err = e
            model = None
    if model is None:
        return {"ok": False, "error": f"from_config: {err!r}", "config_class": type(config).__name__}

    linears: dict[str, int] = defaultdict(int)
    classes: dict[str, int] = defaultdict(int)
    sample_linears: list[dict[str, Any]] = []
    decoder_cls = None
    path_lm_head = None
    path_embed = None
    path_norm = None
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        classes[cls] += 1
        if cls == "Qwen3_5MoeDecoderLayer" and decoder_cls is None:
            decoder_cls = name
        if cls in {"Linear", "Conv1d"} or cls.endswith("Linear"):
            w = getattr(mod, "weight", None)
            shape = list(w.shape) if w is not None else None
            leaf = name.rsplit(".", 1)[-1] if name else cls
            linears[leaf] += 1
            if len(sample_linears) < 60:
                sample_linears.append({"name": name, "cls": cls, "weight": shape})
        if name.endswith("lm_head") or name == "lm_head":
            path_lm_head = name
        if name.endswith("embed_tokens"):
            path_embed = name
        if name in {"model.language_model.norm", "model.norm", "language_model.norm"}:
            path_norm = name

    lora_hits = {n: linears.get(n, 0) for n in LORA_CANDIDATES}

    del model
    return {
        "ok": True,
        "config_class": type(config).__name__,
        "model_class_tried": True,
        "module_class_counts": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
        "linear_leaf_counts": dict(sorted(linears.items(), key=lambda kv: (-kv[1], kv[0]))),
        "lora_candidate_hits": lora_hits,
        "decoder_layer_example": decoder_cls,
        "lm_head_module": path_lm_head,
        "embed_module": path_embed,
        "final_norm_module": path_norm,
        "sample_linears": sample_linears,
    }


def cut_hints(per_layer: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-layer ranking for inspection. Operational ℓ is recommended_cut / pick_cut."""
    if not per_layer:
        return {}
    ranked = []
    for row in per_layer:
        ratio = row.get("delta_ratio_instruct_over_aw")
        ranked.append(
            {
                "layer": row["layer"],
                "layer_type": row.get("layer_type"),
                "cos_aw_instruct": row.get("cos_aw_instruct"),
                "cos_delta": row.get("cos_delta"),
                "rms_delta_aw": row.get("rms_delta_aw"),
                "rms_delta_instruct": row.get("rms_delta_instruct"),
                "delta_ratio_instruct_over_aw": ratio,
            }
        )
    by_ratio = [r for r in ranked if r.get("delta_ratio_instruct_over_aw") is not None]
    by_ratio.sort(key=lambda r: r["delta_ratio_instruct_over_aw"], reverse=True)
    by_low_cos = sorted(
        [r for r in ranked if r.get("cos_aw_instruct") is not None],
        key=lambda r: r["cos_aw_instruct"],
    )
    return {
        "note": (
            "Operational ℓ is report['recommended_cut'] (pick_cut: max tail−front "
            "mean Instruct/AW δ-ratio on 4-layer GDN+attn boundaries). Rankings below are inspection only."
        ),
        "highest_instruct_delta_ratio": by_ratio[:8],
        "lowest_weight_cosine": by_low_cos[:8],
        "full_attention_layers": sorted(FULL_ATTN_LAYERS),
    }


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    a = lambda s: lines.append(s)
    a("BIV train/scripts/probe.py summary")
    a("==========================")
    paths = report.get("paths") or {}
    for role in ("world", "agent", "base"):
        a(f"{role}: {paths.get(role)}")
    a("")
    wc = report.get("configs", {}).get("diff_world_vs_instruct") or []
    a(f"config diffs world vs instruct: {len(wc)}")
    for row in wc[:30]:
        a(f"  {row['key']}: {row['left']!r} → {row['right']!r}")
    a("")
    w = report.get("weights") or {}
    a(f"compared tensors: {w.get('n_compared')}")
    a(f"only in AgentWorld: {w.get('only_world_n')} buckets={w.get('only_world_buckets')}")
    a(f"only in Instruct: {w.get('only_instruct_n')} buckets={w.get('only_instruct_buckets')}")
    a(f"shape mismatches: {len(w.get('shape_mismatch') or [])}")
    a(f"moe 3D tensors: {w.get('moe_3d_n')}")
    a(f"special shapes: {json.dumps(w.get('special_shapes'), ensure_ascii=False)}")
    a("")
    a("per-layer (layer, type, cos(AW,Inst), cos(delta), rmsΔAW, rmsΔInst, ratio Inst/AW)")
    for row in w.get("per_layer") or []:
        a(
            "{layer:2d} {layer_type:16s} cos={cos_aw_instruct} dcos={cos_delta} "
            "dAW={rms_delta_aw} dInst={rms_delta_instruct} ratio={delta_ratio_instruct_over_aw}".format(
                layer=row.get("layer"),
                layer_type=row.get("layer_type"),
                cos_aw_instruct=_fmt(row.get("cos_aw_instruct")),
                cos_delta=_fmt(row.get("cos_delta")),
                rms_delta_aw=_fmt(row.get("rms_delta_aw")),
                rms_delta_instruct=_fmt(row.get("rms_delta_instruct")),
                delta_ratio_instruct_over_aw=_fmt(row.get("delta_ratio_instruct_over_aw")),
            )
        )
    swap = w.get("swap_ok_layers") or []
    bad = [r for r in swap if not r.get("ok")]
    a("")
    a(f"layers copyable Instruct→AW: {sum(1 for r in swap if r.get('ok'))}/40  failing={len(bad)}")
    for r in bad[:10]:
        a(f"  fail L{r.get('layer')}: {r}")
    rec = report.get("recommended_cut") or {}
    a("")
    if rec.get("ell") is not None:
        a(
            f"recommended_cut ℓ={rec.get('ell')}  gap={_fmt(rec.get('gap'))}  "
            f"mean_front={_fmt(rec.get('mean_front'))}  mean_back={_fmt(rec.get('mean_back'))}"
        )
        a("cut rule: " + str(rec.get("rule") or ""))
        for row in rec.get("candidates") or []:
            mark = " <==" if int(row.get("ell", -1)) == int(rec.get("ell")) else ""
            a(
                f"  candidate ℓ={row.get('ell'):2d}  gap={_fmt(row.get('gap'))}  "
                f"front={_fmt(row.get('mean_front'))}  back={_fmt(row.get('mean_back'))}{mark}"
            )
    elif rec.get("error"):
        a(f"recommended_cut error: {rec.get('error')}")
    else:
        a("recommended_cut: (run probe without --skip-weights)")
    hints = report.get("cut_hints") or {}
    a("cut hints: " + (hints.get("note") or ""))
    a("highest Instruct/AW delta ratio: " + json.dumps(hints.get("highest_instruct_delta_ratio"), ensure_ascii=False))
    meta = report.get("meta_tree") or {}
    a("")
    a(f"meta tree: ok={meta.get('ok')} error={meta.get('error')}")
    if meta.get("ok"):
        a(f"  lm_head={meta.get('lm_head_module')} embed={meta.get('embed_module')} norm={meta.get('final_norm_module')}")
        a(f"  lora hits={json.dumps(meta.get('lora_candidate_hits'))}")
        a(f"  decoder example={meta.get('decoder_layer_example')}")
    fw = report.get("frameworks") or {}
    a("")
    a("libraries: " + json.dumps(fw.get("versions"), ensure_ascii=False))
    a("cuda: " + json.dumps(fw.get("cuda"), ensure_ascii=False))
    a("train_library: " + json.dumps((report.get("train_stack") or {}).get("library"), ensure_ascii=False))
    fwd = report.get("forward") or {}
    a("")
    a(f"forward ok={fwd.get('ok')} error={fwd.get('error')}")
    if fwd.get("ok"):
        a(f"  class={fwd.get('model_class')} device_map={fwd.get('device_map')}")
        a(f"  hidden={fwd.get('last_hidden_shape')} logits={fwd.get('logits_shape')}")
        a(f"  n_hidden_states={fwd.get('n_hidden_states')} lm_head={fwd.get('lm_head_name')}")
        a(f"  layers_path={fwd.get('layers_path')} n_layers={fwd.get('n_layers')}")
    spl = report.get("splice_forward") or {}
    a(f"splice_forward ok={spl.get('ok')} error={spl.get('error')} copied={spl.get('n_copied')} missing={spl.get('n_missing')}")
    if spl.get("ok"):
        a(f"  hidden={spl.get('last_hidden_shape')} logits={spl.get('logits_shape')}")
        a(f"  dummy_W_logits={spl.get('dummy_w_logits_shape')} dummy_W_ok={spl.get('dummy_w_ok')}")
    sa = report.get("speed_advice") or {}
    a("")
    if sa.get("ok"):
        mem = sa.get("selective_checkpointing_memory_math") or {}
        thr = sa.get("throughput_estimates") or {}
        base = sa.get("throughput_baseline") or {}
        a("speed_advice ok=True")
        a(f"  current: {base.get('current_h_per_epoch')}h/epoch  original-est: {base.get('original_h_per_epoch_estimate')}h/epoch")
        a(f"  linear layers to uncheckpoint: {mem.get('linear_layers_to_uncheckpoint')}  full-attn remain: {mem.get('full_attention_layers_remain_checkpointed')}")
        a(f"  activation mem at max_len: {mem.get('estimated_activation_gib_at_max_len_65536')} GiB  headroom: {mem.get('observed_headroom_gib_per_gpu')} GiB  fits={mem.get('fits_at_max_len')}")
        a(f"  estimate after selective-ckpt: {thr.get('after_selective_checkpointing_h_per_epoch')}h/epoch")
        a(f"  estimate after both opts: {thr.get('after_both_optimizations_h_per_epoch')}h/epoch")
        audit = sa.get("gc_mechanism_audit") or {}
        sim = audit.get("sim_apply_selective_checkpointing") or {}
        a(f"  gc_mechanism_audit: gcl_inheritors={audit.get('gcl_inheritor_count')} "
          f"layer_idx_gc_found={audit.get('layer_idx_gc_scan_total')} "
          f"path={audit.get('path_discovery_result')} "
          f"sim_uncheckpoint={sim.get('would_uncheckpoint')} sim_ok={sim.get('ok')}")
        a(f"    gc_in_forward={audit.get('gc_in_forward')}")
        a(f"    decoder_classname_matches={audit.get('decoder_classname_matches','[]')[:3]}")
        z1t = sa.get("z1_batch_merge_test") or {}
        a(f"  z1 batch=2 test: tested={z1t.get('tested')} ok={z1t.get('ok')} error={z1t.get('error')}")
        if z1t.get("ok"):
            a(f"    single shape={z1t.get('single_hidden_shape')} double shape={z1t.get('double_hidden_shape')}")
            a(f"    mem batch1={z1t.get('batch1_mem_delta_mb')}MB  batch2={z1t.get('batch2_mem_delta_mb')}MB")
    else:
        a(f"speed_advice: {sa.get('error', 'not run')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def probe_frameworks() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "trl",
        "safetensors",
        "unsloth",
        "axolotl",
        "ms_swift",
        "liger_kernel",
        "modelscope",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "imported")
        except ImportError:
            versions[name] = None
        except Exception as e:
            versions[name] = f"error:{type(e).__name__}"
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "arch": torch.version.cuda,
        }
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cuda["total_gib"] = round(p.total_memory / 1024**3, 2)
    except Exception as e:
        cuda["error"] = repr(e)
    return {
        "versions": versions,
        "cuda": cuda,
        "library": {
            "train": ["torch", "transformers", "peft", "accelerate"],
            "loop": "custom (not TRL SFTTrainer)",
            "unused_on_this_branch": ["unsloth", "axolotl", "ms-swift", "trl.SFTTrainer"],
        },
    }


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _resolve_module(root: Any, dotted: str) -> Any | None:
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model) -> tuple[str | None, Any | None]:
    for path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.model.language_model.layers",
        "model.layers",
    ):
        got = _resolve_module(model, path)
        if got is not None:
            return path, got
    return None, None


def find_lm_head(model) -> tuple[str | None, Any | None]:
    for path in ("lm_head", "model.lm_head"):
        got = _resolve_module(model, path)
        if got is not None:
            return path, got
    return None, None


def _load_qwen_moe(model_dir: Path, *, dtype, device_map: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }
    last = None
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            return loader.from_pretrained(str(model_dir), **kwargs), loader.__name__, kwargs
        except Exception as e:
            last = e
    raise RuntimeError(f"from_pretrained failed: {_exc(last)}")


def _forward_once(model, tokenizer, prompt: str, device) -> dict[str, Any]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = getattr(out, "hidden_states", None)
    last = getattr(out, "last_hidden_state", None)
    if last is None and hidden:
        last = hidden[-1]
    logits = getattr(out, "logits", None)
    tok = None
    if logits is not None:
        tok = int(logits[0, -1].argmax().item())
    return {
        "input_ids_shape": list(inputs["input_ids"].shape),
        "n_hidden_states": len(hidden) if hidden is not None else 0,
        "last_hidden_shape": list(last.shape) if last is not None else None,
        "last_hidden_dtype": str(last.dtype) if last is not None else None,
        "logits_shape": list(logits.shape) if logits is not None else None,
        "argmax_last_token_id": tok,
        "argmax_last_token": tokenizer.decode([tok]) if tok is not None else None,
    }


def _param_lookup(model) -> dict[str, Any]:
    out = {}
    for name, p in model.named_parameters():
        out[name] = p
    for name, b in model.named_buffers():
        out.setdefault(name, b)
    return out


def copy_keys_into_model(model, store: TensorStore, keys: list[str]) -> dict[str, Any]:
    import torch

    lookup = _param_lookup(model)
    copied = 0
    missing: list[str] = []
    mismatch: list[dict[str, Any]] = []
    aliases_used: list[str] = []
    for key in keys:
        candidates = [key]
        if key.startswith("model."):
            candidates.append(key[len("model.") :])
        else:
            candidates.append("model." + key)
        target = None
        hit = None
        for c in candidates:
            if c in lookup:
                target = lookup[c]
                hit = c
                break
        if target is None:
            missing.append(key)
            continue
        src = store.get(key)
        if src is None:
            missing.append(key)
            continue
        if tuple(src.shape) != tuple(target.shape):
            mismatch.append({"key": key, "src": list(src.shape), "dst": list(target.shape)})
            del src
            continue
        with torch.no_grad():
            target.data.copy_(src.to(device=target.device, dtype=target.dtype))
        copied += 1
        if hit != key:
            aliases_used.append(f"{key} -> {hit}")
        del src
    return {
        "n_copied": copied,
        "n_missing": len(missing),
        "n_mismatch": len(mismatch),
        "missing_head": missing[:30],
        "mismatch_head": mismatch[:20],
        "alias_head": aliases_used[:20],
    }


def try_forward_and_surgery(
    *,
    world_dir: Path,
    agent_dir: Path,
    agent_map: dict[str, str],
    cut: int,
    device_map: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_for_inputs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fwd: dict[str, Any] = {"ok": False}
    splice: dict[str, Any] = {"ok": False, "cut": cut}

    log(f"loading AgentWorld via transformers device_map={device_map} dtype={dtype}")
    try:
        model, loader_name, load_kwargs = _load_qwen_moe(
            world_dir, dtype=dtype, device_map=device_map
        )
    except Exception as e:
        fwd["error"] = _exc(e)
        return fwd, splice

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(agent_dir), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        layers_path, layers = find_decoder_layers(model)
        lm_name, lm_head = find_lm_head(model)
        # inputs on the embed device
        embed = _resolve_module(model, "model.language_model.embed_tokens") or _resolve_module(
            model, "language_model.embed_tokens"
        )
        if embed is not None and hasattr(embed, "weight"):
            device_for_inputs = embed.weight.device
        fwd_stats = _forward_once(model, tokenizer, PROBE_PROMPT, device_for_inputs)
        leaf_counts: dict[str, int] = defaultdict(int)
        for name, _p in model.named_parameters():
            leaf_counts[name.rsplit(".", 1)[-1]] += 1
        fwd = {
            "ok": True,
            "loader": loader_name,
            "model_class": type(model).__name__,
            "device_map": device_map,
            "load_kwargs": {k: str(v) for k, v in load_kwargs.items()},
            "layers_path": layers_path,
            "n_layers": len(layers) if layers is not None else None,
            "lm_head_name": lm_name,
            "lm_head_shape": list(lm_head.weight.shape) if lm_head is not None else None,
            "linear_leaf_counts": dict(sorted(leaf_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            **fwd_stats,
        }
        if layers is None or lm_head is None:
            splice["error"] = f"cannot locate layers/lm_head path={layers_path} lm={lm_name}"
            return fwd, splice

        keys = [
            k
            for k in agent_map
            if k == "lm_head.weight" or k.endswith(".lm_head.weight")
            or any(f"layers.{i}." in k for i in range(cut, 40))
        ]
        keys = [k for k in keys if not is_visual_key(k) and not k.startswith("mtp.")]
        log(f"copying Instruct keys onto live AgentWorld: cut={cut} n={len(keys)}")
        store = TensorStore(agent_dir, agent_map)
        try:
            copy_info = copy_keys_into_model(model, store, keys)
        finally:
            store.close()
        splice.update(copy_info)
        splice["n_keys_attempted"] = len(keys)
        spl_stats = _forward_once(model, tokenizer, PROBE_PROMPT, device_for_inputs)
        splice.update(spl_stats)

        hidden = None
        # dummy W: last hidden -> W -> original lm_head
        inputs = tokenizer(PROBE_PROMPT, return_tensors="pt")
        inputs = {k: v.to(device_for_inputs) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1] if out.hidden_states else out.last_hidden_state
            vec = hidden[:, -1, :]
            w = torch.nn.Linear(vec.shape[-1], vec.shape[-1], bias=False)
            w = w.to(device=vec.device, dtype=vec.dtype)
            mapped = w(vec)
            w_logits = lm_head(mapped)
        splice["dummy_w_ok"] = True
        splice["dummy_w_logits_shape"] = list(w_logits.shape)
        splice["dummy_w_hidden_in"] = list(vec.shape)
        splice["ok"] = True
        del w, mapped, w_logits, hidden, vec
    except Exception as e:
        if not fwd.get("ok"):
            fwd["error"] = _exc(e)
        splice["error"] = _exc(e)
        splice["ok"] = False
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return fwd, splice


def probe_speed_advice(
    *,
    world_dir: Path,
    agent_dir: Path,
    cut: int,
    device_map: str,
) -> dict[str, Any]:
    """Measure the two space-for-time knobs on the real GPU host.

    1. Selective gradient checkpointing: only full-attention layers keep
       checkpointing (O(L²) activation memory); linear-attention layers are
       uncheckpointed (O(L), safe given the observed ~33 GiB/GPU headroom).
       Per-layer toggle works via GradientCheckpointingLayer.gradient_checkpointing
       instance attribute — confirmed from transformers/modeling_layers.py.

    2. z/z1 batch-merge: the no-grad z forward and the live z1 forward both
       process the same o_ids (different dropout draws). Batching them into one
       call (batch_size×2 on o_ids) eliminates one full-backbone FSDP2
       all-gather round per micro-batch.

    Writes to report['speed_advice'].  Run with::

        # Stage 1: uses world_dir (AgentWorld backbone) — defaults already set
        cd train && python scripts/probe.py --speed-advice
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    out: dict[str, Any] = {"ok": False}

    # ── 1. Per-layer gradient_checkpointing architecture check (meta device) ──
    # Stage 1 trains AgentWorld (world_dir); use its config, not agent_dir.
    try:
        config = AutoConfig.from_pretrained(str(world_dir), trust_remote_code=True)
        tc = getattr(config, "text_config", config)
        layer_types: list[str] = getattr(tc, "layer_types", [])
        hidden_size: int = getattr(tc, "hidden_size", 2048)
        num_layers: int = getattr(tc, "num_hidden_layers", 40)
        full_attn_idx = sorted(i for i, t in enumerate(layer_types) if t == "full_attention")
        linear_attn_idx = sorted(i for i, t in enumerate(layer_types) if t != "full_attention")
        if not layer_types:
            full_attn_idx = sorted({3, 7, 11, 15, 19, 23, 27, 31, 35, 39} & set(range(num_layers)))
            linear_attn_idx = sorted(set(range(num_layers)) - set(full_attn_idx))
        out["layer_types_from_config"] = {
            "total": num_layers,
            "full_attention_count": len(full_attn_idx),
            "linear_attention_count": len(linear_attn_idx),
            "full_attention_indices": full_attn_idx,
            "hidden_size": hidden_size,
        }

        try:
            with torch.device("meta"):
                meta_model = None
                meta_loader = None
                for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
                    try:
                        meta_model = loader.from_config(config, trust_remote_code=True)
                        meta_loader = loader.__name__
                        break
                    except Exception:
                        pass
            if meta_model is not None:
                # ── 1a. Scan ALL modules BEFORE gradient_checkpointing_enable ──
                # Record which modules have gradient_checkpointing / layer_idx
                import inspect

                def _scan_modules(model_):
                    has_gc = {}   # name -> bool (has gradient_checkpointing attr)
                    has_idx = {}  # name -> value
                    for name, mod in model_.named_modules():
                        gc = getattr(mod, "gradient_checkpointing", _SENTINEL := object())
                        if gc is not _SENTINEL:
                            has_gc[name] = gc
                        idx = getattr(mod, "layer_idx", None)
                        if idx is not None:
                            has_idx[name] = idx
                    return has_gc, has_idx

                gc_before, idx_before = _scan_modules(meta_model)

                # ── 1b. gradient_checkpointing_enable and re-scan ──
                meta_model.gradient_checkpointing_enable()
                gc_after, idx_after = _scan_modules(meta_model)

                # Which modules gained / changed gradient_checkpointing after enable?
                gc_changed = {
                    k: {"before": gc_before.get(k), "after": gc_after[k]}
                    for k in gc_after
                    if gc_before.get(k) != gc_after[k]
                }

                # ── 1c. Identify decoder-layer candidates ──
                # Strategy A: path-based
                layers_found = None
                found_path = None
                for path in (
                    "model.language_model.layers",
                    "language_model.layers",
                    "model.model.layers",
                    "model.layers",
                ):
                    m = meta_model
                    for part in path.split("."):
                        m = getattr(m, part, None)
                        if m is None:
                            break
                    if m is not None:
                        layers_found = list(m)
                        found_path = path
                        break

                # Strategy B: layer_idx scan (what training-time code now uses)
                by_idx: dict[int, Any] = {}
                for mod in meta_model.modules():
                    idx = getattr(mod, "layer_idx", None)
                    if isinstance(idx, int):
                        by_idx[idx] = mod
                layer_idx_scan_count = len(by_idx)

                # Strategy B filtered: layer_idx + gradient_checkpointing
                by_idx_gc: dict[int, Any] = {}
                for mod in meta_model.modules():
                    idx = getattr(mod, "layer_idx", None)
                    if isinstance(idx, int) and hasattr(mod, "gradient_checkpointing"):
                        by_idx_gc[idx] = mod
                layer_idx_gc_scan_count = len(by_idx_gc)

                # Strategy C: class name contains "DecoderLayer" or "Block"
                by_classname: list[str] = []
                for name, mod in meta_model.named_modules():
                    cn = type(mod).__name__
                    if any(k in cn for k in ("DecoderLayer", "TransformerBlock", "HybridLayer")):
                        by_classname.append(f"{name}: {cn}")

                # ── 1d. GradientCheckpointingLayer inheritance check ──
                try:
                    from transformers.modeling_layers import GradientCheckpointingLayer as GCL
                    gcl_available = True
                except ImportError:
                    GCL = None
                    gcl_available = False

                gcl_inheritors = []
                for name, mod in meta_model.named_modules():
                    if GCL is not None and isinstance(mod, GCL):
                        gcl_inheritors.append(f"{name}: {type(mod).__name__}")
                    elif type(mod).__name__ in ("GradientCheckpointingLayer",):
                        gcl_inheritors.append(f"{name}: {type(mod).__name__} [by name]")

                # ── 1e. Check if model forward uses per-layer or model-level GC ──
                # Look at the inner language model's forward source for checkpoint calls
                gc_in_forward: dict[str, str] = {}
                for attr_name in ("model", "language_model"):
                    inner = getattr(meta_model, attr_name, None)
                    if inner is None:
                        continue
                    try:
                        src = inspect.getsource(type(inner).forward)
                        if "gradient_checkpointing_func" in src or "_gradient_checkpointing_func" in src:
                            gc_in_forward[attr_name] = "model-level: uses self._gradient_checkpointing_func in forward loop"
                        elif "gradient_checkpointing" in src:
                            gc_in_forward[attr_name] = "model-level: checks self.gradient_checkpointing in forward loop"
                        else:
                            gc_in_forward[attr_name] = "no gradient_checkpointing in forward source"
                    except (TypeError, OSError):
                        gc_in_forward[attr_name] = "could not get source"

                # ── 1f. Simulate apply_selective_checkpointing with modules() scan ──
                # (what the current training code does)
                full_attn = set(full_attn_idx)
                sim_layers = by_idx_gc  # idx -> mod
                sim_uncheckpointed = 0
                for idx, mod in sim_layers.items():
                    if idx not in full_attn:
                        mod.gradient_checkpointing = False
                        sim_uncheckpointed += 1
                flags_after_sim = [
                    getattr(by_idx_gc[i], "gradient_checkpointing", None)
                    for i in sorted(by_idx_gc)
                ]

                out["gc_mechanism_audit"] = {
                    "meta_loader": meta_loader,
                    # What modules have gradient_checkpointing BEFORE enable?
                    "gc_attr_before_enable_count": len(gc_before),
                    "gc_attr_before_sample": dict(list(gc_before.items())[:5]),
                    # What changed after enable?
                    "gc_attr_changed_after_enable_count": len(gc_changed),
                    "gc_attr_changed_sample": dict(list(gc_changed.items())[:5]),
                    # layer_idx scan
                    "layer_idx_scan_total": layer_idx_scan_count,
                    "layer_idx_gc_scan_total": layer_idx_gc_scan_count,
                    "layer_idx_gc_sample_names": list(by_idx_gc.keys())[:10],
                    # path-based discovery
                    "path_discovery_result": found_path,
                    "path_discovery_count": len(layers_found) if layers_found else 0,
                    # class-name discovery
                    "decoder_classname_matches": by_classname[:10],
                    # GradientCheckpointingLayer inheritance
                    "gcl_available_in_transformers": gcl_available,
                    "gcl_inheritor_count": len(gcl_inheritors),
                    "gcl_inheritor_sample": gcl_inheritors[:5],
                    # forward source analysis
                    "gc_in_forward": gc_in_forward,
                    # Simulation: what apply_selective_checkpointing (training) would do
                    "sim_apply_selective_checkpointing": {
                        "found_via_layer_idx_gc": layer_idx_gc_scan_count,
                        "would_uncheckpoint": sim_uncheckpointed,
                        "flags_after_sim_sample": flags_after_sim[:10],
                        "ok": sim_uncheckpointed > 0,
                    },
                }
                del meta_model
        except Exception as e:
            import traceback
            out["gc_mechanism_audit"] = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]}

        # ── 2. Static memory arithmetic for selective checkpointing ──
        # Residual-stream activation per uncheckpointed linear layer (bf16):
        #   batch=1 × seq_len × hidden_size × 2 bytes
        MAX_LEN = 65536
        AVG_LEN = 35000
        bytes_per_linear_layer_max = 1 * MAX_LEN * hidden_size * 2
        bytes_per_linear_layer_avg = 1 * AVG_LEN * hidden_size * 2
        # Full-attention layers at max_len keep O(L²) KV; linear layers are O(L).
        # Overhead estimate (residual stream only — real is ~1.5-2x due to attn
        # intermediates per layer, but linear-attn intermediates are O(L)).
        gib = lambda b: round(b / 1024**3, 3)
        total_linear_max_gib = gib(len(linear_attn_idx) * bytes_per_linear_layer_max * 1.8)
        total_linear_avg_gib = gib(len(linear_attn_idx) * bytes_per_linear_layer_avg * 1.8)
        out["selective_checkpointing_memory_math"] = {
            "linear_layers_to_uncheckpoint": len(linear_attn_idx),
            "full_attention_layers_remain_checkpointed": len(full_attn_idx),
            "estimated_activation_gib_at_max_len_65536": total_linear_max_gib,
            "estimated_activation_gib_at_avg_len_35000": total_linear_avg_gib,
            "observed_headroom_gib_per_gpu": 33.0,
            "fits_at_max_len": total_linear_max_gib < 33.0,
            "note": (
                "Estimate uses 1.8× residual-stream size as proxy for all O(L) "
                "activations per linear-attention layer. Full-attention layers are "
                "never uncheckpointed because their O(L²) KV activations at "
                f"L={MAX_LEN} would be ~{gib(MAX_LEN**2 * 2 * 2):.1f} GiB/layer."
            ),
        }

        # ── 3. Throughput arithmetic (using empirical step times) ──
        # Known from training run: 192.92 s/step with 1 extra live o forward.
        # Original (no anti-collapse): ~158 s/step.
        total_steps_per_epoch = 6815  # 13630 total / 2 epochs
        current_sps = 192.92
        out["throughput_baseline"] = {
            "current_s_per_step": current_sps,
            "current_h_per_epoch": round(total_steps_per_epoch * current_sps / 3600, 1),
            "original_s_per_step_estimate": 158.0,
            "original_h_per_epoch_estimate": round(total_steps_per_epoch * 158.0 / 3600, 1),
        }
        # Selective checkpointing saves backward recompute for 30 linear layers.
        # Recompute = 1 extra forward per checkpointed layer per backward.
        # For c and u (both live): 30 fewer recomputes each. z1 (live): same.
        # 3 paths × 30 saved recomputes = 90 layer-passes saved out of ~400 total
        # (rough model: 3 no-backward forwards + 3×(forward+backward) with recompute
        # = 3×40 + 3×80 = 120+240 = 360 layer-passes; selective: 3×40 + 3×50 = 270).
        saved_fraction_ckpt = (360 - 270) / 360  # ~0.25
        sps_after_ckpt = round(current_sps * (1 - saved_fraction_ckpt * 0.8), 1)  # 0.8: recompute ≠ full forward cost
        # z/z1 batch-merge: eliminates 1 of 4 distinct backbone calls (the no-grad z
        # forward merges with z1 into one batched call). Estimated saving: ~15-20%.
        sps_after_both = round(sps_after_ckpt * 0.87, 1)
        out["throughput_estimates"] = {
            "after_selective_checkpointing_s_per_step": sps_after_ckpt,
            "after_selective_checkpointing_h_per_epoch": round(total_steps_per_epoch * sps_after_ckpt / 3600, 1),
            "after_both_optimizations_s_per_step": sps_after_both,
            "after_both_optimizations_h_per_epoch": round(total_steps_per_epoch * sps_after_both / 3600, 1),
            "note": "Static estimates; run a smoke step and measure to calibrate.",
        }

        # ── 4. z/z1 batch-merge: GPU forward test ──
        z1_batch_test: dict[str, Any] = {"tested": False}
        if torch.cuda.is_available():
            try:
                dtype = torch.bfloat16
                model, _, _ = _load_qwen_moe(world_dir, dtype=dtype, device_map=device_map)
                tokenizer = AutoTokenizer.from_pretrained(str(agent_dir), trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                sample = "ls /tmp\n"
                toks = tokenizer(sample, return_tensors="pt")
                o_ids = toks["input_ids"].to("cuda")
                o_mask = toks["attention_mask"].to("cuda")
                # batch=2: two independent dropout views of the same o_ids
                o_ids2 = torch.cat([o_ids, o_ids], dim=0)
                o_mask2 = torch.cat([o_mask, o_mask], dim=0)
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
                with torch.no_grad():
                    out_single = model(input_ids=o_ids, attention_mask=o_mask, output_hidden_states=True, use_cache=False)
                    h_single = out_single.hidden_states[-1][:, -1, :]
                mem_single = torch.cuda.memory_allocated() - mem_before
                torch.cuda.reset_peak_memory_stats()
                mem_before2 = torch.cuda.memory_allocated()
                with torch.no_grad():
                    out_double = model(input_ids=o_ids2, attention_mask=o_mask2, output_hidden_states=True, use_cache=False)
                    h_double = out_double.hidden_states[-1][:, -1, :]
                mem_double = torch.cuda.memory_allocated() - mem_before2
                shapes_match = h_single.shape == h_double[:1].shape
                z1_batch_test = {
                    "tested": True,
                    "ok": shapes_match,
                    "single_hidden_shape": list(h_single.shape),
                    "double_hidden_shape": list(h_double.shape),
                    "batch1_mem_delta_mb": round(mem_single / 1024**2, 2),
                    "batch2_mem_delta_mb": round(mem_double / 1024**2, 2),
                }
                del model, h_single, h_double, out_single, out_double
                torch.cuda.empty_cache()
            except Exception as e:
                z1_batch_test = {"tested": True, "ok": False, "error": repr(e)}
        out["z1_batch_merge_test"] = z1_batch_test
        out["ok"] = True
    except Exception as e:
        out["error"] = repr(e)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--no-base", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
    )
    p.add_argument("--skip-weights", action="store_true", help="Skip 3-way tensor cosine")
    p.add_argument("--skip-forward", action="store_true", help="Skip transformers load + forward")
    p.add_argument(
        "--cut",
        type=int,
        default=36,
        help="Copy Instruct layers [cut, 40) + lm_head onto loaded AgentWorld before the second forward",
    )
    p.add_argument("--device-map", default="auto", help="transformers device_map")
    p.add_argument(
        "--meta",
        action="store_true",
        help="Also build module tree on meta device (forward already dumps the live tree)",
    )
    p.add_argument(
        "--speed-advice",
        action="store_true",
        help=(
            "Probe selective gradient checkpointing feasibility and z/z1 batch-merge "
            "on the GPU host. Writes report['speed_advice']. Uses meta device for "
            "architecture checks (no weights needed) plus a live GPU micro-forward "
            "if CUDA is available."
        ),
    )
    p.add_argument(
        "--stage1-config",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "Read train.model_dir (and optionally train.torch_dtype) from a Stage-1 "
            "YAML config (e.g. configs/jepa/stage1.yaml) and override --world. "
            "Makes probe and training guaranteed-identical in model path. "
            "Example: python scripts/probe.py --stage1-config configs/jepa/stage1.yaml "
            "--speed-advice --skip-weights --skip-forward"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= int(args.cut) <= 39:
        raise SystemExit("--cut must be in 0..39")
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --stage1-config: read model_dir from a Stage-1 YAML and override --world.
    # Guarantees probe uses the exact same model as train_jepa.py.
    if args.stage1_config is not None:
        import yaml as _yaml

        s1_path = args.stage1_config
        if not s1_path.is_absolute():
            s1_path = ROOT / "train" / s1_path
        try:
            s1 = _yaml.safe_load(s1_path.read_text(encoding="utf-8"))
            s1_model = (s1 or {}).get("model_dir") or (s1 or {}).get("train", {}).get("model_dir")
            if s1_model:
                print(f"[stage1-config] overriding --world with model_dir={s1_model!r} from {s1_path}")
                args.world = s1_model
            else:
                print(f"[stage1-config] WARNING: no model_dir found in {s1_path}, using default")
        except Exception as exc:
            print(f"[stage1-config] WARNING: could not read {s1_path}: {exc}")

    world_dir = resolve_model(args.world, source=args.source, cache_dir=cache_dir, role="world")
    agent_dir = resolve_model(args.agent, source=args.source, cache_dir=cache_dir, role="agent")
    base_dir = None
    if not args.no_base:
        base_dir = resolve_model(
            args.base_model, source=args.source, cache_dir=cache_dir, role="base"
        )

    world_cfg = load_json(world_dir / "config.json") or load_json(world_dir / "configuration.json")
    agent_cfg = load_json(agent_dir / "config.json") or load_json(agent_dir / "configuration.json")
    base_cfg = None
    if base_dir is not None:
        base_cfg = load_json(base_dir / "config.json") or load_json(base_dir / "configuration.json")

    report: dict[str, Any] = {
        "train_stack": {
            "library": ["torch", "transformers", "peft", "accelerate"],
            "fsdp": "accelerate FSDP2, wrap Qwen3_5MoeDecoderLayer",
            "loop": "custom PyTorch (JEPA then draft/W); not TRL SFTTrainer",
            "not_used": ["unsloth", "axolotl", "ms-swift", "trl.SFTTrainer"],
            "hf_class": "Qwen3_5MoeForConditionalGeneration",
            "text_prefix": "model.language_model",
            "lm_head": "lm_head (untied)",
            "new_modules": ["JEPA Pred", "draft head", "scorer", "W 2048x2048"],
            "leave_unused": ["model.visual", "mtp.*"],
            "hidden_size": (text_config(world_cfg) or {}).get("hidden_size"),
            "num_hidden_layers": (text_config(world_cfg) or {}).get("num_hidden_layers"),
            "vocab_size": (text_config(world_cfg) or {}).get("vocab_size")
            or (world_cfg or {}).get("vocab_size"),
        },
        "paths": {
            "world": str(world_dir),
            "agent": str(agent_dir),
            "base": str(base_dir) if base_dir else None,
            "out": str(out_dir),
        },
        "sidecars": {
            "world": list_sidecar_files(world_dir),
            "agent": list_sidecar_files(agent_dir),
            "base": list_sidecar_files(base_dir) if base_dir else None,
        },
        "tokenizers": {
            "world": tokenizer_snapshot(world_dir),
            "agent": tokenizer_snapshot(agent_dir),
            "base": tokenizer_snapshot(base_dir) if base_dir else None,
        },
        "configs": {
            "world_root_keys": sorted((world_cfg or {}).keys()),
            "instruct_root_keys": sorted((agent_cfg or {}).keys()),
            "world_language_model_only": (world_cfg or {}).get("language_model_only"),
            "instruct_language_model_only": (agent_cfg or {}).get("language_model_only"),
            "world_architectures": (world_cfg or {}).get("architectures"),
            "instruct_architectures": (agent_cfg or {}).get("architectures"),
            "tie_word_embeddings": {
                "world": (world_cfg or {}).get("tie_word_embeddings"),
                "instruct": (agent_cfg or {}).get("tie_word_embeddings"),
                "base": (base_cfg or {}).get("tie_word_embeddings") if base_cfg else None,
            },
            "text_world": text_config(world_cfg),
            "text_instruct": text_config(agent_cfg),
            "diff_world_vs_instruct": cfg_diff(world_cfg, agent_cfg),
            "diff_instruct_vs_base": cfg_diff(agent_cfg, base_cfg) if base_cfg else None,
            "diff_world_vs_base": cfg_diff(world_cfg, base_cfg) if base_cfg else None,
        },
    }

    log("indexing safetensors maps")
    world_map = load_weight_map(world_dir)
    agent_map = load_weight_map(agent_dir)
    base_map = load_weight_map(base_dir) if base_dir else None
    report["inventories"] = {
        "world": key_inventory(world_map),
        "instruct": key_inventory(agent_map),
        "base": key_inventory(base_map) if base_map else None,
    }

    if not args.skip_weights:
        log("streaming weight stats (one tensor at a time)")
        lt = text_config(world_cfg).get("layer_types")
        report["weights"] = compare_weights(
            world_dir=world_dir,
            agent_dir=agent_dir,
            base_dir=base_dir,
            world_map=world_map,
            agent_map=agent_map,
            base_map=base_map,
            layer_types=lt if isinstance(lt, list) else None,
        )
        report["cut_hints"] = cut_hints(report["weights"].get("per_layer") or [])
        try:
            report["recommended_cut"] = pick_cut(report["weights"].get("per_layer") or [])
            rec = report["recommended_cut"]
            log(f"recommended_cut ℓ={rec['ell']} gap={rec['gap']:.6f}")
        except Exception as e:
            report["recommended_cut"] = {"error": str(e)}
            log(f"recommended_cut failed: {e}")
    else:
        report["weights"] = None
        report["cut_hints"] = None
        report["recommended_cut"] = None

    if args.meta:
        log("meta-device module tree (Instruct)")
        report["meta_tree"] = try_meta_tree(agent_dir)
    else:
        report["meta_tree"] = {"ok": False, "error": "skipped (pass --meta)"}

    if args.speed_advice:
        log("probing selective-checkpointing + z/z1 batch-merge (--speed-advice)")
        report["speed_advice"] = probe_speed_advice(
            world_dir=world_dir,
            agent_dir=agent_dir,
            cut=int(args.cut),
            device_map=str(args.device_map),
        )
        sa = report["speed_advice"]
        audit = sa.get("gc_mechanism_audit") or {}
        sim = audit.get("sim_apply_selective_checkpointing") or {}
        mem = sa.get("selective_checkpointing_memory_math") or {}
        thr = sa.get("throughput_estimates") or {}
        z1t = sa.get("z1_batch_merge_test") or {}

        sep = "=" * 70
        print(sep)
        print("SPEED-ADVICE DIAGNOSTIC RESULTS")
        print(sep)

        print("\n── GC mechanism (how does this model do gradient checkpointing?) ──")
        print(f"  meta_loader (model class):      {audit.get('meta_loader')}")
        print(f"  path_discovery_result:          {audit.get('path_discovery_result')}")
        print(f"  path_discovery_layer_count:     {audit.get('path_discovery_count')}")
        print(f"  gcl_inheritor_count:            {audit.get('gcl_inheritor_count')}")
        print(f"    (>0 = layers inherit GradientCheckpointingLayer, per-layer flag works)")
        print(f"  layer_idx_scan_total:           {audit.get('layer_idx_scan_total')}")
        print(f"  layer_idx_gc_scan_total:        {audit.get('layer_idx_gc_scan_total')}")
        print(f"    (= what apply_selective_checkpointing finds via model.modules())")
        print(f"  gc_attr_changed_after_enable:   {audit.get('gc_attr_changed_after_enable_count')}")
        print(f"    (>0 = gradient_checkpointing_enable() sets per-layer flags)")
        print(f"  gc_in_forward:                  {audit.get('gc_in_forward')}")
        print(f"  decoder_classname_matches:      {audit.get('decoder_classname_matches', [])[:5]}")
        print(f"  gcl_inheritor_sample:           {audit.get('gcl_inheritor_sample', [])[:3]}")

        print("\n── Simulation: what apply_selective_checkpointing() would do ──")
        print(f"  found_via_layer_idx_gc:         {sim.get('found_via_layer_idx_gc')}")
        print(f"  would_uncheckpoint:             {sim.get('would_uncheckpoint')}")
        print(f"  flags_after_sim_sample:         {sim.get('flags_after_sim_sample', [])[:10]}")
        print(f"  sim_ok:                         {sim.get('ok')}")

        print("\n── Memory math ──")
        print(f"  linear layers to uncheckpoint:  {mem.get('linear_layers_to_uncheckpoint')}")
        print(f"  activation GiB at max_len:      {mem.get('estimated_activation_gib_at_max_len_65536')}")
        print(f"  headroom GiB/GPU:               {mem.get('observed_headroom_gib_per_gpu')}")
        print(f"  fits_at_max_len:                {mem.get('fits_at_max_len')}")

        print("\n── z/z1 batch-merge test ──")
        print(f"  tested:  {z1t.get('tested')}  ok: {z1t.get('ok')}  error: {z1t.get('error')}")
        if z1t.get("ok"):
            print(f"  single hidden shape: {z1t.get('single_hidden_shape')}")
            print(f"  double hidden shape: {z1t.get('double_hidden_shape')}")
            print(f"  mem batch1={z1t.get('batch1_mem_delta_mb')} MB  batch2={z1t.get('batch2_mem_delta_mb')} MB")

        print("\n── Throughput estimates ──")
        print(f"  after selective-ckpt:  {thr.get('after_selective_checkpointing_h_per_epoch')} h/epoch")
        print(f"  after both opts:       {thr.get('after_both_optimizations_h_per_epoch')} h/epoch")

        print(sep)
        if audit.get("error"):
            print(f"WARNING gc_mechanism_audit error: {audit.get('error')}")
            print(audit.get("tb", ""))
        print()
    else:
        report["speed_advice"] = {"ok": False, "error": "skipped (pass --speed-advice)"}

    report["frameworks"] = probe_frameworks()
    log(f"libraries: {report['frameworks']['versions']}")

    if args.skip_forward:
        report["forward"] = {"ok": False, "error": "skipped"}
        report["splice_forward"] = {"ok": False, "error": "skipped"}
    else:
        log("transformers forward + in-memory Instruct tail copy onto AgentWorld")
        fwd, spl = try_forward_and_surgery(
            world_dir=world_dir,
            agent_dir=agent_dir,
            agent_map=agent_map,
            cut=int(args.cut),
            device_map=str(args.device_map),
        )
        report["forward"] = fwd
        report["splice_forward"] = spl

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    summary_path = out_dir / "summary.txt"
    write_summary(report, summary_path)
    log(f"wrote {report_path}")
    log(f"wrote {summary_path}")
    log("send report.json (and summary.txt) back")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
