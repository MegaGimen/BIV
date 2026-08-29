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

Send back train/outputs/probe/report.json and summary.txt.
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
if str(_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_DIR))

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
    """Heuristic only — ℓ is chosen after reading the curve, not here."""
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
        "note": "High Instruct/AW delta ratio at the tail suggests where Instruct specialized for writing; high cosine(delta) in the middle suggests a shared core. Pick ℓ from the curve, do not trust a single argmax.",
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
    a("")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= int(args.cut) <= 39:
        raise SystemExit("--cut must be in 0..39")
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    else:
        report["weights"] = None
        report["cut_hints"] = None

    if args.meta:
        log("meta-device module tree (Instruct)")
        report["meta_tree"] = try_meta_tree(agent_dir)
    else:
        report["meta_tree"] = {"ok": False, "error": "skipped (pass --meta)"}

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
