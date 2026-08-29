#!/usr/bin/env python3
"""Probe AgentWorld / Instruct / Base for fish-cut + JEPA training.

This is the Qwen3.5-35B-A3B line, not Muse. Muse trains one dense CausalLM
with TRL SFTTrainer + LoRA on observation tokens. Here Stage -1 copies
whole decoder layers, Stage 1 trains JEPA on last-layer states, Stage 2
adds a new draft / W and feeds Instruct's original lm_head.

Reuses merge/output/cache/<id> from download.py (same defaults as merge.py).
Does not keep three full 35B copies in RAM: one tensor at a time.

GPU host::

    python merge/probe.py
    python merge/probe.py --meta          # also dump HF module tree (needs transformers)
    python merge/probe.py --skip-weights  # config / tokenizer / key index only

Send back ``merge/output/probe/report.json`` (and summary.txt if present).
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
from merge import TensorStore, is_visual_key, load_weight_map  # noqa: E402

DEFAULT_OUT = ROOT / "merge" / "output" / "probe"

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
    a("BIV merge/probe.py summary")
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


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
    p.add_argument("--skip-weights", action="store_true", help="Skip tensor stats")
    p.add_argument(
        "--meta",
        action="store_true",
        help="Build module tree on meta device (needs transformers with qwen3_5_moe)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
            "line": "Qwen3.5-35B-A3B fish-cut + JEPA + W",
            "not": "Muse TRL SFTTrainer / Chat Vector",
            "hf_class": "Qwen3_5MoeForConditionalGeneration",
            "fsdp_wrap": "Qwen3_5MoeDecoderLayer",
            "text_prefix": "model.language_model",
            "lm_head": "lm_head (untied, 2048 x vocab)",
            "new_modules": ["JEPA Pred", "draft head", "scorer", "W 2048x2048"],
            "leave_unused": ["model.visual", "mtp.* (official speculative draft)"],
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
