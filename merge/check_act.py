#!/usr/bin/env python3
"""Read an ACT merge directory and say what actually landed.

Does not load 35B. Reads ``merge_meta.json`` + the mask copy that merge
wrote next to the weights.

    python merge/check_act.py
    python merge/check_act.py --dir merge/output/act
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "merge" / "output" / "act"

BUCKETS = (
    "packed_gate_up",
    "packed_down",
    "shared_expert",
    "router",
    "attn",
    "ln",
    "embed",
    "lm_head",
    "other",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def mask_bucket(key: str) -> str:
    k = str(key)
    if k == "lm_head" or k.endswith(".lm_head"):
        return "lm_head"
    if "experts.gate_up_proj" in k:
        return "packed_gate_up"
    if "experts.down_proj" in k:
        return "packed_down"
    if "shared_expert" in k:
        return "shared_expert"
    if k.endswith(".mlp.gate") or ".mlp.gate" in k:
        return "router"
    if "linear_attn" in k or "self_attn" in k:
        return "attn"
    if "layernorm" in k or k == "norm" or k.endswith(".norm"):
        return "ln"
    if "embed" in k:
        return "embed"
    return "other"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_rows(payload: dict[str, Any], *, no_lm_head: bool) -> list[dict[str, Any]]:
    if no_lm_head:
        rows = payload.get("mask_no_lm_head")
        if not isinstance(rows, list):
            rows = [
                r
                for r in (payload.get("mask") or [])
                if isinstance(r, dict)
                and r.get("kind") != "lm_head"
                and r.get("key") != "lm_head"
            ]
        return [r for r in rows if isinstance(r, dict)]
    rows = payload.get("mask")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def inspect(out_dir: Path) -> dict[str, Any]:
    meta_path = out_dir / "merge_meta.json"
    mask_path = out_dir / "mask.json"
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    if not meta_path.is_file():
        add("merge_meta.json", False, f"missing {meta_path}")
        return {
            "out_dir": str(out_dir),
            "ok": False,
            "checks": checks,
            "buckets": {},
            "meta": {},
        }
    add("merge_meta.json", True, str(meta_path))
    meta = load_json(meta_path)

    no_lm_head = bool(meta.get("no_lm_head", True))
    n_mask = int(meta.get("n_mask_rows") or 0)
    n_lm = int(meta.get("n_lm_head_rows") or 0)
    n_written = int(meta.get("n_rows_written") or 0)
    n_unmatched = int(meta.get("n_unmatched_modules") or 0)
    n_missing = int(meta.get("n_missing_world") or 0)
    n_shape = int(meta.get("n_shape_mismatch") or 0)
    n_oob = int(meta.get("n_oob_skipped") or 0)

    add("no_lm_head", no_lm_head, f"no_lm_head={no_lm_head} n_lm_head_rows={n_lm}")
    add(
        "lm_head_not_in_mask_used",
        n_lm == 0 if no_lm_head else True,
        f"n_lm_head_rows={n_lm} (0 means this merge did not take vocab rows)",
    )

    if not mask_path.is_file():
        add("mask.json", False, f"missing {mask_path}")
        buckets: dict[str, int] = {}
        n_mask_file = 0
    else:
        add("mask.json", True, str(mask_path))
        payload = load_json(mask_path)
        rows = mask_rows(payload, no_lm_head=no_lm_head)
        n_mask_file = len(rows)
        buckets = dict(Counter(mask_bucket(str(r.get("key", ""))) for r in rows))
        add(
            "mask_row_count",
            n_mask_file == n_mask or n_mask == 0,
            f"mask file {n_mask_file} rows vs merge_meta n_mask_rows={n_mask}",
        )

    packed = int(buckets.get("packed_gate_up") or 0) + int(
        buckets.get("packed_down") or 0
    )
    shared = int(buckets.get("shared_expert") or 0)
    router = int(buckets.get("router") or 0)
    add(
        "mask_has_packed_experts",
        packed > 0,
        f"packed_gate_up={buckets.get('packed_gate_up', 0)} "
        f"packed_down={buckets.get('packed_down', 0)}",
    )
    add(
        "mask_has_shared_or_router",
        True,
        f"shared_expert={shared} router={router} (informational, not required)",
    )
    add(
        "all_mask_modules_matched_instruct",
        n_unmatched == 0,
        f"n_unmatched_modules={n_unmatched} "
        f"unmatched={meta.get('unmatched_modules')}",
    )
    add(
        "no_missing_world_or_shape",
        n_missing == 0 and n_shape == 0,
        f"n_missing_world={n_missing} n_shape_mismatch={n_shape} n_oob_skipped={n_oob}",
    )
    # OOB skips can drop a few rows; a large gap means the mask did not land.
    gap = n_mask - n_written if n_mask else 0
    add(
        "rows_written_near_mask",
        n_mask > 0 and gap <= max(32, int(0.01 * n_mask)),
        f"n_rows_written={n_written} n_mask_rows={n_mask} gap={gap}",
    )

    examples = [str(x) for x in (meta.get("patched_examples") or [])]
    ex_packed = [e for e in examples if "experts." in e]
    add(
        "patched_examples_mention_experts",
        bool(ex_packed) or packed == 0,
        f"{len(ex_packed)}/{len(examples)} examples mention experts "
        f"(list is capped at 16, early shards may fill it first)",
    )

    skip_overall = {
        "patched_examples_mention_experts",
        "mask_has_shared_or_router",
    }
    ok = all(c["ok"] for c in checks if c["name"] not in skip_overall)
    moe_ok = packed > 0 and n_unmatched == 0 and n_written > 0 and gap <= max(
        32, int(0.01 * n_mask) if n_mask else 0
    )
    return {
        "out_dir": str(out_dir),
        "ok": ok,
        "moe_written": moe_ok,
        "checks": checks,
        "buckets": {k: int(buckets.get(k, 0)) for k in BUCKETS},
        "meta": {
            "lambda": meta.get("lambda"),
            "no_lm_head": no_lm_head,
            "mask_source": meta.get("mask"),
            "n_mask_rows": n_mask,
            "n_lm_head_rows": n_lm,
            "n_tensors_patched": meta.get("n_tensors_patched"),
            "n_rows_written": n_written,
            "n_rewritten_shards": meta.get("n_rewritten_shards"),
            "n_copied_shards": meta.get("n_copied_shards"),
            "n_unmatched_modules": n_unmatched,
            "patched_examples": examples,
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "check_act.py — 读 merge 输出目录，不加载模型",
        f"dir: {report['out_dir']}",
        "",
        "merge_meta:",
    ]
    meta = report.get("meta") or {}
    for k in (
        "lambda",
        "no_lm_head",
        "mask_source",
        "n_mask_rows",
        "n_lm_head_rows",
        "n_tensors_patched",
        "n_rows_written",
        "n_rewritten_shards",
        "n_copied_shards",
        "n_unmatched_modules",
    ):
        lines.append(f"  {k}: {meta.get(k)}")
    lines.append("")
    lines.append("这份 merge 实际用的掩码（按通道来源拆）:")
    buckets = report.get("buckets") or {}
    n_all = sum(int(v) for v in buckets.values())
    for k in BUCKETS:
        n = int(buckets.get(k, 0))
        if n == 0:
            continue
        pct = (100.0 * n / n_all) if n_all else 0.0
        lines.append(f"  {k:<16} {n:>8}  {pct:5.1f}%")
    lines.append("")
    lines.append("checks:")
    for c in report.get("checks") or []:
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(f"  {mark}  {c['name']}  {c['detail']}")
    lines.append("")
    if report.get("moe_written"):
        lines.append(
            "MoE: 掩码里有 packed 专家通道，模块名都对上了 Instruct 权重，"
            "写入行数接近掩码行数 → 这些通道已经按 λ 写进 merge/output/act。"
        )
    else:
        lines.append(
            "MoE: 不能从这份目录判定 packed 专家已写入。"
            "看上面 FAIL 项（缺文件 / unmatched / 行数对不上 / 掩码里没有 experts）。"
        )
    lines.append(
        "ACT 不是「除 lm_head 外整份 AgentWorld 拷过去」。"
        "copy 的 shard 仍是 Instruct 原文件；patch 的张量只改掩码选中的行。"
    )
    lines.append("OVERALL: " + ("PASS" if report.get("ok") else "FAIL"))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="ACT merge output (default: merge/output/act)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print the report dict as JSON instead of text",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.dir if args.dir.is_absolute() else (ROOT / args.dir)
    report = inspect(out_dir)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    else:
        log(render(report))
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
