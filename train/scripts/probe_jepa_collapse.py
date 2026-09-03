#!/usr/bin/env python3
"""Forward-only collapse probe for Stage 1 JEPA. No backward, no LoRA.

Left = concat(z_t, u) = [Enc(h); Enc(a)] (4096). Right = Enc(h,a,o) (2048).
Takes the first N complete turns from each mix source in file order, then
ranks cross-source cosine on each side separately. Writes two files:
left top-20 and right top-20. Never stores history strings.

  cd train
  CUDA_VISIBLE_DEVICES=0 bash scripts/probe_jepa_collapse.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
SCRIPTS = Path(__file__).resolve().parent
MERGE = ROOT / "merge"
for p in (str(SRC), str(MERGE), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_jepa as tj  # noqa: E402
from biv_wm.arch import install_hidden_only_forward  # noqa: E402
from biv_wm.hao import split_hao  # noqa: E402
from biv_wm.jepa import attach_pair_snippets, cross_source_topk  # noqa: E402
from download import resolve_model  # noqa: E402

DEFAULT_MAX_ROWS = 25 * 8
HARD_CAP = 400
DEFAULT_TOP_K = 20
_PARALLEL_ENV = (
    "ACCELERATE_USE_PARALLELISM_CONFIG",
    "PARALLELISM_CONFIG_DP_REPLICATE_SIZE",
    "PARALLELISM_CONFIG_DP_SHARD_SIZE",
    "PARALLELISM_CONFIG_TP_SIZE",
    "PARALLELISM_CONFIG_CP_SIZE",
    "PARALLELISM_CONFIG_CP_BACKEND",
    "BIV_CP_SIZE",
    "BIV_PARALLEL",
)


def _clear_parallelism_env() -> None:
    """Drop 4-GPU train leftovers so this 1-GPU probe cannot inherit a mesh."""
    for key in _PARALLEL_ENV:
        os.environ.pop(key, None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=TRAIN / "configs/jepa/stage1.yaml")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None, help="Ignored; probe is always 1 GPU.")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--max-pairs", type=int, default=None, help="Alias of --top-k.")
    p.add_argument("--snippet", type=int, default=120)
    p.add_argument("--print-pairs", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--close-threshold", type=float, default=None, help="Ignored; kept so old flags still parse.")
    p.add_argument("--source", choices=["modelscope", "huggingface"], default=os.environ.get("MERGE_SOURCE", "modelscope"))
    p.add_argument("--last-token", type=int, default=None)
    return p.parse_args()


def load_first_hao(mix_dir: Path, src: str, split: str, n: int) -> list[list]:
    """First ``n`` complete (h,a,o) trajectories in file order. No reservoir."""
    path = mix_dir / src / f"{split}.jsonl"
    if not path.is_file():
        tj.log(f"skip missing {path}")
        return []
    rows: list[list] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or split_hao(msgs) is None:
                continue
            rows.append(msgs)
    tj.log(f"{src}: first {len(rows)} complete turns (file order, cap={n})")
    return rows


def _cell(block: object, key: str) -> str:
    if not isinstance(block, dict) or block.get(key) is None:
        return "na"
    return f"{float(block[key]):.3f}"


def _preview_top(side: str, recs: list[dict[str, object]], n_print: int, snippet: int) -> str:
    from biv_wm.jepa import _one_line

    n_print = max(0, int(n_print))
    if n_print == 0:
        return ""
    lines = [f"{side} showing {min(n_print, len(recs))}/{len(recs)}"]
    for rec in recs[:n_print]:
        cos = float(rec.get("cosine") or 0.0)
        a_i = _one_line(str(rec.get("a_i") or ""), snippet)
        o_i = _one_line(str(rec.get("o_i") or ""), snippet)
        a_j = _one_line(str(rec.get("a_j") or ""), snippet)
        o_j = _one_line(str(rec.get("o_j") or ""), snippet)
        src_i = rec.get("src_i")
        src_j = rec.get("src_j")
        lines.append(
            f"  {cos:.3f}  {src_i}#{rec.get('src_row_i')} a={a_i} o={o_i}  ||  "
            f"{src_j}#{rec.get('src_row_j')} a={a_j} o={o_j}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    max_rows = int(args.max_rows)
    if max_rows < 1:
        raise SystemExit("max-rows must be >= 1")
    if max_rows > HARD_CAP:
        raise SystemExit(f"max-rows cap is {HARD_CAP}; got {max_rows}")
    top_k = int(args.max_pairs if args.max_pairs is not None else args.top_k)
    if top_k < 1:
        raise SystemExit("top-k must be >= 1")

    import torch
    from torch.utils.data import DataLoader

    _clear_parallelism_env()
    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = tj._load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    cp_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rank_log(msg: str) -> None:
        tj.log(msg)

    cache_dir = tj._resolve(args.cache_dir or "merge/output/cache", ROOT)
    model_dir = resolve_model(
        str(args.model_dir or cfg["model_dir"]),
        source=args.source,
        cache_dir=cache_dir,
        role="world",
    )
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    if len(sources) < 2:
        raise SystemExit("cross-dataset probe needs at least two mix sources")
    mix_dir = tj.resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    out_dir = args.out_dir or tj._resolve("outputs/jepa_collapse_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)
    max_length = int(args.max_length or tcfg.get("max_length") or 32768)
    last_token = int(args.last_token if args.last_token is not None else (tcfg.get("last_token") or -3))
    pad_multiple = 0
    per_source = max(1, max_rows // len(sources))

    model, tokenizer = tj.load_backbone(
        model_dir,
        dtype,
        bool(tcfg.get("gradient_checkpointing", True)),
        attn_implementation=None,
        distributed=False,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    install_hidden_only_forward(model, detach_head=False)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)

    tagged: list[tuple[str, int, list]] = []
    for src in sources:
        rows = load_first_hao(mix_dir, src, "train", per_source)
        tagged.extend((src, i, msgs) for i, msgs in enumerate(rows))
    if not tagged:
        raise SystemExit("probe collected 0 rows")

    train_ds = tj.HaoDataset([msgs for _, _, msgs in tagged], tokenizer, max_length, encoding="stage1")
    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: tj.collate(b, pad_id, pad_multiple),
    )

    lefts: list[torch.Tensor] = []
    rights: list[torch.Tensor] = []
    o_texts: list[str] = []
    a_texts: list[str] = []
    src_names: list[str] = []
    src_index: list[int] = []
    n_done = 0
    rank_log(
        f"probe left=[Enc(h);Enc(a)] (4096) vs right=Enc(h,a,o) (2048) "
        f"single-GPU device={device} max_rows={max_rows} per_source={per_source} "
        f"top_k={top_k} last_token={last_token} snippet={args.snippet}"
    )
    with torch.no_grad():
        for batch, (src, src_i, _) in zip(loader, tagged):
            if n_done >= max_rows:
                break
            batch_t = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            h_state = tj.full_hidden(model, batch_t["state_ids"], batch_t["state_mask"], cp_size)
            h_act = tj.full_hidden(model, batch_t["action_ids"], batch_t["action_mask"], cp_size)
            h_full = tj.full_hidden(model, batch_t["full_ids"], batch_t["full_mask"], cp_size)
            z_t = tj.gather_at(
                h_state,
                tj.last_token_index(batch_t["state_ids"], batch_t["state_mask"], last_token),
            )
            u = tj.gather_at(
                h_act,
                tj.last_token_index(batch_t["action_ids"], batch_t["action_mask"], last_token),
            )
            z_next = tj.gather_at(
                h_full,
                tj.last_token_index(batch_t["full_ids"], batch_t["full_mask"], last_token),
            )
            left = torch.cat([z_t, u], dim=-1)
            for i in range(left.size(0)):
                lefts.append(left[i].detach().float().cpu())
                rights.append(z_next[i].detach().float().cpu())
                a_texts.append(batch["a_text"][i] if "a_text" in batch else "")
                o_texts.append(batch["o_text"][i] if "o_text" in batch else "")
                src_names.append(src)
                src_index.append(src_i)
            n_done += left.size(0)
            if n_done % 25 == 0:
                rank_log(f"probe encoded {n_done}/{len(tagged)}")

    if not lefts:
        raise SystemExit("probe collected 0 encodings")
    left = torch.stack(lefts)
    right = torch.stack(rights)
    if int(left.size(-1)) != int(right.size(-1)) * 2:
        raise SystemExit(
            f"expected left dim=2*right dim, got left={tuple(left.shape)} right={tuple(right.shape)}"
        )

    left_stats = cross_source_topk(left, src_names, k=top_k)
    right_stats = cross_source_topk(right, src_names, k=top_k)
    left_recs = attach_pair_snippets(
        list(left_stats["top"]),
        sources=src_names,
        src_index=src_index,
        a_texts=a_texts,
        o_texts=o_texts,
        snippet=int(args.snippet),
    )
    right_recs = attach_pair_snippets(
        list(right_stats["top"]),
        sources=src_names,
        src_index=src_index,
        a_texts=a_texts,
        o_texts=o_texts,
        snippet=int(args.snippet),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    counts = {s: src_names.count(s) for s in sources}
    meta = {
        "stamp": stamp,
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "sources": sources,
        "n_per_source": counts,
        "n_rows": len(src_names),
        "file_order": True,
        "last_token": last_token,
        "top_k": top_k,
        "snippet": int(args.snippet),
        "note": (
            "Cross-dataset cosine only (wm_code vs wm_os). Same-source pairs dropped. "
            "Samples are the first complete turns in each JSONL, file order. "
            "JSON stores clipped a/o only, never h."
        ),
    }
    left_payload: dict[str, Any] = {
        **meta,
        "side": "left",
        "vector": "[z_t; u] = concat(Enc(h), Enc(a))",
        "dim": int(left.size(-1)),
        "n_cross": left_stats["n_cross"],
        "cosine": left_stats["cosine"],
        "top": left_recs,
    }
    right_payload: dict[str, Any] = {
        **meta,
        "side": "right",
        "vector": "Enc(h,a,o)",
        "dim": int(right.size(-1)),
        "n_cross": right_stats["n_cross"],
        "cosine": right_stats["cosine"],
        "top": right_recs,
    }

    def _write(payload: dict[str, Any], stem: str) -> Path:
        dest = out_dir / f"{stem}-{stamp}.json"
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        dest.write_text(text, encoding="utf-8")
        (out_dir / f"{stem}-latest.json").write_text(text, encoding="utf-8")
        return dest

    left_path = _write(left_payload, "left_zt_u")
    right_path = _write(right_payload, "right_hao")
    rank_log(
        f"left dim={left.size(-1)} cross={left_stats['n_cross']} "
        f"med={_cell(left_stats['cosine'], 'median')} "
        f"p90={_cell(left_stats['cosine'], 'p90')}"
    )
    rank_log(
        f"right dim={right.size(-1)} cross={right_stats['n_cross']} "
        f"med={_cell(right_stats['cosine'], 'median')} "
        f"p90={_cell(right_stats['cosine'], 'p90')}"
    )
    preview_n = int(args.print_pairs)
    preview_snip = min(80, int(args.snippet))
    left_prev = _preview_top("left [z_t;u]", left_recs, preview_n, preview_snip)
    right_prev = _preview_top("right Enc(h,a,o)", right_recs, preview_n, preview_snip)
    if left_prev:
        rank_log(left_prev)
    if right_prev:
        rank_log(right_prev)
    rank_log(f"wrote {left_path}")
    rank_log(f"wrote {right_path}")


if __name__ == "__main__":
    main()
