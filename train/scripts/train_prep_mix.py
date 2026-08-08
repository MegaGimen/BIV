#!/usr/bin/env python3
"""Train-time structure-preserving right trunc + optional 1:1:0.35 rebalance.

ms-swift ``cached_dataset`` keeps ``messages`` + ``lengths`` (not token ``labels``).

Fast path (no per-row chat_template — that was ~hours):
  - scan: only read stored ``lengths`` + check an assistant exists
  - build: trim trailing messages with a char-budget heuristic so the prefix
    ends on a complete ``assistant``; write scalar ``lengths`` ≈ budget

1) Print survivors / delete-ref (seconds).
2) Interactive choice: 1=as-is / 2=1:1:0.35 / 3=abort
3) Sample then trim only selected rows; emit shell exports via --write-env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "swift" / "coder_next_qlora.yaml"
SOURCE_KEYS = ("wm_code", "wm_os", "anti_forget")
MANIFEST_NAME = "tokenize_manifest.json"
REBALANCE_RATIOS = {"wm_code": 1.0, "wm_os": 1.0, "anti_forget": 0.35}
LABEL_IGNORE = -100


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable if iterable is not None else None
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    if iterable is None:
        return tqdm(**kwargs)
    return tqdm(iterable, **kwargs)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML: {path}")
    return data


def _resolve(raw: str | Path) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _find_manifest(cache_root: Path, tag: str | None) -> Path:
    if tag:
        p = cache_root / tag / MANIFEST_NAME
        if not p.is_file():
            raise SystemExit(f"Manifest not found: {p}")
        return p
    latest = cache_root / "LATEST"
    if latest.is_file():
        t = latest.read_text(encoding="utf-8").strip()
        p = cache_root / t / MANIFEST_NAME
        if p.is_file():
            return p
    cands = sorted(cache_root.glob(f"*/{MANIFEST_NAME}"), key=lambda x: x.stat().st_mtime)
    if not cands:
        raise SystemExit(
            f"No {MANIFEST_NAME} under {cache_root}. Run:\n  python scripts/tokenize_data.py"
        )
    return cands[-1]


def _load_dataset(path: Path):
    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise SystemExit(f"datasets required: pip install datasets\n({e})") from e
    return load_from_disk(str(path))


def _length_column(ds) -> str:
    for name in ("lengths", "length"):
        if name in ds.column_names:
            return name
    raise SystemExit(
        f"Dataset missing length column (got {ds.column_names}). "
        "Re-run tokenize_data.py with ms-swift>=3.11."
    )


def _as_int_length(v: Any) -> int | None:
    """Normalize ms-swift length field (scalar or list) to one int."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            v = v.tolist()
        except Exception:
            pass
    if isinstance(v, list):
        if not v:
            return None
        # common: [seq_len]
        if len(v) == 1 and isinstance(v[0], (int, float)):
            return int(v[0])
        # packing / multi-segment: sum segment lengths
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return int(sum(v))
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_seq(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if hasattr(v, "tolist"):
        return list(v.tolist())
    return list(v)


# Empirically safe-ish for Qwen code/JSON chat (slightly conservative).
_CHARS_PER_TOKEN = 3.0
_MSG_OVERHEAD_CHARS = 64


def _msg_char_cost(m: dict) -> int:
    n = _MSG_OVERHEAD_CHARS
    c = m.get("content")
    if isinstance(c, str):
        n += len(c)
    elif c is not None:
        n += len(json.dumps(c, ensure_ascii=False))
    tc = m.get("tool_calls")
    if tc is not None:
        n += len(json.dumps(tc, ensure_ascii=False))
    return n


def approx_token_len_messages(messages: list) -> int:
    if not messages:
        return 0
    chars = 0
    for m in messages:
        if isinstance(m, dict):
            chars += _msg_char_cost(m)
    return max(1, int(chars / _CHARS_PER_TOKEN))


def _has_assistant(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    return any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)


def struct_right_cut_labels(
    labels: Any, max_length: int, *, ignore_index: int = LABEL_IGNORE
) -> int | None:
    """Legacy path if tokenized labels exist."""
    labs = _as_seq(labels)
    n = len(labs)
    lim = min(n, int(max_length))
    if lim <= 0:
        return None

    def is_complete_assistant_end(c: int) -> bool:
        if c <= 0 or c > n:
            return False
        if int(labs[c - 1]) == ignore_index:
            return False
        if c == n:
            return True
        return int(labs[c]) == ignore_index

    for c in range(lim, 0, -1):
        if is_complete_assistant_end(c):
            return c
    return None


def struct_right_cut_messages(
    messages: Any, max_length: int
) -> tuple[list[dict] | None, int]:
    """Keep longest prefix ending on ``assistant`` under char-budget ≈ max_length.

    No tokenizer calls — fast enough for full-corpus prep.
    Returns (kept_messages_or_None, approx_token_length).
    """
    if not isinstance(messages, list) or not messages:
        return None, 0
    ends = [
        i + 1
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    if not ends:
        return None, 0

    lo, hi = 0, len(ends) - 1
    best_msgs: list[dict] | None = None
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cut = ends[mid]
        prefix = messages[:cut]
        n = approx_token_len_messages(prefix)
        if n <= max_length:
            best_msgs = prefix
            best_len = n
            lo = mid + 1
        else:
            hi = mid - 1
    return best_msgs, best_len


def _sample_indices(n_keep: int, n_take: int, seed: int) -> list[int]:
    if n_take >= n_keep:
        return list(range(n_keep))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_keep), n_take))


def _max_fill_targets(kept: dict[str, int]) -> dict[str, int]:
    for k in SOURCE_KEYS:
        if kept[k] <= 0:
            raise SystemExit(
                f"After struct-right, {k} has 0 rows — cannot rebalance. "
                "Raise --max-length."
            )
    ratios = REBALANCE_RATIOS
    s = min(kept[k] / ratios[k] for k in SOURCE_KEYS)
    targets = {k: max(1, int(s * ratios[k])) for k in SOURCE_KEYS}
    n_wm = min(targets["wm_code"], targets["wm_os"], kept["wm_code"], kept["wm_os"])
    targets["wm_code"] = n_wm
    targets["wm_os"] = n_wm
    targets["anti_forget"] = min(
        kept["anti_forget"],
        max(1, int(round(n_wm * ratios["anti_forget"] / ratios["wm_code"]))),
    )
    for k in SOURCE_KEYS:
        if targets[k] > kept[k]:
            raise SystemExit(f"Internal error: target {k}={targets[k]} > kept={kept[k]}")
    if targets["wm_code"] != targets["wm_os"]:
        raise SystemExit("Internal error: code/os not equal after flatten")
    return targets


def _prompt_choice(preselected: int | None) -> int:
    if preselected is not None:
        if preselected not in {1, 2, 3}:
            raise SystemExit("--choice must be 1, 2, or 3")
        print(f"Choice (non-interactive): {preselected}", flush=True)
        return preselected
    print("", flush=True)
    print("选择下一步：", flush=True)
    print("  1 = 不按 1:1 拉平 code/os，结构右截断后原样开训", flush=True)
    print("  2 = 在存活池上按 code:os:anti = 1:1:0.35 再 sample，然后开训", flush=True)
    print("  3 = 中断", flush=True)
    while True:
        raw = input("请输入 1 / 2 / 3: ").strip()
        if raw in {"1", "2", "3"}:
            return int(raw)
        print("无效输入，请重新输入 1、2 或 3。", flush=True)


def _write_env(path: Path, exports: dict[str, str]) -> None:
    lines = [f"export {k}={shlex.quote(v)}" for k, v in exports.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_fast(
    ds,
    *,
    max_length: int,
    length_col: str,
    desc: str,
) -> dict[str, int]:
    """Seconds-scale scan: lengths + has-assistant only (no chat_template)."""
    cols = set(ds.column_names)
    has_messages = "messages" in cols
    if not has_messages and "labels" not in cols:
        raise SystemExit(f"{desc}: need 'messages' or 'labels' (got {ds.column_names})")

    total = len(ds)
    survivors = 0
    dropped = 0
    need_trunc = 0
    delete_keep = 0
    lengths = ds[length_col]
    messages_all = ds["messages"] if has_messages else None

    bar = _tqdm(total=total, unit="rows", desc=desc)
    for i in range(total):
        L = _as_int_length(lengths[i])
        if L is not None and L <= max_length:
            delete_keep += 1
        else:
            need_trunc += 1

        if has_messages:
            ok = _has_assistant(messages_all[i])
        else:
            # labels path: treat any row as potentially truncatable; drop decided at build
            ok = True
        if ok:
            survivors += 1
        else:
            dropped += 1
        if bar is not None:
            bar.update(1)
            if i % 256 == 0:
                bar.set_postfix(ok=survivors, drop=dropped, refresh=False)
    if bar is not None:
        bar.close()

    return {
        "total": total,
        "survivors": survivors,
        "dropped": dropped,
        "truncated": need_trunc,  # rows with L>max (will char-trim at build)
        "delete_keep": delete_keep,
        "backend": "messages+lengths" if has_messages else "labels",
    }


def _build_truncated_sample(
    ds,
    *,
    max_length: int,
    length_col: str,
    n_take: int,
    seed: int,
    name: str,
    num_proc: int,
):
    """Sample first, then char-budget struct-right trim (fast)."""
    cols = set(ds.column_names)
    has_messages = "messages" in cols
    n_all = len(ds)
    if n_take > n_all:
        raise SystemExit(f"{name}: need {n_take} but dataset only has {n_all}")
    # Oversample a bit so trim drops (no assistant / first turn too long) can be topped up.
    n_pick = min(n_all, max(n_take, int(n_take * 1.02) + 8))
    idxs = _sample_indices(n_all, n_pick, seed)
    subset = ds.select(idxs)
    print(f"  {name}: selected {len(subset):,} rows → char-budget trim …", flush=True)

    def _map(ex):
        if has_messages:
            msgs = ex["messages"]
            L = _as_int_length(ex[length_col])
            if L is not None and L <= max_length and _has_assistant(msgs):
                out = dict(ex)
                out[length_col] = int(L)
                if "lengths" in out:
                    out["lengths"] = int(L)
                out["_biv_keep"] = 1
                return out
            kept, nlen = struct_right_cut_messages(msgs, max_length)
            if kept is None:
                out = dict(ex)
                out["_biv_keep"] = 0
                return out
            out = dict(ex)
            out["messages"] = kept
            out[length_col] = int(nlen)
            if "lengths" in out:
                out["lengths"] = int(nlen)
            if "length" in out:
                out["length"] = int(nlen)
            out["_biv_keep"] = 1
            return out
        cut = struct_right_cut_labels(ex["labels"], max_length)
        if cut is None:
            out = dict(ex)
            out["_biv_keep"] = 0
            return out
        labs = _as_seq(ex["labels"])
        n = len(labs)
        out = dict(ex)
        out["labels"] = labs[:cut]
        for k, v in ex.items():
            if k in {"labels", "_biv_keep", "length", "lengths"}:
                continue
            if isinstance(v, list) and len(v) == n and v and isinstance(v[0], (int, float)):
                out[k] = v[:cut]
        out[length_col] = int(cut)
        out["_biv_keep"] = 1
        return out

    mapped = subset.map(_map, num_proc=max(1, min(num_proc, 4)), desc=f"trim {name}")
    kept_ds = mapped.filter(
        lambda x: int(x["_biv_keep"]) == 1,
        num_proc=max(1, min(num_proc, 4)),
        desc=f"keep {name}",
    )
    if "_biv_keep" in kept_ds.column_names:
        kept_ds = kept_ds.remove_columns(["_biv_keep"])
    if len(kept_ds) == 0:
        raise SystemExit(f"{name}: all selected rows failed struct-right trim")
    if len(kept_ds) > n_take:
        kept_ds = kept_ds.select(range(n_take))
    elif len(kept_ds) < n_take:
        print(
            f"  WARNING: {name} trim kept {len(kept_ds):,}/{n_take:,} "
            f"(some rows had no assistant or first turn too long)",
            flush=True,
        )
    return kept_ds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--choice", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--write-env", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.max_length <= 0:
        raise SystemExit("--max-length must be > 0")

    cfg = _load_yaml(_resolve(args.config))
    cache_root = _resolve(cfg.get("cache_root", "outputs/swift_cache/coder_next_mix_v2"))
    manifest_path = _find_manifest(cache_root, args.tag)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = manifest.get("cached_train") or {}
    tag = str(manifest.get("tag") or manifest_path.parent.name)
    model = str(manifest.get("model") or cfg.get("model") or "Qwen/Qwen3-Coder-Next")
    train_cfg = cfg.get("train") or {}

    print(f"Manifest:    {manifest_path}", flush=True)
    print(f"Tag:         {tag}", flush=True)
    print(f"Model:       {model}", flush=True)
    print(
        f"max_length:  {args.max_length}  "
        "(fast scan on lengths; build trims messages by char-budget)",
        flush=True,
    )
    print("", flush=True)
    print("=== structure-preserving right trunc @ max_length ===", flush=True)

    datasets_raw: dict[str, Any] = {}
    length_cols: dict[str, str] = {}
    kept_counts: dict[str, int] = {}
    scan_stats: dict[str, dict[str, int]] = {}

    for name in SOURCE_KEYS:
        rel = cached.get(name)
        if not rel:
            raise SystemExit(f"manifest missing cached_train.{name}")
        path = _resolve(rel)
        if not path.exists():
            raise SystemExit(f"Missing cached dataset: {path}")
        ds = _load_dataset(path)
        length_col = _length_column(ds)
        print(f"  {name}: columns={ds.column_names}", flush=True)
        st = _scan_fast(
            ds,
            max_length=args.max_length,
            length_col=length_col,
            desc=f"scan {name}",
        )
        datasets_raw[name] = ds
        length_cols[name] = length_col
        scan_stats[name] = st
        kept_counts[name] = st["survivors"]
        tot = st["total"]
        print(
            f"  {name:12s}: survivors {st['survivors']:7,} / {tot:,} "
            f"(need-trim {st['truncated']:,}, drop {st['dropped']:,}); "
            f"delete-ref keep {st['delete_keep']:,}  [{st['backend']}]",
            flush=True,
        )

    total_keep = sum(kept_counts.values())
    total_all = sum(scan_stats[k]["total"] for k in SOURCE_KEYS)
    print(f"  {'TOTAL':12s}: survivors {total_keep:7,} / {total_all:,}", flush=True)
    print(
        "  note: scan 很快（只看 lengths + 是否有 assistant）；"
        "选 1/2 后才按轮裁剪 messages（字符预算，不跑 chat_template）",
        flush=True,
    )

    if total_keep == 0:
        raise SystemExit("No rows survive struct-right at this max_length. Raise --max-length.")

    preview2: dict[str, int] | None = None
    if all(kept_counts[k] > 0 for k in SOURCE_KEYS):
        preview2 = _max_fill_targets(kept_counts)
        print("", flush=True)
        print("若选 2（1:1:0.35 max_fill）预计 sample：", flush=True)
        tsum = sum(preview2.values())
        for k in SOURCE_KEYS:
            share = 100.0 * preview2[k] / tsum if tsum else 0.0
            print(f"  {k:12s}: {preview2[k]:7,} ({share:5.1f}%)", flush=True)
        print(f"  {'TOTAL':12s}: {tsum:7,}", flush=True)
    else:
        print("\nWARNING: 至少一个源存活为 0，选项 2 不可用。", flush=True)

    choice = _prompt_choice(args.choice)
    if choice == 3:
        print("已中断。", flush=True)
        raise SystemExit(3)
    if choice == 2 and preview2 is None:
        raise SystemExit("选项 2 需要三个源存活后都 >0。")

    if choice == 1:
        targets = dict(kept_counts)
        mode = "struct_right_as_is"
    else:
        targets = preview2  # type: ignore[assignment]
        mode = "struct_right_rebalance_1_1_0.35"

    if any(int(targets[k]) <= 0 for k in SOURCE_KEYS):
        raise SystemExit("At least one source has 0 rows after struct-right/rebalance.")

    blob = (
        f"tag={tag}|ml={args.max_length}|mode={mode}|seed={args.seed}|"
        f"t={targets['wm_code']}:{targets['wm_os']}:{targets['anti_forget']}|backend=messages"
    )
    run_id = (
        f"ml{args.max_length}_{mode}"
        f"_n{targets['wm_code']}_{targets['wm_os']}_{targets['anti_forget']}_"
        f"{hashlib.sha1(blob.encode()).hexdigest()[:8]}"
    )
    run_root = manifest_path.parent / "train_runs" / run_id
    out_paths: dict[str, Path] = {k: run_root / k for k in SOURCE_KEYS}

    reuse = (
        not args.force
        and all((out_paths[k] / "state.json").is_file() for k in SOURCE_KEYS)
        and (run_root / "run_manifest.json").is_file()
    )

    if reuse:
        print(f"\nReusing prepared run cache: {run_root}", flush=True)
    else:
        if run_root.exists() and args.force:
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"\nBuilding train mix under {run_root} …", flush=True)
        for i, name in enumerate(SOURCE_KEYS):
            final = _build_truncated_sample(
                datasets_raw[name],
                max_length=args.max_length,
                length_col=length_cols[name],
                n_take=targets[name],
                seed=args.seed + i * 17,
                name=name,
                num_proc=args.num_proc,
            )
            out = out_paths[name]
            if out.exists():
                shutil.rmtree(out)
            final.save_to_disk(str(out))
            print(f"  wrote {out} ({len(final):,} rows)", flush=True)

        run_manifest = {
            "tag": tag,
            "parent_manifest": str(manifest_path.relative_to(ROOT)),
            "max_length": args.max_length,
            "mode": mode,
            "choice": choice,
            "seed": args.seed,
            "ratios": REBALANCE_RATIOS if choice == 2 else None,
            "scan": scan_stats,
            "kept_after_struct_right": kept_counts,
            "targets": targets,
            "cached_train": {k: str(out_paths[k].relative_to(ROOT)) for k in SOURCE_KEYS},
            "truncation_strategy": "delete",
            "prep_truncation": "struct_right_messages_char_budget",
            "chars_per_token": _CHARS_PER_TOKEN,
            "note": (
                "messages trimmed to end on complete assistant via char budget; "
                "train uses delete as safety net for tokenizer mismatch."
            ),
        }
        (run_root / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2), encoding="utf-8"
        )

    cached_list = " ".join(str(out_paths[k]) for k in SOURCE_KEYS)
    out_dir = train_cfg.get("output_dir", "outputs/swift_coder_next_wm_mix")
    out_dir = f"{out_dir}_ml{args.max_length}_c{choice}"

    exports = {
        "TAG": tag,
        "RUN_ID": run_id,
        "RUN_ROOT": str(run_root),
        "MANIFEST": str(run_root / "run_manifest.json"),
        "MODEL": str(model),
        "MAX_LENGTH": str(args.max_length),
        "TRUNC": "delete",
        "CACHED_DATASETS": cached_list,
        "OUT_DIR": str(out_dir),
        "TRAIN_CHOICE": str(choice),
        "LR": str(train_cfg.get("learning_rate", 1e-4)),
        "EPOCHS": str(train_cfg.get("num_epochs", 2)),
        "LORA_RANK": str(train_cfg.get("lora_rank", 16)),
        "LORA_ALPHA": str(train_cfg.get("lora_alpha", 16)),
        "BS": str(train_cfg.get("per_device_train_batch_size", 1)),
        "GAS": str(train_cfg.get("gradient_accumulation_steps", 8)),
        "DEEPSPEED": str(train_cfg.get("deepspeed", "zero2")),
        "DTYPE": str(train_cfg.get("torch_dtype", "bfloat16")),
        "WARMUP": str(train_cfg.get("warmup_ratio", 0.03)),
        "LOG_STEPS": str(train_cfg.get("logging_steps", 10)),
        "SAVE_STEPS": str(train_cfg.get("save_steps", 200)),
        "SAVE_LIMIT": str(train_cfg.get("save_total_limit", 3)),
        "TARGET_MODULES": " ".join(
            train_cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    }
    env_path = Path(args.write_env)
    if not env_path.is_absolute():
        env_path = ROOT / env_path
    env_path.parent.mkdir(parents=True, exist_ok=True)
    _write_env(env_path, exports)

    print("", flush=True)
    print(f"Ready: choice={choice} mode={mode}", flush=True)
    print(f"  run: {run_root}", flush=True)
    print(
        "  targets: "
        + ", ".join(f"{k}={targets[k]:,}" for k in SOURCE_KEYS)
        + f"  total={sum(targets.values()):,}",
        flush=True,
    )
    print(f"  env → {env_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。", flush=True)
        sys.exit(3)
