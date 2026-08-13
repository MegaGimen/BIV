#!/usr/bin/env python3
"""Inspect mix / prep / final train samples (readable).

Stages:
  raw   — mix JSONL messages (pre-trim)
  prep  — train_prep_mix HF cache (struct-right trimmed messages)
  final — what Muse TRL actually trains on: normalize → chat_template
          (+ assistant_only loss spans when tokenizer is available)

Examples:
  # One sample per source in final train format (default for --final)
  python scripts/debug.py --final --run-root outputs/trl_cache/.../train_runs/ml8192_...
  python scripts/debug.py --final --max-length 8192

  # Single source
  python scripts/debug.py --final --source wm_code --index 0

  # Legacy raw JSONL view
  python scripts/debug.py --stage raw --source wm_os
  python scripts/debug.py --source anti_forget --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_MIX = ROOT / "data" / "processed" / "mix_v2"
FALLBACK_MIX = ROOT / "data" / "processed" / "mix_v1"
DEFAULT_CONFIG = ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"
SOURCES = ("wm_code", "wm_os", "anti_forget")


def _resolve(p: Path | str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def _default_mix() -> Path:
    if DEFAULT_MIX.is_dir():
        return DEFAULT_MIX
    return FALLBACK_MIX


def _load_muse_mod():
    path = ROOT / "scripts" / "train_muse_trl.py"
    spec = importlib.util.spec_from_file_location("biv_train_muse_trl", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_nth_jsonl(path: Path, index: int) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as f:
        n = -1
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            n += 1
            if n < index:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON at row {n} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Row {n} is not an object")
            return obj
    raise SystemExit(f"Only found {n + 1} rows in {path}; --index {index} out of range")


def _fmt_block(role: str, content: str | None, extra: str | None = None) -> str:
    bar = "=" * 72
    head = f"{bar}\n[{role}]"
    if extra:
        head += f"  {extra}"
    body = content if content is not None else ""
    if isinstance(body, str) and body.strip()[:1] in "{[":
        try:
            parsed = json.loads(body)
            body = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return f"{head}\n{body}\n"


def _print_messages(messages: list, *, show_tools: bool) -> None:
    if not messages:
        print("(empty messages)", flush=True)
        return
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            print(f"--- msg[{i}] (non-dict): {m!r}\n", flush=True)
            continue
        role = str(m.get("role", "?"))
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        extra_bits: list[str] = []
        tool_calls = m.get("tool_calls")
        if show_tools and tool_calls:
            extra_bits.append(f"tool_calls×{len(tool_calls)}")
        tid = m.get("tool_call_id")
        if tid:
            extra_bits.append(f"tool_call_id={tid}")
        name = m.get("name")
        if name and role == "tool":
            extra_bits.append(f"name={name}")
        rc = m.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            extra_bits.append("reasoning_content")

        print(
            _fmt_block(role, content, " | ".join(extra_bits) if extra_bits else None),
            flush=True,
        )

        if isinstance(rc, str) and rc.strip():
            print("- reasoning_content (→ Muse assistant to=self) -", flush=True)
            print(rc, flush=True)
            print("", flush=True)

        if show_tools and tool_calls:
            print("- tool_calls -", flush=True)
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2), flush=True)
            print("", flush=True)


def _find_train_runs(cache_root: Path) -> list[Path]:
    runs: list[Path] = []
    if not cache_root.is_dir():
        return runs
    for p in cache_root.rglob("run_manifest.json"):
        root = p.parent
        if all((root / k).is_dir() for k in SOURCES):
            runs.append(root)
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs


def _resolve_run_root(
    *,
    run_root: Path | None,
    cache_root: Path | None,
    config: dict,
) -> Path | None:
    if run_root is not None:
        root = _resolve(run_root)
        if not root.is_dir():
            raise SystemExit(f"--run-root not found: {root}")
        return root
    roots: list[Path] = []
    if cache_root is not None:
        roots.append(_resolve(cache_root))
    cfg_cache = config.get("cache_root")
    if cfg_cache:
        roots.append(_resolve(cfg_cache))
    roots.append(ROOT / "outputs" / "trl_cache")
    seen: set[Path] = set()
    for cr in roots:
        cr = cr.resolve() if cr.exists() else cr
        if cr in seen:
            continue
        seen.add(cr)
        found = _find_train_runs(cr)
        if found:
            print(f"[debug] auto run-root → {found[0]}", flush=True)
            return found[0]
    return None


def _load_prep_row(run_root: Path, source: str, index: int) -> tuple[dict, Path]:
    from datasets import load_from_disk

    path = run_root / source
    if not path.is_dir():
        raise SystemExit(f"Missing prep dataset: {path}")
    ds = load_from_disk(str(path))
    if index < 0 or index >= len(ds):
        raise SystemExit(f"{source}: index {index} out of range (n={len(ds)})")
    row = ds[index]
    if not isinstance(row, dict):
        row = dict(row)
    return row, path


def _load_raw_row(mix: Path, source: str, split: str, index: int) -> tuple[dict, Path]:
    path = mix / source / f"{split}.jsonl"
    return _load_nth_jsonl(path, index), path


def _load_tokenizer(model_path: str, muse_mod):
    return muse_mod._load_tokenizer_only(model_path)


def _render_final(
    messages: list,
    *,
    tokenizer,
    muse_mod,
    max_length: int,
    assistant_only: bool,
    truncation_mode: str,
) -> dict[str, Any]:
    from biv_wm.adapters.normalize import messages_for_chat_template

    norm = messages_for_chat_template(messages)
    rendered = tokenizer.apply_chat_template(
        norm,
        tokenize=False,
        add_generation_prompt=False,
    )
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "add_generation_prompt": False,
    }
    if assistant_only:
        kwargs["return_assistant_tokens_mask"] = True
    out = tokenizer.apply_chat_template(norm, **kwargs)
    ids = out["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = list(ids)
    if assistant_only:
        mask = out.get("assistant_masks")
        if mask is None:
            mask = out.get("assistant_mask")
        if mask is not None and mask and isinstance(mask[0], (list, tuple)):
            mask = mask[0]
        if mask is None or len(mask) != len(ids):
            labels = list(ids)
        else:
            labels = [tid if bool(mask[i]) else -100 for i, tid in enumerate(ids)]
    else:
        labels = list(ids)

    n_full = len(ids)
    if truncation_mode == "keep_end":
        ids_t = ids[-max_length:]
        labels_t = labels[-max_length:]
    else:
        ids_t = ids[:max_length]
        labels_t = labels[:max_length]

    return {
        "normalized_messages": norm,
        "rendered": rendered if isinstance(rendered, str) else str(rendered),
        "input_ids": ids_t,
        "labels": labels_t,
        "n_tokens_full": n_full,
        "n_tokens": len(ids_t),
        "n_loss": sum(1 for x in labels_t if x != -100),
        "truncated": n_full > max_length,
    }


def _decode_loss_spans(tokenizer, input_ids: list[int], labels: list[int]) -> list[tuple[str, str]]:
    """Return contiguous (kind, text) spans: kind in {context, loss}."""
    spans: list[tuple[str, str]] = []
    cur_kind: str | None = None
    buf: list[int] = []

    def flush() -> None:
        nonlocal cur_kind, buf
        if cur_kind is None or not buf:
            cur_kind = None
            buf = []
            return
        text = tokenizer.decode(buf, skip_special_tokens=False)
        spans.append((cur_kind, text))
        cur_kind = None
        buf = []

    for tid, lab in zip(input_ids, labels):
        kind = "loss" if lab != -100 else "context"
        if kind != cur_kind:
            flush()
            cur_kind = kind
        buf.append(int(tid))
    flush()
    return spans


def _print_banner(title: str) -> None:
    print("\n" + "#" * 72, flush=True)
    print(f"# {title}", flush=True)
    print("#" * 72 + "\n", flush=True)


def _print_final_sample(
    *,
    source: str,
    path: Path,
    index: int,
    messages: list,
    tokenizer,
    muse_mod,
    max_length: int,
    assistant_only: bool,
    truncation_mode: str,
    show_messages: bool,
    max_chars: int,
) -> None:
    _print_banner(f"SOURCE={source}  index={index}")
    print(f"path:    {path}", flush=True)
    print(f"turns:   {len(messages)} (raw/prep messages before normalize)", flush=True)

    info = _render_final(
        messages,
        tokenizer=tokenizer,
        muse_mod=muse_mod,
        max_length=max_length,
        assistant_only=assistant_only,
        truncation_mode=truncation_mode,
    )
    print(
        f"tokens:  full={info['n_tokens_full']}  after_trunc={info['n_tokens']}  "
        f"loss={info['n_loss']}  truncated={info['truncated']}  "
        f"max_length={max_length}  trunc_mode={truncation_mode}  "
        f"assistant_only={assistant_only}",
        flush=True,
    )
    print("", flush=True)

    if show_messages:
        print("--- normalized messages (Muse-ready; tool_calls kept structured) ---", flush=True)
        _print_messages(info["normalized_messages"], show_tools=False)
        print("", flush=True)

    rendered = info["rendered"]
    if max_chars > 0 and len(rendered) > max_chars:
        rendered_show = rendered[:max_chars] + f"\n… [truncated rendered text @ {max_chars} chars]"
    else:
        rendered_show = rendered
    print("=" * 72, flush=True)
    print("[chat_template rendered — full string model tokenizes]", flush=True)
    print("=" * 72, flush=True)
    print(rendered_show, flush=True)
    print("", flush=True)

    print("=" * 72, flush=True)
    print("[loss spans after tokenize + assistant_only mask + max_length]", flush=True)
    print("=" * 72, flush=True)
    spans = _decode_loss_spans(tokenizer, info["input_ids"], info["labels"])
    for kind, text in spans:
        tag = "LOSS (trained)" if kind == "loss" else "CONTEXT (labels=-100)"
        body = text
        if max_chars > 0 and len(body) > max_chars:
            body = body[:max_chars] + f"\n… [truncated span @ {max_chars} chars]"
        print(f"\n<<< {tag} >>>\n{body}", flush=True)
    print("", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--final",
        action="store_true",
        help="shortcut for --stage final; dump all three sources unless --source set",
    )
    p.add_argument(
        "--stage",
        choices=("raw", "prep", "final"),
        default=None,
        help="raw=JSONL, prep=train_prep_mix HF, final=normalize+chat_template+loss",
    )
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument(
        "--source",
        choices=(*SOURCES, "all"),
        default=None,
        help="one source, or all (default: all for --final/prep, wm_code for raw)",
    )
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument("--index", type=int, default=0, help="0-based row index (per source)")
    p.add_argument("--json", action="store_true", help="also print raw/prep row JSON")
    p.add_argument("--no-tool-details", action="store_true")
    p.add_argument("--path", type=Path, default=None, help="explicit JSONL (raw stage only)")
    p.add_argument("--run-root", type=Path, default=None, help="train_prep_mix run dir")
    p.add_argument("--cache-root", type=Path, default=None, help="search train_runs under here")
    p.add_argument(
        "--from-mix",
        action="store_true",
        help="final: use mix JSONL only (ignore stale train_runs prep cache)",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", type=str, default=None, help="tokenizer/model path override")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="truncate printed rendered/span text (0=unlimited)",
    )
    p.add_argument(
        "--no-messages",
        action="store_true",
        help="final: skip normalized messages dump (only rendered + loss spans)",
    )
    args = p.parse_args()

    stage = args.stage
    if args.final:
        stage = "final"
    if stage is None:
        stage = "raw"

    if args.source is None:
        sources: tuple[str, ...] = SOURCES if stage in {"final", "prep"} else ("wm_code",)
    elif args.source == "all":
        sources = SOURCES
    else:
        sources = (args.source,)

    cfg: dict = {}
    cfg_path = _resolve(args.config)
    if cfg_path.is_file():
        import yaml

        with cfg_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            cfg = loaded
    train_cfg = cfg.get("train") or {}
    max_length = int(args.max_length or train_cfg.get("max_length") or 8192)
    # trainmodel always passes --max-length; yaml may not have train.max_length
    if args.max_length is None and "max_length" not in train_cfg:
        # Prefer run_manifest if present
        pass
    assistant_only = bool(train_cfg.get("assistant_only_loss", True))
    truncation_mode = str(train_cfg.get("truncation_mode") or "keep_start")

    muse_mod = None
    tokenizer = None
    if stage == "final":
        muse_mod = _load_muse_mod()
        from biv_wm.model_store import resolve_model_for_train

        model_path = args.model or resolve_model_for_train(cfg, root=ROOT)
        print(f"[debug] model/tokenizer: {model_path}", flush=True)
        tokenizer = _load_tokenizer(model_path, muse_mod)

    run_root = None
    if stage in {"prep", "final"} and not getattr(args, "from_mix", False):
        run_root = _resolve_run_root(
            run_root=args.run_root,
            cache_root=args.cache_root,
            config=cfg,
        )
        if run_root is None and stage == "prep":
            raise SystemExit(
                "No train_prep_mix run found. Pass --run-root "
                "…/train_runs/<run_id> (contains wm_code/wm_os/anti_forget)."
            )
        if run_root is not None:
            man = run_root / "run_manifest.json"
            if man.is_file():
                try:
                    meta = json.loads(man.read_text(encoding="utf-8"))
                    if args.max_length is None and meta.get("max_length"):
                        max_length = int(meta["max_length"])
                    print(
                        f"[debug] run_manifest max_length={meta.get('max_length')} "
                        f"choice={meta.get('choice')} targets={meta.get('targets')}",
                        flush=True,
                    )
                except json.JSONDecodeError:
                    pass
    elif getattr(args, "from_mix", False):
        print("[debug] --from-mix: skipping train_runs prep cache", flush=True)
    mix = _resolve(args.mix_dir) if args.mix_dir else _default_mix()

    for source in sources:
        if args.path and len(sources) > 1:
            raise SystemExit("--path only works with a single --source")

        if stage == "raw" or (stage == "final" and run_root is None):
            if args.path:
                path = _resolve(args.path)
                row = _load_nth_jsonl(path, args.index)
            else:
                if stage == "final":
                    print(
                        "[debug] WARNING: no prep run-root; using raw mix JSONL "
                        "(not struct-right trimmed). Prefer --run-root on the train host.",
                        flush=True,
                    )
                row, path = _load_raw_row(mix, source, args.split, args.index)
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise SystemExit(f"{source}: no messages list (keys={list(row.keys())})")

            if stage == "raw":
                _print_banner(f"SOURCE={source}  stage=raw  index={args.index}")
                print(f"file:    {path}", flush=True)
                meta = {
                    k: row[k]
                    for k in ("source", "n_turns", "instance_id", "trajectory_id")
                    if k in row
                }
                if meta:
                    print(f"meta:    {json.dumps(meta, ensure_ascii=False)}", flush=True)
                print(f"turns:   {len(messages)} messages", flush=True)
                print("", flush=True)
                _print_messages(messages, show_tools=not args.no_tool_details)
                if args.json:
                    print("=" * 72, flush=True)
                    print("[raw JSON]", flush=True)
                    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
                continue

            # final from raw
            assert muse_mod is not None and tokenizer is not None
            _print_final_sample(
                source=source,
                path=path,
                index=args.index,
                messages=messages,
                tokenizer=tokenizer,
                muse_mod=muse_mod,
                max_length=max_length,
                assistant_only=assistant_only,
                truncation_mode=truncation_mode,
                show_messages=not args.no_messages,
                max_chars=args.max_chars,
            )
            if args.json:
                print(json.dumps({"messages": messages}, ensure_ascii=False, indent=2), flush=True)
            continue

        # prep or final-from-prep
        assert run_root is not None
        row, path = _load_prep_row(run_root, source, args.index)
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise SystemExit(f"{source}: prep row has no messages (keys={list(row.keys())})")

        if stage == "prep":
            _print_banner(f"SOURCE={source}  stage=prep  index={args.index}")
            print(f"path:    {path}", flush=True)
            print(f"turns:   {len(messages)} messages", flush=True)
            print("", flush=True)
            _print_messages(messages, show_tools=not args.no_tool_details)
            if args.json:
                # messages may contain non-JSON-serializable; best-effort
                try:
                    print(json.dumps(row, ensure_ascii=False, indent=2, default=str), flush=True)
                except TypeError:
                    print(repr(row), flush=True)
            continue

        assert muse_mod is not None and tokenizer is not None
        _print_final_sample(
            source=source,
            path=path,
            index=args.index,
            messages=messages,
            tokenizer=tokenizer,
            muse_mod=muse_mod,
            max_length=max_length,
            assistant_only=assistant_only,
            truncation_mode=truncation_mode,
            show_messages=not args.no_messages,
            max_chars=args.max_chars,
        )
        if args.json:
            print(
                json.dumps({"messages": messages}, ensure_ascii=False, indent=2, default=str),
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        sys.exit(130)
