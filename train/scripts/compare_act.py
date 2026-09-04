#!/usr/bin/env python3
"""Compare AgentWorld vs Instruct with ACT module-channel Δa.

ACT 2601.09398 §3.1 / eq. (2) / §4.1: same tokens into both models; a channel
is one output dimension of a trainable module. Average |a_AW − a_Instruct|
over pooled answer tokens, rank every channel together, take top p% as the
ability mask. No residual stream, no layer-cut table.

    python train/scripts/compare_act.py
    python train/scripts/compare_act.py --jsonl train/data/processed/mix_v2/train.jsonl --max-rows 8

Writes under ``train/outputs/compare_act/``:
  summary.txt, report.json, mask.json, channels.jsonl

Merge those mask rows into Instruct::

    python merge/act.py
    python merge/eval.py --act --max-model-len 32768
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_MERGE_DIR = ROOT / "merge"
_SRC = ROOT / "train" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_DIR))

from biv_wm.act import (  # noqa: E402
    DEFAULT_TOP_P,
    add_abs_sum,
    analyze_channels,
    canonical_module_key,
    finalize_running,
    hook_kind,
    token_abs_sum,
)
from biv_wm.arch import language_model, lm_head_module  # noqa: E402
from biv_wm.hao import split_hao  # noqa: E402
from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_CACHE,
    DEFAULT_SOURCE,
    DEFAULT_WORLD,
    resolve_model,
)

DEFAULT_OUT = ROOT / "train" / "outputs" / "compare_act"

DEFAULT_MESSAGES = [
    {"role": "user", "content": "ls"},
    {"role": "assistant", "content": "a.txt"},
    {"role": "user", "content": "rm a.txt"},
    {"role": "assistant", "content": "gone"},
]


def log(msg: str) -> None:
    print(msg, flush=True)


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


def _as_btc(out: Any):
    import torch

    t = out[0] if isinstance(out, (tuple, list)) else out
    if not torch.is_tensor(t):
        return None
    if t.ndim == 4:
        b, s, h, d = t.shape
        t = t.reshape(b, s, h * d)
    if t.ndim != 3:
        return None
    return t


def _as_device(dev):
    if dev is None:
        return None
    if hasattr(dev, "type"):
        return dev
    try:
        import torch

        return torch.device(dev)
    except Exception:
        return dev


def _module_exec_device(mod):
    """Real device for a module under device_map=auto. Never return meta."""
    if mod is None:
        return None
    params = list(getattr(mod, "parameters", lambda: [])())
    for p in params:
        dev = getattr(p, "device", None)
        if dev is not None and getattr(dev, "type", None) != "meta":
            return _as_device(dev)
    hook = getattr(mod, "_hf_hook", None)
    exec_dev = getattr(hook, "execution_device", None) if hook is not None else None
    return _as_device(exec_dev)


def _module_dtype(mod, fallback):
    for p in getattr(mod, "parameters", lambda: [])():
        dt = getattr(p, "dtype", None)
        if dt is not None:
            return dt
    return fallback


def _embed_device(model) -> Any:
    inner = language_model(model)
    emb = None
    if inner is not None:
        emb = getattr(inner, "embed_tokens", None) or getattr(inner, "embed", None)
    if emb is None:
        getter = getattr(model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
    dev = _module_exec_device(emb)
    if dev is None:
        raise RuntimeError("cannot find a non-meta device for input_ids")
    return dev


def _apply_lm_head(head, last_ans):
    """Run lm_head without copying onto a meta (offloaded) tensor."""
    import torch

    dt = _module_dtype(head, last_ans.dtype)
    dev = _module_exec_device(head)
    x = last_ans if dev is None else last_ans.to(device=dev, dtype=dt)
    with torch.inference_mode():
        return head(x)


def _filter_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    allowed = set(inspect.signature(fn).parameters)
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _to_token_ids(tokenizer, raw) -> list[int]:
    """Coerce apply_chat_template / encode output to a flat list of ints.

    Qwen tokenizers on current transformers often ignore tokenize=True and
    return the rendered string. ``list(that_string)`` is characters, which
    later blows up in ``tokenizer.decode``.
    """
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, dict):
        raw = raw.get("input_ids", raw)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
    if isinstance(raw, str):
        return list(tokenizer(raw, truncation=False, add_special_tokens=False)["input_ids"])
    if isinstance(raw, list):
        if not raw:
            return []
        if isinstance(raw[0], list):
            raw = raw[0]
        if raw and isinstance(raw[0], str):
            return list(
                tokenizer("".join(raw), truncation=False, add_special_tokens=False)["input_ids"]
            )
        return [int(x) for x in raw]
    raise TypeError(f"cannot read token ids from {type(raw)}")


def chat_ids(tokenizer, messages: list, *, add_generation_prompt: bool) -> list[int]:
    """Render chat to text, then tokenize — same path as train_jepa.py."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(text, str):
        return list(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
    return _to_token_ids(tokenizer, text)


def encode_prompt(
    tokenizer,
    *,
    text: str | None,
    messages: list[dict[str, str]] | None,
    token_mode: str,
) -> tuple[list[int], list[int], str]:
    """Return (input_ids, answer_positions, note)."""
    if text is not None:
        ids = _to_token_ids(tokenizer, tokenizer.encode(text, add_special_tokens=True))
        if not ids:
            raise ValueError("empty --text")
        if token_mode == "answer":
            raise ValueError(
                "--text has no chat answer span; pass --tokens all (not ACT) "
                "or use the default chat / --jsonl"
            )
        pos = list(range(len(ids)))
        return ids, pos, "raw --text; mean over all tokens (not ACT answer mask)"

    if not messages:
        raise ValueError("need --text or chat messages")
    full = chat_ids(tokenizer, messages, add_generation_prompt=False)
    if not full:
        raise ValueError("empty chat tokenization")
    if token_mode == "all":
        return full, list(range(len(full))), "chat; mean over all tokens (not ACT answer mask)"

    prefix = chat_ids(tokenizer, messages[:-1], add_generation_prompt=True)
    start = len(prefix)
    if start >= len(full) or full[: min(start, len(full))] != prefix[: min(start, len(full))]:
        start = max(0, len(full) - max(1, len(full) // 4))
        note = (
            f"chat; prefix not a prefix of full (start clamped to {start}); "
            "ACT mean over that suffix"
        )
    else:
        note = f"chat; ACT mean over last-assistant tokens [{start}:{len(full)}]"
    return full, list(range(start, len(full))), note


def _language_model_only(model_dir: Path) -> bool:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(cfg.get("language_model_only"))


def _load_model(model_dir: Path, *, dtype, device_map: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }
    # AgentWorld is language_model_only (no ViT). ImageTextToText would
    # randomly init model.visual and print a MISSING dump; CausalLM first.
    loaders = (AutoModelForCausalLM, AutoModelForImageTextToText)
    if not _language_model_only(model_dir):
        loaders = (AutoModelForImageTextToText, AutoModelForCausalLM)
    last = None
    for loader in loaders:
        try:
            return loader.from_pretrained(str(model_dir), **kwargs), loader.__name__
        except TypeError:
            kwargs.pop("dtype", None)
            try:
                return loader.from_pretrained(str(model_dir), **kwargs), loader.__name__
            except Exception as e:
                last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"from_pretrained failed: {last}")


def _free(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def capture_one(
    model,
    input_ids: list[int],
    answer_pos: list[int],
) -> dict[str, Any]:
    """Forward once; CPU float32 *module* outputs at answer positions."""
    import torch

    device = _embed_device(model)
    ids = torch.tensor([input_ids], device=device)
    mask = torch.ones_like(ids)

    captures: dict[str, Any] = {}
    handles = []

    def _answer(t):
        return t[0].detach().float().cpu()[answer_pos]

    def make_hook(name: str):
        def _hook(_mod, _inp, out):
            t = _as_btc(out)
            if t is None:
                return
            captures[name] = _answer(t)

        return _hook

    seen: set[str] = set()
    named = list(model.named_modules())
    named.sort(key=lambda x: (0 if "language_model." in x[0] else 1, x[0]))
    for name, mod in named:
        kind = hook_kind(name)
        if kind is None or kind == "lm_head":
            continue
        canon = canonical_module_key(name)
        if canon in seen:
            continue
        seen.add(canon)
        handles.append(mod.register_forward_hook(make_hook(canon)))

    inner = language_model(model)
    fwd = inner.forward if inner is not None else model.forward
    kwargs = _filter_kwargs(
        fwd,
        {
            "input_ids": ids,
            "attention_mask": mask,
            "output_hidden_states": False,
            "use_cache": False,
            "return_dict": True,
        },
    )
    with torch.inference_mode():
        out = fwd(**kwargs)

    for h in handles:
        h.remove()

    last = getattr(out, "last_hidden_state", None)
    if last is None:
        raise RuntimeError("forward returned no last_hidden_state")
    last_ans = _answer(last)

    head = lm_head_module(model)
    if head is not None:
        logits = _apply_lm_head(head, last_ans)
        captures["lm_head"] = logits.detach().float().cpu()

    return captures


def acc_pair(
    running: dict[str, tuple[list[float], int]],
    left: dict[str, Any],
    right: dict[str, Any],
    skipped: set[str],
) -> None:
    for k in sorted(set(left) & set(right)):
        a, b = left[k], right[k]
        if tuple(a.shape) != tuple(b.shape):
            skipped.add(f"{k}:shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        s, n = token_abs_sum(a, b)
        add_abs_sum(running, k, s, n)
    skipped.update(set(left) ^ set(right))


def _fmt_ccdf(st: dict[str, Any], thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.5, 8.0, 16.0)) -> list[str]:
    c = st.get("ccdf") or {}
    lines = [f"  {'t':>6}  {'P(Δa>t)':>10}"]
    for t in thresholds:
        lines.append(f"  {t:6.2f}  {float(c.get(str(t), 0.0)):10.4f}")
    return lines


def format_summary(report: dict[str, Any]) -> str:
    kinds = ("attn", "ffn", "ln", "embed", "lm_head")
    by_kind = report.get("by_kind") or {}
    g = report.get("global") or {}
    gnh = report.get("global_no_lm_head") or {}
    p = float(report.get("p") or 0.01)

    lines: list[str] = []
    lines.append("compare_act.py — ACT 通道 Δa（按通道排序取 top-p%，不是按层切）")
    lines.append(str(report.get("method") or ""))
    paths = report.get("paths") or {}
    lines.append(f"world     {paths.get('world')}")
    lines.append(f"instruct  {paths.get('instruct')}")
    lines.append(f"prompt    {report.get('prompt_note', '')}")
    lines.append(
        f"n_samples={report.get('n_samples')} n_tokens={report.get('n_tokens')} "
        f"n_answer={report.get('n_answer_tokens')} "
        f"decoded_answer={report.get('decoded_answer')!r}"
    )
    lines.append(
        f"n_channels={report.get('n_channels')}  no_lm_head={report.get('n_channels_no_lm_head')}  "
        f"p={p}  |mask|={report.get('n_mask')}  "
        f"mask_threshold={report.get('mask_threshold')}"
    )
    lines.append("")
    lines.append("全局 CCDF（ACT Figure 1a，全部模块通道混在一起）")
    lines.append(
        f"  all: n={g.get('n_channels')} mean={float(g.get('mean') or 0):.4e} "
        f"p50={float(g.get('p50') or 0):.4e} p99={float(g.get('p99') or 0):.4e} "
        f"max={float(g.get('max') or 0):.4e}"
    )
    lines.extend(_fmt_ccdf(g))
    lines.append(
        f"  no_lm_head: n={gnh.get('n_channels')} mean={float(gnh.get('mean') or 0):.4e} "
        f"p99={float(gnh.get('p99') or 0):.4e}"
    )
    lines.extend(_fmt_ccdf(gnh))

    lines.append("")
    lines.append("按模块种类 CCDF（ACT Figure 1c）")
    lines.append(f"  {'kind':<8} {'n':>8} {'mean':>10} {'p99':>10} {'P(>4.5)':>10}")
    for k in kinds:
        st = by_kind.get(k) or {}
        if int(st.get("n_channels") or 0) == 0:
            continue
        frac = float((st.get("ccdf") or {}).get("4.5", 0.0))
        lines.append(
            f"  {k:<8} {int(st.get('n_channels') or 0):8d} "
            f"{float(st.get('mean') or 0):10.4e} {float(st.get('p99') or 0):10.4e} "
            f"{frac:10.4f}"
        )

    lines.append("")
    lines.append("top-p% 掩码落在各层的通道数（看是否铺开，不是切点）")
    lines.append(f"  {'L':>3} {'kind':<16} {'n_ch':>8} {'n_in_mask':>10} {'p99':>10} {'P(>4.5)':>10}")
    for row in report.get("by_layer") or []:
        n_mask = int(row.get("n_in_mask") or 0)
        if int(row.get("n_channels") or 0) == 0 and n_mask == 0:
            continue
        frac = float((row.get("ccdf") or {}).get("4.5", 0.0))
        lines.append(
            f"  {int(row['layer']):3d} {str(row.get('kind') or '?')[:16]:<16} "
            f"{int(row.get('n_channels') or 0):8d} {n_mask:10d} "
            f"{float(row.get('p99') or 0):10.4e} {frac:10.4f}"
        )

    lines.append("")
    lines.append("各 hooked 模块（按 p99 降序，前 40；全量在 report.json / modules）")
    lines.append(f"  {'p99':>10} {'mean':>10} {'n':>8} {'mask':>6}  key")
    for row in (report.get("modules") or [])[:40]:
        lines.append(
            f"  {float(row.get('p99') or 0):10.4e} {float(row.get('mean') or 0):10.4e} "
            f"{int(row.get('n_channels') or 0):8d} {int(row.get('n_in_mask') or 0):6d}  "
            f"{row.get('key')}"
        )

    lines.append("")
    lines.append(f"ACT 掩码 top {p:.2%}（前 40；全量 mask.json）")
    for row in (report.get("mask") or [])[:40]:
        li = row.get("layer")
        layer_s = f"L{li}" if li is not None else "  "
        lines.append(
            f"  #{int(row.get('rank') or 0):<6d} {float(row.get('delta') or 0):.4e}  "
            f"{str(row.get('kind') or '?'):<8} {layer_s:<4}  "
            f"{row.get('key')}[{row.get('channel')}]"
        )

    lines.append("")
    lines.append(f"去掉 lm_head 后的 top {p:.2%}（前 20；全量 mask.json 的 mask_no_lm_head）")
    for row in (report.get("mask_no_lm_head") or [])[:20]:
        li = row.get("layer")
        layer_s = f"L{li}" if li is not None else "  "
        lines.append(
            f"  #{int(row.get('rank') or 0):<6d} {float(row.get('delta') or 0):.4e}  "
            f"{str(row.get('kind') or '?'):<8} {layer_s:<4}  "
            f"{row.get('key')}[{row.get('channel')}]"
        )

    skipped = report.get("skipped") or []
    if skipped:
        lines.append("")
        lines.append(f"unpaired module keys ({len(skipped)}):")
        for s in skipped[:30]:
            lines.append(f"  {s}")
    lines.append("")
    return "\n".join(lines)


def load_jsonl_chats(path: Path, max_rows: int) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") if isinstance(obj, dict) else None
            if not isinstance(msgs, list) or split_hao(msgs) is None:
                continue
            rows.append(msgs)
            if len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"no complete (h,a,o) chats in {path}")
    return rows


def run_compare_act(
    world_dir: Path,
    agent_dir: Path,
    *,
    text: str | None,
    jsonl: Path | None,
    max_rows: int,
    device_map: str,
    token_mode: str,
    p: float,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(agent_dir), trust_remote_code=True)
    jobs: list[tuple[list[int], list[int], str]] = []
    if jsonl is not None:
        for msgs in load_jsonl_chats(jsonl, max_rows):
            jobs.append(
                encode_prompt(tokenizer, text=None, messages=msgs, token_mode=token_mode)
            )
        prompt_note = f"jsonl={jsonl} n={len(jobs)}; " + jobs[0][2]
    elif text is not None:
        jobs.append(encode_prompt(tokenizer, text=text, messages=None, token_mode=token_mode))
        prompt_note = jobs[0][2]
    else:
        jobs.append(
            encode_prompt(
                tokenizer, text=None, messages=DEFAULT_MESSAGES, token_mode=token_mode
            )
        )
        prompt_note = jobs[0][2]

    n_tokens = sum(len(ids) for ids, _, _ in jobs)
    n_answer = sum(len(pos) for _, pos, _ in jobs)
    ans_ids = [int(jobs[0][0][i]) for i in jobs[0][1]]
    decoded = tokenizer.decode(ans_ids, skip_special_tokens=False) if ans_ids else ""
    log(f"samples={len(jobs)} tokens={n_tokens} answer={n_answer} ({prompt_note})")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    types = load_layer_types(world_dir) or load_layer_types(agent_dir)

    log("loading AgentWorld")
    world, wname = _load_model(world_dir, dtype=dtype, device_map=device_map)
    log(f"  class={wname}")
    world_caps = [capture_one(world, ids, pos) for ids, pos, _ in jobs]
    log(f"  captured {len(world_caps[0]) if world_caps else 0} modules")
    _free(world)

    log("loading Instruct")
    agent, aname = _load_model(agent_dir, dtype=dtype, device_map=device_map)
    log(f"  class={aname}")
    running: dict[str, tuple[list[float], int]] = {}
    skipped: set[str] = set()
    n_inst = 0
    for cap_w, (ids, pos, _) in zip(world_caps, jobs, strict=True):
        cap_a = capture_one(agent, ids, pos)
        n_inst = len(cap_a)
        acc_pair(running, cap_w, cap_a, skipped)
        del cap_w, cap_a
    _free(agent)
    del world_caps
    n_body = sum(1 for k in running if k != "lm_head")
    log(f"  captured {n_inst} modules; paired={len(running)} (non-lm_head={n_body}) skipped={len(skipped)}")
    if n_body == 0:
        log("WARNING: only lm_head paired — module names did not align")

    mod_delta = finalize_running(running)
    analysis = analyze_channels(mod_delta, p=p, layer_types=types)
    ranked = analysis.pop("ranked")

    return {
        "method": (
            "ACT §3.1 eq. (2) and §4.1: module-output channel |a_AW - a_Instruct|, "
            "mean over pooled answer tokens, rank all channels, keep top p% "
            "(arXiv:2601.09398). Residual stream is not used. Routed MoE experts "
            "skipped. Shared-expert gate/up/down is the dense-MLP analog."
        ),
        "prompt_note": prompt_note,
        "n_samples": len(jobs),
        "n_tokens": n_tokens,
        "n_answer_tokens": n_answer,
        "decoded_answer": decoded,
        "skipped": sorted(skipped),
        "paths": {"world": str(world_dir), "instruct": str(agent_dir)},
        "ranked": ranked,
        **analysis,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--agent", default=DEFAULT_AGENT, help="Instruct hub id or local dir")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=DEFAULT_SOURCE,
    )
    p.add_argument(
        "--text",
        default=None,
        help="raw string (requires --tokens all). default: ls / a.txt / rm / gone chat",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="mix JSONL; last assistant span is the ACT answer; average across rows",
    )
    p.add_argument("--max-rows", type=int, default=8, help="with --jsonl, how many complete chats")
    p.add_argument(
        "--tokens",
        choices=["answer", "all"],
        default="answer",
        help="answer = last assistant (ACT). all = every token (not ACT)",
    )
    p.add_argument(
        "--p",
        type=float,
        default=DEFAULT_TOP_P,
        help="ACT top-p fraction for the channel mask (paper default 0.01)",
    )
    p.add_argument("--device-map", default="auto")
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

    token_mode = args.tokens
    jsonl = args.jsonl
    if jsonl is not None and not jsonl.is_absolute():
        for cand in (jsonl, ROOT / jsonl, ROOT / "train" / jsonl):
            if cand.is_file():
                jsonl = cand
                break
    report = run_compare_act(
        world_dir,
        agent_dir,
        text=args.text,
        jsonl=jsonl,
        max_rows=args.max_rows,
        device_map=args.device_map,
        token_mode=token_mode,
        p=args.p,
    )
    ranked = report.pop("ranked")
    mask = {
        "p": report.get("p"),
        "n_channels": report.get("n_channels"),
        "n_mask": report.get("n_mask"),
        "mask_threshold": report.get("mask_threshold"),
        "mask": report.get("mask"),
        "mask_no_lm_head": report.get("mask_no_lm_head"),
    }
    text = format_summary(report)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "mask.json").write_text(
        json.dumps(mask, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "channels.jsonl").open("w", encoding="utf-8") as f:
        for row in ranked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(text)
    log(f"wrote {out_dir / 'summary.txt'}")
    log(f"wrote {out_dir / 'report.json'}")
    log(f"wrote {out_dir / 'mask.json'}")
    log(f"wrote {out_dir / 'channels.jsonl'} ({len(ranked)} channels)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
