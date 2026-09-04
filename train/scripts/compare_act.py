#!/usr/bin/env python3
"""Compare AgentWorld vs Instruct activations with ACT channel Δa (text bars).

Same tokens into both checkpoints. Per output channel of each hooked module
(and of the residual stream), ACT 2601.09398:

    Δa_i = mean over answer tokens of |a_i^{AgentWorld} − a_i^{Instruct}|

The table looks like ``compare.py`` (per-layer text bars), but the number is
activation difference, not row-MAV of weights. No Base checkpoint.

    python train/scripts/compare_act.py
    python train/scripts/compare_act.py --text "ls\\na.txt\\nrm a.txt\\ngone"

Writes ``train/outputs/compare_act/summary.txt`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import gc
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

from biv_wm.act import (  # noqa: E402
    CCDF_THRESHOLDS,
    bar,
    ccdf,
    channel_delta,
    channel_stats,
    hook_kind,
    layer_index,
    top_channels,
)
from biv_wm.arch import language_model, lm_head_module  # noqa: E402
from download import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_CACHE,
    DEFAULT_SOURCE,
    DEFAULT_WORLD,
    resolve_model,
)

DEFAULT_OUT = ROOT / "train" / "outputs" / "compare_act"
BAR_W = 24
N_LAYERS = 40

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


def _embed_device(model) -> Any:
    inner = language_model(model)
    emb = None
    if inner is not None:
        emb = getattr(inner, "embed_tokens", None) or getattr(inner, "embed", None)
    if emb is None:
        getter = getattr(model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
    if emb is None or getattr(emb, "weight", None) is None:
        raise RuntimeError("cannot find input embeddings to place input_ids")
    return emb.weight.device


def _filter_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    allowed = set(inspect.signature(fn).parameters)
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def encode_prompt(
    tokenizer,
    *,
    text: str | None,
    messages: list[dict[str, str]] | None,
    token_mode: str,
) -> tuple[list[int], list[int], str]:
    """Return (input_ids, answer_positions, note)."""
    if text is not None:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not ids:
            raise ValueError("empty --text")
        pos = list(range(len(ids)))
        return ids, pos, "raw --text; ACT mean over all tokens"

    if not messages:
        raise ValueError("need --text or chat messages")
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if isinstance(full, dict):
        full = full["input_ids"]
    full = list(full)
    if token_mode == "all":
        return full, list(range(len(full))), "chat; ACT mean over all tokens"

    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    if isinstance(prefix, dict):
        prefix = prefix["input_ids"]
    prefix = list(prefix)
    start = len(prefix)
    if start >= len(full):
        start = max(0, len(full) - max(1, len(full) // 4))
        note = (
            f"chat; prefix not a prefix of full (start clamped to {start}); "
            "ACT mean over that suffix"
        )
    else:
        note = f"chat; ACT mean over last-assistant tokens [{start}:{len(full)}]"
    return full, list(range(start, len(full))), note


def _load_model(model_dir: Path, *, dtype, device_map: str):
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
            return loader.from_pretrained(str(model_dir), **kwargs), loader.__name__
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
    """Forward once; CPU float32 activations at answer positions, keyed by name."""
    import torch

    device = _embed_device(model)
    ids = torch.tensor([input_ids], device=device)
    mask = torch.ones_like(ids)

    captures: dict[str, Any] = {}
    handles = []

    def _answer(t):
        # Slice on CPU so device_map=auto layers are not required to share idx's device.
        return t[0].detach().float().cpu()[answer_pos]

    def make_hook(name: str):
        def _hook(_mod, _inp, out):
            t = _as_btc(out)
            if t is None:
                return
            captures[name] = _answer(t)

        return _hook

    for name, mod in model.named_modules():
        kind = hook_kind(name)
        if kind is None:
            continue
        if kind == "lm_head":
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))

    inner = language_model(model)
    fwd = inner.forward if inner is not None else model.forward
    kwargs = _filter_kwargs(
        fwd,
        {
            "input_ids": ids,
            "attention_mask": mask,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        },
    )
    with torch.inference_mode():
        out = fwd(**kwargs)

    for h in handles:
        h.remove()

    hidden = getattr(out, "hidden_states", None)
    residual: dict[str, Any] = {}
    last = getattr(out, "last_hidden_state", None)
    last_ans = None
    if hidden:
        for i, h in enumerate(hidden):
            key = "embed" if i == 0 else f"layers.{i - 1}.residual"
            residual[key] = _answer(h)
    if last is not None:
        last_cpu = _answer(last)
        if not hidden or last.data_ptr() != hidden[-1].data_ptr():
            residual["final_norm"] = last_cpu
        last_ans = last_cpu
    elif residual:
        last_ans = residual["embed"]
        if hidden and len(hidden) > 1:
            last_ans = residual[f"layers.{len(hidden) - 2}.residual"]
    else:
        raise RuntimeError("forward returned no hidden_states / last_hidden_state")

    head = lm_head_module(model)
    if head is not None and last_ans is not None:
        weight = next(head.parameters())
        with torch.inference_mode():
            logits = head(last_ans.to(device=weight.device, dtype=weight.dtype))
        captures["lm_head"] = logits.detach().float().cpu()

    return {"modules": captures, "residual": residual}


def _pair_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, list[float]], list[str]]:
    keys = sorted(set(left) & set(right))
    missing = sorted(set(left) ^ set(right))
    out: dict[str, list[float]] = {}
    for k in keys:
        a, b = left[k], right[k]
        if tuple(a.shape) != tuple(b.shape):
            missing.append(f"{k}:shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        out[k] = channel_delta(a, b)
    return out, missing


def _empty_layer(i: int, kind: str) -> dict[str, Any]:
    z = channel_stats([])
    return {
        "layer": i,
        "kind": kind,
        "residual": z,
        "groups": {},
    }


def build_layers(
    residual_delta: dict[str, list[float]],
    module_delta: dict[str, list[float]],
    types: list[str],
) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[str, Any]] = {}
    for i in range(N_LAYERS):
        kind = types[i] if i < len(types) else "?"
        by_layer[i] = _empty_layer(i, kind)

    for key, vec in residual_delta.items():
        li = layer_index(key)
        if li is None:
            continue
        if li not in by_layer:
            by_layer[li] = _empty_layer(li, "?")
        by_layer[li]["residual"] = channel_stats(vec)

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for key, vec in module_delta.items():
        li = layer_index(key)
        kind = hook_kind(key)
        if li is None or kind not in {"attn", "ffn", "ln"}:
            continue
        grouped[(li, kind)].extend(vec)

    for (li, g), vec in grouped.items():
        if li not in by_layer:
            by_layer[li] = _empty_layer(li, "?")
        by_layer[li]["groups"][g] = channel_stats(vec)

    return [by_layer[i] for i in sorted(by_layer)]


def special_stats(residual_delta: dict[str, list[float]], module_delta: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, vec in residual_delta.items():
        if layer_index(key) is None:
            out[key] = channel_stats(vec)
    for key, vec in module_delta.items():
        kind = hook_kind(key)
        if kind in {"embed", "lm_head"} or (kind == "ln" and layer_index(key) is None):
            out[key] = channel_stats(vec)
    return out


def format_summary(report: dict[str, Any]) -> str:
    layers: list[dict[str, Any]] = report.get("layers") or []
    peak = max((float(r["residual"]["mean"]) for r in layers), default=0.0)
    peak = max(peak, 1e-12)
    gpeak = 0.0
    for r in layers:
        for g in (r.get("groups") or {}).values():
            gpeak = max(gpeak, float(g["mean"]))
    gpeak = max(gpeak, peak, 1e-12)

    lines: list[str] = []
    lines.append(
        "compare_act.py — 激活差 = |a_AW − a_Instruct|，答案 token 上按通道平均 (ACT Δa_i)"
    )
    lines.append(report["method"])
    lines.append(f"world     {report['paths']['world']}")
    lines.append(f"instruct  {report['paths']['instruct']}")
    lines.append(f"prompt    {report.get('prompt_note', '')}")
    lines.append(
        f"n_tokens={report.get('n_tokens')} n_answer={report.get('n_answer_tokens')} "
        f"decoded_answer={report.get('decoded_answer')!r}"
    )
    lines.append("")
    lines.append(
        f"{'L':>3} {'kind':<16} {'Δa_mean':>10} {'p99':>10} {'top1%':>10}  "
        f"{'ACT_Δa':<{BAR_W}}"
    )
    lines.append("-" * (3 + 1 + 16 + 1 + 10 + 1 + 10 + 1 + 10 + 2 + BAR_W))
    for r in layers:
        i = int(r["layer"])
        st = r["residual"]
        mean = float(st["mean"])
        lines.append(
            f"{i:3d} {str(r['kind'])[:16]:<16} {mean:10.4e} "
            f"{float(st['p99']):10.4e} {float(st['top1pct_mean']):10.4e}  "
            f"{bar(mean, peak, BAR_W)}"
        )

    lines.append("")
    lines.append("per layer split attn vs ffn vs ln (module outputs, same Δa_i mean):")
    lines.append(
        f"{'L':>3} {'g':<5} {'Δa_mean':>10} {'p99':>10} {'n_ch':>8}  "
        f"{'ACT_Δa':<{BAR_W}}"
    )
    for r in layers:
        i = int(r["layer"])
        for g in ("attn", "ffn", "ln"):
            gg = (r.get("groups") or {}).get(g)
            if not gg or int(gg.get("n_channels") or 0) == 0:
                continue
            mean = float(gg["mean"])
            lines.append(
                f"{i:3d} {g:<5} {mean:10.4e} {float(gg['p99']):10.4e} "
                f"{int(gg['n_channels']):8d}  {bar(mean, gpeak, BAR_W)}"
            )

    scored = [(float(r["residual"]["mean"]), int(r["layer"])) for r in layers]
    scored.sort(reverse=True)
    top = [f"L{i}={v:.4e}" for v, i in scored[:8]]
    lines.append("")
    lines.append("残差流激活差最大层: " + ", ".join(top))

    c_all = report.get("ccdf_all") or {}
    c_no_head = report.get("ccdf_no_lm_head") or {}
    lines.append("")
    lines.append("CCDF = 通道里 Δa 超过阈值的比例 (ACT Figure 1):")
    lines.append(f"  {'t':>6}  {'all':>8}  {'no_lm_head':>10}")
    for t in CCDF_THRESHOLDS:
        key = str(t)
        lines.append(
            f"  {t:6.1f}  {float(c_all.get(key, 0.0)):8.4f}  "
            f"{float(c_no_head.get(key, 0.0)):10.4f}"
        )

    spec = report.get("special") or {}
    if spec:
        lines.append("")
        lines.append("non-layer (embed / lm_head / final norm):")
        for k, v in sorted(spec.items()):
            lines.append(
                f"  {k}: Δa_mean={v['mean']:.4e}  p99={v['p99']:.4e}  "
                f"n_ch={v['n_channels']}"
            )

    tops = report.get("top_channels") or []
    if tops:
        lines.append("")
        lines.append("Δa 最大的通道 (module key, channel, Δa):")
        for row in tops[:20]:
            lines.append(
                f"  {row['key']} [{row['channel']}]  {row['delta']:.4e}"
            )

    skipped = report.get("skipped") or []
    if skipped:
        lines.append("")
        lines.append(f"unpaired module keys ({len(skipped)}):")
        for s in skipped[:30]:
            lines.append(f"  {s}")
    lines.append("")
    return "\n".join(lines)


def run_compare_act(
    world_dir: Path,
    agent_dir: Path,
    *,
    text: str | None,
    device_map: str,
    token_mode: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(agent_dir), trust_remote_code=True)
    messages = None if text is not None else DEFAULT_MESSAGES
    ids, answer_pos, prompt_note = encode_prompt(
        tokenizer, text=text, messages=messages, token_mode=token_mode
    )
    decoded = tokenizer.decode([ids[i] for i in answer_pos], skip_special_tokens=False)
    log(f"tokens={len(ids)} answer={len(answer_pos)} ({prompt_note})")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    types = load_layer_types(world_dir) or load_layer_types(agent_dir)

    log("loading AgentWorld")
    world, wname = _load_model(world_dir, dtype=dtype, device_map=device_map)
    log(f"  class={wname}")
    cap_w = capture_one(world, ids, answer_pos)
    _free(world)

    log("loading Instruct")
    agent, aname = _load_model(agent_dir, dtype=dtype, device_map=device_map)
    log(f"  class={aname}")
    cap_a = capture_one(agent, ids, answer_pos)
    _free(agent)

    res_delta, res_skip = _pair_delta(cap_w["residual"], cap_a["residual"])
    mod_delta, mod_skip = _pair_delta(cap_w["modules"], cap_a["modules"])
    layers = build_layers(res_delta, mod_delta, types)

    all_ch: list[float] = []
    no_head: list[float] = []
    named: dict[str, list[float]] = {}
    named.update(res_delta)
    named.update(mod_delta)
    for key, vec in named.items():
        all_ch.extend(vec)
        if hook_kind(key) != "lm_head" and key != "lm_head":
            no_head.extend(vec)

    return {
        "method": (
            "ACT channel-wise |a_AW - a_Instruct|, mean over answer tokens "
            "(arXiv:2601.09398 §3.1). Residual stream = per-layer bars; "
            "hooks = attn/ffn/ln projections (MoE routed experts skipped). "
            "Text bars like compare.py, quantity is activation not weight MAV."
        ),
        "prompt_note": prompt_note,
        "n_tokens": len(ids),
        "n_answer_tokens": len(answer_pos),
        "decoded_answer": decoded,
        "layers": layers,
        "special": special_stats(res_delta, mod_delta),
        "ccdf_all": ccdf(all_ch),
        "ccdf_no_lm_head": ccdf(no_head),
        "n_channels_all": len(all_ch),
        "n_channels_no_lm_head": len(no_head),
        "top_channels": top_channels(named, n=32),
        "skipped": res_skip + mod_skip,
        "paths": {"world": str(world_dir), "instruct": str(agent_dir)},
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
        help="raw string (same tokens to both models). default: ls / a.txt / rm / gone chat",
    )
    p.add_argument(
        "--tokens",
        choices=["answer", "all"],
        default="answer",
        help="answer = last assistant span on the default chat; ignored for --text (always all)",
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

    token_mode = "all" if args.text is not None else args.tokens
    report = run_compare_act(
        world_dir,
        agent_dir,
        text=args.text,
        device_map=args.device_map,
        token_mode=token_mode,
    )
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
