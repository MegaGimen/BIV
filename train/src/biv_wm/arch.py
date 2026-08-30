"""Print fish-cut backbone + heads without dumping 40 identical layers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from biv_wm.cut import GROUP, N_LAYERS

LAYER_IN_NAME = re.compile(r"layers\.(\d+)\.")


def freeze_instruct_tail(model: Any, ell: int) -> int:
    """Freeze decoder layers[ell:] and final norm (Instruct half)."""
    lm = language_model(model)
    n = 0
    layers = list(getattr(lm, "layers", []) or [])
    for i, layer in enumerate(layers):
        if i < ell:
            continue
        for p in layer.parameters():
            p.requires_grad = False
            n += 1
    norm = getattr(lm, "norm", None)
    if norm is not None:
        for p in norm.parameters():
            p.requires_grad = False
            n += 1
    return n


def lora_targets_world_only(module_names: list[str], ell: int) -> list[str]:
    """Keep LoRA only on decoder layers before the cut."""
    out: list[str] = []
    for name in module_names:
        m = LAYER_IN_NAME.search(name)
        if m is None:
            continue
        if int(m.group(1)) < ell:
            out.append(name)
    return out


def _span_status(layers: list[Any], start: int, end: int) -> str:
    n_train = 0
    n_all = 0
    for ly in layers[start:end]:
        for p in ly.parameters():
            n_all += 1
            if p.requires_grad:
                n_train += 1
    if n_all == 0:
        return "empty"
    if n_train == 0:
        return "frozen"
    return f"LoRA/trainable tensors={n_train}/{n_all}"


def _kind(layer: Any) -> str:
    name = type(layer).__name__
    if getattr(layer, "linear_attn", None) is not None:
        return f"{name}(linear_attn)"
    if getattr(layer, "self_attn", None) is not None:
        return f"{name}(self_attn)"
    return name


def _runs(items: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for x in items:
        if out and out[-1][0] == x:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return out


def _runs_text(items: list[str]) -> str:
    parts = []
    for name, n in _runs(items):
        parts.append(f"{name}*{n}" if n > 1 else name)
    return " + ".join(parts)


def collapse_block(kinds: list[str], group: int = GROUP) -> str:
    """One repeating group as `block *n`, else run-length of the span."""
    if not kinds:
        return "(empty)"
    if len(kinds) >= group and len(kinds) % group == 0:
        unit = kinds[:group]
        n = len(kinds) // group
        if unit * n == kinds:
            return f"[{_runs_text(unit)}] *{n}"
    return _runs_text(kinds)


def unwrap_base(model: Any) -> Any:
    m = model
    get = getattr(m, "get_base_model", None)
    if callable(get):
        try:
            m = get()
        except Exception:
            pass
    return m


def language_model(model: Any) -> Any | None:
    m = unwrap_base(model)
    inner = getattr(m, "model", m)
    return getattr(inner, "language_model", None) or (
        inner if hasattr(inner, "layers") else None
    )


def detach_lm_head(model: Any) -> bool:
    """Remove lm_head from the live CausalLM. Disk checkpoint is unchanged."""
    m = unwrap_base(model)
    head = getattr(m, "lm_head", None)
    if head is None:
        return False
    m.lm_head = None
    del head
    return True


def install_hidden_only_forward(model: Any) -> None:
    """CausalLM.forward → language_model only (no token logits). Call via wrapped model()."""
    import inspect

    m = unwrap_base(model)
    inner = language_model(model)
    if inner is None:
        raise RuntimeError("no language_model; cannot strip lm_head path")
    allowed = set(inspect.signature(inner.forward).parameters)
    allowed.discard("self")

    def _fwd(*args, **kwargs):
        kwargs.pop("labels", None)
        kwargs.pop("logits_to_keep", None)
        kwargs["output_hidden_states"] = True
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        if args:
            # transformers sometimes passes input_ids positional
            if "input_ids" in allowed and "input_ids" not in kwargs:
                kwargs["input_ids"] = args[0]
            if len(args) > 1 and "attention_mask" in allowed and "attention_mask" not in kwargs:
                kwargs["attention_mask"] = args[1]
        filt = {k: v for k, v in kwargs.items() if k in allowed}
        return inner(**filt)

    m.forward = _fwd
    detach_lm_head(model)


def lm_head_module(model: Any) -> Any | None:
    m = unwrap_base(model)
    return getattr(m, "lm_head", None)


def read_ell(model_dir: Path) -> int | None:
    meta = model_dir / "cut_meta.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ell = data.get("ell")
    return int(ell) if ell is not None else None


def dump_tree(mod: Any, log: Callable[[str], None], prefix: str = "  ") -> None:
    for name, child in mod.named_children():
        extra = ""
        if hasattr(child, "in_features") and hasattr(child, "out_features"):
            extra = f"  {child.in_features}→{child.out_features}"
        log(f"{prefix}{name}: {type(child).__name__}{extra}")
        if list(child.children()):
            dump_tree(child, log, prefix + "  ")


def log_train_architecture(
    *,
    model: Any,
    extra: dict[str, Any],
    model_dir: Path,
    log: Callable[[str], None],
    ell: int | None = None,
) -> None:
    """Backbone: collapse repeating decoder groups. After backbone: print every module.

    Stage 1 extra is JEPA. Stage 2 adds draft / scorer / W the same way.
    """
    if ell is None:
        ell = read_ell(model_dir)
    if ell is None:
        ell = 12
    lm = language_model(model)
    layers = list(getattr(lm, "layers", []))
    n = len(layers) or N_LAYERS
    ell = max(0, min(int(ell), n))
    kinds = [_kind(ly) for ly in layers]
    front, back = kinds[:ell], kinds[ell:]

    log("=== architecture ===")
    log(f"backbone  {model_dir}  ell={ell}  (AgentWorld[:{ell}] + Instruct[{ell}:{n}])")
    embed = getattr(lm, "embed_tokens", None) or getattr(lm, "embedding", None)
    if embed is not None:
        log(f"  embed_tokens  {type(embed).__name__}")
    log(f"  layers[0:{ell}]   AgentWorld  {collapse_block(front)}  {_span_status(layers, 0, ell)}")
    log(f"  layers[{ell}:{n}]  Instruct   {collapse_block(back)}  {_span_status(layers, ell, n)}")
    norm = getattr(lm, "norm", None)
    if norm is not None:
        frozen = not any(p.requires_grad for p in norm.parameters())
        log(f"  norm  {type(norm).__name__}  {'frozen' if frozen else 'trainable'}")
    log("world path after backbone (no tokens):")
    if extra:
        for name, mod in extra.items():
            log(f"  {name}")
            dump_tree(mod, log, prefix="    ")
    else:
        log("  (none)")
    if lm_head_module(model) is not None:
        log("ERROR: lm_head still attached; Stage 1 must detach it")
    else:
        log("lm_head: detached this stage (reload from stage1_cut for Stage 2)")
    log("=== end architecture ===")
