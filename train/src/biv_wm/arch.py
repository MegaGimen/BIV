"""Print fish-cut backbone + heads without dumping 40 identical layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from biv_wm.cut import GROUP, N_LAYERS


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
    log(f"  layers[0:{ell}]   AgentWorld  {collapse_block(front)}")
    log(f"  layers[{ell}:{n}]  Instruct   {collapse_block(back)}")
    norm = getattr(lm, "norm", None)
    if norm is not None:
        log(f"  norm  {type(norm).__name__}")
    log("world path after backbone (no tokens):")
    if extra:
        for name, mod in extra.items():
            log(f"  {name}")
            dump_tree(mod, log, prefix="    ")
    else:
        log("  (none)")
    head = lm_head_module(model)
    if head is not None:
        frozen = not any(p.requires_grad for p in head.parameters())
        shape = ""
        w = getattr(head, "weight", None)
        if w is not None:
            shape = f"  {tuple(w.shape)}"
        log(
            f"unused this stage: lm_head  {type(head).__name__}{shape}  "
            f"frozen={frozen}  (HF CausalLM leftover; not wired to JEPA, no token loss)"
        )
    log("=== end architecture ===")
