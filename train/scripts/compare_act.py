#!/usr/bin/env python3
"""Compare AgentWorld vs Instruct with ACT module-channel Δa.

ACT 2601.09398 §3.1 / eq. (2) / §4.1: same tokens into both models; a channel
is one output dimension of a trainable module. Average |a_AW − a_Instruct|
over pooled answer tokens, rank every channel together, take top p% as the
ability mask. No residual stream, no layer-cut table.

    python train/scripts/compare_act.py
    python train/scripts/compare_act.py --jsonl train/data/processed/mix_v2 --max-rows 1500
    cd train && CUDA_VISIBLE_DEVICES=0,1 bash scripts/compare_act.sh

Writes under ``train/outputs/act/``:
  summary.txt, report.json, mask.json, channels.jsonl

Merge those mask rows into Instruct::

    python merge/act.py
    python merge/eval.py --act --max-model-len 32768
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
    is_packed_experts_module,
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

DEFAULT_OUT = ROOT / "train" / "outputs" / "act"

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
    """Module output as ``[B, S, C]``.

    MoE Linear / router run after ``view(-1, hidden)``, so the hook sees
    ``[S, C]``. Treat that as batch=1. Router returns a tuple; take logits.
    """
    import torch

    t = out[0] if isinstance(out, (tuple, list)) else out
    if not torch.is_tensor(t):
        return None
    if t.ndim == 4:
        b, s, h, d = t.shape
        t = t.reshape(b, s, h * d)
    if t.ndim == 2:
        t = t.unsqueeze(0)
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
    max_length: int | None = None,
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
        if max_length is not None and len(ids) > max_length:
            ids = ids[-max_length:]
        pos = list(range(len(ids)))
        return ids, pos, "raw --text; mean over all tokens (not ACT answer mask)"

    if not messages:
        raise ValueError("need --text or chat messages")
    full = chat_ids(tokenizer, messages, add_generation_prompt=False)
    if not full:
        raise ValueError("empty chat tokenization")
    if token_mode == "all":
        if max_length is not None and len(full) > max_length:
            full = full[-max_length:]
        return full, list(range(len(full))), "chat; mean over all tokens (not ACT answer mask)"

    prefix = chat_ids(tokenizer, messages[:-1], add_generation_prompt=True)
    start = len(prefix)
    if start >= len(full) or full[: min(start, len(full))] != prefix[: min(start, len(full))]:
        start = max(0, len(full) - max(1, len(full) // 4))

    if max_length is not None and len(full) > max_length:
        ans_len = len(full) - start
        if ans_len >= max_length:
            full = full[start : start + max_length]
            start = 0
        else:
            offset = len(full) - max_length
            full = full[offset:]
            start = max(0, start - offset)
        note = f"chat; truncated to {max_length}; ACT mean over last-assistant tokens [{start}:{len(full)}]"
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


def _gpu_max_memory(*, reserve_gib: float, n_gpu: int | None = None) -> dict[Any, Any] | None:
    """Cap per-GPU weight placement so a 32k MoE forward still has activation room."""
    import torch

    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count() if n_gpu is None else n_gpu
    if n <= 0:
        return None
    out: dict[Any, Any] = {}
    for i in range(n):
        total = torch.cuda.get_device_properties(i).total_memory
        reserve = int(reserve_gib * (1024**3))
        cap = total - reserve
        floor = int(total * 0.55)
        if cap < floor:
            cap = floor
        out[i] = cap
    return out


def _one_gpu_max_memory(gpu_id: int, reserve_gib: float) -> dict[Any, Any] | None:
    """Place this model on one GPU; forbid the other card so dual-resident fits."""
    import torch

    if not torch.cuda.is_available():
        return None
    out: dict[Any, Any] = {}
    for i in range(torch.cuda.device_count()):
        if i == gpu_id:
            total = torch.cuda.get_device_properties(i).total_memory
            cap = total - int(reserve_gib * (1024**3))
            out[i] = max(cap, int(total * 0.55))
        else:
            out[i] = 1
    out["cpu"] = "256GiB"
    return out


def use_dual_gpus(n_gpu: int, device_map: str) -> bool:
    """Two visible cards + default auto map → one model per GPU, stay resident."""
    return n_gpu >= 2 and device_map == "auto"


def _offload_unused_multimodal(model) -> None:
    """ACT only runs the text backbone. Park ViT / MTP on CPU if they landed on GPU."""
    import torch

    root = getattr(model, "model", model)
    moved: list[str] = []
    for name in ("visual", "vision_tower", "mtp"):
        mod = getattr(model, name, None)
        if mod is None:
            mod = getattr(root, name, None)
        if mod is None:
            continue
        try:
            mod.to("cpu")
            moved.append(name)
        except Exception:
            continue
    if moved:
        log(f"  offloaded to cpu: {', '.join(moved)}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()


def _try_from_pretrained(model_dir: Path, kwargs: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # AgentWorld is language_model_only (no ViT). ImageTextToText would
    # randomly init model.visual and print a MISSING dump; CausalLM first.

    loaders = (AutoModelForCausalLM, AutoModelForImageTextToText)
    if not _language_model_only(model_dir):
        loaders = (AutoModelForImageTextToText, AutoModelForCausalLM)
    last = None
    local = dict(kwargs)
    for loader in loaders:
        try:
            model = loader.from_pretrained(str(model_dir), **local)
            _offload_unused_multimodal(model)
            return model, loader.__name__
        except TypeError:
            local.pop("dtype", None)
            try:
                model = loader.from_pretrained(str(model_dir), **local)
                _offload_unused_multimodal(model)
                return model, loader.__name__
            except Exception as e:
                last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"from_pretrained failed: {last}")


def _load_model(model_dir: Path, *, dtype, device_map: str | dict[str, Any] | int):
    import torch

    pin_id: int | None = None
    if isinstance(device_map, int):
        pin_id = device_map
        mapped: str | dict[str, Any] = {"": f"cuda:{pin_id}"}
    elif isinstance(device_map, str) and device_map.startswith("cuda:"):
        pin_id = int(device_map.split(":")[-1])
        mapped = {"": f"cuda:{pin_id}"}
    else:
        mapped = device_map

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": mapped,
    }
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    multimodal = not _language_model_only(model_dir)
    if pin_id is not None:
        log(f"  pin gpu={pin_id} ({'Instruct VLM' if multimodal else 'AgentWorld'})")
    elif mapped == "auto":
        reserve = 28.0 if multimodal else 18.0
        mem = _gpu_max_memory(reserve_gib=reserve)
        if mem is not None:
            kwargs["max_memory"] = mem
            log(
                f"  device_map=auto n_gpu={n_gpu} reserve={reserve:.0f}GiB max_memory_gib="
                + ",".join(f"{i}:{v / 1024**3:.0f}" for i, v in mem.items() if isinstance(i, int))
            )
    try:
        return _try_from_pretrained(model_dir, kwargs)
    except Exception as e:
        err = str(e).lower()
        oom = "out of memory" in err or ("cuda" in err and "memory" in err)
        if pin_id is None or not oom:
            raise
        log(f"  pin gpu={pin_id} OOM, retry with CPU offload on this card only")
        kwargs["device_map"] = "auto"
        mem = _one_gpu_max_memory(pin_id, reserve_gib=18.0 if multimodal else 12.0)
        if mem is not None:
            kwargs["max_memory"] = mem
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _try_from_pretrained(model_dir, kwargs)


def _free(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def thin_answer_pos(pos: list[int], max_answer: int | None) -> list[int]:
    """Evenly keep at most ``max_answer`` positions from the answer span."""
    if not pos or max_answer is None or max_answer <= 0 or len(pos) <= max_answer:
        return list(pos)
    if max_answer == 1:
        return [pos[-1]]
    last = len(pos) - 1
    picks = [round(i * last / (max_answer - 1)) for i in range(max_answer)]
    return [pos[i] for i in picks]


def _answer_cpu(t, answer_idx):
    """Index answer tokens on-device, then one small bf16 copy to CPU.

    Do not ``.cpu()`` the full sequence first: a 32k × 8k activation is ~1GB
    per module, and hundreds of those modules are why 1500-row runs fill RAM.
    """
    import torch

    row = t[0] if t.ndim == 3 else t
    if row.ndim != 2:
        return None
    if answer_idx.numel() == 0:
        return None
    n = int(row.shape[0])
    if int(answer_idx[-1]) >= n:
        return None
    idx = answer_idx
    if idx.device != row.device:
        idx = idx.to(device=row.device)
    sl = row.index_select(0, idx)
    return sl.detach().to(device="cpu", dtype=torch.bfloat16)


def _hidden_answer(hidden, answer_idx):
    """Answer-token rows of a MoE hidden ``[S, H]`` or ``[B, S, H]``.

    Keep them on the module GPU. Round-tripping 40 layers through CPU was the
    bulk of dual-GPU MoE dense-eval time.
    """
    import torch

    if hidden is None or not torch.is_tensor(hidden) or answer_idx is None:
        return None
    h = hidden[0] if hidden.ndim == 3 else hidden
    if h.ndim != 2:
        return None
    idx = answer_idx
    if idx.device != h.device:
        idx = idx.to(device=h.device)
    if idx.numel() == 0 or int(idx[-1]) >= int(h.shape[0]):
        return None
    return h.index_select(0, idx).detach()


def packed_expert_outputs(mod, hidden_th, *, to_cpu: bool = True):
    """Dense-eval every packed expert on answer-token hiddens.

    Same ``h`` for both models, ignoring routing, so channel ``e*O + o`` is
    expert ``e``'s projection output dim ``o`` — the row merge writes in the
    flattened 3D Parameter. Dual-GPU compare keeps the result on device and
    only the per-channel ``|Δ|`` sum crosses to CPU.
    """
    import torch
    import torch.nn.functional as F

    Wgu = mod.gate_up_proj
    Wd = mod.down_proj
    device = Wgu.device
    if getattr(device, "type", None) == "cuda":
        torch.cuda.set_device(device)
        ctx = torch.cuda.device(device)
    else:
        ctx = nullcontext()
    h = hidden_th
    if h.ndim == 3:
        h = h.reshape(-1, h.shape[-1])
    act = getattr(mod, "act_fn", None) or (lambda x: F.silu(x))
    chunk = 32
    gu_parts = []
    dn_parts = []
    with ctx, torch.inference_mode():
        h = h.to(device=device, dtype=Wgu.dtype)
        t, _hin = h.shape
        e, o, _ = Wgu.shape
        hout = int(Wd.shape[1])
        for e0 in range(0, e, chunk):
            sl = slice(e0, min(e, e0 + chunk))
            gu = torch.einsum("th,eoh->teo", h, Wgu[sl])
            gate, up = gu.chunk(2, dim=-1)
            inter = act(gate) * up
            dn = torch.einsum("tei,ehi->teh", inter, Wd[sl])
            gu_parts.append(gu)
            dn_parts.append(dn)
            del gate, up, inter
        gu = torch.cat(gu_parts, dim=1).reshape(t, e * o)
        dn = torch.cat(dn_parts, dim=1).reshape(t, e * hout)
    if to_cpu:
        return (
            gu.detach().to(device="cpu", dtype=torch.bfloat16),
            dn.detach().to(device="cpu", dtype=torch.bfloat16),
        )
    return gu.detach(), dn.detach()


def _channel_abs_sum_diff(left, right, *, col_chunk: int = 65536):
    """``sum_t |a − b|`` as a 1-D CPU tensor. Copies ``b`` in column slices."""
    import torch

    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"packed delta shape {tuple(left.shape)} vs {tuple(right.shape)}")
    t, c = int(left.shape[0]), int(left.shape[1])
    parts = []
    step = max(int(col_chunk), 1)
    with torch.inference_mode():
        for i0 in range(0, c, step):
            sl = slice(i0, min(c, i0 + step))
            a = left[:, sl].float()
            b = right[:, sl].to(device=left.device, non_blocking=True).float()
            parts.append((a - b).abs().sum(0))
            del a, b
        out = torch.cat(parts).contiguous()
    return out.cpu(), t


def acc_packed_experts(cap_w, cap_a, left, right, running, skipped, pool=None) -> None:
    """One layer at a time: |Pred_AW − Pred_Instruct| on packed expert channels.

    World and Instruct already sit on two GPUs; eval them together, keep the
    ``[T, E*O]`` activations on-device, and only ship the channel sums.
    """
    keys = sorted(k for k in left if str(k).startswith("_expert_h:"))

    def _eval(mod, hidden):
        return packed_expert_outputs(mod, hidden, to_cpu=False)

    for hk in keys:
        canon = str(hk).split(":", 1)[1]
        if hk not in right:
            skipped.add(str(hk))
            continue
        mod_w = cap_w.expert_mods.get(canon)
        mod_a = cap_a.expert_mods.get(canon)
        if mod_w is None or mod_a is None:
            skipped.add(f"{canon}:missing_mod")
            continue
        if pool is not None:
            fut_w = pool.submit(_eval, mod_w, left[hk])
            fut_a = pool.submit(_eval, mod_a, right[hk])
            gu_w, dn_w = fut_w.result()
            gu_a, dn_a = fut_a.result()
        else:
            gu_w, dn_w = _eval(mod_w, left[hk])
            gu_a, dn_a = _eval(mod_a, right[hk])
        d, n = _channel_abs_sum_diff(gu_w, gu_a)
        add_abs_sum(running, f"{canon}.gate_up_proj", d, n)
        d2, _n2 = _channel_abs_sum_diff(dn_w, dn_a)
        add_abs_sum(running, f"{canon}.down_proj", d2, n2)
        del gu_w, gu_a, dn_w, dn_a, d, d2


def fill_packed_expert_captures(cap, captured: dict[str, Any]) -> None:
    """1-GPU: turn saved hiddens into [T, E*O] activations before the model is freed."""
    for hk in [k for k in list(captured) if str(k).startswith("_expert_h:")]:
        canon = str(hk).split(":", 1)[1]
        mod = cap.expert_mods.get(canon)
        hidden = captured.pop(hk)
        if mod is None or hidden is None:
            continue
        gu, dn = packed_expert_outputs(mod, hidden)
        captured[f"{canon}.gate_up_proj"] = gu
        captured[f"{canon}.down_proj"] = dn
        del hidden, gu, dn


class ActCapture:
    """Register ACT hooks once; reuse across samples on a resident model."""

    def __init__(
        self,
        model,
        *,
        include_experts: bool = True,
        include_lm_head: bool = False,
        eager_packed_experts: bool = False,
    ) -> None:
        self.model = model
        self.include_lm_head = include_lm_head
        self.include_experts = include_experts
        self.eager_packed_experts = eager_packed_experts
        self.captures: dict[str, Any] = {}
        self.answer_idx = None
        self.handles = []
        self.expert_mods: dict[str, Any] = {}
        seen: set[str] = set()
        named = list(model.named_modules())
        named.sort(key=lambda x: (0 if "language_model." in x[0] else 1, x[0]))
        for name, mod in named:
            canon = canonical_module_key(name)
            if include_experts and is_packed_experts_module(mod, name):
                if canon in seen:
                    continue
                seen.add(canon)
                self.expert_mods[canon] = mod
                self.handles.append(mod.register_forward_hook(self._make_expert_h_hook(canon)))
                continue
            kind = hook_kind(name, include_experts=include_experts)
            if kind is None or kind == "lm_head":
                continue
            if canon in seen:
                continue
            seen.add(canon)
            self.handles.append(mod.register_forward_hook(self._make_hook(canon)))

    def _make_expert_h_hook(self, canon: str):
        def _hook(_mod, inp, _out):
            hidden = inp[0] if isinstance(inp, (tuple, list)) else inp
            sl = _hidden_answer(hidden, self.answer_idx)
            if sl is not None:
                self.captures[f"_expert_h:{canon}"] = sl

        return _hook

    def _make_hook(self, name: str):
        def _hook(_mod, _inp, out):
            t = _as_btc(out)
            if t is None or self.answer_idx is None:
                return
            sl = _answer_cpu(t, self.answer_idx)
            if sl is not None:
                self.captures[name] = sl

        return _hook

    def forward(self, input_ids: list[int], answer_pos: list[int]) -> dict[str, Any]:
        import torch

        device = _embed_device(self.model)
        cuda_ctx = (
            torch.cuda.device(device)
            if getattr(device, "type", None) == "cuda"
            else nullcontext()
        )
        if getattr(device, "type", None) == "cuda":
            torch.cuda.set_device(device)
        ids = torch.tensor([input_ids], device=device)
        mask = torch.ones_like(ids)
        self.answer_idx = torch.tensor(answer_pos, device=device, dtype=torch.long)
        self.captures = {}
        inner = language_model(self.model)
        fwd = inner.forward if inner is not None else self.model.forward
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
        with cuda_ctx, torch.inference_mode():
            out = fwd(**kwargs)
        if self.include_lm_head:
            last = getattr(out, "last_hidden_state", None)
            if last is None:
                raise RuntimeError("forward returned no last_hidden_state")
            last_ans = last[0].index_select(0, self.answer_idx)
            head = lm_head_module(self.model)
            if head is not None:
                logits = _apply_lm_head(head, last_ans)
                self.captures["lm_head"] = logits.detach().to(
                    device="cpu", dtype=torch.bfloat16
                )
        if self.eager_packed_experts and self.expert_mods:
            fill_packed_expert_captures(self, self.captures)
        return self.captures

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.answer_idx = None
        self.captures = {}


def capture_one(
    model,
    input_ids: list[int],
    answer_pos: list[int],
    *,
    include_experts: bool = True,
    include_lm_head: bool = False,
) -> dict[str, Any]:
    """Forward once; keep only answer-token module outputs on CPU (bf16)."""
    cap = ActCapture(
        model,
        include_experts=include_experts,
        include_lm_head=include_lm_head,
        eager_packed_experts=True,
    )
    try:
        return cap.forward(input_ids, answer_pos)
    finally:
        cap.close()


def acc_pair(
    running: dict[str, tuple[Any, int]],
    left: dict[str, Any],
    right: dict[str, Any],
    skipped: set[str],
) -> None:
    for k in sorted(set(left) & set(right)):
        if k.startswith("_"):
            continue
        a, b = left[k], right[k]
        if hasattr(a, "shape") and hasattr(b, "shape") and tuple(a.shape) != tuple(b.shape):
            skipped.add(f"{k}:shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if hasattr(a, "detach") and hasattr(b, "detach"):
            d = (a.detach().float() - b.detach().float()).abs()
            if d.ndim == 1:
                d = d.unsqueeze(0)
            add_abs_sum(running, k, d.sum(dim=0).reshape(-1), int(d.shape[0]))
            continue
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


BROKEN_SMOKE_N_CHANNELS = 712576


def verify_moe_coverage(out_dir: Path, *, include_experts: bool = True) -> dict[str, Any]:
    """Read ``report.json`` + ``mask.json`` and say whether MoE channels landed.

    Broken 20-row smoke had n_channels=712576, kinds attn/ln/embed, no ffn,
    no ``mlp.experts.*`` / ``shared_expert`` in the mask.
    """
    report_path = out_dir / "report.json"
    mask_path = out_dir / "mask.json"
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if not report_path.is_file():
        add("report.json", False, f"missing {report_path}")
        return {"ok": False, "checks": checks, "report": str(report_path), "mask": str(mask_path)}
    if not mask_path.is_file():
        add("mask.json", False, f"missing {mask_path}")
        return {"ok": False, "checks": checks, "report": str(report_path), "mask": str(mask_path)}
    add("report.json", True, str(report_path))
    add("mask.json", True, str(mask_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
    by_kind = report.get("by_kind") or {}
    n_ffn = int((by_kind.get("ffn") or {}).get("n_channels") or 0)
    add("ffn_in_by_kind", n_ffn > 0, f"by_kind.ffn.n_channels={n_ffn}")

    n_ch = int(report.get("n_channels") or 0)
    add(
        "n_channels_above_broken_smoke",
        n_ch > BROKEN_SMOKE_N_CHANNELS,
        f"n_channels={n_ch} (broken smoke was {BROKEN_SMOKE_N_CHANNELS})",
    )

    mod_keys = [str(m.get("key") or "") for m in (report.get("modules") or []) if isinstance(m, dict)]
    n_shared_mod = sum(1 for k in mod_keys if ".mlp.shared_expert." in k)
    add(
        "modules_shared_expert",
        n_shared_mod > 0,
        f"{n_shared_mod} shared_expert modules in report.json",
    )
    if include_experts:
        n_gu = sum(1 for k in mod_keys if k.endswith(".mlp.experts.gate_up_proj"))
        n_dn = sum(1 for k in mod_keys if k.endswith(".mlp.experts.down_proj"))
        add("modules_packed_gate_up", n_gu > 0, f"{n_gu} layers.mlp.experts.gate_up_proj")
        add("modules_packed_down", n_dn > 0, f"{n_dn} layers.mlp.experts.down_proj")

    mask_rows = [
        r
        for r in (mask_payload.get("mask") or mask_payload.get("mask_no_lm_head") or [])
        if isinstance(r, dict)
    ]
    mask_keys = [str(r.get("key") or "") for r in mask_rows]
    n_mask_ffn = sum(
        1
        for r, k in zip(mask_rows, mask_keys, strict=True)
        if r.get("kind") == "ffn"
        or "shared_expert" in k
        or ".experts." in k
        or k.endswith(".mlp.gate")
    )
    add("mask_has_ffn", n_mask_ffn > 0, f"{n_mask_ffn}/{len(mask_keys)} mask rows are MoE/FFN")
    add(
        "mask_shared_expert",
        any(".mlp.shared_expert." in k for k in mask_keys),
        "shared_expert in mask.json",
    )
    if include_experts:
        add(
            "mask_packed_gate_up",
            any(k.endswith(".mlp.experts.gate_up_proj") for k in mask_keys),
            "experts.gate_up_proj in mask.json",
        )
        add(
            "mask_packed_down",
            any(k.endswith(".mlp.experts.down_proj") for k in mask_keys),
            "experts.down_proj in mask.json",
        )

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "report": str(report_path),
        "mask": str(mask_path),
        "include_experts": include_experts,
    }


def print_moe_self_check(verdict: dict[str, Any]) -> None:
    log("[compare_act] self-check (read report.json + mask.json from disk)")
    for c in verdict.get("checks") or []:
        tag = "PASS" if c.get("ok") else "FAIL"
        log(f"  {tag}  {c.get('name')}  {c.get('detail')}")
    log("SELF-CHECK: PASS" if verdict.get("ok") else "SELF-CHECK: FAIL")


def load_jsonl_chats(path: Path, max_rows: int) -> list[list[dict[str, Any]]]:
    """Read full chat messages from a single JSONL or a multi-source mix directory."""
    paths: list[Path] = []
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        # Like train_jepa.py: check subdirectories (wm_code, wm_os, etc.) or root
        for src in ("wm_code", "wm_os", "anti_forget"):
            cand = path / src / "train.jsonl"
            if cand.is_file():
                paths.append(cand)
        if not paths:
            for cand in path.glob("*.jsonl"):
                if "train" in cand.name:
                    paths.append(cand)
        if not paths:
            raise FileNotFoundError(f"no .jsonl files found under directory {path}")
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

    rows: list[list[dict[str, Any]]] = []
    per_file_limit = max(1, math.ceil(max_rows / len(paths))) if paths else max_rows
    for p in paths:
        sub_count = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages") if isinstance(obj, dict) else None
                if not isinstance(msgs, list) or split_hao(msgs) is None:
                    continue
                rows.append(msgs)
                sub_count += 1
                if sub_count >= per_file_limit or len(rows) >= max_rows:
                    break
        if len(rows) >= max_rows:
            break

    if not rows:
        raise ValueError(f"no complete (h,a,o) chats found via {path}")
    return rows


def run_compare_act(
    world_dir: Path,
    agent_dir: Path,
    *,
    text: str | None,
    jsonl: Path | None,
    max_rows: int,
    max_length: int | None = 32768,
    max_answer_tokens: int = 256,
    device_map: str,
    token_mode: str,
    p: float,
    include_experts: bool = True,
    include_lm_head: bool = False,
    chunk_size: int = 32,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(agent_dir), trust_remote_code=True)
    jobs: list[tuple[list[int], list[int], str]] = []
    if jsonl is not None:
        for msgs in load_jsonl_chats(jsonl, max_rows):
            jobs.append(
                encode_prompt(
                    tokenizer,
                    text=None,
                    messages=msgs,
                    token_mode=token_mode,
                    max_length=max_length,
                )
            )
        prompt_note = f"jsonl={jsonl} n={len(jobs)}; " + jobs[0][2]
    elif text is not None:
        jobs.append(
            encode_prompt(
                tokenizer,
                text=text,
                messages=None,
                token_mode=token_mode,
                max_length=max_length,
            )
        )
        prompt_note = jobs[0][2]
    else:
        jobs.append(
            encode_prompt(
                tokenizer,
                text=None,
                messages=DEFAULT_MESSAGES,
                token_mode=token_mode,
                max_length=max_length,
            )
        )
        prompt_note = jobs[0][2]

    n_tokens_raw = sum(len(ids) for ids, _, _ in jobs)
    n_answer_raw = sum(len(pos) for _, pos, _ in jobs)
    jobs = [
        (ids, thin_answer_pos(pos, max_answer_tokens), note)
        for ids, pos, note in jobs
    ]
    n_tokens = sum(len(ids) for ids, _, _ in jobs)
    n_answer = sum(len(pos) for _, pos, _ in jobs)
    ans_ids = [int(jobs[0][0][i]) for i in jobs[0][1]]
    decoded = tokenizer.decode(ans_ids, skip_special_tokens=False) if ans_ids else ""
    log(
        f"samples={len(jobs)} tokens={n_tokens} answer={n_answer} "
        f"(raw_tokens={n_tokens_raw} raw_answer={n_answer_raw} "
        f"max_answer_tokens={max_answer_tokens}) ({prompt_note})"
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dual = use_dual_gpus(n_gpu, device_map)
    log(f"cuda_gpus={n_gpu} device_map={device_map} layout={'dual' if dual else 'swap'}")
    types = load_layer_types(world_dir) or load_layer_types(agent_dir)
    n_jobs = len(jobs)

    def _format_eta(elapsed: float, done: int, total: int) -> str:
        if done <= 0:
            return "--:--:--"
        rem = (elapsed / done) * (total - done)
        return time.strftime("%H:%M:%S", time.gmtime(rem))

    def _log_fwd(phase: str, idx: int, ids: list[int], pos: list[int], t0: float, done: int, total: int) -> None:
        if idx == 1 or idx % 5 == 0 or idx == n_jobs:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 and done else 0.0
            log(
                f"  {phase} sample {idx}/{n_jobs} "
                f"(len={len(ids)}, ans={len(pos)}) "
                f"[{speed:.2f} it/s, ETA: {_format_eta(elapsed, done, total)}]"
            )

    running: dict[str, tuple[Any, int]] = {}
    skipped: set[str] = set()
    n_inst = 0
    t0 = time.time()
    cap_kw = dict(include_experts=include_experts, include_lm_head=include_lm_head)

    if dual:
        log("loading AgentWorld on cuda:0 (stays resident)")
        world, wname = _load_model(world_dir, dtype=dtype, device_map=0)
        log(f"  class={wname}")
        log("loading Instruct on cuda:1 (stays resident)")
        agent, aname = _load_model(agent_dir, dtype=dtype, device_map=1)
        log(f"  class={aname}")
        cap_w = ActCapture(world, eager_packed_experts=False, **cap_kw)
        cap_a = ActCapture(agent, eager_packed_experts=False, **cap_kw)
        log(
            f"  packed_expert_blocks={len(cap_w.expert_mods)} "
            "(dense-eval all experts on-GPU, both models in parallel)"
        )
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for idx, (ids, pos, _) in enumerate(jobs, start=1):
                    _log_fwd("pair", idx, ids, pos, t0, idx - 1, n_jobs)
                    fut_w = pool.submit(cap_w.forward, ids, pos)
                    fut_a = pool.submit(cap_a.forward, ids, pos)
                    left, right = fut_w.result(), fut_a.result()
                    n_inst = sum(1 for k in right if not str(k).startswith("_"))
                    acc_pair(running, left, right, skipped)
                    if include_experts:
                        acc_packed_experts(
                            cap_w, cap_a, left, right, running, skipped, pool=pool
                        )
                    del left, right
        finally:
            cap_w.close()
            cap_a.close()
            _free(world)
            world = None
            _free(agent)
            agent = None
        layout_note = (
            "Dual-GPU resident: AgentWorld on cuda:0, Instruct on cuda:1, "
            "parallel forward per sample, hooks registered once. "
            "Packed routed experts: save answer-token hiddens on GPU, then dense-eval "
            "all experts on both cards in parallel; only channel |Δ| sums hit CPU. "
        )
    else:
        if include_experts and chunk_size > 1:
            log("packed MoE experts on 1-GPU: chunk_size=1 (activations must exist before unload)")
            chunk_size = 1
        n_chunks = max(1, math.ceil(n_jobs / chunk_size))
        total_fwds = 2 * n_jobs
        fwds_done = 0
        for c in range(n_chunks):
            sl = slice(c * chunk_size, min(n_jobs, (c + 1) * chunk_size))
            chunk = jobs[sl]
            log(f"chunk {c + 1}/{n_chunks} ({len(chunk)} samples): loading AgentWorld")
            world, wname = _load_model(world_dir, dtype=dtype, device_map=device_map)
            if c == 0:
                log(f"  class={wname}")
            cap_w = ActCapture(world, eager_packed_experts=True, **cap_kw)
            world_caps: list[dict[str, Any]] = []
            try:
                for j, (ids, pos, _) in enumerate(chunk):
                    idx = c * chunk_size + j + 1
                    _log_fwd("AgentWorld", idx, ids, pos, t0, fwds_done, total_fwds)
                    world_caps.append(cap_w.forward(ids, pos))
                    fwds_done += 1
            finally:
                cap_w.close()
                _free(world)

            log(f"chunk {c + 1}/{n_chunks}: loading Instruct")
            agent, aname = _load_model(agent_dir, dtype=dtype, device_map=device_map)
            if c == 0:
                log(f"  class={aname}")
            cap_a = ActCapture(agent, eager_packed_experts=True, **cap_kw)
            try:
                for j, ((ids, pos, _), left) in enumerate(zip(chunk, world_caps, strict=True)):
                    idx = c * chunk_size + j + 1
                    _log_fwd("Instruct", idx, ids, pos, t0, fwds_done, total_fwds)
                    right = cap_a.forward(ids, pos)
                    n_inst = sum(1 for k in right if not str(k).startswith("_"))
                    acc_pair(running, left, right, skipped)
                    del left, right
                    fwds_done += 1
            finally:
                cap_a.close()
                _free(agent)
            del world_caps
        layout_note = (
            f"Chunked in-RAM (chunk_size={chunk_size}): World then Instruct per chunk, "
            "accumulate Δa, drop activations. Packed experts materialized before unload. "
        )

    n_body = sum(1 for k in running if k != "lm_head")
    log(
        f"  captured {n_inst} modules; paired={len(running)} "
        f"(non-lm_head={n_body}) skipped={len(skipped)} in {time.time() - t0:.1f}s"
    )
    if n_body == 0:
        log("WARNING: only lm_head paired — module names did not align")

    mod_delta = finalize_running(running)
    analysis = analyze_channels(mod_delta, p=p, layer_types=types)
    ranked = analysis.pop("ranked")

    method_note = (
        "ACT §3.1 eq. (2) and §4.1: module-output channel |a_AW - a_Instruct|, "
        "mean over pooled answer tokens, rank all channels, keep top p% "
        "(arXiv:2601.09398). Residual stream is not used. "
        + layout_note
    )
    if include_experts:
        method_note += (
            "Routed MoE: packed 3D experts dense-eval'd on answer-token hiddens "
            "(channel e*O+o). Shared-expert Linear + mlp.gate router hooked; "
            "2D MoE activations kept. "
        )
    else:
        method_note += "Routed MoE experts skipped. Shared-expert gate/up/down is the dense-MLP analog. "
    if not include_lm_head:
        method_note += " lm_head not captured (merge default --no-lm-head)."

    return {
        "method": method_note,
        "prompt_note": prompt_note,
        "n_samples": len(jobs),
        "n_tokens": n_tokens,
        "n_answer_tokens": n_answer,
        "decoded_answer": decoded,
        "skipped": sorted(skipped),
        "paths": {"world": str(world_dir), "instruct": str(agent_dir)},
        "layout": "dual" if dual else "swap",
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
        "--output",
        type=Path,
        default=None,
        help="alias for --out-dir; a .json path uses the parent directory",
    )
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
    p.add_argument("--max-rows", type=int, default=1500, help="with --jsonl, how many complete chats")
    p.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="truncate input tokens (preserves last assistant answer). default: 32768",
    )
    p.add_argument(
        "--max-answer-tokens",
        type=int,
        default=256,
        help="evenly subsample the ACT answer span (default 256). "
        "keeps RAM/disk small; the mean is still over this pooled set",
    )
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
    p.add_argument(
        "--skip-experts",
        action="store_true",
        default=False,
        help="skip packed routed experts (still hook shared_expert + router)",
    )
    p.add_argument(
        "--lm-head",
        action="store_true",
        default=False,
        help="also capture lm_head (large; merge uses --no-lm-head by default)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="1-GPU only: World/Instruct swap every N samples. Dual GPU ignores this.",
    )
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    out_dir = args.output if args.output is not None else args.out_dir
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    if out_dir.suffix == ".json":
        out_dir = out_dir.parent
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
            if cand.exists():
                jsonl = cand
                break
    report = run_compare_act(
        world_dir,
        agent_dir,
        text=args.text,
        jsonl=jsonl,
        max_rows=args.max_rows,
        max_length=args.max_length,
        max_answer_tokens=args.max_answer_tokens,
        cache_dir=cache_dir,
        device_map=args.device_map,
        token_mode=token_mode,
        p=args.p,
        include_experts=not args.skip_experts,
        include_lm_head=args.lm_head,
        chunk_size=max(1, args.chunk_size),
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
    log(f"merge reuses {out_dir / 'mask.json'}: python merge/act.py")
    verdict = verify_moe_coverage(out_dir, include_experts=not args.skip_experts)
    (out_dir / "self_check.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {out_dir / 'self_check.json'}")
    print_moe_self_check(verdict)
    if not verdict["ok"]:
        raise SystemExit("SELF-CHECK: FAIL — MoE channels did not land in report.json / mask.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
