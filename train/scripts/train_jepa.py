#!/usr/bin/env python3
"""Stage 1 LLM-JEPA on AgentWorld. Wiring copied from galilai-group/llm-jepa.

Their RepresentationTrainer (finetune.py) encodes Text and Code as two
independent chat-templated sequences, takes hidden at `len(unpad)+last_token`,
and adds 1-cosine to a next-token CE on the full conversation. We keep that
graph. Pairing is ours: Text = history+command, Code = observation. CE
unmasks only the observation (not earlier assistant turns in h).

Over-long trajectories drop later complete turns (keep the left, chop the
right) so (h, a, o) is the last pair that still fits. Token lists are only
prefix-truncated if that first remaining turn still overflows.

What we do not copy: HuggingFace Trainer, concatenating 3 sequences into one
batch, padding every row to max_length, full-seq lm_head. Those blow up at
35B / 65k. Sequential hidden-only forwards + lm_head only on labeled tokens.

  cd train && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_jepa.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
MERGE = ROOT / "merge"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MERGE) not in sys.path:
    sys.path.insert(0, str(MERGE))

from biv_wm.ckpt import (  # noqa: E402
    canonical_lora_key,
    epoch_end_name,
    find_latest_ckpt,
    rotate_rolling,
    rolling_name,
    write_trainer_state,
)
from biv_wm.hao import (  # noqa: E402
    complete_turn_end_indices,
    messages_through_n_turns,
    split_hao,
)
from download import resolve_model  # noqa: E402

DEFAULT_CONFIG = TRAIN / "configs" / "jepa" / "stage1.yaml"


def log(msg: str) -> None:
    print(msg, flush=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"bad yaml {path}")
    return data


def _resolve(raw: str | Path, base: Path = TRAIN) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (base / p)


def resolve_mix(raw: str | Path, sources: list[str]) -> Path:
    wanted = _resolve(raw)
    candidates = [wanted]
    for name in ("mix_v2", "mix_v1"):
        alt = TRAIN / "data" / "processed" / name
        if alt not in candidates:
            candidates.append(alt)
    for path in candidates:
        if any((path / src / "train.jsonl").is_file() for src in sources):
            if path != wanted:
                log(f"mix {wanted} missing or empty, using {path}")
            return path
    raise SystemExit(
        f"no mix JSONL at {wanted} or mix_v2/mix_v1; "
        "run python train/scripts/prepare_data.py --wm-code --wm-os "
        "--out-dir train/data/processed/mix_v2"
    )


def two_d_lora_targets(model, suffixes: list[str]) -> list[str]:
    """PEFT LoRA needs 2D Linear. MoE expert gate/up/down are 3D — skip those."""
    want = set(suffixes)
    names: list[str] = []
    for n, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or getattr(w, "ndim", 0) != 2:
            continue
        leaf = n.rsplit(".", 1)[-1]
        if leaf in want:
            names.append(n)
    if not names:
        raise SystemExit(f"no 2D LoRA modules matching {suffixes}")
    return names


def _content(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def apply_template(tokenizer, messages: list) -> str:
    """Same as llm-jepa: chat template, no generation prompt."""
    msgs = list(messages) if messages else [{"role": "user", "content": ""}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        try:
            alt = []
            for m in msgs:
                if (m or {}).get("role") == "assistant":
                    alt.append({**m, "role": "user"})
                else:
                    alt.append(m)
            return tokenizer.apply_chat_template(
                alt, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return "\n".join(_content(m) for m in msgs)


def tokenize_ids(tokenizer, text: str) -> list[int]:
    return list(
        tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    )


def _fit(ids: list[int], max_length: int, *, keep: str) -> list[int]:
    if len(ids) <= max_length:
        return ids
    if keep == "suffix":
        return ids[-max_length:]
    return ids[:max_length]


def chat_token_len(tokenizer, messages: list) -> int:
    return len(tokenize_ids(tokenizer, apply_template(tokenizer, messages)))


def trim_messages_keep_prefix(tokenizer, messages: list, max_length: int) -> list:
    """Drop later complete turns until the remaining chat fits in ``max_length``.

    If turns are a, b, c and a+b already fills the window, the sample becomes
    a,b — c is treated as never having happened. Never drop the left of the
    trajectory. If even the first turn overflows, still return that first turn;
    ``encode_texts`` then chops tokens from the right (never the left).
    """
    ends = complete_turn_end_indices(messages)
    if not ends:
        return list(messages)
    n = len(ends)
    last = messages[: ends[-1]]
    if chat_token_len(tokenizer, last) <= max_length:
        return last
    lo, hi = 1, n - 1
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        prefix = messages_through_n_turns(messages, mid)
        if chat_token_len(tokenizer, prefix) <= max_length:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return messages_through_n_turns(messages, best)


def fit_hao(tokenizer, messages: list, max_length: int):
    """Last fitting prefix of complete turns, then ``split_hao``."""
    return split_hao(trim_messages_keep_prefix(tokenizer, messages, max_length))


def _find_span(haystack: list[int], needle: list[int]) -> int | None:
    """Last occurrence (observation sits at the end of the full chat)."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i : i + n] == needle:
            return i
    return None


def create_o_labels(full_ids: list[int], o_content_ids: list[int]) -> list[int]:
    """GitHub create_masked_labels, but only the observation turn."""
    labels = [-100] * len(full_ids)
    start = _find_span(full_ids, o_content_ids)
    if start is None:
        return labels
    for j in range(start, start + len(o_content_ids)):
        labels[j] = full_ids[j]
    return labels


def encode_texts(
    tokenizer,
    h_msgs: list,
    a_msg: dict,
    o_msg: dict,
    max_length: int,
) -> dict[str, Any]:
    """Three sequences, same split as llm-jepa user / assistant / full.

    full  — chat(h + a + o), for next-token CE on o
    left  — chat(h + a) independently, Enc(Text)
    right — chat([o]) independently, Enc(Code)
    Over-long token lists (first turn still past the window after later turns
    were dropped): chop the right, keep the left. Do not pad to max_length here.
    """
    full_ids = _fit(
        tokenize_ids(tokenizer, apply_template(tokenizer, list(h_msgs) + [a_msg, o_msg])),
        max_length,
        keep="prefix",
    )
    left_ids = _fit(
        tokenize_ids(tokenizer, apply_template(tokenizer, list(h_msgs) + [a_msg])),
        max_length,
        keep="prefix",
    )
    right_ids = _fit(
        tokenize_ids(tokenizer, apply_template(tokenizer, [o_msg])),
        max_length,
        keep="prefix",
    )
    o_content_ids = tokenizer.encode(_content(o_msg), add_special_tokens=False)
    return {
        "full_ids": full_ids,
        "full_labels": create_o_labels(full_ids, o_content_ids),
        "left_ids": left_ids,
        "right_ids": right_ids,
    }


def encode_mediated(
    tokenizer,
    h_msgs: list,
    a_msg: dict,
    o_msg: dict,
    max_length: int,
) -> dict[str, Any]:
    """History-mediated pair used by the collapse probe (not the live train loss).

    \(z_t=\mathrm{Enc}(h)\), \(z_{t+1}=\mathrm{Enc}(h,a,o)\). Observation is not
    encoded alone. Truncation still chops the right of each token list.
    """
    state_ids = _fit(
        tokenize_ids(tokenizer, apply_template(tokenizer, list(h_msgs))),
        max_length,
        keep="prefix",
    )
    next_ids = _fit(
        tokenize_ids(tokenizer, apply_template(tokenizer, list(h_msgs) + [a_msg, o_msg])),
        max_length,
        keep="prefix",
    )
    return {
        "state_ids": state_ids,
        "next_ids": next_ids,
        "a_text": _content(a_msg),
        "o_text": _content(o_msg),
    }


def sequence_lengths(tokenizer, h_msgs: list, a_msg: dict, o_msg: dict) -> dict[str, int]:
    """Untruncated token counts for the three LLM-JEPA sequences."""
    full = tokenize_ids(
        tokenizer, apply_template(tokenizer, list(h_msgs) + [a_msg, o_msg])
    )
    left = tokenize_ids(
        tokenizer, apply_template(tokenizer, list(h_msgs) + [a_msg])
    )
    right = tokenize_ids(tokenizer, apply_template(tokenizer, [o_msg]))
    return {"full": len(full), "left": len(left), "right": len(right)}


def load_rows(mix_dir: Path, sources: list[str], split: str, limit: int | None) -> list[list]:
    """Read mix chat ``messages`` lists (full trajectories) for each source.

    Truncation to ``max_length`` happens later: drop later complete turns, then
    ``split_hao`` on what remains. Do not pre-split to the last (a, o) here.

    `limit`, if set, is a *global* row budget split evenly across `sources`
    and filled by reservoir sampling (Algorithm R) within each source. This
    keeps every source represented in a downsampled quick run instead of the
    old behavior, which just took the first `limit` rows in file order and
    could return zero rows from any source after the first once the budget
    was hit (e.g. wm_os would get nothing if wm_code alone exceeded limit).
    Reservoir sampling also avoids reading whole multi-GB files into memory
    just to shuffle, and avoids a positional bias if a file happens to be
    sorted/grouped (e.g. by task or time) rather than pre-shuffled.
    """
    per_source_limit = None if limit is None else max(1, math.ceil(limit / len(sources)))
    rng = random.Random(42)
    rows: list[list] = []
    for src in sources:
        path = mix_dir / src / f"{split}.jsonl"
        if not path.is_file():
            log(f"skip missing {path}")
            continue
        reservoir: list[list] = []
        seen = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                hao = split_hao(msgs) if isinstance(msgs, list) else None
                if hao is None:
                    continue
                item = msgs
                if per_source_limit is None:
                    reservoir.append(item)
                    continue
                seen += 1
                if len(reservoir) < per_source_limit:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < per_source_limit:
                        reservoir[j] = item
        suffix = f" (reservoir-sampled from {seen})" if per_source_limit is not None else ""
        log(f"{src}: {len(reservoir)} rows{suffix}")
        rows.extend(reservoir)
    return rows


class HaoDataset:
    def __init__(
        self,
        rows: list,
        tokenizer,
        max_length: int,
        encoding: str = "llm-jepa",
    ) -> None:
        if encoding not in ("llm-jepa", "mediated"):
            raise ValueError(f"encoding must be llm-jepa or mediated, got {encoding!r}")
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoding = encoding
        self._fitted: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        hao = self._fitted.get(idx)
        if hao is None:
            hao = fit_hao(self.tokenizer, self.rows[idx], self.max_length)
            if hao is None:
                raise RuntimeError(
                    f"row {idx} has no complete (h, a, o) after prefix trim"
                )
            self._fitted[idx] = hao
        h, a, o = hao
        if self.encoding == "mediated":
            return encode_mediated(self.tokenizer, h, a, o, self.max_length)
        enc = encode_texts(self.tokenizer, h, a, o, self.max_length)
        enc["o_text"] = _content(o)
        enc["left_text"] = _content(a)
        return enc


def _pad_len(n: int, multiple: int) -> int:
    if multiple <= 1:
        return n
    return ((n + multiple - 1) // multiple) * multiple


def collate(batch: list[dict[str, Any]], pad_id: int, pad_multiple: int = 1) -> dict[str, Any]:
    import torch

    def pad_rows(rows: list[list[int]], fill: int) -> tuple[torch.Tensor, torch.Tensor]:
        mlen = _pad_len(max((len(x) for x in rows), default=1), pad_multiple)
        mlen = max(mlen, 1)
        out = torch.full((len(rows), mlen), fill, dtype=torch.long)
        mask = torch.zeros((len(rows), mlen), dtype=torch.long)
        for i, row in enumerate(rows):
            if not row:
                continue
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return out, mask

    full_ids, full_mask = pad_rows([ex["full_ids"] for ex in batch], pad_id)
    left_ids, left_mask = pad_rows([ex["left_ids"] for ex in batch], pad_id)
    right_ids, right_mask = pad_rows([ex["right_ids"] for ex in batch], pad_id)
    labels_rows = [ex["full_labels"] for ex in batch]
    mlen = full_ids.size(1)
    full_labels = torch.full((len(batch), mlen), -100, dtype=torch.long)
    for i, row in enumerate(labels_rows):
        n = min(len(row), mlen)
        if n:
            full_labels[i, :n] = torch.tensor(row[:n], dtype=torch.long)
    return {
        "full_ids": full_ids,
        "full_mask": full_mask,
        "full_labels": full_labels,
        "left_ids": left_ids,
        "left_mask": left_mask,
        "right_ids": right_ids,
        "right_mask": right_mask,
        "full_len": torch.tensor([len(ex["full_ids"]) for ex in batch], dtype=torch.long),
        "left_len": torch.tensor([len(ex["left_ids"]) for ex in batch], dtype=torch.long),
        "right_len": torch.tensor([len(ex["right_ids"]) for ex in batch], dtype=torch.long),
        "o_text": [ex.get("o_text", "") for ex in batch],
        "left_text": [ex.get("left_text", "") for ex in batch],
    }


def collate_mediated(batch: list[dict[str, Any]], pad_id: int, pad_multiple: int = 1) -> dict[str, Any]:
    """Pad \(z_t=\mathrm{Enc}(h)\) / \(z_{t+1}=\mathrm{Enc}(h,a,o)\). No history strings."""
    import torch

    def pad_rows(rows: list[list[int]], fill: int) -> tuple[torch.Tensor, torch.Tensor]:
        mlen = _pad_len(max((len(x) for x in rows), default=1), pad_multiple)
        mlen = max(mlen, 1)
        out = torch.full((len(rows), mlen), fill, dtype=torch.long)
        mask = torch.zeros((len(rows), mlen), dtype=torch.long)
        for i, row in enumerate(rows):
            if not row:
                continue
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return out, mask

    state_ids, state_mask = pad_rows([ex["state_ids"] for ex in batch], pad_id)
    next_ids, next_mask = pad_rows([ex["next_ids"] for ex in batch], pad_id)
    return {
        "state_ids": state_ids,
        "state_mask": state_mask,
        "next_ids": next_ids,
        "next_mask": next_mask,
        "state_len": torch.tensor([len(ex["state_ids"]) for ex in batch], dtype=torch.long),
        "next_len": torch.tensor([len(ex["next_ids"]) for ex in batch], dtype=torch.long),
        "a_text": [ex.get("a_text", "") for ex in batch],
        "o_text": [ex.get("o_text", "") for ex in batch],
    }


def all_gather_seq(x, group=None):
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return x
    world = dist.get_world_size(group)
    if world <= 1:
        return x
    parts = [x.new_empty(x.shape) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=group)
    return __import__("torch").cat(parts, dim=1)


def gather_hidden(h, attention_mask, cp_size: int = 1):
    """CP shards seq across ranks — gather to full length before indexing."""
    if cp_size > 1 and h.size(1) < attention_mask.size(1):
        return all_gather_seq(h)
    if cp_size > 1 and h.size(1) * cp_size == attention_mask.size(1):
        return all_gather_seq(h)
    return h


def full_hidden(model, input_ids, attention_mask, cp_size: int = 1):
    """Full last-layer hidden [B, S, D]. Forward is hidden-only (no full-seq logits)."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hs = getattr(out, "hidden_states", None)
    if hs:
        h = hs[-1]
    else:
        h = getattr(out, "last_hidden_state", None)
    if h is None:
        raise RuntimeError(
            f"{type(out).__name__} has no hidden_states; "
            f"fields={getattr(out, '__dataclass_fields__', {})}"
        )
    return gather_hidden(h, attention_mask, cp_size)


def last_token_index(input_ids, attention_mask, last_token: int):
    """llm-jepa RepresentationTrainer._last_token_index (right padding)."""
    import torch

    index = []
    seqs = input_ids.tolist()
    masks = attention_mask.tolist()
    max_i = input_ids.size(1) - 1
    for ids, mask in zip(seqs, masks):
        unpadded = []
        seen = False
        for tid, m in zip(ids, mask):
            if m != 0:
                seen = True
            if m == 0 and seen:
                break
            unpadded.append(tid)
        index.append(min(max(len(unpadded) + last_token, 0), max_i))
    return torch.tensor(index, device=input_ids.device, dtype=torch.long)


def gather_at(hidden, index):
    import torch

    idx = index.clamp(min=0, max=hidden.size(1) - 1)
    b = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[b, idx]


def jepa_cosine(user_embedding, assistant_embedding):
    """llm-jepa default: 1 - mean(cosine). Both sides live."""
    import torch
    import torch.nn.functional as F

    cosine_similarity = F.cosine_similarity(user_embedding, assistant_embedding, dim=-1)
    return 1.0 - torch.mean(cosine_similarity)


# AgentWorld vocab is 248320. A 25k-token observation as one [N, V] fp32
# logits tensor is ~23 GiB and OOMs a 95 GB card that already holds the model.
# Chunk so peak logits stay under ~1 GiB regardless of how long o is.
CE_LOGIT_CHUNK = 512


def shifted_ce(hidden, labels, lm_head, chunk_size: int = CE_LOGIT_CHUNK):
    """HF CausalLM shift: hidden[t] predicts labels[t+1]. lm_head only on labeled rows.

    Prefix-turn trim can leave a complete multi-ten-thousand-token observation
    inside the window. Materializing that many vocab logits at once OOMs;
    this is mean-token CE over chunks, same value as one shot.
    """
    import torch
    import torch.nn.functional as F

    pred = hidden[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    if not bool(mask.any()):
        return hidden.new_zeros(())
    h = pred[mask]
    y = tgt[mask]
    n = h.size(0)
    step = max(int(chunk_size), 1)
    total = h.new_zeros(())
    for i in range(0, n, step):
        logits = lm_head(h[i : i + step])
        total = total + F.cross_entropy(logits.float(), y[i : i + step], reduction="sum")
    return total / n


def force_attn_implementation(model, impl: str) -> None:
    configs = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        configs.append(cfg)
        tc = getattr(cfg, "text_config", None)
        if tc is not None:
            configs.append(tc)
    base = getattr(model, "get_base_model", lambda: None)()
    if base is not None and hasattr(base, "config"):
        configs.append(base.config)
        tc = getattr(base.config, "text_config", None)
        if tc is not None:
            configs.append(tc)
    for c in configs:
        try:
            c._attn_implementation = impl
        except Exception:
            pass
    for path in (
        lambda: model.get_base_model().model.language_model,
        lambda: model.model.language_model,
        lambda: model.language_model,
    ):
        try:
            lm = path()
            if hasattr(lm, "config"):
                lm.config._attn_implementation = impl
        except Exception:
            continue
    log(f"attn_implementation={impl}")


def load_backbone(model_dir: Path, dtype, checkpointing: bool, *, attn_implementation=None, distributed=False):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # Card says 131k; we count full chat then _fit to 65k. Silence the false "indexing errors" warn.
    tok.model_max_length = int(1e12)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
        log(f"load attn_implementation={attn_implementation}")
    model = None
    err = None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = loader.from_pretrained(str(model_dir), **kwargs)
            break
        except TypeError:
            kwargs.pop("dtype", None)
            try:
                model = loader.from_pretrained(str(model_dir), **kwargs)
                break
            except Exception as e:
                err = e
        except Exception as e:
            err = e
    if model is None:
        raise SystemExit(f"failed to load {model_dir}: {err}")
    if attn_implementation:
        force_attn_implementation(model, attn_implementation)
    if checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tok


def open_tb(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))
    log(f"tensorboard → {log_dir}  (tensorboard --logdir {log_dir.parent})")
    return writer


def resolve_tb_dir(tcfg: dict[str, Any], _out_dir: Path, cli: Path | None, run_tag: str = "jepa") -> Path:
    """AutoDL's TensorBoard panel watches ``/root/tf-logs`` (same as Muse / eval).

    One writer, one run, opened by global main only — even with 2
    dp_replicate groups. See `merge_group_stats` for why: both groups' losses
    get all-reduced into a single combined number before anything is logged,
    so there is exactly one curve representing the whole job, not one curve
    per group.
    """
    raw = (
        cli
        or os.environ.get("LOGGING_DIR")
        or os.environ.get("TF_LOGS")
        or tcfg.get("logging_dir")
        or "/root/tf-logs"
    )
    root = Path(str(raw))
    if not root.is_absolute():
        root = _resolve(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return root / f"{run_tag}-{stamp}"


def merge_group_stats(accelerator, *sums: float) -> tuple[float, ...]:
    """All-reduce (sum) windowed accumulators across every rank, then return
    them unchanged in shape — the caller still divides by its own local
    `n_loss`-style denominator, also passed through this same call.

    Why summing (not averaging) is the right op here: ranks inside one CP
    group hold identical local values (CP all-gathers hidden states before
    the loss, so the loss itself isn't split). So the global sum equals
    `cp_size * (group0_sum + group1_sum)`, and the global sum of the
    denominator (e.g. n_loss) equals `cp_size * (group0_n + group1_n)`. Taking
    the *ratio* of the two global sums cancels the `cp_size` factor and lands
    exactly on the combined average across both dp_replicate groups — no
    separate "divide by dp_size" step needed, and it's correct regardless of
    cp_size. This is the actual communication: one small collective per log
    point (a handful of floats), so one TensorBoard record represents the
    whole job's data, not just whichever rank happens to be global main.
    """
    if accelerator.num_processes <= 1:
        return sums
    import torch

    t = torch.tensor(sums, device=accelerator.device, dtype=torch.float64)
    t = accelerator.reduce(t, reduction="sum")
    return tuple(t.tolist())


def resolve_cp_size(cli: int | None) -> int:
    if cli is not None and int(cli) > 0:
        return int(cli)
    return int(os.environ.get("BIV_CP_SIZE") or os.environ.get("PARALLELISM_CONFIG_CP_SIZE") or "1")


def build_parallelism_config():
    """Explicit ParallelismConfig, only to route around one overly-blunt guard.

    accelerate's ParallelismConfig.__post_init__ hard-blocks dp_replicate>1 +
    cp>1 with dp_shard_size==1 ("pure DP + CP"), because in general that would
    mean literal DDP composed with TP/CP, which they don't support. But CP
    doesn't need that guard: accelerate's own ParallelismConfig.fsdp_dim_names
    (consumed verbatim by fsdp2_prepare_model as `mesh[fsdp_dim_names]` passed
    to torch's `fully_shard`) folds cp into a `dp_shard_cp` joint dim whenever
    cp_enabled, even with dp_shard disabled — that's exactly the mechanism our
    already-working single-group config (dp_shard=1, cp=NGPU, dp_replicate=1)
    relies on. Adding dp_replicate>1 on top just makes fsdp_dim_names =
    ("dp_replicate", "dp_shard_cp"), a plain 2D (replicate, shard) mesh that
    torch's fully_shard natively treats as HSDP — grad sync across the
    replicate dim is automatic, no manual all_reduce needed.

    So: construct with a throwaway dp_shard_size=2 to satisfy __post_init__'s
    check, then patch it back to the real value (1) — __post_init__ only runs
    at construction, total_size/fsdp_dim_names/_sizes are live-attribute
    properties, unaffected by the later mutation. Untested off-GPU; if this is
    wrong, torch's fully_shard should fail loudly (wrong mesh rank/shape), not
    silently train wrong.

    Returns None when dp_replicate isn't in play, so Accelerator() falls back
    to its normal env-var-driven construction — single-group CP (this file's
    original, verified path) is untouched.
    """
    from accelerate.utils import ParallelismConfig

    dp_replicate = int(os.environ.get("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1"))
    if dp_replicate <= 1:
        return None
    dp_shard = int(os.environ.get("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1"))
    cp = int(os.environ.get("PARALLELISM_CONFIG_CP_SIZE", "1"))
    tp = int(os.environ.get("PARALLELISM_CONFIG_TP_SIZE", "1"))
    cp_backend = os.environ.get("PARALLELISM_CONFIG_CP_BACKEND", "torch")
    needs_hack = dp_shard <= 1 and cp > 1 and tp <= 1
    pc = ParallelismConfig(
        dp_replicate_size=dp_replicate,
        dp_shard_size=2 if needs_hack else dp_shard,
        tp_size=tp,
        cp_size=cp,
        cp_backend=cp_backend,
    )
    if needs_hack:
        pc.dp_shard_size = dp_shard
        pc._sizes["dp_shard"] = dp_shard
    return pc


def dp_replicate_info(accelerator, cp_size: int) -> tuple[int, int]:
    """(dp_rank, dp_size) — which data-parallel replicate group this rank is in.

    2 CP groups of 2 GPUs each (32k smoke) means dp_size=2: group 0 = ranks
    0-1, group 1 = ranks 2-3. Prefers accelerate's actual device mesh (correct
    regardless of dim ordering); falls back to rank // cp_size, which matches
    accelerate's (dp_replicate, dp_shard, cp, tp) mesh convention. Untested
    off-GPU — first run should log and eyeball this before trusting it.
    """
    mesh = getattr(accelerator, "torch_device_mesh", None)
    try:
        if mesh is not None and "dp_replicate" in getattr(mesh, "mesh_dim_names", ()):
            sub = mesh["dp_replicate"]
            return int(sub.get_local_rank()), int(sub.size())
    except Exception:
        pass
    nproc = accelerator.num_processes
    if cp_size > 0 and nproc % cp_size == 0:
        dp_size = nproc // cp_size
        if dp_size > 1:
            return accelerator.process_index // cp_size, dp_size
    return 0, 1


class ReplicaSampler:
    """Same global shuffle order on every rank; each dp_replicate group takes
    a disjoint interleaved slice, so the two groups train on different data
    (real throughput gain, not redundant compute). Ranks inside one CP group
    share dp_rank, so they get the identical slice — required, since CP
    splits one sample's sequence across those ranks, not the batch.

    Truncates to `(n // dp_size) * dp_size` *before* slicing by dp_rank, so
    every replicate group gets exactly the same number of rows per epoch —
    drops at most `dp_size - 1` rows/epoch, but guarantees identical
    `len(loader)` (hence identical steps_per_epoch) across groups. Without
    this, `n % dp_size != 0` can leave one group's DataLoader one micro-batch
    short; since the two groups' gradients are all-reduced every accumulation
    boundary (a blocking collective), whichever group's loop exits its epoch
    first would leave the other group's rank waiting on a collective call
    that never comes — an NCCL hang, not a crash. This is what keeps
    save_steps/log_steps from drifting between the two groups: they are the
    same integer `step` on every rank, computed off equal-length loaders, not
    negotiated at runtime."""

    def __init__(self, n: int, dp_rank: int, dp_size: int, generator) -> None:
        self.n = n
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.generator = generator

    def __iter__(self):
        import torch

        order = torch.randperm(self.n, generator=self.generator).tolist()
        usable = (self.n // self.dp_size) * self.dp_size
        return iter(order[:usable][self.dp_rank :: self.dp_size])

    def __len__(self) -> int:
        return self.n // self.dp_size


def assert_equal_loader_len(accelerator, local_len: int) -> None:
    """Belt-and-suspenders check: every rank's local `len(loader)` must match.

    ReplicaSampler already guarantees this by construction (equal split before
    slicing), so this should never fire. It exists because a hang from
    mismatched loader lengths (see ReplicaSampler docstring) is silent and
    happens much later — mid-epoch, at whatever step one group's DataLoader
    runs dry — which is a miserable thing to debug on a live 4-GPU job. This
    turns that into a loud, immediate, pre-training error naming the actual
    per-rank lengths, using one all_gather (negligible cost, runs once).
    """
    if accelerator.num_processes <= 1:
        return
    import torch.distributed as dist

    lens = [None] * accelerator.num_processes
    dist.all_gather_object(lens, int(local_len))
    if len(set(lens)) > 1:
        raise SystemExit(
            f"loader length mismatch across ranks: {lens} "
            f"(rank {accelerator.process_index}={local_len}) — ReplicaSampler "
            "should make these identical; something upstream changed n or "
            "dp_size per-rank. Fix before training: mismatched lengths hang "
            "mid-epoch at the accumulation boundary, not at startup."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model-dir", type=str, default=None, help="AgentWorld hub id or local dir")
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Override yaml save_steps (default 25).",
    )
    p.add_argument(
        "--log-steps",
        type=int,
        default=None,
        help="Loss + collapse log every N optimizer steps (default 5). Not save.",
    )
    p.add_argument(
        "--collapse-steps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume weights. Bare --resume picks the newest ckpt (epoch, then step).",
    )
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None, help="Context-parallel size (Ring Attention).")
    p.add_argument("--cache-dir", type=Path, default=None, help="Default: merge/output/cache")
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
        help="Where to download AgentWorld from if not already cached.",
    )
    p.add_argument(
        "--logging-dir",
        type=Path,
        default=None,
        help="TensorBoard root (default: $LOGGING_DIR or $TF_LOGS or /root/tf-logs)",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default="jepa",
        help="Console/TensorBoard prefix (default jepa). 2x2 and single-group "
        "runs share this tag unless you override it.",
    )
    return p.parse_args()


def _full_cpu(param):
    """Materialize a (possibly DTensor) param on CPU. Collective if DTensor."""
    t = param.full_tensor() if hasattr(param, "full_tensor") else param
    t = t.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    return t.contiguous().clone()


def gather_lora_cpu(model, *, keep: bool) -> dict:
    """All ranks must call this: DTensor.full_tensor is a collective.

    Only LoRA tensors are gathered (full 35B gather OOMs). Cloned onto CPU so
    safetensors never sees an FSDP/DTensor storage pointer — that was the
    ``invalid python storage`` crash from ``PeftModel.save_pretrained``.
    """
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        full = _full_cpu(param)
        if keep:
            out[canonical_lora_key(name)] = full
        del full
    return out


def save_adapter(unwrapped, lora_cpu: dict, path: Path) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    if not lora_cpu:
        raise RuntimeError("no LoRA tensors gathered; refuse to write an empty adapter")
    save_file(lora_cpu, str(path / "adapter_model.safetensors"))
    adapter = getattr(unwrapped, "active_adapter", "default")
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0] if adapter else "default"
    cfg_map = getattr(unwrapped, "peft_config", None) or {}
    cfg = cfg_map.get(adapter) if isinstance(cfg_map, dict) else None
    if cfg is not None and hasattr(cfg, "save_pretrained"):
        cfg.save_pretrained(str(path))



def save_ckpt(
    accelerator,
    model,
    tokenizer,
    path: Path,
    extra: dict | None = None,
    *,
    epoch: int,
    step: int,
) -> None:
    accelerator.wait_for_everyone()
    lora_cpu = gather_lora_cpu(model, keep=accelerator.is_main_process)
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        save_adapter(unwrapped, lora_cpu, path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        meta = dict(extra or {})
        write_trainer_state(path, epoch=epoch, global_step=step, extra=meta)
        (path / "train_meta.json").write_text(
            json.dumps({"epoch": epoch, "global_step": step, **meta}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        log(f"saved {path}")
    accelerator.wait_for_everyone()


def load_resume_dir(out_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    if raw == "auto":
        found = find_latest_ckpt(out_dir, require_jepa=False)
        if found is None:
            raise SystemExit(f"--resume: no complete checkpoint under {out_dir}")
        return found
    pth = Path(raw)
    if not pth.is_absolute():
        pth = _resolve(pth)
    if (
        pth.is_dir()
        and (pth / "trainer_state.json").is_file()
        and (pth / "adapter_model.safetensors").is_file()
    ):
        return pth
    raise SystemExit(f"--resume path is not a checkpoint: {pth}")


def load_lora_into_model(model, sd: dict, log_fn) -> None:
    """Copy LoRA tensors by name. Skip PEFT's MoE WeightConverter (PEFT/transformers skew)."""
    live = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if not live:
        raise SystemExit("resume: model has no LoRA parameters")
    by_canon = {canonical_lora_key(n): n for n in live}
    mapped: dict[str, Any] = {}
    unmatched = 0
    shape_bad: list[str] = []
    for k, v in sd.items():
        if "lora_" not in k:
            continue
        dest = by_canon.get(canonical_lora_key(k))
        if dest is None:
            unmatched += 1
            continue
        if tuple(live[dest].shape) != tuple(v.shape):
            shape_bad.append(f"{dest} file={tuple(v.shape)} model={tuple(live[dest].shape)}")
            continue
        mapped[dest] = v
    if shape_bad:
        raise SystemExit("resume LoRA shape mismatch: " + "; ".join(shape_bad[:8]))
    if not mapped:
        raise SystemExit(
            f"resume: 0 LoRA tensors matched (file_keys={len(sd)} model_lora={len(live)})"
        )
    model.load_state_dict(mapped, strict=False)
    log_fn(
        f"resume adapter matched={len(mapped)}/{len(live)} "
        f"unmatched_saved={unmatched}"
    )


def load_adapter(model, path: Path, log_fn) -> tuple[int, int]:
    from safetensors.torch import load_file

    adapter = path / "adapter_model.safetensors"
    if adapter.is_file():
        load_lora_into_model(model, load_file(str(adapter)), log_fn)
    else:
        raise SystemExit(f"resume: missing {adapter}")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state.get("epoch") or 0), int(state.get("global_step") or 0)


def decode_ids_texts(tokenizer, ids, mask) -> list[str]:
    out: list[str] = []
    for i in range(ids.size(0)):
        tok = ids[i][mask[i].bool()].tolist()
        out.append(tokenizer.decode(tok, skip_special_tokens=True))
    return out


def decode_o_texts(tokenizer, batch) -> list[str]:
    return decode_ids_texts(tokenizer, batch["right_ids"], batch["right_mask"])


def _warmup_lambda(warmup: int):
    def fn(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup))

    return fn


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag

    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
    from biv_wm.arch import install_hidden_only_forward, lm_head_module, log_world_architecture
    from biv_wm.jepa import collapse_stats, format_collapse_line

    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        alt = ROOT / args.config
        if alt.is_file():
            cfg_path = alt
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {args.config} (tried {TRAIN / args.config})")
    cfg = _load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    accum = int(tcfg.get("grad_accum") or 8)
    cp_size = resolve_cp_size(args.cp_size)
    if cp_size > 1:
        os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
        os.environ.setdefault("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_TP_SIZE", "1")
        os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cp_size)
        os.environ.setdefault("PARALLELISM_CONFIG_CP_BACKEND", "torch")

    accelerator = Accelerator(
        gradient_accumulation_steps=accum, parallelism_config=build_parallelism_config()
    )
    is_main = accelerator.is_main_process

    def rank_log(msg: str) -> None:
        if is_main:
            log(msg)

    cache_dir = _resolve(args.cache_dir or "merge/output/cache", ROOT)
    model_dir = resolve_model(
        str(args.model_dir or cfg["model_dir"]),
        source=args.source,
        cache_dir=cache_dir,
        role="world",
    )
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    out_dir = _resolve(tcfg.get("output_dir") or "outputs/jepa_stage1")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)

    max_length = int(args.max_length or tcfg.get("max_length") or 32768)
    pad_multiple = cp_size * 2 if cp_size > 1 else int(tcfg.get("pad_to_multiple_of") or 0)
    attn_impl = "sdpa" if cp_size > 1 else None
    distributed = accelerator.num_processes > 1
    rank_log(f"model={model_dir}")
    rank_log(f"mix={mix_dir} sources={sources}")
    rank_log(
        f"max_length={max_length} cp_size={cp_size} pad_multiple={pad_multiple} "
        f"ce_logit_chunk={CE_LOGIT_CHUNK} nproc={accelerator.num_processes} attn={attn_impl}"
    )

    model, tokenizer = load_backbone(
        model_dir,
        dtype,
        bool(tcfg.get("gradient_checkpointing", True)),
        attn_implementation=attn_impl,
        distributed=distributed,
    )
    for name, p in model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False

    suffixes = list(tcfg.get("target_modules") or [])
    targets = two_d_lora_targets(model, suffixes)
    if not targets:
        raise SystemExit(f"no LoRA targets matching {suffixes} in AgentWorld backbone")
    rank_log(f"lora 2D targets (full AgentWorld backbone, no cut)={len(targets)}")
    lora = LoraConfig(
        r=int(tcfg.get("lora_rank") or 16),
        lora_alpha=int(tcfg.get("lora_alpha") or 32),
        lora_dropout=float(tcfg.get("lora_dropout") or 0.05),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if is_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    install_hidden_only_forward(model, detach_head=False)
    resume_dir = load_resume_dir(out_dir, args.resume)
    resume_epoch, resume_step = 0, 0
    if resume_dir is not None:
        resume_epoch, resume_step = load_adapter(model, resume_dir, rank_log)
        rank_log(
            f"resume {resume_dir.name} trainer_state epoch={resume_epoch} "
            f"global_step={resume_step} (newest by epoch, then step)"
        )
    if is_main:
        log_world_architecture(
            model=model,
            extra={},
            model_dir=model_dir,
            log=log,
            expect_lm_head="attached",
        )

    train_rows = load_rows(mix_dir, sources, "train", tcfg.get("max_train_samples"))
    if not train_rows:
        raise SystemExit(f"no mix rows under {mix_dir}/{sources}/train.jsonl")
    rank_log(f"train_rows={len(train_rows)}")

    train_ds = HaoDataset(train_rows, tokenizer, max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch_size = int(tcfg.get("batch_size") or 1)
    dp_rank, dp_size = dp_replicate_info(accelerator, cp_size)
    log(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"dp_replicate rank={dp_rank}/{dp_size} cp_size={cp_size} — "
        f"first run: check dp_rank groups match {{0,1}} and {{2,3}} (or your GPU order), "
        f"not all-0 or all-different."
    )
    if dp_size > 1:
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=ReplicaSampler(len(train_ds), dp_rank, dp_size, gen),
            collate_fn=lambda b: collate(b, pad_id, pad_multiple),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
            collate_fn=lambda b: collate(b, pad_id, pad_multiple),
            num_workers=0,
        )
    assert_equal_loader_len(accelerator, len(loader))

    backbone_lr = float(tcfg.get("lr") or 5e-5)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=backbone_lr,
        weight_decay=float(tcfg.get("weight_decay") or 0.01),
    )
    model, opt = accelerator.prepare(model, opt)
    lm_head = lm_head_module(accelerator.unwrap_model(model))
    if lm_head is None:
        raise SystemExit("LLM-JEPA Stage 1 needs AgentWorld lm_head attached")

    max_norm = float(tcfg.get("max_grad_norm") or 1.0)
    log_every = int(
        args.log_steps
        if args.log_steps is not None
        else (
            args.collapse_steps
            if args.collapse_steps is not None
            else (tcfg.get("log_steps") or tcfg.get("logging_steps") or 5)
        )
    )
    save_every = int(
        args.save_steps if args.save_steps is not None else (tcfg.get("save_steps") or 25)
    )
    if save_every < 1:
        raise SystemExit(f"save_steps must be >= 1, got {save_every}")
    if log_every < 1:
        raise SystemExit(f"log_steps must be >= 1, got {log_every}")
    save_limit = int(tcfg.get("save_total_limit") or 3)
    gamma = float(
        tcfg["gamma"]
        if tcfg.get("gamma") is not None
        else (tcfg["ce_weight"] if tcfg.get("ce_weight") is not None else 1.0)
    )
    lbd = float(
        tcfg["lbd"]
        if tcfg.get("lbd") is not None
        else (tcfg["jepa_weight"] if tcfg.get("jepa_weight") is not None else 0.1)
    )
    last_token = int(tcfg["last_token"] if tcfg.get("last_token") is not None else -3)
    epochs = int(tcfg.get("num_epochs") or 2)
    max_steps = args.max_steps
    steps_per_epoch = math.ceil(len(loader) / accum)
    planned = steps_per_epoch * epochs
    total_opt = planned if max_steps is None else min(planned, int(max_steps))
    warmup = int(tcfg.get("warmup_steps") or 50)
    warmup = max(0, min(warmup, max(total_opt - 1, 0)))
    # last_epoch = steps already done. Do not sched.step() here: that warns and
    # skips the first warmup value because optimizer has not stepped yet.
    sched = LambdaLR(
        opt,
        _warmup_lambda(warmup),
        last_epoch=(resume_step - 1) if resume_step > 0 else -1,
    )
    rank_log(
        f"epochs={epochs} steps_per_epoch≈{steps_per_epoch} accum={accum} "
        f"lr_lora={backbone_lr} warmup={warmup} "
        f"save_steps={save_every} log_steps={log_every} "
        f"gamma={gamma} lbd={lbd} last_token={last_token} "
        f"save_total_limit={save_limit} resume_step={resume_step}"
    )

    ckpt_extra = {
        "model_dir": str(model_dir),
        "mix_dir": str(mix_dir),
        "sources": sources,
        "max_length": max_length,
        "cp_size": cp_size,
        "dp_replicate_size": dp_size,
        "run_tag": run_tag,
        "lm_head": "attached_frozen_base",
        "recipe": "llm-jepa RepresentationTrainer (independent Enc + last_token cosine + shifted CE)",
        "last_token": last_token,
        "backbone": "AgentWorld only, no fish-cut, no Instruct tail",
    }
    seen_pred: deque = deque()
    seen_z: deque = deque()
    seen_o: deque = deque()
    writer = None

    def emit(msg: str) -> None:
        if is_main:
            try:
                from tqdm.auto import tqdm as _tqdm

                _tqdm.write(msg)
            except Exception:
                log(msg)

    def check_collapse(step_i: int) -> None:
        def _clear() -> None:
            seen_pred.clear()
            seen_z.clear()
            seen_o.clear()

        if not seen_pred:
            _clear()
            return
        stats = collapse_stats(
            torch.stack(list(seen_pred)),
            torch.stack(list(seen_z)),
            list(seen_o),
        )
        if not is_main:
            _clear()
            return
        emit(f"[{run_tag}] collapse step={step_i} {format_collapse_line(stats)}")
        payload = json.dumps(dict(stats), indent=2, ensure_ascii=False) + "\n"
        (out_dir / "collapse.json").write_text(payload, encoding="utf-8")
        (out_dir / f"collapse-s{step_i}.json").write_text(payload, encoding="utf-8")
        if writer is not None:
            def _sc(name: str, block: object, key: str) -> None:
                if isinstance(block, dict) and block.get(key) is not None:
                    writer.add_scalar(name, float(block[key]), step_i)

            paired = stats["paired"]
            mis = stats["mismatch"]
            z_self = stats["z_self"]
            pred_self = stats["pred_self"]
            writer.add_scalar("collapse/n", float(stats["n"]), step_i)
            writer.add_scalar("collapse/skipped_same_o", float(stats["skipped_same_o"]), step_i)
            _sc("collapse/paired_median", paired, "median")
            _sc("collapse/paired_mean", paired, "mean")
            _sc("collapse/mismatch_median", mis, "median")
            _sc("collapse/mismatch_p90", mis, "p90")
            _sc("collapse/z_self_median", z_self, "median")
            _sc("collapse/pred_self_median", pred_self, "median")
            writer.add_text("collapse/verdict", str(stats.get("verdict")), step_i)
            writer.flush()
        _clear()

    def dump_ckpt(kind: str, epoch_i: int, step_i: int) -> None:
        extra = {**ckpt_extra, "kind": kind}
        if kind == "epoch-end":
            dest = out_dir / epoch_end_name(epoch_i, step_i)
        else:
            dest = out_dir / rolling_name(epoch_i, step_i)
        save_ckpt(
            accelerator,
            model,
            tokenizer,
            dest,
            extra=extra,
            epoch=epoch_i,
            step=step_i,
        )
        if is_main:
            emit(f"[{run_tag}] checkpoint ({kind}) → {dest.name}")
            rotate_rolling(out_dir, save_limit, log=lambda m: emit(f"[{run_tag}] {m}"))

    tb_dir = resolve_tb_dir(tcfg, out_dir, args.logging_dir, run_tag)
    ckpt_extra["tensorboard"] = str(tb_dir)
    if is_main:
        writer = open_tb(tb_dir)
        writer.add_text("data/mix_dir", str(mix_dir), 0)
        writer.add_text("data/sources", ", ".join(sources), 0)
        writer.add_text("train/max_length", str(max_length), 0)
        writer.add_text("train/cp_size", str(cp_size), 0)
        writer.add_text("train/dp_replicate_size", str(dp_size), 0)
        writer.add_text(
            "train/step_semantics",
            "x-axis is step*dp_replicate_size (single-group-step-equivalent), "
            "loss is all-reduced across both dp_replicate groups — one record "
            "for the whole job, not one per group.",
            0,
        )
        writer.add_text("train/run_tag", run_tag, 0)
        writer.add_text("train/epochs", str(epochs), 0)
        writer.add_text("train/save_steps", str(save_every), 0)
        writer.add_text("train/log_steps", str(log_every), 0)
        writer.add_text("train/gamma", str(gamma), 0)
        writer.add_text("train/lbd", str(lbd), 0)
        writer.add_text("train/last_token", str(last_token), 0)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None  # type: ignore[assignment]

    pbar = None
    if is_main and tqdm is not None:
        pbar = tqdm(
            total=total_opt,
            initial=min(resume_step, total_opt),
            desc=run_tag,
            unit="step",
            dynamic_ncols=True,
        )

    if resume_step >= total_opt:
        rank_log(f"resume_step={resume_step} already covers total_opt={total_opt}; nothing to do")
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.close()
        return

    model.train()
    step = resume_step
    skip_micro = resume_step * accum
    micro_seen = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    run_ce = 0.0
    run_jepa = 0.0
    n_loss = 0
    run_h = run_a = run_o = 0.0
    run_nlab = 0.0
    last_train_loss = None
    hit_max = False
    last_log_time = time.monotonic()
    throughput_postfix: dict[str, str] = {}

    try:
        for epoch in range(epochs):
            epoch_base = epoch * steps_per_epoch
            epoch_opt = max(0, min(resume_step - epoch_base, steps_per_epoch))
            for batch in loader:
                if micro_seen < skip_micro:
                    micro_seen += 1
                    continue
                micro_seen += 1
                batch = {
                    k: v.to(accelerator.device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                full_len = float(batch["full_len"].float().mean().item())
                left_len = float(batch["left_len"].float().mean().item())
                right_len = float(batch["right_len"].float().mean().item())
                n_lab = float((batch["full_labels"] != -100).sum().item())
                with accelerator.accumulate(model):
                    h_full = full_hidden(
                        model, batch["full_ids"], batch["full_mask"], cp_size
                    )
                    ce_loss = shifted_ce(h_full, batch["full_labels"], lm_head)
                    h_left_seq = full_hidden(
                        model, batch["left_ids"], batch["left_mask"], cp_size
                    )
                    h_right_seq = full_hidden(
                        model, batch["right_ids"], batch["right_mask"], cp_size
                    )
                    idx_l = last_token_index(
                        batch["left_ids"], batch["left_mask"], last_token
                    )
                    idx_r = last_token_index(
                        batch["right_ids"], batch["right_mask"], last_token
                    )
                    h_left = gather_at(h_left_seq, idx_l)
                    h_right = gather_at(h_right_seq, idx_r)
                    jepa_loss = jepa_cosine(h_left, h_right)
                    loss = gamma * ce_loss + lbd * jepa_loss
                    o_texts = batch.get("o_text") or decode_o_texts(tokenizer, batch)
                    for i in range(h_left.size(0)):
                        seen_pred.append(h_left[i].detach().float().cpu())
                        seen_z.append(h_right[i].detach().float().cpu())
                        seen_o.append(o_texts[i])
                    accelerator.backward(loss)
                    running += float(loss.detach().float().item())
                    run_ce += float(ce_loss.detach().float().item())
                    run_jepa += float(jepa_loss.detach().float().item())
                    n_loss += 1
                    run_h += full_len
                    run_a += left_len
                    run_o += right_len
                    run_nlab += n_lab
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm)
                        opt.step()
                        sched.step()
                        opt.zero_grad(set_to_none=True)
                        step += 1
                        epoch_opt += 1
                        last_train_loss = running / max(n_loss, 1)
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(
                                epoch=f"{epoch + 1}/{epochs}",
                                loss=f"{last_train_loss:.4f}",
                                refresh=False,
                                **throughput_postfix,
                            )
                        is_last_in_epoch = epoch_opt >= steps_per_epoch
                        if step % log_every == 0:
                            # Collective: every rank must call this (it's an
                            # all-reduce), not just is_main — see
                            # merge_group_stats docstring for why summing (not
                            # averaging) both groups' windowed accumulators and
                            # then taking a ratio gives the correct combined
                            # average regardless of cp_size.
                            running_g, ce_g, jepa_g, h_g, a_g, o_g, nlab_g, n_g = merge_group_stats(
                                accelerator,
                                running,
                                run_ce,
                                run_jepa,
                                run_h,
                                run_a,
                                run_o,
                                run_nlab,
                                float(n_loss),
                            )
                            denom_g = max(n_g, 1.0)
                            # step*dp_size = single-group-step-equivalent — lines
                            # this run's x-axis up with a non-parallel run's step
                            # count (see AGENTS.md), so this is the number that
                            # "represents having trained step*dp_size steps", not
                            # the raw (per-group) step counter. Used for every
                            # scalar in this run, including collapse/*, so all
                            # tags in one TensorBoard run share the same x-axis.
                            eff_step = step * dp_size
                            if is_main:
                                loss_g = running_g / denom_g
                                now = time.monotonic()
                                elapsed = max(now - last_log_time, 1e-6)
                                rows = dp_size * accum * batch_size * log_every
                                toks = rows * (h_g / denom_g)
                                rows_s = rows / elapsed
                                toks_s = toks / elapsed
                                last_log_time = now
                                throughput_postfix = {"rows/s": f"{rows_s:.2f}", "tok/s": f"{toks_s:.0f}"}
                                emit(
                                    f"epoch={epoch} step={step} eff_step={eff_step} loss={loss_g:.4f} "
                                    f"ce={ce_g / denom_g:.4f} jepa={jepa_g / denom_g:.4f} "
                                    f"len(full/left/right)={h_g / denom_g:.0f}/{a_g / denom_g:.0f}/{o_g / denom_g:.0f} "
                                    f"n_ce={nlab_g / denom_g:.0f} "
                                    f"throughput≈{rows_s:.2f} rows/s {toks_s:.0f} tok/s "
                                    f"(dp_size={dp_size} groups combined)"
                                )
                                if writer is not None:
                                    writer.add_scalar("train/loss", loss_g, eff_step)
                                    writer.add_scalar("train/loss_ce", ce_g / denom_g, eff_step)
                                    writer.add_scalar("train/loss_jepa", jepa_g / denom_g, eff_step)
                                    writer.add_scalar("train/lr_backbone", opt.param_groups[0]["lr"], eff_step)
                                    writer.add_scalar("train/len_h", h_g / denom_g, eff_step)
                                    writer.add_scalar("train/len_a", a_g / denom_g, eff_step)
                                    writer.add_scalar("train/len_o", o_g / denom_g, eff_step)
                                    writer.add_scalar("train/n_ce", nlab_g / denom_g, eff_step)
                                    writer.add_scalar("train/rows_per_sec_global", rows_s, eff_step)
                                    writer.add_scalar("train/tokens_per_sec_global", toks_s, eff_step)
                            running = 0.0
                            run_ce = 0.0
                            run_jepa = 0.0
                            n_loss = 0
                            run_h = run_a = run_o = 0.0
                            run_nlab = 0.0
                            # Unconditional: check_collapse() clears seen_pred/
                            # seen_z/seen_o on every rank internally (only
                            # is_main computes+logs stats) — skipping this on
                            # non-main ranks would leak those deques forever.
                            check_collapse(eff_step)
                        if is_last_in_epoch:
                            emit(
                                f"[{run_tag}] epoch {epoch + 1} end: force checkpoint "
                                "(permanent epoch ckpt)"
                            )
                            dump_ckpt("epoch-end", epoch + 1, step)
                        elif step % save_every == 0:
                            dump_ckpt("rolling", epoch, step)
                        if max_steps is not None and step >= max_steps:
                            hit_max = True
                            break
            if hit_max:
                break
    finally:
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.flush()
            writer.close()

    if step > 0 and not hit_max:
        rank_log(f"wrote last epoch-end under {out_dir}")
    elif step > 0 and hit_max:
        rank_log(f"max_steps={max_steps} stop at step={step} under {out_dir}")
    else:
        rank_log("no optimizer steps; nothing saved")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
