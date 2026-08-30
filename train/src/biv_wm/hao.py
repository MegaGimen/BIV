"""Split mix JSONL chat messages into (h, a, o) for Stage 1 JEPA.

mix rows from prepare_data.py / formatting.py use:
  user = tool-call JSON (the action a)
  assistant = real observation JSON (the next state o)
"""

from __future__ import annotations

from typing import Any


def split_hao(messages: list[dict[str, Any]]) -> tuple[list, dict, dict] | None:
    """Last user/assistant pair is (a, o); everything before that user is h."""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    o_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i] or {}).get("role") == "assistant":
            o_idx = i
            break
    if o_idx is None or o_idx == 0:
        return None
    a_idx = None
    for i in range(o_idx - 1, -1, -1):
        if (messages[i] or {}).get("role") == "user":
            a_idx = i
            break
    if a_idx is None:
        return None
    return messages[:a_idx], messages[a_idx], messages[o_idx]
