"""Split mix JSONL chat messages into (h, a, o) for Stage 1 JEPA.

mix rows from prepare_data.py / formatting.py use:
  user = tool-call JSON (the action a)
  assistant = real observation JSON (the next state o)
"""

from __future__ import annotations

from typing import Any


def complete_turn_end_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Exclusive end index after each complete user → assistant turn, in order.

    Leading system (and other non-user) messages stay in every prefix. A
    dangling user with no assistant is not a turn and is dropped with later
    context — same as treating those later messages as never having happened.
    """
    if not isinstance(messages, list):
        return []
    ends: list[int] = []
    i = 0
    n = len(messages)
    while i < n:
        if (messages[i] or {}).get("role") != "user":
            i += 1
            continue
        j = i + 1
        while j < n and (messages[j] or {}).get("role") != "assistant":
            j += 1
        if j >= n:
            break
        ends.append(j + 1)
        i = j + 1
    return ends


def messages_through_n_turns(
    messages: list[dict[str, Any]], n_turns: int
) -> list[dict[str, Any]]:
    """Keep system + the first ``n_turns`` complete (user, assistant) pairs."""
    ends = complete_turn_end_indices(messages)
    if n_turns < 1 or n_turns > len(ends):
        raise ValueError(f"n_turns={n_turns} invalid for {len(ends)} complete turns")
    return messages[: ends[n_turns - 1]]


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
