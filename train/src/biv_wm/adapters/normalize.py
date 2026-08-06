"""Adapters that normalize heterogeneous corpora into unified SFT rows."""

from __future__ import annotations

import json
from typing import Any

from biv_wm.data import (
    WM_SKIP_TOOLS,
    expand_trajectory_samples,
    extract_turns_from_openai_tool_messages,
    extract_turns_from_record,
    normalize_turn,
)
from biv_wm.formatting import (
    DEFAULT_WM_SYSTEM,
    OS_WM_SYSTEM,
    SOURCE_ANTI_FORGET,
    SOURCE_WM_CODE,
    SOURCE_WM_OS,
    sample_to_chat_dict,
)

ISE_EXTRA_SKIP = frozenset()


def record_ids(record: dict[str, Any]) -> tuple[str | None, str | None]:
    instance_id = (
        record.get("instance_id")
        or record.get("session_id")
        or record.get("intent_id")
        or record.get("id")
    )
    trajectory_id = (
        record.get("trajectory_id")
        or record.get("session_id")
        or record.get("traj_id")
        or instance_id
    )
    if instance_id is not None:
        instance_id = str(instance_id)
    if trajectory_id is not None:
        trajectory_id = str(trajectory_id)
    return instance_id, trajectory_id


def wm_row_from_openhands_record(
    record: dict[str, Any],
    *,
    source: str = SOURCE_WM_CODE,
    system: str | None = None,
    shuffle_obs: bool = False,
    shuffled_obs: str | None = None,
    min_turns: int = 1,
    max_prefix: int | None = None,
    expand_prefixes: bool = False,
    every_k: int = 1,
) -> list[dict[str, Any]]:
    turns = extract_turns_from_record(record)
    if not turns:
        return []
    instance_id, trajectory_id = record_ids(record)
    sys_prompt = system
    if sys_prompt is None:
        sys_prompt = OS_WM_SYSTEM if source == SOURCE_WM_OS else DEFAULT_WM_SYSTEM
    rows: list[dict[str, Any]] = []
    for prefix in expand_trajectory_samples(
        turns,
        min_turns=min_turns,
        max_prefix=max_prefix,
        every_k=every_k,
        expand_prefixes=expand_prefixes,
    ):
        rows.append(
            sample_to_chat_dict(
                prefix,
                system=sys_prompt,
                source=source,
                instance_id=instance_id,
                trajectory_id=trajectory_id,
                shuffle_observation=shuffle_obs,
                shuffled_obs=shuffled_obs,
            )
        )
    return rows


def wm_row_from_isetrace_record(
    record: dict[str, Any],
    *,
    shuffle_obs: bool = False,
    shuffled_obs: str | None = None,
    min_turns: int = 1,
    max_prefix: int | None = None,
    expand_prefixes: bool = False,
    every_k: int = 1,
) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    turns = extract_turns_from_openai_tool_messages(
        messages, skip_tools=ISE_EXTRA_SKIP | WM_SKIP_TOOLS
    )
    if not turns:
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            nt = normalize_turn(
                {
                    "tool": msg.get("name") or "tool",
                    "arguments": {},
                    "observation": msg.get("content", ""),
                    "is_error": (
                        (not msg["success"]) if msg.get("success") is not None else None
                    ),
                }
            )
            if nt:
                turns.append(nt)
    if not turns:
        return []
    instance_id, trajectory_id = record_ids(record)
    rows: list[dict[str, Any]] = []
    for prefix in expand_trajectory_samples(
        turns,
        min_turns=min_turns,
        max_prefix=max_prefix,
        every_k=every_k,
        expand_prefixes=expand_prefixes,
    ):
        rows.append(
            sample_to_chat_dict(
                prefix,
                system=OS_WM_SYSTEM,
                source=SOURCE_WM_OS,
                instance_id=instance_id,
                trajectory_id=trajectory_id,
                shuffle_observation=shuffle_obs,
                shuffled_obs=shuffled_obs,
            )
        )
    return rows


def _clip_content(text: Any, max_chars: int) -> Any:
    if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def policy_row_from_openhands_record(
    record: dict[str, Any],
    *,
    max_tool_chars: int = 8000,
    drop_think_tool: bool = True,
) -> dict[str, Any] | None:
    traj = record.get("trajectory")
    if not isinstance(traj, list) or not traj:
        return None

    skip = {"think", "finish"} if drop_think_tool else set()
    cleaned: list[dict[str, Any]] = []
    skip_ids: set[str] = set()

    for msg in traj:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            kept_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                name = fn.get("name") if isinstance(fn, dict) else None
                tid = tc.get("id") if isinstance(tc, dict) else None
                if name in skip:
                    if tid:
                        skip_ids.add(str(tid))
                    continue
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), dict):
                    tc = dict(tc)
                    fn = dict(fn)
                    fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                    tc["function"] = fn
                kept_calls.append(tc)
            if not kept_calls and not (msg.get("content") or "").strip():
                continue
            new_msg = dict(msg)
            if kept_calls:
                new_msg["tool_calls"] = kept_calls
            else:
                new_msg.pop("tool_calls", None)
            if isinstance(new_msg.get("content"), str):
                new_msg["content"] = _clip_content(new_msg["content"], max_tool_chars)
            cleaned.append(new_msg)
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid is not None and str(tid) in skip_ids:
                continue
            new_msg = dict(msg)
            new_msg["content"] = _clip_content(new_msg.get("content", ""), max_tool_chars)
            cleaned.append(new_msg)
        else:
            new_msg = dict(msg)
            if isinstance(new_msg.get("content"), str):
                new_msg["content"] = _clip_content(new_msg["content"], max_tool_chars)
            cleaned.append(new_msg)

    if len(cleaned) < 2:
        return None
    has_act = any(m.get("role") == "assistant" and m.get("tool_calls") for m in cleaned)
    if not has_act:
        return None
    instance_id, trajectory_id = record_ids(record)
    return {
        "messages": cleaned,
        "source": SOURCE_ANTI_FORGET,
        "instance_id": instance_id,
        "trajectory_id": trajectory_id,
        "n_turns": sum(1 for m in cleaned if m.get("role") == "assistant"),
    }
