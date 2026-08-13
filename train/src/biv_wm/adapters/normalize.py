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


def _clip_content(text: Any, max_chars: int) -> tuple[Any, bool, int]:
    """Return (maybe_clipped_text, was_clipped, overflow_chars)."""
    if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
        return text, False, 0
    overflow = len(text) - max_chars
    return (
        text[:max_chars] + f"\n...[truncated {overflow} chars]",
        True,
        overflow,
    )


# OpenHands "think" tool replies often lose tool_call_id after HF Arrow unify.
_THINK_REPLY_EXACT = frozenset(
    {
        "your thought has been logged.",
        "your thought has been logged",
    }
)
_POLICY_SKIP_TOOLS = frozenset({"think", "finish"})


def _is_think_tool_reply(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    return content.strip().lower() in _THINK_REPLY_EXACT


def _as_str_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def _canon_tool_call(tc: Any) -> dict[str, Any] | None:
    """Normalize one OpenAI-style tool_call for Muse chat_template.

    Muse / Onyx ATEM jinja requires ``function.arguments`` to be a **dict**
    (JSON strings are rejected in the HF sandbox).
    """
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function", tc)
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not name:
        return None
    args = fn.get("arguments", {})
    if isinstance(args, str):
        s = args.strip()
        if not s:
            args = {}
        else:
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = {"_raw": args}
            args = parsed if isinstance(parsed, dict) else {"_raw": parsed}
    elif args is None:
        args = {}
    elif not isinstance(args, dict):
        args = {"_raw": args}
    out: dict[str, Any] = {
        "id": str(tc.get("id") or ""),
        "type": str(tc.get("type") or "function"),
        "function": {"name": str(name), "arguments": args},
    }
    return out


def messages_for_arrow(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize tool_call arguments to JSON strings for HF Arrow Features."""
    out: list[dict[str, Any]] = []
    for m in messages:
        row = {
            "role": m.get("role") or "user",
            "content": _as_str_content(m.get("content")),
            "name": str(m.get("name") or ""),
            "tool_call_id": str(m.get("tool_call_id") or ""),
            "tool_calls": [],
        }
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args = fn.get("arguments", {})
            if isinstance(args, dict):
                args_s = json.dumps(args, ensure_ascii=False)
            else:
                args_s = str(args) if args is not None else "{}"
            row["tool_calls"].append(
                {
                    "id": str(tc.get("id") or ""),
                    "type": str(tc.get("type") or "function"),
                    "function": {
                        "name": str(fn.get("name") or ""),
                        "arguments": args_s,
                    },
                }
            )
        out.append(row)
    return out


def messages_for_chat_template(messages: Any) -> list[dict[str, Any]]:
    """Ensure Muse-ready messages (tool_calls.arguments as dict; drop empty fields)."""
    normed = normalize_train_messages(messages)
    out: list[dict[str, Any]] = []
    for m in normed:
        row: dict[str, Any] = {"role": m["role"], "content": m["content"]}
        if m["role"] == "tool":
            if m.get("name"):
                row["name"] = m["name"]
            if m.get("tool_call_id"):
                row["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            # _canon already made arguments dict via normalize_train_messages
            row["tool_calls"] = m["tool_calls"]
        out.append(row)
    return out


def normalize_openhands_messages(
    messages: Any,
    *,
    drop_think_tool: bool = True,
) -> list[dict[str, Any]]:
    """Repair OpenHands / anti_forget messages for Muse native tool rendering.

    - Drop ``think`` / ``finish`` tool_calls and their replies (incl. orphan
      ``Your thought has been logged.`` when ids were stripped by Arrow).
    - Keep structured ``tool_calls`` on assistant (do **not** fold into content).
    - Re-attach ``name`` / ``tool_call_id`` on ``role=tool`` by sequential pairing
      with the preceding assistant's tool_calls when metadata was lost.
    - Emit a uniform per-message key set for HF concatenate.
    """
    if not isinstance(messages, list):
        return []

    skip = _POLICY_SKIP_TOOLS if drop_think_tool else frozenset()
    skip_ids: set[str] = set()
    interim: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = _as_str_content(msg.get("content"))

        if role == "assistant":
            raw_calls = msg.get("tool_calls")
            kept: list[dict[str, Any]] = []
            if isinstance(raw_calls, list):
                for tc in raw_calls:
                    canon = _canon_tool_call(tc)
                    if canon is None:
                        continue
                    name = canon["function"]["name"]
                    tid = canon["id"]
                    if name in skip:
                        if tid:
                            skip_ids.add(tid)
                        continue
                    kept.append(canon)
            if not kept and not content.strip():
                continue
            row: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "name": "",
                "tool_call_id": "",
                "tool_calls": kept,
            }
            interim.append(row)
            continue

        if role == "tool":
            if drop_think_tool and _is_think_tool_reply(content):
                continue
            tid = msg.get("tool_call_id")
            tid_s = str(tid) if tid is not None and str(tid) else ""
            if tid_s and tid_s in skip_ids:
                continue
            tname = msg.get("name")
            interim.append(
                {
                    "role": "tool",
                    "content": content,
                    "name": str(tname) if tname else "",
                    "tool_call_id": tid_s,
                    "tool_calls": [],
                }
            )
            continue

        interim.append(
            {
                "role": role,
                "content": content,
                "name": "",
                "tool_call_id": "",
                "tool_calls": [],
            }
        )

    # Sequential re-pair: after each assistant with tool_calls, fill following tools.
    out: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []  # (id, name)
    for row in interim:
        if row["role"] == "assistant":
            pending = [
                (str(tc.get("id") or ""), str(tc["function"]["name"]))
                for tc in row["tool_calls"]
            ]
            out.append(row)
            continue
        if row["role"] == "tool":
            if pending:
                tid, tname = pending.pop(0)
                if not row["tool_call_id"] and tid:
                    row["tool_call_id"] = tid
                if not row["name"] and tname:
                    row["name"] = tname
            if not row["name"]:
                row["name"] = "tool"
            out.append(row)
            continue
        pending = []
        out.append(row)
    return out


def normalize_train_messages(messages: Any) -> list[dict[str, Any]]:
    """Train-time message normalize for all mix sources.

    WM rows (user/assistant JSON only) pass through with empty tool fields.
    Anti-forget / OpenHands rows get ``normalize_openhands_messages``.
    """
    if not isinstance(messages, list):
        return []
    needs_policy = any(
        isinstance(m, dict)
        and (
            m.get("role") == "tool"
            or m.get("tool_calls")
            or m.get("tool_call_id")
            or (m.get("role") == "assistant" and m.get("name"))
        )
        for m in messages
    )
    if needs_policy:
        return normalize_openhands_messages(messages, drop_think_tool=True)
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        out.append(
            {
                "role": str(m.get("role") or "user"),
                "content": _as_str_content(m.get("content")),
                "name": "",
                "tool_call_id": "",
                "tool_calls": [],
            }
        )
    return out


def policy_row_from_openhands_record(
    record: dict[str, Any],
    *,
    max_tool_chars: int = 8000,
    drop_think_tool: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Build anti-forget row.

    Returns ``(row_or_none, clip_stats)`` where clip_stats always has:
      messages_seen, messages_clipped, traj_clipped (0/1), chars_overflow
    """
    empty_stats = {
        "messages_seen": 0,
        "messages_clipped": 0,
        "traj_clipped": 0,
        "chars_overflow": 0,
    }
    traj = record.get("trajectory")
    if not isinstance(traj, list) or not traj:
        return None, empty_stats

    skip = _POLICY_SKIP_TOOLS if drop_think_tool else frozenset()
    cleaned: list[dict[str, Any]] = []
    skip_ids: set[str] = set()
    messages_seen = 0
    messages_clipped = 0
    chars_overflow = 0

    def _maybe_clip_field(msg: dict[str, Any], key: str = "content") -> None:
        nonlocal messages_seen, messages_clipped, chars_overflow
        if key not in msg:
            return
        messages_seen += 1
        new_val, clipped, overflow = _clip_content(msg.get(key, ""), max_tool_chars)
        msg[key] = new_val
        if clipped:
            messages_clipped += 1
            chars_overflow += overflow

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
                _maybe_clip_field(new_msg)
            cleaned.append(new_msg)
        elif role == "tool":
            if drop_think_tool and _is_think_tool_reply(msg.get("content")):
                continue
            tid = msg.get("tool_call_id")
            if tid is not None and str(tid) in skip_ids:
                continue
            new_msg = dict(msg)
            _maybe_clip_field(new_msg)
            cleaned.append(new_msg)
        else:
            new_msg = dict(msg)
            if isinstance(new_msg.get("content"), str):
                _maybe_clip_field(new_msg)
            cleaned.append(new_msg)

    # Final repair: pair name/tool_call_id, drop residual think, uniform keys.
    cleaned = normalize_openhands_messages(cleaned, drop_think_tool=drop_think_tool)

    clip_stats = {
        "messages_seen": messages_seen,
        "messages_clipped": messages_clipped,
        "traj_clipped": 1 if messages_clipped else 0,
        "chars_overflow": chars_overflow,
    }
    if len(cleaned) < 2:
        return None, clip_stats
    has_act = any(m.get("role") == "assistant" and m.get("tool_calls") for m in cleaned)
    if not has_act:
        return None, clip_stats
    instance_id, trajectory_id = record_ids(record)
    return {
        "messages": cleaned,
        "source": SOURCE_ANTI_FORGET,
        "instance_id": instance_id,
        "trajectory_id": trajectory_id,
        "n_turns": sum(1 for m in cleaned if m.get("role") == "assistant"),
    }, clip_stats

