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
_POLICY_DROP_TOOLS = frozenset({"finish"})  # never train finish as a tool call
_THINK_TOOL = "think"


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


def _thought_text_from_args(args: Any) -> str:
    """Extract OpenHands think-tool body (usually ``{\"thought\": \"...\"}``)."""
    if args is None:
        return ""
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return ""
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return s
        return _thought_text_from_args(parsed)
    if isinstance(args, dict):
        for key in ("thought", "content", "text", "reasoning"):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Single-string dict values fallback
        strs = [v.strip() for v in args.values() if isinstance(v, str) and v.strip()]
        if len(strs) == 1:
            return strs[0]
        if strs:
            return "\n\n".join(strs)
        try:
            return json.dumps(args, ensure_ascii=False)
        except TypeError:
            return str(args)
    return str(args).strip()


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
            "reasoning_content": str(m.get("reasoning_content") or ""),
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
        rc = m.get("reasoning_content") or ""
        if rc:
            # Muse template → <|start|>assistant to=self<|message|>…<|eom|>
            row["reasoning_content"] = rc
        if m["role"] == "tool":
            if m.get("name"):
                row["name"] = m["name"]
            if m.get("tool_call_id"):
                row["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            row["tool_calls"] = m["tool_calls"]
        out.append(row)
    return out


def normalize_openhands_messages(
    messages: Any,
    *,
    map_think_to_reasoning: bool = True,
) -> list[dict[str, Any]]:
    """Repair OpenHands / anti_forget messages for Muse native tool + CoT rendering.

    - Map ``think`` tool_calls → ``reasoning_content`` (Muse ``assistant to=self``).
    - Drop ``finish`` tool_calls and think/finish tool replies.
    - Keep structured env ``tool_calls`` on assistant (do **not** fold into content).
    - Re-attach ``name`` / ``tool_call_id`` on ``role=tool`` by sequential pairing
      when metadata was lost in Arrow.
    - Emit a uniform per-message key set for HF concatenate.
    """
    if not isinstance(messages, list):
        return []

    skip_ids: set[str] = set()
    interim: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = _as_str_content(msg.get("content"))
        existing_rc = msg.get("reasoning_content")
        existing_rc_s = (
            existing_rc.strip()
            if isinstance(existing_rc, str) and existing_rc.strip()
            else ""
        )

        if role == "assistant":
            raw_calls = msg.get("tool_calls")
            kept: list[dict[str, Any]] = []
            thoughts: list[str] = []
            if existing_rc_s:
                thoughts.append(existing_rc_s)
            if isinstance(raw_calls, list):
                for tc in raw_calls:
                    canon = _canon_tool_call(tc)
                    if canon is None:
                        continue
                    name = canon["function"]["name"]
                    tid = canon["id"]
                    if name == _THINK_TOOL:
                        if tid:
                            skip_ids.add(tid)
                        if map_think_to_reasoning:
                            thought = _thought_text_from_args(canon["function"]["arguments"])
                            if thought:
                                thoughts.append(thought)
                        continue
                    if name in _POLICY_DROP_TOOLS:
                        if tid:
                            skip_ids.add(tid)
                        continue
                    kept.append(canon)
            if not kept and not content.strip() and not thoughts:
                continue
            interim.append(
                {
                    "role": "assistant",
                    "content": content,
                    "name": "",
                    "tool_call_id": "",
                    "reasoning_content": "\n\n".join(thoughts),
                    "tool_calls": kept,
                }
            )
            continue

        if role == "tool":
            # Always drop think/finish ack replies (no training signal).
            if _is_think_tool_reply(content):
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
                    "reasoning_content": "",
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
                "reasoning_content": "",
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
            or m.get("reasoning_content")
            or (m.get("role") == "assistant" and m.get("name"))
        )
        for m in messages
    )
    if needs_policy:
        return normalize_openhands_messages(messages, map_think_to_reasoning=True)
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
                "reasoning_content": "",
                "tool_calls": [],
            }
        )
    return out


def policy_row_from_openhands_record(
    record: dict[str, Any],
    *,
    max_tool_chars: int = 8000,
    map_think_to_reasoning: bool = True,
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

    # Keep raw-ish messages (clip content only); final think→reasoning + pairing
    # happens in ``normalize_openhands_messages``.
    cleaned: list[dict[str, Any]] = []
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
        if not isinstance(msg, dict):
            continue
        new_msg = dict(msg)
        if isinstance(new_msg.get("content"), str):
            _maybe_clip_field(new_msg)
        # Also clip think arguments (often long CoT) if present as string.
        tcs = new_msg.get("tool_calls")
        if isinstance(tcs, list) and max_tool_chars > 0:
            new_tcs = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    new_tcs.append(tc)
                    continue
                tc2 = dict(tc)
                fn = tc2.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    args = fn.get("arguments")
                    if isinstance(args, str) and len(args) > max_tool_chars:
                        messages_seen += 1
                        clipped_s, was, overflow = _clip_content(args, max_tool_chars)
                        fn["arguments"] = clipped_s
                        if was:
                            messages_clipped += 1
                            chars_overflow += overflow
                    tc2["function"] = fn
                new_tcs.append(tc2)
            new_msg["tool_calls"] = new_tcs
        cleaned.append(new_msg)

    cleaned = normalize_openhands_messages(
        cleaned, map_think_to_reasoning=map_think_to_reasoning
    )

    clip_stats = {
        "messages_seen": messages_seen,
        "messages_clipped": messages_clipped,
        "traj_clipped": 1 if messages_clipped else 0,
        "chars_overflow": chars_overflow,
    }
    if len(cleaned) < 2:
        return None, clip_stats
    has_act = any(m.get("role") == "assistant" and m.get("tool_calls") for m in cleaned)
    has_rc = any(
        m.get("role") == "assistant" and (m.get("reasoning_content") or "").strip()
        for m in cleaned
    )
    if not has_act and not has_rc:
        return None, clip_stats
    # Prefer trajectories that still have at least one env tool call.
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

