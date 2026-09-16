"""Unwrap mix JSON wrappers before they enter Enc(h) / Enc(a) / Enc(h,a,o).

mix_v2 (AutoDL `data/processed/mix_v2`) stores every user turn as
`{"tool": ..., "arguments": ...}` and every assistant turn as
`{"output": ..., "isError": ...}`. AgentWorld last_token still lands on
those shared keys unless the wrapper is removed from the chat text.

Observation: keep the payload (`output`, with `result` as an alias).
`isError: true` becomes a leading `error` line so the flag survives
without the JSON key. Tool call: keep the tool name plus each argument
on its own line, not the outer `{"tool","arguments"}` object.

Non-JSON content (system prompt, unit-test fixtures) is left unchanged.
Unwrapping twice is a no-op: the second pass no longer starts with `{`.
"""

from __future__ import annotations

import json
from typing import Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _looks_json_object(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("{") or s.startswith("[")


def _unwrap_observation(obj: dict[str, Any]) -> str:
    if "output" in obj:
        payload = obj.get("output")
    else:
        payload = obj.get("result")
    text = _as_text(payload)
    if obj.get("isError") is True:
        return f"error\n{text}" if text else "error"
    return text


def _unwrap_tool(obj: dict[str, Any]) -> str:
    tool = obj.get("tool") or obj.get("name") or obj.get("tool_name") or ""
    args = obj.get("arguments", obj.get("input", obj.get("parameters")))
    lines: list[str] = []
    if tool:
        lines.append(str(tool))
    if isinstance(args, dict):
        for key, value in args.items():
            lines.append(f"{key}: {_as_text(value)}")
    elif args is not None:
        rendered = _as_text(args)
        if rendered:
            lines.append(rendered)
    return "\n".join(lines).strip()


def unwrap_content(text: Any) -> str:
    """Strip one mix wrapper from a message body. Identity if not mix JSON."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    if not text or not _looks_json_object(text):
        return text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(obj, dict):
        return text
    if "output" in obj or "result" in obj:
        return _unwrap_observation(obj)
    if "tool" in obj or "arguments" in obj or "name" in obj:
        return _unwrap_tool(obj)
    return text


def unwrap_message(msg: dict[str, Any] | None) -> dict[str, Any]:
    """Copy a chat message and unwrap its ``content``."""
    if not isinstance(msg, dict):
        return {"role": "user", "content": unwrap_content(msg)}
    out = dict(msg)
    out["content"] = unwrap_content(out.get("content"))
    return out


def unwrap_messages(messages: list | None) -> list[dict[str, Any]]:
    if not messages:
        return []
    return [unwrap_message(m) for m in messages]
