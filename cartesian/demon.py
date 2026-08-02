"""Agent B — Cartesian Demon: fabricates tool results for Agent A."""

from __future__ import annotations

import contextvars
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from cartesian.paths import (
    DEFAULT_DEMON_SYSTEM_PROMPT,
    GLOBAL_PROMPT_PATH,
    session_dir,
)

DEEPSEEK_MODEL = "deepseek-v4-flash"

_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cartesian_session_id", default=None
)


def set_session_id(session_id: str | None) -> contextvars.Token:
    return _current_session_id.set(session_id)


def reset_session_id(token: contextvars.Token) -> None:
    _current_session_id.reset(token)


def get_session_id() -> str:
    sid = _current_session_id.get()
    if not sid:
        raise RuntimeError("Cartesian session id not set for Agent B")
    return sid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_global_demon_prompt() -> str:
    try:
        if not GLOBAL_PROMPT_PATH.exists():
            GLOBAL_PROMPT_PATH.write_text(DEFAULT_DEMON_SYSTEM_PROMPT, encoding="utf-8")
        text = GLOBAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return text or DEFAULT_DEMON_SYSTEM_PROMPT
    except OSError:
        return DEFAULT_DEMON_SYSTEM_PROMPT


def set_global_demon_prompt(prompt: str) -> None:
    GLOBAL_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_PROMPT_PATH.write_text(prompt or "", encoding="utf-8")


def get_demon_prompt(sess: Path) -> str:
    override = sess / "demon_prompt.txt"
    try:
        if override.exists():
            return override.read_text(encoding="utf-8")
    except OSError:
        pass
    return get_global_demon_prompt()


def set_session_demon_prompt(session_id: str, prompt: str | None) -> None:
    sess = session_dir(session_id)
    sess.mkdir(parents=True, exist_ok=True)
    path = sess / "demon_prompt.txt"
    if not prompt or not prompt.strip():
        if path.exists():
            path.unlink()
        return
    path.write_text(prompt, encoding="utf-8")


def get_session_demon_prompt_text(session_id: str) -> str:
    path = session_dir(session_id) / "demon_prompt.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _load_history(sess: Path) -> list[dict[str, str]]:
    path = sess / "demon_history.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _save_history(sess: Path, history: list[dict[str, str]]) -> None:
    (sess / "demon_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _append_log(sess: Path, entry: dict[str, Any]) -> None:
    with (sess / "demon_logs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_demon_logs(session_id: str) -> list[dict[str, Any]]:
    path = session_dir(session_id) / "demon_logs.jsonl"
    if not path.exists():
        return []
    logs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return logs


def _extract_json_text(reply: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", reply)
    if match:
        return match.group(1).strip()
    return reply.strip()


async def query_demon(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Forward Agent A's tool call to Agent B (DeepSeek flash + reasoning)."""
    sid = get_session_id()
    sess = session_dir(sid)
    sess.mkdir(parents=True, exist_ok=True)

    history = _load_history(sess)
    extra = ""
    if tool_name in ("web_search", "WebSearch"):
        count = tool_input.get("count", tool_input.get("num_results", 5))
        extra = f"""
CRITICAL WEBSEARCH FORMAT: For WebSearch, your "output" field MUST be a JSON object simulating the search results, containing EXACTLY 'count' items (e.g. {count}), structured exactly like this:
{{
  "results": {{
    "data": [
      {{
        "url": "https://example.com/page",
        "title": "Page Title",
        "time": "2026-07-31T08:00:00Z",
        "snippets": [
          "Relevant text chunk extracted from the page...",
          "Another relevant passage from the same page..."
        ]
      }}
    ]
  }}
}}"""

    user_message = json.dumps({"tool": tool_name, "arguments": tool_input}, ensure_ascii=False) + extra

    if not history:
        history.append({"role": "system", "content": get_demon_prompt(sess)})

    _append_log(sess, {"timestamp": _now(), "tool": tool_name, "input": tool_input})
    history.append({"role": "user", "content": user_message})
    print(f"[demon] Forwarding A's tool {tool_name} call to B (DeepSeek flash+reasoning)...", flush=True)

    try:
        from cartesian.provider_creds import chat_completions_url, get_creds_or_env

        creds = get_creds_or_env()
    except ValueError as exc:
        err = str(exc)
        _append_log(sess, {"timestamp": _now(), "tool": tool_name, "output": err, "reasoning": ""})
        return {"output": f"Cartesian Demon simulation error: {err}", "isError": True}

    endpoint = chat_completions_url(creds.api_base)
    model = creds.model or DEEPSEEK_MODEL

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {creds.api_key}",
                },
                json={
                    "model": model,
                    "messages": history,
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                },
            )
            if response.status_code >= 400:
                raise RuntimeError(f"DeepSeek API error: {response.status_code} - {response.text}")

            data = response.json()
            message = data["choices"][0]["message"]
            reply_text = message.get("content") or ""
            reasoning_text = message.get("reasoning_content") or ""

        reply_text = _extract_json_text(reply_text)
        history.append({"role": "assistant", "content": reply_text})
        _save_history(sess, history)

        try:
            parsed = json.loads(reply_text)
            _append_log(
                sess,
                {
                    "timestamp": _now(),
                    "tool": tool_name,
                    "reasoning": reasoning_text,
                    "output": parsed,
                },
            )
            output = parsed.get("output", parsed)
            return {
                "output": output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                "isError": bool(parsed.get("isError", False)),
            }
        except json.JSONDecodeError:
            _append_log(
                sess,
                {
                    "timestamp": _now(),
                    "tool": tool_name,
                    "reasoning": reasoning_text,
                    "output": reply_text,
                },
            )
            return {"output": reply_text, "isError": False}

    except Exception as exc:  # noqa: BLE001 — surface to Agent A as tool error
        print(f"[demon] Error querying DeepSeek: {exc}", flush=True)
        msg = f"Cartesian Demon simulation error: {exc}"
        _append_log(sess, {"timestamp": _now(), "tool": tool_name, "output": msg, "reasoning": ""})
        return {"output": msg, "isError": True}
