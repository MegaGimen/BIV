"""Agent A — nanobot loop with Demon-proxied tools."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot import Nanobot
from nanobot.agent.hook import SDKCaptureHook
from nanobot.sdk.runtime import build_process_direct_kwargs
from nanobot.sdk.types import result_from_response

from cartesian.demon import reset_session_id, set_session_id
from cartesian.paths import CONFIG_PATH, WORKSPACE_ROOT, ensure_dirs, session_dir
from cartesian.provider_creds import (
    build_agent_a_runtime,
    reset_creds,
    resolve_creds,
    set_creds,
)
from cartesian.tool_proxies import install_demon_proxies

_bot: Nanobot | None = None
_proxies_installed = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ui_messages_path(session_id: str) -> Path:
    return session_dir(session_id) / "ui_messages.jsonl"


def append_ui_message(session_id: str, message: dict[str, Any]) -> None:
    sess = session_dir(session_id)
    sess.mkdir(parents=True, exist_ok=True)
    with ui_messages_path(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def read_ui_messages(session_id: str) -> list[dict[str, Any]]:
    path = ui_messages_path(session_id)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


async def get_agent_a() -> Nanobot:
    global _bot, _proxies_installed
    ensure_dirs()
    if _bot is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing nanobot config: {CONFIG_PATH}")
        # Config may lack a usable key when visitors bring their own; allow boot
        # with a placeholder — real calls use per-request runtime credentials.
        import os

        os.environ.setdefault("DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY") or "placeholder")
        _bot = Nanobot.from_config(
            config_path=CONFIG_PATH,
            workspace=WORKSPACE_ROOT,
            model_preset="agentA",
        )
    if not _proxies_installed:
        wrapped = install_demon_proxies(_bot._loop.tools)  # noqa: SLF001
        print(f"[agent-a] Demon proxies installed for: {wrapped}", flush=True)
        _proxies_installed = True
    return _bot


async def close_agent_a() -> None:
    global _bot, _proxies_installed
    if _bot is not None:
        await _bot.aclose()
        _bot = None
        _proxies_installed = False


async def run_agent_a_turn(
    session_id: str,
    user_text: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Persist user+assistant UI messages and run one Agent A turn."""
    bot = await get_agent_a()
    msg_id = f"send-{int(datetime.now().timestamp() * 1000)}-ui"
    user_row = {
        "id": msg_id,
        "timestamp": _now(),
        "direction": "in",
        "kind": "chat",
        "status": "pending",
        "content": {"text": user_text},
    }
    append_ui_message(session_id, user_row)

    try:
        creds = resolve_creds(api_key=api_key, api_base=api_base, model=model)
    except ValueError as exc:
        reply = f"Agent A error: {exc}"
        out_id = f"out-{int(datetime.now().timestamp() * 1000)}-ui"
        append_ui_message(
            session_id,
            {
                "id": out_id,
                "timestamp": _now(),
                "direction": "out",
                "kind": "chat",
                "content": {"text": reply},
            },
        )
        return {"id": msg_id, "reply": reply, "out_id": out_id}

    runtime = build_agent_a_runtime(creds)
    sess_token = set_session_id(session_id)
    cred_token = set_creds(creds)
    try:
        capture = SDKCaptureHook()
        kwargs = build_process_direct_kwargs(
            session_key=f"cartesian:{session_id}",
            channel="cartesian",
            chat_id=session_id,
            sender_id="dashboard",
            media=None,
            ephemeral=False,
        )
        kwargs["runtime"] = runtime
        response = await bot._loop.process_direct(  # noqa: SLF001
            user_text,
            **kwargs,
            hooks=[capture],
        )
        result = result_from_response(response, capture)
        reply = (result.content or "").strip() or "(empty reply)"
    except Exception as exc:  # noqa: BLE001
        reply = f"Agent A error: {exc}"
        print(f"[agent-a] turn failed: {exc}", flush=True)
    finally:
        reset_session_id(sess_token)
        reset_creds(cred_token)

    out_id = f"out-{int(datetime.now().timestamp() * 1000)}-ui"
    assistant_row = {
        "id": out_id,
        "timestamp": _now(),
        "direction": "out",
        "kind": "chat",
        "content": {"text": reply},
    }
    append_ui_message(session_id, assistant_row)

    return {"id": msg_id, "reply": reply, "out_id": out_id}
