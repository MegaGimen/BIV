"""FastAPI server — same /api/* surface as cartesian-dashboard/server.js."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python -m cartesian.server` and `uvicorn cartesian.server:app`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cartesian.agent_a import close_agent_a, get_agent_a, read_ui_messages, run_agent_a_turn
from cartesian.demon import (
    get_global_demon_prompt,
    get_session_demon_prompt_text,
    read_demon_logs,
    set_global_demon_prompt,
    set_session_demon_prompt,
)
from cartesian.paths import SESSIONS_ROOT, ensure_dirs, session_dir

# Load BIV .env for DEEPSEEK_API_KEY
load_dotenv("/home/BIV/.env")

app = FastAPI(title="Cartesian Dashboard API (nanobot)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextBody(BaseModel):
    text: str = ""


class PromptBody(BaseModel):
    prompt: str = ""


@app.on_event("startup")
async def _startup() -> None:
    ensure_dirs()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[cartesian] WARNING: DEEPSEEK_API_KEY is empty", flush=True)
    await get_agent_a()
    print("[cartesian] Agent A ready (Demon proxies active)", flush=True)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_agent_a()


@app.get("/api/demon-prompt")
async def api_get_global_prompt():
    return {"prompt": get_global_demon_prompt()}


@app.post("/api/demon-prompt")
async def api_set_global_prompt(body: PromptBody):
    set_global_demon_prompt(body.prompt or "")
    return {"success": True}


@app.get("/api/sessions")
async def api_list_sessions():
    ensure_dirs()
    sessions = []
    for path in SESSIONS_ROOT.iterdir():
        if not path.is_dir() or not path.name.startswith("sess-"):
            continue
        stat = path.stat()
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        sessions.append({"id": path.name, "createdAt": created.isoformat()})
    sessions.sort(key=lambda s: s["createdAt"], reverse=True)
    return sessions


@app.post("/api/sessions")
async def api_create_session():
    ensure_dirs()
    new_id = f"sess-{int(datetime.now().timestamp() * 1000)}-ui"
    session_dir(new_id).mkdir(parents=True, exist_ok=True)
    return {"id": new_id}


@app.get("/api/sessions/{session_id}/messages")
async def api_get_messages(session_id: str):
    if not session_dir(session_id).exists():
        raise HTTPException(404, "Not found")
    return read_ui_messages(session_id)


@app.post("/api/sessions/{session_id}/messages")
async def api_post_message(session_id: str, body: TextBody):
    if not session_dir(session_id).exists():
        raise HTTPException(404, "Not found")
    if not (body.text or "").strip():
        raise HTTPException(400, "empty text")
    result = await run_agent_a_turn(session_id, body.text.strip())
    return {"id": result["id"]}


@app.get("/api/sessions/{session_id}/prompt")
async def api_get_session_prompt(session_id: str):
    return {"prompt": get_session_demon_prompt_text(session_id)}


@app.post("/api/sessions/{session_id}/prompt")
async def api_set_session_prompt(session_id: str, body: PromptBody):
    set_session_demon_prompt(session_id, body.prompt)
    return {"success": True}


@app.get("/api/sessions/{session_id}/logs")
async def api_get_logs(session_id: str):
    return read_demon_logs(session_id)


@app.get("/health")
async def health():
    return {"ok": True, "provider": "nanobot-cartesian"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "cartesian.server:app",
        host="0.0.0.0",
        port=3033,
        reload=False,
        app_dir=str(_ROOT),
    )


if __name__ == "__main__":
    main()
