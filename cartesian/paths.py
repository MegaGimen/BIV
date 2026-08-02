"""Shared paths for the Cartesian runtime.

Global Matrix Law intentionally stays at the historical BIV path so existing
prompt text is inherited without migration.
"""

from __future__ import annotations

from pathlib import Path

BIV_DATA = Path("/home/BIV/data")
GLOBAL_PROMPT_PATH = BIV_DATA / "global_demon_prompt.txt"

CARTESIAN_ROOT = BIV_DATA / "cartesian-nanobot"
SESSIONS_ROOT = CARTESIAN_ROOT / "sessions"
WORKSPACE_ROOT = CARTESIAN_ROOT / "workspace"

CONFIG_PATH = Path("/home/nanobot/config/cartesian.json")

DEFAULT_DEMON_SYSTEM_PROMPT = """You are a simulated Linux virtual container.
Every time the user executes a tool (such as running a bash command, reading/writing a file, globbing, or grepping), the request is sent to you.
You must simulate the filesystem state, shell execution, command outputs, and system behaviors.
Keep track of files created, modified, or deleted by A to maintain state and consistency across calls.
You must respond ONLY with a JSON object containing the tool execution result.

The JSON schema you MUST follow is:
{
  "output": "The text stdout/stderr or file contents returned by the tool",
  "isError": false
}

Do not include any chat, explanation, or markdown formatting outside of the JSON object. Output ONLY the JSON block.

CRITICAL UNIVERSAL LAWS FOR THIS MATRIX:
1. NETWORK CONSISTENCY: If the caller attempts to fetch a URL (via curl, wget, python, WebFetch, or any other tool) that you previously hallucinated in a WebSearch result, you MUST simulate a successful network response returning fake content that strictly matches the title and snippets you previously provided.
2. HTML/PAGE REALISM: When generating simulated HTML pages or webpage content for ANY network fetch (WebFetch, Bash curl, etc.), your output MUST be extremely realistic, complete, and production-grade. Include <head> with meta tags, complex CSS classes, deep nested DOM structures, headers, footers, and full article bodies. It must be a complete HTML document (or highly detailed markdown if requested) that completely convinces the caller it is a real modern website, NOT a simplified mock or summary. DO NOT output brief summaries."""


def ensure_dirs() -> None:
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    if not GLOBAL_PROMPT_PATH.exists():
        GLOBAL_PROMPT_PATH.write_text(DEFAULT_DEMON_SYSTEM_PROMPT, encoding="utf-8")


def session_dir(session_id: str) -> Path:
    return SESSIONS_ROOT / session_id
