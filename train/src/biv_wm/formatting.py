"""Minimal chat formatting for environment next-observation SFT.

No Matrix Law: the objective is to predict real tool results, not role-play a demon.
"""

from __future__ import annotations

import json
from typing import Any

# Short, domain-agnostic system prompts (language world-model style).
DEFAULT_WM_SYSTEM = (
    "You are an environment dynamics model for an OpenHands-style coding agent. "
    "Given the interaction history and the agent's latest tool call "
    "(e.g. execute_bash, str_replace_editor), "
    "predict the exact tool observation that a real sandbox would return. "
    "Stay faithful to prior state; do not invent contradictions. "
    "Reply with only the tool observation JSON."
)

OS_WM_SYSTEM = (
    "You are an environment dynamics model for an OS/desktop agent. "
    "Given the interaction history and the agent's latest tool call "
    "(e.g. exec, read, write, and other OS tools), "
    "predict the exact tool observation a real isolated OS workspace would return. "
    "Stay faithful to prior state; do not invent contradictions. "
    "Reply with only the tool observation JSON."
)

SOURCE_WM_CODE = "wm_code"
SOURCE_WM_OS = "wm_os"
SOURCE_ANTI_FORGET = "anti_forget"


def format_tool_call(tool_name: str, arguments: dict[str, Any] | str | None) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    payload = {"tool": tool_name, "arguments": arguments or {}}
    return json.dumps(payload, ensure_ascii=False)


def format_observation(output: Any, is_error: bool | None = None) -> str:
    if isinstance(output, dict) and "output" in output:
        text = output.get("output", "")
        err = bool(output.get("isError", is_error or False))
    else:
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        err = bool(is_error) if is_error is not None else False
    return json.dumps({"output": text, "isError": err}, ensure_ascii=False)


def history_to_messages(
    turns: list[dict[str, Any]],
    *,
    system: str = DEFAULT_WM_SYSTEM,
) -> list[dict[str, str]]:
    """Build chat messages ending with the assistant observation label.

    Each turn dict:
      - tool / name: tool name
      - arguments / input / parameters: tool args
      - observation / output / content: real tool result
      - is_error / isError: optional
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in turns:
        tool = turn.get("tool") or turn.get("name") or turn.get("tool_name")
        args = (
            turn.get("arguments")
            or turn.get("input")
            or turn.get("parameters")
            or turn.get("args")
            or {}
        )
        obs = turn.get("observation")
        if obs is None:
            obs = turn.get("output", turn.get("content", ""))
        is_error = turn.get("is_error", turn.get("isError"))

        # Plain tool-call JSON as user; observation JSON as assistant.
        # No "Latest tool call" / step-count wrappers (history is prior messages).
        messages.append({"role": "user", "content": format_tool_call(str(tool), args)})
        messages.append(
            {
                "role": "assistant",
                "content": format_observation(obs, is_error=is_error),
            }
        )
    return messages


def sample_to_chat_dict(
    turns: list[dict[str, Any]],
    *,
    up_to: int | None = None,
    system: str = DEFAULT_WM_SYSTEM,
    shuffle_observation: bool = False,
    shuffled_obs: str | None = None,
    source: str | None = None,
    instance_id: str | None = None,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Prefix sample: keep turns[:up_to] as the supervised sequence.

    If shuffle_observation, replace the final assistant label (control arm).
    """
    if not turns:
        raise ValueError("empty turns")
    end = len(turns) if up_to is None else up_to
    if end < 1 or end > len(turns):
        raise ValueError(f"up_to={up_to} invalid for {len(turns)} turns")
    prefix = [dict(t) for t in turns[:end]]
    if shuffle_observation:
        if shuffled_obs is None:
            raise ValueError("shuffled_obs required when shuffle_observation=True")
        prefix[-1] = dict(prefix[-1])
        prefix[-1]["observation"] = shuffled_obs
        prefix[-1].pop("output", None)
        prefix[-1].pop("content", None)
    messages = history_to_messages(prefix, system=system)
    row: dict[str, Any] = {"messages": messages, "n_turns": end}
    if source:
        row["source"] = source
    if instance_id is not None:
        row["instance_id"] = instance_id
    if trajectory_id is not None:
        row["trajectory_id"] = trajectory_id
    return row
