#!/usr/bin/env python3
"""Minimal offline tests for multi-source adapters (no hub download)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biv_wm.adapters.normalize import (
    policy_row_from_openhands_record,
    wm_row_from_isetrace_record,
    wm_row_from_openhands_record,
)


def test_wm_hero_shaped() -> None:
    rec = {
        "instance_id": "owner__repo-1",
        "trajectory_id": "t1",
        "trajectory": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "execute_bash",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "a.txt\n", "success": True},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "2",
                        "function": {"name": "think", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "2", "content": "ok"},
        ],
    }
    rows = wm_row_from_openhands_record(rec)
    assert len(rows) == 1
    assert rows[0]["source"] == "wm_code"
    assert rows[0]["n_turns"] == 1  # think skipped
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert "isError" in rows[0]["messages"][-1]["content"]


def test_wm_isetrace_shaped() -> None:
    rec = {
        "session_id": "traj_abc",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": json.dumps({"command": "uname -a"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "exec",
                "content": "Linux ...",
                "success": True,
            },
        ],
    }
    rows = wm_row_from_isetrace_record(rec)
    assert len(rows) == 1
    assert rows[0]["source"] == "wm_os"
    assert "OS" in rows[0]["messages"][0]["content"] or "environment" in rows[0]["messages"][0][
        "content"
    ]


def test_anti_forget_policy() -> None:
    rec = {
        "instance_id": "x",
        "trajectory": [
            {"role": "system", "content": "You are OpenHands."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "I'll list files.",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "execute_bash",
                            "arguments": {"command": "ls"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "main.py"},
        ],
    }
    row, clip_stats = policy_row_from_openhands_record(rec)
    assert row is not None
    assert row["source"] == "anti_forget"
    assert any(m.get("tool_calls") for m in row["messages"] if m.get("role") == "assistant")
    assert clip_stats["messages_clipped"] == 0

    long_rec = {
        "instance_id": "y",
        "trajectory": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "execute_bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A" * 100},
        ],
    }
    row2, st2 = policy_row_from_openhands_record(long_rec, max_tool_chars=50)
    assert row2 is not None
    assert st2["messages_clipped"] >= 1
    assert st2["traj_clipped"] == 1


if __name__ == "__main__":
    test_wm_hero_shaped()
    test_wm_isetrace_shaped()
    test_anti_forget_policy()
    print("ok")
