#!/usr/bin/env python3
"""Alias: ACT merge lives in merge/act.py."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "merge" / "act.py"),
    run_name="__main__",
)
