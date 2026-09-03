#!/usr/bin/env python3
"""CPU checks for stat.py length cache. Seqlen must not drop tokenize hits."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("biv_stat", SCRIPTS / "stat.py")
assert _SPEC is not None and _SPEC.loader is not None
st = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(st)


def test_seqlen_reuses_untrunc_and_splits_fitted() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "wm_code.jsonl"
        cache = st.LengthCache(path, enabled=True)
        untrunc = {"full": 40000, "state": 12000, "action": 80, "n_turns": 6}
        fit16 = {
            "fitted_full": 16000,
            "fitted_state": 8000,
            "fitted_action": 80,
            "n_turns_kept": 3,
        }
        cache.put("row-a", untrunc, 16384, fit16)
        again = st.LengthCache(path, enabled=True)
        assert again.get_untrunc("row-a") == untrunc
        assert again.get_fitted("row-a", 16384) == fit16
        assert again.get_fitted("row-a", 32768) is None
        fit32 = {
            "fitted_full": 30000,
            "fitted_state": 12000,
            "fitted_action": 80,
            "n_turns_kept": 5,
        }
        again.put("row-a", untrunc, 32768, fit32)
        third = st.LengthCache(path, enabled=True)
        assert third.get_untrunc("row-a") == untrunc
        assert third.get_fitted("row-a", 16384) == fit16
        assert third.get_fitted("row-a", 32768) == fit32


def test_legacy_flat_without_seqlen_keeps_untrunc_drops_fitted() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "wm_code.jsonl"
        path.write_text(
            json.dumps(
                {
                    "k": "row-b",
                    "full": 10,
                    "state": 4,
                    "action": 2,
                    "n_turns": 1,
                    "fitted_full": 10,
                    "fitted_state": 4,
                    "fitted_action": 2,
                    "n_turns_kept": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cache = st.LengthCache(path, enabled=True)
        assert cache.get_untrunc("row-b") == {
            "full": 10,
            "state": 4,
            "action": 2,
            "n_turns": 1,
        }
        assert cache.get_fitted("row-b", 16384) is None


def test_legacy_flat_with_seqlen_keeps_fitted() -> None:
    parsed = st.parse_cache_record(
        {
            "k": "row-c",
            "full": 10,
            "state": 4,
            "action": 2,
            "n_turns": 1,
            "seqlen": 16384,
            "fitted_full": 8,
            "fitted_state": 3,
            "fitted_action": 2,
            "n_turns_kept": 1,
        }
    )
    assert parsed is not None
    key, rec = parsed
    assert key == "row-c"
    assert rec["fitted"]["16384"]["fitted_full"] == 8


def test_v1_left_right_line_is_skipped() -> None:
    parsed = st.parse_cache_record(
        {
            "k": "old",
            "full": 10,
            "left": 8,
            "right": 2,
            "n_turns": 1,
            "fitted_full": 10,
            "fitted_left": 8,
            "fitted_right": 2,
            "n_turns_kept": 1,
        }
    )
    assert parsed is None


def main() -> None:
    test_seqlen_reuses_untrunc_and_splits_fitted()
    test_legacy_flat_without_seqlen_keeps_untrunc_drops_fitted()
    test_legacy_flat_with_seqlen_keeps_fitted()
    test_v1_left_right_line_is_skipped()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
