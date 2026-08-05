#!/usr/bin/env python3
"""CPU unit checks for SWE-Hero-style dataset formatting (safe without GPU)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biv_wm.data import (  # noqa: E402
    WM_SKIP_TOOLS,
    extract_turns_from_record,
    load_local_trajectories,
    records_to_sft_rows,
    write_jsonl,
)
from biv_wm.formatting import sample_to_chat_dict  # noqa: E402
from biv_wm.metrics import score_pair  # noqa: E402


def main() -> None:
    sample = ROOT / "data" / "examples" / "sample_trajectories.jsonl"
    recs = load_local_trajectories(sample)
    assert len(recs) >= 3, recs

    turns0 = extract_turns_from_record(recs[0])
    assert len(turns0) >= 3, turns0
    # finish / think must be dropped from WM turns
    assert all(t["tool"] not in WM_SKIP_TOOLS for t in turns0)
    assert any(t["tool"] == "execute_bash" for t in turns0)
    assert any(t["tool"] == "str_replace_editor" for t in turns0)

    chat = sample_to_chat_dict(turns0, up_to=2)
    assert chat["messages"][0]["role"] == "system"
    assert chat["messages"][-1]["role"] == "assistant"
    assert "isError" in chat["messages"][-1]["content"]
    user0 = chat["messages"][1]["content"]
    assert "str_replace_editor" in user0 or "execute_bash" in user0

    rows = records_to_sft_rows(recs, min_turns=1)
    # Default: one training row per trajectory with usable turns (no prefix expansion).
    n_with_turns = sum(1 for r in recs if extract_turns_from_record(r))
    assert len(rows) == n_with_turns, (len(rows), n_with_turns, len(recs))

    shuffled = records_to_sft_rows(recs, min_turns=1, shuffle_obs=True)
    assert len(shuffled) == len(rows)

    legacy = records_to_sft_rows(recs, min_turns=1, expand_prefixes=True)
    assert len(legacy) >= len(rows)

    s = score_pair(
        '{"output": "5\\n", "isError": false}',
        '{"output": "5\\n", "isError": false}',
    )
    assert s["exact_norm"] == 1.0

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "train.jsonl"
        chunk = rows[: min(5, len(rows))]
        n = write_jsonl(out, chunk)
        assert n == len(chunk)
        assert out.exists()

    print(
        json.dumps(
            {
                "ok": True,
                "n_records": len(recs),
                "n_rows": len(rows),
                "n_legacy_prefix_rows": len(legacy),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
