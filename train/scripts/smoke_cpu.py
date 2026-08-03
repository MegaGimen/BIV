#!/usr/bin/env python3
"""CPU unit checks for dataset formatting (safe on this non-GPU host)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biv_wm.data import (  # noqa: E402
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
    chat = sample_to_chat_dict(turns0, up_to=2)
    assert chat["messages"][0]["role"] == "system"
    assert chat["messages"][-1]["role"] == "assistant"
    assert "isError" in chat["messages"][-1]["content"]

    rows = records_to_sft_rows(recs, min_turns=1)
    assert len(rows) > len(recs)

    shuffled = records_to_sft_rows(recs, min_turns=1, shuffle_obs=True)
    assert len(shuffled) == len(rows)

    s = score_pair(
        '{"output": "5\\n", "isError": false}',
        '{"output": "5\\n", "isError": false}',
    )
    assert s["exact_norm"] == 1.0

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "train.jsonl"
        n = write_jsonl(out, rows[:5])
        assert n == 5
        assert out.exists()

    # messages-schema record
    msg_rec = [r for r in recs if "messages" in r][0]
    assert extract_turns_from_record(msg_rec)

    print(json.dumps({"ok": True, "n_records": len(recs), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
