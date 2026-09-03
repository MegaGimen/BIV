#!/usr/bin/env python3
"""CPU checks for LLM-JEPA encode/loss helpers. No GPU, no 35B load."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_jepa as tj  # noqa: E402

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]


class _FakeTok:
    """Deterministic stand-in: each character → ord, chat template is join."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        return "\n".join(str((m or {}).get("content") or "") for m in messages)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) for c in text]

    def __call__(self, text, truncation=False, add_special_tokens=True):
        del truncation
        ids = ([1] if add_special_tokens else []) + [ord(c) for c in text]
        return {"input_ids": ids}


def test_create_o_labels_last_span() -> None:
    full = [1, 2, 3, 10, 11, 12, 99]
    assert tj.create_o_labels(full, [10, 11, 12]) == [-100, -100, -100, 10, 11, 12, -100]
    assert tj.create_o_labels([1, 2, 3], [9]) == [-100, -100, -100]


def test_fit_keeps_suffix_or_prefix() -> None:
    ids = list(range(10))
    assert tj._fit(ids, 4, keep="suffix") == [6, 7, 8, 9]
    assert tj._fit(ids, 4, keep="prefix") == [0, 1, 2, 3]
    assert tj._fit(ids, 20, keep="suffix") == ids


def test_encode_texts_chops_right_not_left() -> None:
    tok = _FakeTok()
    h = [{"role": "user", "content": "HELLO"}]
    a = {"role": "user", "content": "WORLD"}
    o = {"role": "assistant", "content": "gone"}
    raw = tj.tokenize_ids(tok, tj.apply_template(tok, h + [a, o]))
    assert len(raw) > 8
    row = tj.encode_texts(tok, h, a, o, max_length=8)
    assert row["full_ids"] == raw[:8]
    assert row["full_ids"] != raw[-8:]
    left_raw = tj.tokenize_ids(tok, tj.apply_template(tok, h + [a]))
    cap = min(8, len(left_raw))
    assert row["left_ids"] == left_raw[:cap]


def test_trim_drops_later_turns() -> None:
    tok = _FakeTok()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "AAA"},
        {"role": "user", "content": "bbb"},
        {"role": "assistant", "content": "BBB"},
        {"role": "user", "content": "ccc"},
        {"role": "assistant", "content": "CCC"},
    ]
    len1 = tj.chat_token_len(tok, msgs[:3])
    len2 = tj.chat_token_len(tok, msgs[:5])
    len3 = tj.chat_token_len(tok, msgs)
    assert len1 < len2 < len3
    keep2 = tj.trim_messages_keep_prefix(tok, msgs, max_length=len2)
    assert keep2 == msgs[:5]
    h, a, o = tj.split_hao(keep2)
    assert a["content"] == "bbb"
    assert o["content"] == "BBB"
    keep1 = tj.trim_messages_keep_prefix(tok, msgs, max_length=len1)
    assert keep1 == msgs[:3]
    _, a1, o1 = tj.split_hao(keep1)
    assert a1["content"] == "aaa"
    assert o1["content"] == "AAA"
    keep_all = tj.trim_messages_keep_prefix(tok, msgs, max_length=len3)
    assert keep_all == msgs
    overflow1 = tj.trim_messages_keep_prefix(tok, msgs, max_length=max(1, len1 - 1))
    assert overflow1 == msgs[:3]


def test_dataset_drops_later_turn() -> None:
    tok = _FakeTok()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "AAA"},
        {"role": "user", "content": "bbb"},
        {"role": "assistant", "content": "BBB"},
        {"role": "user", "content": "ccc"},
        {"role": "assistant", "content": "CCC"},
    ]
    cap = tj.chat_token_len(tok, msgs[:5])
    ds = tj.HaoDataset([msgs], tok, cap)
    row = ds[0]
    assert row["o_text"] == "BBB"
    assert row["left_text"] == "bbb"


def test_encode_texts_three_independent_chats() -> None:
    tok = _FakeTok()
    h = [{"role": "user", "content": "ls"}]
    a = {"role": "user", "content": "rm a.txt"}
    o = {"role": "assistant", "content": "gone"}
    row = tj.encode_texts(tok, h, a, o, max_length=65536)
    assert set(row) == {"full_ids", "full_labels", "left_ids", "right_ids"}
    assert row["left_ids"] != row["right_ids"]
    assert row["full_ids"] != row["left_ids"]
    o_ids = tok.encode("gone", add_special_tokens=False)
    assert any(x != -100 for x in row["full_labels"])
    start = tj._find_span(row["full_ids"], o_ids)
    assert start is not None
    assert row["full_labels"][start : start + len(o_ids)] == o_ids


def test_encode_mediated_is_h_vs_hao() -> None:
    tok = _FakeTok()
    h = [{"role": "user", "content": "ls"}, {"role": "assistant", "content": "a.txt"}]
    a = {"role": "user", "content": "rm a.txt"}
    o = {"role": "assistant", "content": "gone"}
    row = tj.encode_mediated(tok, h, a, o, max_length=65536)
    assert set(row) == {"state_ids", "next_ids", "a_text", "o_text"}
    assert row["a_text"] == "rm a.txt"
    assert row["o_text"] == "gone"
    assert row["state_ids"] != row["next_ids"]
    assert len(row["next_ids"]) > len(row["state_ids"])
    right_only = tj.tokenize_ids(tok, tj.apply_template(tok, [o]))
    assert row["next_ids"] != right_only


def test_dataset_mediated_has_no_history_string() -> None:
    tok = _FakeTok()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "AAA"},
        {"role": "user", "content": "rm a.txt"},
        {"role": "assistant", "content": "gone"},
    ]
    ds = tj.HaoDataset([msgs], tok, 65536, encoding="mediated")
    row = ds[0]
    blob = " ".join(str(v) for k, v in row.items() if k not in ("state_ids", "next_ids"))
    assert "aaa" not in blob and "AAA" not in blob
    assert row["a_text"] == "rm a.txt"
    assert row["o_text"] == "gone"


def test_last_token_index_matches_unpad_plus_offset() -> None:
    if torch is None:
        return
    ids = torch.tensor([[7, 8, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    idx = tj.last_token_index(ids, mask, last_token=-3)
    assert int(idx.item()) == 0
    idx2 = tj.last_token_index(ids, mask, last_token=-1)
    assert int(idx2.item()) == 2


def test_jepa_cosine_both_sides_live() -> None:
    if torch is None:
        return
    left = torch.tensor([[1.0, 0.0]], requires_grad=True)
    right = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = tj.jepa_cosine(left, right)
    assert abs(float(loss.item()) - 1.0) < 1e-5
    loss.backward()
    assert left.grad is not None and torch.any(left.grad != 0)
    assert right.grad is not None and torch.any(right.grad != 0)


def test_collate_keeps_text_and_device_move_skips_it() -> None:
    if torch is None:
        return
    batch = tj.collate(
        [
            {
                "full_ids": [1, 2, 3],
                "full_labels": [-100, -100, 3],
                "left_ids": [1, 2],
                "right_ids": [9],
                "o_text": "gone",
                "left_text": "rm a.txt",
            }
        ],
        pad_id=0,
        pad_multiple=1,
    )
    assert batch["o_text"] == ["gone"]
    moved = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in batch.items()}
    assert moved["o_text"] == ["gone"]
    assert moved["full_ids"].shape[0] == 1


def test_shifted_ce_only_labeled_rows() -> None:
    if torch is None:
        return
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 5]])
    head = torch.nn.Linear(8, 16)
    loss = tj.shifted_ce(hidden, labels, head)
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None


def test_shifted_ce_chunk_matches_unchunked() -> None:
    if torch is None:
        return
    torch.manual_seed(0)
    hidden = torch.randn(1, 20, 8, requires_grad=True)
    labels = torch.full((1, 20), -100, dtype=torch.long)
    labels[0, 2:18] = torch.arange(16) % 12
    head = torch.nn.Linear(8, 12)
    loss_one = tj.shifted_ce(hidden, labels, head, chunk_size=4096)
    loss_chunk = tj.shifted_ce(hidden, labels, head, chunk_size=3)
    assert torch.isfinite(loss_one) and torch.isfinite(loss_chunk)
    assert abs(float(loss_one - loss_chunk)) < 1e-5


def main() -> None:
    test_create_o_labels_last_span()
    test_fit_keeps_suffix_or_prefix()
    test_encode_texts_chops_right_not_left()
    test_trim_drops_later_turns()
    test_dataset_drops_later_turn()
    test_encode_texts_three_independent_chats()
    test_encode_mediated_is_h_vs_hao()
    test_dataset_mediated_has_no_history_string()
    test_last_token_index_matches_unpad_plus_offset()
    test_jepa_cosine_both_sides_live()
    test_collate_keeps_text_and_device_move_skips_it()
    test_shifted_ce_only_labeled_rows()
    test_shifted_ce_chunk_matches_unchunked()
    skipped = " (torch helpers skipped)" if torch is None else ""
    print("ok" + skipped, flush=True)


if __name__ == "__main__":
    main()
