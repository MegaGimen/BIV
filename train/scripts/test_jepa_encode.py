#!/usr/bin/env python3
"""CPU checks for Stage 1 encode/loss helpers. No GPU, no 35B load."""

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
    left_raw = tj.tokenize_ids(tok, tj.apply_template(tok, h))
    cap = min(8, len(left_raw))
    assert row["state_ids"] == left_raw[:cap]


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
    assert row["a_text"] == "bbb"


def test_encode_texts_three_independent_chats() -> None:
    tok = _FakeTok()
    h = [{"role": "user", "content": "ls"}]
    a = {"role": "user", "content": "rm a.txt"}
    o = {"role": "assistant", "content": "gone"}
    row = tj.encode_texts(tok, h, a, o, max_length=65536)
    assert set(row) == {
        "full_ids",
        "full_labels",
        "state_ids",
        "action_ids",
        "a_label_ids",
        "skip_key",
        "a_text",
        "o_text",
    }
    assert "right_ids" not in row
    assert row["state_ids"] != row["full_ids"]
    assert row["action_ids"] != row["full_ids"]
    o_only = tj.tokenize_ids(tok, tj.apply_template(tok, [o]))
    assert row["full_ids"] != o_only
    o_ids = tok.encode("gone", add_special_tokens=False)
    assert any(x != -100 for x in row["full_labels"])
    start = tj._find_span(row["full_ids"], o_ids)
    assert start is not None
    assert row["full_labels"][start : start + len(o_ids)] == o_ids
    assert row["a_label_ids"] == tok.encode("rm a.txt", add_special_tokens=False)
    assert row["a_text"] == "rm a.txt"
    assert row["o_text"] == "gone"


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


def test_pred_align_both_sides_live() -> None:
    if torch is None:
        return
    from biv_wm.jepa import cosine_align_loss, pred_align_loss

    left = torch.tensor([[1.0, 0.0]], requires_grad=True)
    right = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = pred_align_loss(left, right)
    loss.backward()
    assert left.grad is not None and torch.any(left.grad != 0)
    assert right.grad is not None and torch.any(right.grad != 0)

    stopped = torch.tensor([[0.0, 1.0]], requires_grad=True)
    cosine_align_loss(left.detach().clone().requires_grad_(True), stopped).backward()
    assert stopped.grad is None or torch.all(stopped.grad == 0)


def test_ldad_takes_delta_not_concat() -> None:
    if torch is None:
        return
    from biv_wm.jepa import LDAD

    net = LDAD(dim=4, hidden=8)
    delta = torch.randn(2, 4, requires_grad=True)
    out = net(delta)
    assert out.shape == (2, 4)
    out.sum().backward()
    assert delta.grad is not None


def test_ldad_action_ce_first_token_from_cond() -> None:
    if torch is None:
        return
    torch.manual_seed(0)
    cond = torch.randn(1, 8, requires_grad=True)
    a_ids = torch.tensor([[3, 5]])
    a_mask = torch.tensor([[1, 1]])
    embed = torch.nn.Embedding(16, 8)
    head = torch.nn.Linear(8, 16)
    loss = tj.ldad_action_ce(cond, a_ids, a_mask, embed, head)
    assert torch.isfinite(loss)
    loss.backward()
    assert cond.grad is not None and torch.any(cond.grad != 0)


def test_collate_keeps_text_and_device_move_skips_it() -> None:
    if torch is None:
        return
    batch = tj.collate(
        [
            {
                "full_ids": [1, 2, 3],
                "full_labels": [-100, -100, 3],
                "state_ids": [1, 2],
                "action_ids": [4],
                "a_label_ids": [5, 6],
                "o_text": "gone",
                "a_text": "rm a.txt",
                "skip_key": "abcd",
            }
        ],
        pad_id=0,
        pad_multiple=1,
    )
    assert batch["o_text"] == ["gone"]
    assert batch["a_text"] == ["rm a.txt"]
    assert batch["skip_key"] == ["abcd"]
    assert "right_ids" not in batch
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


def test_encode_texts_strips_mix_json() -> None:
    import json

    tok = _FakeTok()
    h = [
        {
            "role": "user",
            "content": json.dumps({"tool": "execute_bash", "arguments": {"command": "ls"}}),
        },
        {
            "role": "assistant",
            "content": json.dumps({"output": "a.txt", "isError": False}),
        },
    ]
    a = {
        "role": "user",
        "content": json.dumps(
            {"tool": "execute_bash", "arguments": {"command": "rm a.txt"}}
        ),
    }
    o = {"role": "assistant", "content": json.dumps({"output": "", "isError": False})}
    row = tj.encode_texts(tok, h, a, o, max_length=65536)
    assert "rm a.txt" in row["a_text"]
    assert "execute_bash" in row["a_text"]
    assert '"tool"' not in row["a_text"]
    assert row["o_text"] == ""
    full_txt = tok.apply_chat_template(h + [a, o])
    wrapped_ids = tj.tokenize_ids(tok, full_txt)
    assert row["full_ids"] != wrapped_ids

    def _decode(ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if 32 <= i < 0x110000)

    state_txt = _decode(row["state_ids"])
    full_body = _decode(row["full_ids"])
    assert "a.txt" in state_txt
    assert "isError" not in state_txt and "isError" not in full_body
    assert "isError" not in row["a_text"] and "isError" not in row["o_text"]


def test_sigreg_matches_epps_pulley_gaussian_cf() -> None:
    if torch is None:
        return
    from biv_wm.sigreg import _quadrature, sigreg_loss

    t, phi, _w = _quadrature(3.0, 17, device=torch.device("cpu"), dtype=torch.float32)
    assert t[0].item() == 0.0
    assert abs(float(phi[0]) - 1.0) < 1e-6
    # φ₀(t) = exp(-t²/2) at t=√2 → e^{-1}
    t_sqrt2 = (2.0 ** 0.5)
    idx = int(torch.argmin((t - t_sqrt2).abs()).item())
    assert abs(float(phi[idx]) - float(torch.exp(torch.tensor(-0.5 * t[idx] ** 2)))) < 1e-6

    torch.manual_seed(0)
    gauss = torch.randn(256, 16)
    const = torch.ones(256, 16)
    g = torch.Generator().manual_seed(1)
    loss_g = sigreg_loss(gauss, num_slices=32, generator=g)
    g = torch.Generator().manual_seed(1)
    loss_c = sigreg_loss(const, num_slices=32, generator=g)
    assert torch.isfinite(loss_g) and torch.isfinite(loss_c)
    assert float(loss_g) < float(loss_c)

    live = torch.randn(64, 8, requires_grad=True)
    sigreg_loss(live, num_slices=16).backward()
    assert live.grad is not None and torch.any(live.grad != 0)


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
    test_pred_align_both_sides_live()
    test_ldad_takes_delta_not_concat()
    test_ldad_action_ce_first_token_from_cond()
    test_collate_keeps_text_and_device_move_skips_it()
    test_shifted_ce_only_labeled_rows()
    test_shifted_ce_chunk_matches_unchunked()
    test_encode_texts_strips_mix_json()
    test_sigreg_matches_epps_pulley_gaussian_cf()
    skipped = " (torch helpers skipped)" if torch is None else ""
    print("ok" + skipped, flush=True)


if __name__ == "__main__":
    main()
