"""Simple world-model held-out metrics (no GPU required for score scripts)."""

from __future__ import annotations

import json
import re
from typing import Any


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _WS_RE.sub(" ", text)
    # Drop volatile tokens that hurt exact CE-style string compare.
    text = re.sub(r"\bpid[=:]?\s*\d+\b", "<pid>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}\b", "<ts>", text)
    return text


def parse_observation_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {
                "output": str(obj.get("output", "")),
                "isError": bool(obj.get("isError", False)),
            }
    except json.JSONDecodeError:
        pass
    return {"output": text, "isError": False}


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    from collections import Counter

    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pc.values())
    recall = overlap / sum(gc.values())
    return 2 * precision * recall / (precision + recall)


def score_pair(pred_text: str, gold_text: str) -> dict[str, float]:
    pred = parse_observation_json(pred_text)
    gold = parse_observation_json(gold_text)
    exact = float(normalize_text(pred["output"]) == normalize_text(gold["output"]))
    err_match = float(pred["isError"] == gold["isError"])
    return {
        "exact_norm": exact,
        "token_f1": token_f1(pred["output"], gold["output"]),
        "is_error_acc": err_match,
    }


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {"exact_norm": 0.0, "token_f1": 0.0, "is_error_acc": 0.0, "n": 0.0}
    keys = ["exact_norm", "token_f1", "is_error_acc"]
    out = {k: sum(s[k] for s in scores) / len(scores) for k in keys}
    out["n"] = float(len(scores))
    return out
