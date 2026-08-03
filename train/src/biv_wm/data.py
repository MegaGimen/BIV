"""Trajectory loaders and JSONL writers for world-model SFT."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from biv_wm.formatting import sample_to_chat_dict


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_turn(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize heterogeneous tool-turn schemas into a single shape."""
    tool = raw.get("tool") or raw.get("name") or raw.get("tool_name")
    if not tool and "function" in raw:
        fn = raw["function"]
        if isinstance(fn, dict):
            tool = fn.get("name")
            raw = {
                **raw,
                "arguments": fn.get("arguments", raw.get("arguments")),
            }
    if not tool:
        return None

    args = (
        raw.get("arguments")
        or raw.get("input")
        or raw.get("parameters")
        or raw.get("args")
        or {}
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}

    obs = raw.get("observation")
    if obs is None:
        obs = raw.get("output", raw.get("content", raw.get("tool_output")))
    if obs is None:
        return None

    is_error = raw.get("is_error", raw.get("isError"))
    if is_error is None and isinstance(obs, dict):
        is_error = obs.get("isError")
        obs = obs.get("output", obs)

    return {
        "tool": str(tool),
        "arguments": args,
        "observation": obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False),
        "is_error": bool(is_error) if is_error is not None else False,
    }


def extract_turns_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort turn extraction from ISETrace-like or local schemas."""
    if "turns" in record and isinstance(record["turns"], list):
        turns = [normalize_turn(t) for t in record["turns"]]
        return [t for t in turns if t]

    if "tool_calls" in record and "observations" in record:
        calls = record["tool_calls"]
        obs = record["observations"]
        turns = []
        for c, o in zip(calls, obs):
            merged = dict(c) if isinstance(c, dict) else {"tool": str(c)}
            if isinstance(o, dict):
                merged = {**merged, **o}
            else:
                merged["observation"] = o
            nt = normalize_turn(merged)
            if nt:
                turns.append(nt)
        return turns

    # OpenAI-style messages with tool roles (ISETrace / similar)
    if "messages" in record and isinstance(record["messages"], list):
        turns: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for msg in record["messages"]:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", tc)
                    pending.append(
                        {
                            "tool": fn.get("name"),
                            "arguments": fn.get("arguments", {}),
                            "tool_call_id": tc.get("id"),
                        }
                    )
            elif role == "tool":
                matched: dict[str, Any] | None = None
                tid = msg.get("tool_call_id")
                if tid:
                    for i, p in enumerate(pending):
                        if p.get("tool_call_id") == tid:
                            matched = pending.pop(i)
                            break
                if matched is None and pending:
                    matched = pending.pop(0)
                payload = {
                    **(matched or {}),
                    "tool": (matched or {}).get("tool") or msg.get("name"),
                    "observation": msg.get("content", ""),
                }
                if "success" in msg and msg["success"] is not None:
                    payload["is_error"] = not bool(msg["success"])
                elif "is_error" in msg or "isError" in msg:
                    payload["is_error"] = msg.get("is_error", msg.get("isError"))
                nt = normalize_turn(payload)
                if nt:
                    turns.append(nt)
        return turns

    # Single-step record
    nt = normalize_turn(record)
    return [nt] if nt else []


def expand_trajectory_samples(
    turns: list[dict[str, Any]],
    *,
    min_turns: int = 1,
    max_prefix: int | None = None,
    every_k: int = 1,
) -> list[list[dict[str, Any]]]:
    """Create causal prefixes: turns[:1], turns[:2], ... (environment consistency)."""
    if not turns:
        return []
    last = len(turns) if max_prefix is None else min(max_prefix, len(turns))
    out: list[list[dict[str, Any]]] = []
    for i in range(min_turns, last + 1):
        if (i - min_turns) % every_k != 0:
            continue
        out.append(turns[:i])
    return out


def iter_sft_rows_from_turns(
    turns: list[dict[str, Any]],
    *,
    min_turns: int = 1,
    max_prefix: int | None = None,
    every_k: int = 1,
    shuffle_obs: bool = False,
    obs_pool: list[str] | None = None,
    rng: random.Random | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield SFT chat rows for one trajectory (constant memory aside from `turns`)."""
    rng = rng or random.Random(0)
    for prefix in expand_trajectory_samples(
        turns, min_turns=min_turns, max_prefix=max_prefix, every_k=every_k
    ):
        shuffled = None
        if shuffle_obs and obs_pool:
            cand = rng.choice(obs_pool)
            if cand == prefix[-1]["observation"] and len(obs_pool) > 1:
                cand = rng.choice(obs_pool)
            shuffled = cand
        yield sample_to_chat_dict(
            prefix,
            shuffle_observation=shuffle_obs,
            shuffled_obs=shuffled,
        )


def records_to_sft_rows(
    records: Iterable[dict[str, Any]],
    *,
    min_turns: int = 1,
    max_prefix: int | None = None,
    every_k: int = 1,
    shuffle_obs: bool = False,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """In-memory helper for tiny local fixtures / unit tests only."""
    rng = rng or random.Random(0)
    pool: list[str] = []
    trajs: list[list[dict[str, Any]]] = []
    for rec in records:
        turns = extract_turns_from_record(rec)
        if not turns:
            continue
        trajs.append(turns)
        for t in turns:
            pool.append(t["observation"])

    rows: list[dict[str, Any]] = []
    for turns in trajs:
        rows.extend(
            iter_sft_rows_from_turns(
                turns,
                min_turns=min_turns,
                max_prefix=max_prefix,
                every_k=every_k,
                shuffle_obs=shuffle_obs,
                obs_pool=pool,
                rng=rng,
            )
        )
    return rows


def load_local_trajectories(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "trajectories" in data:
            return list(data["trajectories"])
        return [data]
    return list(_read_jsonl(path))


def open_isetrace_dataset(
    config: str = "trajectories",
    split: str = "train",
    max_rows: int | None = None,
):
    """Return a HF Dataset (Arrow / memory-mapped). Does NOT call to_list()."""
    from datasets import DatasetDict, load_dataset

    ds = load_dataset("valiere/ISETrace", name=config, split=split)
    if isinstance(ds, DatasetDict):
        key = split if split in ds else next(iter(ds.keys()))
        ds = ds[key]
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    return ds


def try_load_isetrace(
    config: str = "trajectories",
    split: str = "train",
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Deprecated for full corpus: materializes rows. Prefer open_isetrace_dataset()."""
    ds = open_isetrace_dataset(config=config, split=split, max_rows=max_rows)
    return [ds[i] for i in range(len(ds))]


def reservoir_add(
    reservoir: list[str],
    item: str,
    *,
    k: int,
    seen: int,
    rng: random.Random,
    max_chars: int = 4000,
) -> int:
    """Algorithm R reservoir sampling; returns updated seen count."""
    clipped = item if len(item) <= max_chars else item[:max_chars]
    seen += 1
    if len(reservoir) < k:
        reservoir.append(clipped)
        return seen
    j = rng.randint(1, seen)
    if j <= k:
        reservoir[j - 1] = clipped
    return seen
