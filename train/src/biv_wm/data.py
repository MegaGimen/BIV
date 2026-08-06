"""Trajectory loaders and JSONL writers for world-model SFT.

Primary corpora (via ``scripts/prepare_data.py`` + ``biv_wm.hub`` / adapters):
  - SWE-Hero OpenHands trajectories → wm_code
  - ISETrace OS trajectories → wm_os
  - SWE-Zero OpenHands trajectories → anti_forget (policy replay)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from biv_wm.formatting import sample_to_chat_dict

# OpenHands tools that are not real environment transitions for WM SFT.
WM_SKIP_TOOLS = frozenset({"think", "finish"})

DEFAULT_SWE_HERO_MS = "nv-community/SWE-Hero-openhands-trajectories"
DEFAULT_SWE_HERO_HF = "nvidia/SWE-Hero-openhands-trajectories"


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
    """Normalize a single (tool, args, observation) dict."""
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
    if str(tool) in WM_SKIP_TOOLS:
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
    if is_error is None and isinstance(obs, str):
        is_error = _guess_is_error(obs)

    return {
        "tool": str(tool),
        "arguments": args,
        "observation": obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False),
        "is_error": bool(is_error) if is_error is not None else False,
    }


def _guess_is_error(obs: str) -> bool:
    low = obs.lower()
    markers = (
        "traceback (most recent call last)",
        "error:",
        "exception:",
        "command failed",
        "no such file or directory",
        "permission denied",
    )
    return any(m in low for m in markers)


def extract_turns_from_openai_tool_messages(
    messages: list[dict[str, Any]],
    *,
    skip_tools: frozenset[str] = WM_SKIP_TOOLS,
) -> list[dict[str, Any]]:
    """Pair assistant.tool_calls with following role=tool messages → WM turns.

    Used for SWE-Hero ``trajectory`` and any OpenAI-style message log.
    Non-env tools in ``skip_tools`` drop both the call and its tool reply.
    """
    turns: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                if not isinstance(fn, dict):
                    fn = {}
                name = fn.get("name")
                pending.append(
                    {
                        "tool": name,
                        "arguments": fn.get("arguments", {}),
                        "tool_call_id": tc.get("id") if isinstance(tc, dict) else None,
                        "_skip": str(name) in skip_tools if name else True,
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
            if matched is None:
                continue
            if matched.get("_skip"):
                continue
            payload = {
                "tool": matched.get("tool") or msg.get("name"),
                "arguments": matched.get("arguments", {}),
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


def extract_turns_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract WM turns from SWE-Hero rows or compact local fixtures."""
    # SWE-Hero / OpenHands
    if "trajectory" in record and isinstance(record["trajectory"], list):
        return extract_turns_from_openai_tool_messages(record["trajectory"])

    # Compact local fixtures: already (tool, args, observation) lists
    if "turns" in record and isinstance(record["turns"], list):
        turns = [normalize_turn(t) for t in record["turns"]]
        return [t for t in turns if t]

    # OpenAI messages without the SWE-Hero wrapper key
    if "messages" in record and isinstance(record["messages"], list):
        return extract_turns_from_openai_tool_messages(record["messages"])

    nt = normalize_turn(record)
    return [nt] if nt else []


def expand_trajectory_samples(
    turns: list[dict[str, Any]],
    *,
    min_turns: int = 1,
    max_prefix: int | None = None,
    every_k: int = 1,
    expand_prefixes: bool = False,
) -> list[list[dict[str, Any]]]:
    """Build training sequences from one trajectory's tool turns.

    Default (``expand_prefixes=False``): **one sample per trajectory** — the full
    chain (optionally truncated by ``max_prefix``). Response-only SFT then puts
    loss on every observation once (no repeated early-``o`` from short prefixes).

    Legacy (``expand_prefixes=True``): causal prefixes ``turns[:1]``, ``[:2]``, …
    with stride ``every_k`` (over-samples early observations).
    """
    if not turns:
        return []
    last = len(turns) if max_prefix is None else min(max_prefix, len(turns))
    if last < min_turns:
        return []
    if not expand_prefixes:
        return [turns[:last]]

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
    expand_prefixes: bool = False,
    shuffle_obs: bool = False,
    obs_pool: list[str] | None = None,
    rng: random.Random | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield SFT chat rows for one trajectory (default: a single full-chain row)."""
    rng = rng or random.Random(0)
    for prefix in expand_trajectory_samples(
        turns,
        min_turns=min_turns,
        max_prefix=max_prefix,
        every_k=every_k,
        expand_prefixes=expand_prefixes,
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
    expand_prefixes: bool = False,
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
                expand_prefixes=expand_prefixes,
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


def open_swe_hero_dataset(
    split: str = "train",
    max_rows: int | None = None,
    *,
    source: str = "modelscope",
    repo_id: str | None = None,
):
    """Return a HF Dataset for SWE-Hero (Arrow / mmap). Does NOT call to_list()."""
    from datasets import DatasetDict, load_dataset

    src = (source or "modelscope").strip().lower()
    if src in {"modelscope", "ms"}:
        rid = repo_id or DEFAULT_SWE_HERO_MS
        ds = _load_swe_hero_modelscope(rid, split=split)
    elif src in {"huggingface", "hf"}:
        rid = repo_id or DEFAULT_SWE_HERO_HF
        ds = load_dataset(rid, split=split)
    else:
        raise ValueError(f"Unknown SWE-Hero source={source!r}; use modelscope|huggingface")

    if isinstance(ds, DatasetDict):
        key = split if split in ds else next(iter(ds.keys()))
        ds = ds[key]
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    return ds


def _load_swe_hero_modelscope(repo_id: str, *, split: str):
    """ModelScope snapshot → datasets.load_dataset; fall back to MsDataset.load."""
    from datasets import Dataset, DatasetDict, load_dataset

    local_dir = None
    snap_err: Exception | None = None
    try:
        try:
            from modelscope.hub.snapshot_download import dataset_snapshot_download
        except ImportError:
            from modelscope import dataset_snapshot_download  # type: ignore

        print(f"ModelScope dataset_snapshot_download: {repo_id}", flush=True)
        try:
            local_dir = dataset_snapshot_download(repo_id, repo_type="dataset")
        except TypeError:
            local_dir = dataset_snapshot_download(repo_id)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        snap_err = exc
        print(f"snapshot_download failed ({exc!r}); trying MsDataset.load", flush=True)

    if local_dir is not None:
        try:
            return load_dataset(str(local_dir), split=split)
        except Exception as exc:  # noqa: BLE001
            print(
                f"load_dataset({local_dir}, split={split!r}) failed ({exc!r}); "
                "trying MsDataset.load",
                flush=True,
            )

    from modelscope.msdatasets import MsDataset

    print(f"MsDataset.load({repo_id!r}, split={split!r})", flush=True)
    raw = MsDataset.load(repo_id, split=split)
    if hasattr(raw, "to_hf_dataset"):
        out = raw.to_hf_dataset()
    elif isinstance(raw, (Dataset, DatasetDict)):
        out = raw
    else:
        raise TypeError(
            f"Unsupported MsDataset result type {type(raw)!r} for {repo_id}. "
            f"snapshot_error={snap_err!r}."
        )
    if isinstance(out, DatasetDict):
        key = split if split in out else next(iter(out.keys()))
        return out[key]
    return out


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
