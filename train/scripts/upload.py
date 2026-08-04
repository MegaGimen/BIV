#!/usr/bin/env python3
"""Upload Bailian-ready SFT JSONL (messages-only) to Aliyun OSS.

Reads credentials from train/.env (see that file for empty keys).

  cd train
  pip install oss2 python-dotenv
  # fill .env then:
  python scripts/upload.py
  python scripts/upload.py --files data/processed/train.jsonl data/processed/eval.jsonl
  python scripts/upload.py --max-samples 320000 --seed 42

Each uploaded line is exactly: {"messages": [...]}  (Bailian / DashScope SFT style).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> Path:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit("Missing python-dotenv. pip install python-dotenv") from exc
    if not env_path.exists():
        raise SystemExit(f"Missing {env_path}. Create it and fill OSS_* keys.")
    load_dotenv(env_path)
    return env_path


def _require_env(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    if not val:
        raise SystemExit(
            f"Empty {name} in environment / .env. "
            "See train/.env comments for where to get credentials."
        )
    return val


def _oss_bucket():
    try:
        import oss2
        from oss2.credentials import EnvironmentVariableCredentialsProvider
    except ImportError as Exc:
        raise SystemExit("Missing oss2. pip install oss2") from Exc

    # Prefer explicit names from .env; also accept OSS_ACCESS_KEY_* via dotenv.
    ak = _require_env("OSS_ACCESS_KEY_ID")
    sk = _require_env("OSS_ACCESS_KEY_SECRET")
    os.environ["OSS_ACCESS_KEY_ID"] = ak
    os.environ["OSS_ACCESS_KEY_SECRET"] = sk

    endpoint = _require_env("OSS_ENDPOINT")
    region = _require_env("OSS_REGION")
    bucket_name = _require_env("OSS_BUCKET")

    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    return oss2.Bucket(auth, endpoint, bucket_name, region=region), bucket_name


def _extract_messages_row(obj: dict, *, line_no: int, path: Path) -> dict:
    if "messages" not in obj or not isinstance(obj["messages"], list):
        raise ValueError(f"{path}:{line_no}: row missing list field 'messages'")
    messages = obj["messages"]
    if not messages:
        raise ValueError(f"{path}:{line_no}: empty messages")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(
                f"{path}:{line_no}: messages[{i}] must have role+content, got {msg!r}"
            )
        if msg["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"{path}:{line_no}: unexpected role {msg['role']!r} "
                "(expected system/user/assistant/tool)"
            )
    # Bailian SFT: upload ONLY messages (drop n_turns / extras).
    return {"messages": messages}


def _export_messages_jsonl(
    src: Path,
    dst: Path,
    *,
    max_samples: int | None,
    seed: int,
) -> int:
    """Stream src → dst with messages-only rows; optional deterministic sample."""
    if not src.exists():
        raise SystemExit(f"Missing input: {src}")

    rows: list[dict] | None = None
    if max_samples is not None and max_samples > 0:
        # Reservoir needs full load for exact shuffle-subset; for large files
        # we two-pass: count then shuffle indices — memory bound by index list.
        import random

        offsets: list[int] = []
        with src.open("rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(off)
        n = len(offsets)
        take = min(max_samples, n)
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(n), take)) if take < n else list(range(n))
        written = 0
        with src.open("r", encoding="utf-8") as fin, dst.open(
            "w", encoding="utf-8"
        ) as fout:
            for idx in chosen:
                fin.seek(offsets[idx])
                line = fin.readline()
                obj = json.loads(line)
                row = _extract_messages_row(obj, line_no=idx + 1, path=src)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        return written

    written = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = _extract_messages_row(obj, line_no=line_no, path=src)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def _upload_file(bucket, key: str, local: Path) -> None:
    import oss2

    size_mb = local.stat().st_size / (1024 * 1024)
    print(f"Uploading {local} ({size_mb:.1f} MiB) → oss://…/{key}", flush=True)
    # Resumable for large JSONL (common for SFT sets).
    oss2.resumable_upload(
        bucket,
        key,
        str(local),
        multipart_threshold=64 * 1024 * 1024,
        part_size=16 * 1024 * 1024,
        num_threads=4,
    )
    print(f"OK: {key}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            ROOT / "data" / "processed" / "train.jsonl",
            ROOT / "data" / "processed" / "eval.jsonl",
        ],
        help="Local JSONL files (default: processed train+eval)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="OSS key prefix (default: OSS_PREFIX from .env)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic sample size per file (Bailian pilot)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export messages-only JSONL locally only; do not upload",
    )
    args = parser.parse_args()

    _load_dotenv()
    prefix = (args.prefix if args.prefix is not None else os.environ.get("OSS_PREFIX", "")).strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    bucket = None
    bucket_name = None
    if not args.dry_run:
        bucket, bucket_name = _oss_bucket()

    out_dir = ROOT / "data" / "export_bailian"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.files:
        src = src if src.is_absolute() else (ROOT / src)
        export_path = out_dir / src.name
        n = _export_messages_jsonl(
            src,
            export_path,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"Exported {n} messages-only rows → {export_path}", flush=True)
        if args.dry_run:
            continue
        key = f"{prefix}{src.name}"
        assert bucket is not None
        _upload_file(bucket, key, export_path)
        print(f"Public-ish URI hint: oss://{bucket_name}/{key}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
