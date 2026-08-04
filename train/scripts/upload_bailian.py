#!/usr/bin/env python3
"""Upload SFT train/eval JSONL to Bailian (DashScope) via Files API — no OSS.

Docs:
  https://help.aliyun.com/zh/model-studio/upload-file-api
  https://help.aliyun.com/zh/model-studio/create-fine-tuning-job-api

Flow:
  1) Export messages-only JSONL (or reuse export_bailian/)
  2) Split into ≤ shard_max_mb (default 190; API fine-tune limit is 300MB)
  3) POST multipart to https://dashscope.aliyuncs.com/api/v1/files  purpose=fine-tune
  4) Write file_id manifest for creating a fine-tune job later

  cd train
  # train/.env must contain DASHSCOPE_API_KEY=
  python scripts/upload_bailian.py
  python scripts/upload_bailian.py --reuse-shards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse export/shard helpers from the OSS uploader.
from upload import (  # noqa: E402
    _export_messages_jsonl,
    _load_dotenv,
    _split_jsonl_shards,
)

DASHSCOPE_FILES_URL = "https://dashscope.aliyuncs.com/api/v1/files"


def _require_dashscope_key() -> str:
    key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "Empty DASHSCOPE_API_KEY in train/.env\n"
            "Create at: https://bailian.console.aliyun.com/ → API-KEY 管理"
        )
    return key


def _upload_file_to_bailian(
    path: Path,
    *,
    api_key: str,
    purpose: str = "fine-tune",
    description: str = "",
) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("Missing requests. pip install requests") from exc

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 300:
        raise SystemExit(
            f"{path} is {size_mb:.1f} MiB > Bailian fine-tune limit 300MB. "
            "Lower --shard-max-mb."
        )
    print(f"Uploading {path.name} ({size_mb:.1f} MiB) → Bailian Files API ...", flush=True)
    with path.open("rb") as fh:
        files = {"files": (path.name, fh, "application/jsonl")}
        data = {"purpose": purpose}
        if description:
            data["descriptions"] = description
        resp = requests.post(
            DASHSCOPE_FILES_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=3600,
        )
    try:
        payload = resp.json()
    except Exception:
        raise SystemExit(f"Non-JSON response HTTP {resp.status_code}: {resp.text[:500]}")
    if resp.status_code != 200:
        raise SystemExit(f"Upload failed HTTP {resp.status_code}: {payload}")

    uploaded = (payload.get("data") or {}).get("uploaded_files") or []
    failed = (payload.get("data") or {}).get("failed_uploads") or []
    if failed:
        raise SystemExit(f"Bailian rejected {path.name}: {failed}")
    if not uploaded:
        raise SystemExit(f"No uploaded_files in response: {payload}")
    info = uploaded[0]
    print(f"OK file_id={info.get('file_id')} name={info.get('name')}", flush=True)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "processed" / "train.jsonl",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data" / "processed" / "eval.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shard-max-mb",
        type=int,
        default=190,
        help="Per-file shard size (Bailian fine-tune max 300MB; default 190)",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip export; reuse data/export_bailian/<train|eval>.jsonl then shard+upload",
    )
    parser.add_argument(
        "--reuse-shards",
        action="store_true",
        help="Skip export/split; upload existing export_bailian/shards/*",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Upload training shards only",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "export_bailian" / "bailian_file_ids.json",
    )
    args = parser.parse_args()

    _load_dotenv()
    api_key = _require_dashscope_key()
    shard_max_bytes = args.shard_max_mb * 1024 * 1024

    out_dir = ROOT / "data" / "export_bailian"
    shard_root = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = [("train", args.train)]
    if not args.skip_eval:
        jobs.append(("eval", args.eval))

    manifest: dict = {
        "training_file_ids": [],
        "validation_file_ids": [],
        "training_files": [],
        "validation_files": [],
    }

    for split, src in jobs:
        src = src if src.is_absolute() else (ROOT / src)
        export_path = out_dir / src.name

        if args.reuse_shards:
            shards = sorted(shard_root.glob(f"{src.stem}-*.jsonl"))
            if not shards:
                raise SystemExit(f"No shards for {src.stem} under {shard_root}")
            print(f"Reusing {len(shards)} {split} shards", flush=True)
        else:
            if args.upload_only:
                if not export_path.exists():
                    raise SystemExit(f"--upload-only missing {export_path}")
                print(f"Reusing export {export_path}", flush=True)
            else:
                if not src.exists():
                    raise SystemExit(f"Missing {src}")
                n = _export_messages_jsonl(
                    src,
                    export_path,
                    max_samples=args.max_samples if split == "train" else None,
                    seed=args.seed + (0 if split == "train" else 1),
                )
                print(f"Exported {n} {split} rows → {export_path}", flush=True)
            shards = _split_jsonl_shards(
                export_path, shard_root, max_bytes=shard_max_bytes
            )

        for shard in shards:
            info = _upload_file_to_bailian(
                shard,
                api_key=api_key,
                purpose="fine-tune",
                description=f"biv-wm {split} {shard.name}",
            )
            entry = {
                "file_id": info.get("file_id"),
                "name": info.get("name") or shard.name,
                "local": str(shard),
                "split": split,
            }
            if split == "train":
                manifest["training_file_ids"].append(entry["file_id"])
                manifest["training_files"].append(entry)
            else:
                manifest["validation_file_ids"].append(entry["file_id"])
                manifest["validation_files"].append(entry)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest → {args.manifest}", flush=True)
    print(
        "Create fine-tune with training_datasets / validation_datasets "
        "data_source_type=file_id using these ids.\n"
        "See: https://help.aliyun.com/zh/model-studio/create-fine-tuning-job-api",
        flush=True,
    )


if __name__ == "__main__":
    main()
