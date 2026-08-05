#!/usr/bin/env python3
"""Upload processed WM SFT JSONL (train/eval) to a ModelScope dataset repo.

  export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
  python scripts/upload_processed_modelscope.py --processed-dir data/processed \\
      --ms-repo YourOrg/wm-sft-processed
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _upload_files(api, *, repo_id: str, files: list[Path], commit_message: str) -> None:
    if not hasattr(api, "upload_file"):
        raise SystemExit(
            "HubApi.upload_file missing. Upgrade modelscope, or run for each file:\n"
            f"  ms upload {repo_id} <file> --repo-type dataset"
        )

    for path in files:
        remote_name = path.name
        size_gb = path.stat().st_size / (1024**3)
        print(f"Uploading {path} ({size_gb:.2f} GiB) -> {repo_id}/{remote_name}", flush=True)
        print(
            "Note: ModelScope first SHA256-hashes the whole file (often NO bar). "
            "After hashing finishes you should see a tqdm like `[Uploading] train.jsonl`.",
            flush=True,
        )

        stop = threading.Event()

        def _heartbeat() -> None:
            t0 = time.time()
            while not stop.wait(30.0):
                elapsed = int(time.time() - t0)
                print(
                    f"  ... still working on {remote_name} ({elapsed}s elapsed; "
                    "hashing or uploading)",
                    flush=True,
                )

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"{commit_message}: {remote_name}",
                disable_tqdm=False,
            )
        finally:
            stop.set()
            hb.join(timeout=1.0)
        print(f"Finished {remote_name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ms-repo",
        required=True,
        help="Target ModelScope dataset repo, e.g. YourOrg/wm-sft-processed",
    )
    p.add_argument("--files", type=Path, nargs="+", default=None)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN"),
    )
    p.add_argument("--commit-message", default="Add processed WM SFT train/eval JSONL")
    args = p.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing ModelScope token. export MODELSCOPE_API_TOKEN=... "
            "(https://www.modelscope.cn/my/myaccesstoken)"
        )

    if args.files:
        files = [f.expanduser().resolve() for f in args.files]
    else:
        proc = (args.processed_dir or (ROOT / "data" / "processed")).expanduser().resolve()
        files = [proc / "train.jsonl", proc / "eval.jsonl"]
        print(f"Using processed pair under {proc}", flush=True)

    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing))

    from modelscope import HubApi

    api = HubApi()
    api.login(args.token)
    _upload_files(
        api,
        repo_id=args.ms_repo,
        files=files,
        commit_message=args.commit_message,
    )
    print("Upload done.", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
