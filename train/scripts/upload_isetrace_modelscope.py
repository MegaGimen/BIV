#!/usr/bin/env python3
"""Upload local ISETrace artifacts to ModelScope (no re-download by default).

Default for our pipeline: only upload processed SFT JSONL files
  data/processed/train.jsonl
  data/processed/eval.jsonl

  export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
  python scripts/upload_isetrace_modelscope.py \\
      --files data/processed/train.jsonl data/processed/eval.jsonl \\
      --ms-repo LambdaLinker/ISETrace

Or shorthand (same two files under --processed-dir):

  python scripts/upload_isetrace_modelscope.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _upload_files(api, *, repo_id: str, files: list[Path], commit_message: str) -> None:
    if not hasattr(api, "upload_file"):
        raise SystemExit(
            "HubApi.upload_file missing. Upgrade modelscope, or run for each file:\n"
            f"  ms upload {repo_id} <file> --repo-type dataset"
        )
    import threading
    import time

    for path in files:
        remote_name = path.name
        size = path.stat().st_size
        size_gb = size / (1024**3)
        print(f"Uploading {path} ({size_gb:.2f} GiB) -> {repo_id}/{remote_name}", flush=True)
        print(
            "Note: ModelScope first SHA256-hashes the whole file (often NO bar). "
            "After hashing finishes you should see a tqdm like `[Uploading] train.jsonl`. "
            f"For ~{size_gb:.0f} GiB this silent hash can take many minutes.",
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
    p.add_argument("--ms-repo", default="LambdaLinker/ISETrace")
    p.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit files to upload (e.g. train.jsonl eval.jsonl)",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Upload only train.jsonl + eval.jsonl from this dir (default if no --files)",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Upload an entire folder (discouraged; prefer --files / --processed-dir)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN"),
    )
    p.add_argument("--commit-message", default="Add processed ISETrace train/eval JSONL")
    args = p.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing ModelScope token. export MODELSCOPE_API_TOKEN=... "
            "(https://www.modelscope.cn/my/myaccesstoken)"
        )

    files: list[Path] = []
    if args.files:
        files = [f.expanduser().resolve() for f in args.files]
    elif args.processed_dir or args.local_dir is None:
        proc = (args.processed_dir or (ROOT / "data" / "processed")).expanduser().resolve()
        files = [proc / "train.jsonl", proc / "eval.jsonl"]
        print(f"Using processed pair under {proc}", flush=True)
    elif args.local_dir is not None:
        # Legacy whole-folder path
        local = args.local_dir.expanduser().resolve()
        if not local.is_dir():
            raise SystemExit(f"Not a directory: {local}")
        from modelscope import HubApi

        api = HubApi()
        api.login(args.token)
        if not hasattr(api, "upload_folder"):
            raise SystemExit("HubApi.upload_folder missing; upgrade modelscope")
        print(f"Uploading entire folder {local} -> {args.ms_repo}", flush=True)
        api.upload_folder(
            repo_id=args.ms_repo,
            folder_path=str(local),
            path_in_repo="",
            repo_type="dataset",
            commit_message=args.commit_message,
        )
        print("Upload done.", flush=True)
        return

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
    print("Repo files should include: train.jsonl, eval.jsonl", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
