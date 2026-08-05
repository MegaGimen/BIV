#!/usr/bin/env python3
"""Download HuggingFace valiere/ISETrace and upload to ModelScope LambdaLinker/ISETrace.

Run this on the inference/training server (where network can reach HF, or use mirror).

Prereqs:
  pip install -U modelscope datasets huggingface_hub
  # ModelScope token: https://www.modelscope.cn/my/myaccesstoken
  export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
  # optional CN HF mirror:
  # export HF_ENDPOINT=https://hf-mirror.com

Examples:
  python scripts/upload_isetrace_modelscope.py
  python scripts/upload_isetrace_modelscope.py \\
      --hf-repo valiere/ISETrace \\
      --ms-repo LambdaLinker/ISETrace \\
      --work-dir /tmp/isetrace_ms_upload
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-repo", default="valiere/ISETrace")
    p.add_argument("--ms-repo", default="LambdaLinker/ISETrace")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/isetrace_ms_upload"),
        help="Local folder for HF snapshot before upload",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN"),
        help="ModelScope SDK token (or set MODELSCOPE_API_TOKEN)",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing --work-dir contents (do not re-download from HF)",
    )
    p.add_argument(
        "--commit-message",
        default="Mirror valiere/ISETrace from HuggingFace",
    )
    args = p.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing ModelScope token. export MODELSCOPE_API_TOKEN=... "
            "(https://www.modelscope.cn/my/myaccesstoken)"
        )

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if work.exists() and any(work.iterdir()):
            print(f"Clearing {work} before download...", flush=True)
            shutil.rmtree(work)
            work.mkdir(parents=True, exist_ok=True)
        print(f"Downloading HF dataset {args.hf_repo} -> {work}", flush=True)
        print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}", flush=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("pip install huggingface_hub") from exc
        snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            local_dir=str(work),
            local_dir_use_symlinks=False,
        )
    else:
        if not any(work.iterdir()):
            raise SystemExit(f"--skip-download but {work} is empty")

    n_files = sum(1 for _ in work.rglob("*") if _.is_file())
    print(f"Local snapshot ready: {work} ({n_files} files)", flush=True)

    from modelscope import HubApi

    api = HubApi()
    api.login(args.token)
    print(f"Uploading folder -> ModelScope dataset {args.ms_repo}", flush=True)
    # Prefer upload_folder when available.
    if hasattr(api, "upload_folder"):
        api.upload_folder(
            repo_id=args.ms_repo,
            folder_path=str(work),
            path_in_repo="",
            repo_type="dataset",
            commit_message=args.commit_message,
        )
    else:
        # Fallback: modelscope CLI-compatible path via file loop
        raise SystemExit(
            "HubApi.upload_folder missing. Upgrade modelscope, or run:\n"
            f"  modelscope login --token $MODELSCOPE_API_TOKEN\n"
            f"  ms upload {args.ms_repo} {work} --repo-type dataset"
        )

    print("Upload done.", flush=True)
    print("Verify:", flush=True)
    print(
        "  from modelscope.msdatasets import MsDataset\n"
        f"  ds = MsDataset.load({args.ms_repo!r})\n"
        "  # or: python -c \"from biv_wm.data import open_isetrace_dataset; "
        "print(len(open_isetrace_dataset()))\"",
        flush=True,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
