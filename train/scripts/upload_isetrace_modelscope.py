#!/usr/bin/env python3
"""Upload an already-local ISETrace tree to ModelScope (no re-download by default).

Prefer pointing at an existing snapshot on the inference server:

  # 1) Find HF hub cache (raw ISETrace, best for LambdaLinker/ISETrace mirror)
  ls ~/.cache/huggingface/hub/datasets--valiere--ISETrace/
  # often: .../snapshots/<hash>/

  # 2) Or upload our prepared JSONL (SFT-ready, not the HF trajectories config)
  #    ~/BIV/train/data/processed/{train,eval}.jsonl

  export MODELSCOPE_API_TOKEN=ms-xxxxxxxx
  python scripts/upload_isetrace_modelscope.py \\
      --local-dir /path/to/existing/isetrace_or_processed \\
      --ms-repo LambdaLinker/ISETrace

Optional: only if local tree is missing, pass --download-hf to fetch valiere/ISETrace once.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _guess_hf_isetrace_dirs() -> list[Path]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    root = hub / "datasets--valiere--ISETrace"
    cands: list[Path] = []
    snap = root / "snapshots"
    if snap.is_dir():
        cands.extend(sorted(p for p in snap.iterdir() if p.is_dir()))
    if root.is_dir():
        cands.append(root)
    # datasets library arrow cache (less ideal for hub upload)
    ds = Path.home() / ".cache" / "huggingface" / "datasets"
    if ds.is_dir():
        for p in ds.rglob("*"):
            if p.is_dir() and "isetrace" in p.name.lower():
                cands.append(p)
                break
    return cands


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms-repo", default="LambdaLinker/ISETrace")
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Existing local folder to upload (HF snapshot or data/processed). No download.",
    )
    p.add_argument(
        "--download-hf",
        action="store_true",
        help="ONLY if local data is missing: download valiere/ISETrace into --work-dir first",
    )
    p.add_argument("--hf-repo", default="valiere/ISETrace")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/isetrace_ms_upload"),
        help="Used only with --download-hf",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN"),
        help="ModelScope SDK token (or set MODELSCOPE_API_TOKEN)",
    )
    p.add_argument(
        "--commit-message",
        default="Upload local ISETrace snapshot",
    )
    p.add_argument(
        "--list-guesses",
        action="store_true",
        help="Print guessed local HF ISETrace paths and exit",
    )
    args = p.parse_args()

    guesses = _guess_hf_isetrace_dirs()
    if args.list_guesses:
        if not guesses:
            print("No guessed ISETrace dirs under ~/.cache/huggingface", flush=True)
        for g in guesses:
            print(g, flush=True)
        return

    local: Path | None = args.local_dir
    if local is None and not args.download_hf:
        if guesses:
            local = guesses[0]
            print(f"Auto-using guessed local dir: {local}", flush=True)
        else:
            raise SystemExit(
                "No --local-dir and no HF cache guess.\n"
                "Pass --local-dir ~/BIV/train/data/processed\n"
                "  or --local-dir ~/.cache/huggingface/hub/datasets--valiere--ISETrace/snapshots/<hash>\n"
                "  or --list-guesses / --download-hf"
            )

    if args.download_hf:
        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)
        print(f"Downloading HF dataset {args.hf_repo} -> {work}", flush=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            local_dir=str(work),
            local_dir_use_symlinks=False,
        )
        local = work

    assert local is not None
    local = local.expanduser().resolve()
    if not local.is_dir() or not any(local.iterdir()):
        raise SystemExit(f"Local dir missing or empty: {local}")

    if not args.token:
        raise SystemExit(
            "Missing ModelScope token. export MODELSCOPE_API_TOKEN=... "
            "(https://www.modelscope.cn/my/myaccesstoken)"
        )

    n_files = sum(1 for f in local.rglob("*") if f.is_file())
    print(f"Uploading {local} ({n_files} files) -> ModelScope dataset {args.ms_repo}", flush=True)

    from modelscope import HubApi

    api = HubApi()
    api.login(args.token)
    if not hasattr(api, "upload_folder"):
        raise SystemExit(
            "HubApi.upload_folder missing. Upgrade modelscope, or run:\n"
            f"  ms upload {args.ms_repo} {local} --repo-type dataset"
        )
    api.upload_folder(
        repo_id=args.ms_repo,
        folder_path=str(local),
        path_in_repo="",
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print("Upload done.", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
