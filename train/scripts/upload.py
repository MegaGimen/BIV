#!/usr/bin/env python3
"""Upload Bailian-ready SFT JSONL (messages-only) to Aliyun OSS.

Non-secret defaults: configs/oss.yaml (Tokyo / ap-northeast-1).
Secrets only: train/.env → OSS_ACCESS_KEY_ID + OSS_ACCESS_KEY_SECRET.

  cd train
  pip install oss2 python-dotenv pyyaml
  # set bucket in configs/oss.yaml; put AccessKey pair in .env
  python scripts/upload.py
  python scripts/upload.py --max-samples 320000 --seed 42

Each uploaded line is exactly: {"messages": [...]}  (Bailian / DashScope SFT style).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_oss_public_cfg(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "configs" / "oss.yaml")
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit("Missing python-dotenv. pip install python-dotenv") from exc
    if env_path.exists():
        load_dotenv(env_path)


def _require_secret(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    if not val:
        raise SystemExit(
            f"Empty {name}. Put the AccessKey pair in train/.env only. "
            "Create at https://ram.console.aliyun.com/manage/ak"
        )
    return val


def _oss_bucket(public: dict):
    try:
        import oss2
        from oss2.credentials import EnvironmentVariableCredentialsProvider
    except ImportError as Exc:
        raise SystemExit("Missing oss2. pip install oss2") from Exc

    ak = _require_secret("OSS_ACCESS_KEY_ID")
    sk = _require_secret("OSS_ACCESS_KEY_SECRET")
    os.environ["OSS_ACCESS_KEY_ID"] = ak
    os.environ["OSS_ACCESS_KEY_SECRET"] = sk

    endpoint = str(public.get("endpoint") or "").strip()
    region = str(public.get("region") or "").strip()
    bucket_name = str(public.get("bucket") or "").strip()
    if not endpoint or not region:
        raise SystemExit("configs/oss.yaml must set endpoint and region")
    if not bucket_name:
        raise SystemExit(
            "Set non-secret `bucket:` in configs/oss.yaml "
            "(OSS console → create bucket in 东京 ap-northeast-1)"
        )

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
    return {"messages": messages}


def _export_messages_jsonl(
    src: Path,
    dst: Path,
    *,
    max_samples: int | None,
    seed: int,
) -> int:
    if not src.exists():
        raise SystemExit(f"Missing input: {src}")

    if max_samples is not None and max_samples > 0:
        import random

        from tqdm.auto import tqdm

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
            for idx in tqdm(chosen, desc=f"export {src.name}", unit="ex"):
                fin.seek(offsets[idx])
                line = fin.readline()
                obj = json.loads(line)
                row = _extract_messages_row(obj, line_no=idx + 1, path=src)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        return written

    from tqdm.auto import tqdm

    written = 0
    # Approximate progress by file bytes for full export.
    total_bytes = src.stat().st_size
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"export {src.name}")
        try:
            while True:
                line = fin.readline()
                if not line:
                    break
                pbar.update(len(line.encode("utf-8")))
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                row = _extract_messages_row(obj, line_no=written + 1, path=src)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        finally:
            pbar.close()
    return written


def _probe_oss_write(bucket, prefix: str) -> None:
    """Fail fast on AccessDenied before starting a multi-GB upload."""
    key = f"{prefix.rstrip('/')}/.biv_upload_probe.txt" if prefix else ".biv_upload_probe.txt"
    body = b"biv oss probe\n"
    print(f"Probing PutObject → {key} ...", flush=True)
    try:
        bucket.put_object(key, body)
        bucket.delete_object(key)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Probe failed ({type(exc).__name__}): {exc}\n"
            "Fix RAM/OSS permissions before uploading 50GB+ files.\n"
            "Need at least: PutObject, DeleteObject, and multipart APIs "
            "(InitiateMultipartUpload/UploadPart/CompleteMultipartUpload/ListParts)."
        ) from exc
    print("Probe OK (write permission works).", flush=True)


def _upload_file(
    bucket,
    key: str,
    local: Path,
    *,
    part_size_mb: int = 4,
    num_threads: int = 2,
    store_dir: Path | None = None,
) -> None:
    import oss2
    from tqdm.auto import tqdm

    total = local.stat().st_size
    size_mb = total / (1024 * 1024)
    part_size = max(part_size_mb, 1) * 1024 * 1024
    store = str(store_dir or (ROOT / "data" / "export_bailian" / ".oss2_checkpoint"))
    Path(store).mkdir(parents=True, exist_ok=True)

    print(
        f"Uploading {local} ({size_mb:.1f} MiB) → oss://…/{key}\n"
        f"  part_size={part_size_mb}MiB threads={num_threads} checkpoint={store}\n"
        f"  Note: first progress tick waits for the first part to finish; "
        f"China→Tokyo can be slow.",
        flush=True,
    )
    pbar = tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"oss {local.name}",
        miniters=1,
        mininterval=0.5,
    )
    last = {"n": 0}

    def _progress(consumed_bytes: int, total_bytes: int) -> None:
        pbar.total = total_bytes or total
        n = min(int(consumed_bytes), int(pbar.total))
        if n >= last["n"]:
            pbar.n = n
            last["n"] = n
            pbar.refresh()

    try:
        oss2.resumable_upload(
            bucket,
            key,
            str(local),
            store=oss2.ResumableStore(root=store),
            multipart_threshold=part_size,
            part_size=part_size,
            num_threads=max(num_threads, 1),
            progress_callback=_progress,
        )
        pbar.n = pbar.total
        pbar.refresh()
    except KeyboardInterrupt:
        pbar.close()
        raise SystemExit(
            "\nInterrupted. Partial multipart state is kept under "
            f"{store}; re-run with --upload-only to resume."
        )
    finally:
        pbar.close()
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
    )
    parser.add_argument(
        "--oss-config",
        type=Path,
        default=ROOT / "configs" / "oss.yaml",
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip export; upload existing files under data/export_bailian/ matching --files basenames",
    )
    parser.add_argument(
        "--part-size-mb",
        type=int,
        default=4,
        help="Multipart part size in MiB (smaller → earlier progress ticks; default 4)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Multipart upload threads (default 2; use 1 to debug)",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip the tiny PutObject permission probe",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Delete local oss2 resumable checkpoints before upload",
    )
    args = parser.parse_args()

    _load_dotenv()
    public = _load_oss_public_cfg(args.oss_config)
    prefix = (
        args.prefix if args.prefix is not None else str(public.get("prefix") or "")
    ).strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    bucket = None
    bucket_name = None
    if not args.dry_run:
        bucket, bucket_name = _oss_bucket(public)
        print(
            f"OSS region={public.get('region')} endpoint={public.get('endpoint')} "
            f"bucket={bucket_name} prefix={prefix!r}",
            flush=True,
        )
        if not args.skip_probe:
            _probe_oss_write(bucket, prefix)

    ckpt_dir = ROOT / "data" / "export_bailian" / ".oss2_checkpoint"
    if args.clear_checkpoint and ckpt_dir.exists():
        import shutil

        shutil.rmtree(ckpt_dir)
        print(f"Cleared checkpoint dir {ckpt_dir}", flush=True)

    out_dir = ROOT / "data" / "export_bailian"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.files:
        src = src if src.is_absolute() else (ROOT / src)
        export_path = out_dir / src.name
        if args.upload_only:
            if not export_path.exists():
                raise SystemExit(f"--upload-only but missing {export_path}")
            print(f"Reusing exported file {export_path}", flush=True)
        else:
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
        try:
            _upload_file(
                bucket,
                key,
                export_path,
                part_size_mb=args.part_size_mb,
                num_threads=args.threads,
                store_dir=ckpt_dir,
            )
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if "AccessDenied" in err or "403" in err:
                raise SystemExit(
                    f"OSS AccessDenied for oss://{bucket_name}/{key}\n"
                    "Export is OK and kept on disk — fix RAM permissions, then:\n"
                    "  python scripts/upload.py --upload-only\n"
                    "Checks:\n"
                    "  1) AccessKey belongs to a RAM user with oss:PutObject (e.g. AliyunOSSFullAccess)\n"
                    "  2) Bucket agenttools is in 东京 ap-northeast-1\n"
                    "  3) Bucket policy does not deny this RAM user\n"
                    f"Raw error: {exc}"
                ) from exc
            raise
        print(f"URI hint: oss://{bucket_name}/{key}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
