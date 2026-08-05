#!/usr/bin/env python3
"""Upload Bailian-ready SFT JSONL (messages-only) to Aliyun OSS.

Non-secret defaults: configs/oss.yaml (Ulanqab / cn-wulanchabu internal by default).
Secrets only: train/.env → OSS_ACCESS_KEY_ID + OSS_ACCESS_KEY_SECRET.

JSONL is split into ≤190MB shards (line-aligned) then uploaded as:
  {prefix}train-00001.jsonl, train-00002.jsonl, ...

  cd train
  pip install oss2 python-dotenv pyyaml
  python scripts/upload.py
  python scripts/upload.py --upload-only
  python scripts/upload.py --max-samples 320000 --seed 42

Each line is exactly: {"messages": [...]}  (Bailian / DashScope SFT style).
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
        raise SystemExit("Set `bucket:` in configs/oss.yaml")

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
    total_bytes = src.stat().st_size
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        pbar = tqdm(
            total=total_bytes, unit="B", unit_scale=True, desc=f"export {src.name}"
        )
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


def _split_jsonl_shards(
    src: Path,
    shard_dir: Path,
    *,
    max_bytes: int,
) -> list[Path]:
    """Split JSONL into ≤max_bytes files on line boundaries (never break a row)."""
    from tqdm.auto import tqdm

    if max_bytes < 1024 * 1024:
        raise SystemExit(f"shard_max too small: {max_bytes}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    for old in shard_dir.glob(f"{stem}-*.jsonl"):
        old.unlink()

    shards: list[Path] = []
    idx = 1
    current_path: Path | None = None
    current_fh = None
    current_size = 0
    rows = 0

    def _open_new() -> None:
        nonlocal idx, current_path, current_fh, current_size
        if current_fh is not None:
            current_fh.close()
        current_path = shard_dir / f"{stem}-{idx:05d}.jsonl"
        idx += 1
        current_fh = current_path.open("w", encoding="utf-8")
        current_size = 0
        shards.append(current_path)

    _open_new()
    total = src.stat().st_size
    with src.open("r", encoding="utf-8") as fin:
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"shard {src.name}")
        try:
            while True:
                line = fin.readline()
                if not line:
                    break
                raw = line if line.endswith("\n") else line + "\n"
                if not raw.strip():
                    pbar.update(len(line.encode("utf-8")))
                    continue
                blen = len(raw.encode("utf-8"))
                if blen > max_bytes:
                    raise SystemExit(
                        f"Single JSONL row is {blen} bytes > shard limit {max_bytes}. "
                        f"Raise shard_max_mb or skip that row."
                    )
                assert current_fh is not None
                if current_size > 0 and current_size + blen > max_bytes:
                    _open_new()
                    assert current_fh is not None
                current_fh.write(raw)
                current_size += blen
                rows += 1
                pbar.update(len(line.encode("utf-8")))
        finally:
            pbar.close()
            if current_fh is not None:
                current_fh.close()

    sizes = [p.stat().st_size / (1024 * 1024) for p in shards]
    print(
        f"Split {src.name}: {rows} rows → {len(shards)} shards "
        f"(~{min(sizes):.1f}–{max(sizes):.1f} MiB each, limit {max_bytes / (1024 * 1024):.0f} MiB)",
        flush=True,
    )
    return shards


def _probe_oss_write(bucket, prefix: str) -> None:
    key = f"{prefix.rstrip('/')}/.biv_upload_probe.txt" if prefix else ".biv_upload_probe.txt"
    body = b"biv oss probe\n"
    print(f"Probing PutObject → {key} ...", flush=True)
    try:
        bucket.put_object(key, body)
        bucket.delete_object(key)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Probe failed ({type(exc).__name__}): {exc}\n"
            "Fix RAM/OSS permissions before uploading. Need PutObject + multipart APIs."
        ) from exc
    print("Probe OK (write permission works).", flush=True)


def _upload_file(
    bucket,
    key: str,
    local: Path,
    *,
    part_size_mb: int = 16,
    num_threads: int = 8,
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
        f"Uploading {local.name} ({size_mb:.1f} MiB) → oss://…/{key}\n"
        f"  part_size={part_size_mb}MiB threads={num_threads}",
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
            f"\nInterrupted. Checkpoint kept under {store}; re-run --upload-only to resume."
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
        help="Skip export; reuse data/export_bailian/<name>.jsonl then re-shard + upload",
    )
    parser.add_argument(
        "--reuse-shards",
        action="store_true",
        help="Skip split; upload existing data/export_bailian/shards/<stem>-*.jsonl",
    )
    parser.add_argument(
        "--shard-max-mb",
        type=int,
        default=None,
        help="Override configs/oss.yaml shard_max_mb (default 190)",
    )
    parser.add_argument("--part-size-mb", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--clear-checkpoint", action="store_true")
    args = parser.parse_args()

    _load_dotenv()
    public = _load_oss_public_cfg(args.oss_config)
    prefix = (
        args.prefix if args.prefix is not None else str(public.get("prefix") or "")
    ).strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    shard_max_mb = int(
        args.shard_max_mb
        if args.shard_max_mb is not None
        else public.get("shard_max_mb", 190)
    )
    shard_max_bytes = shard_max_mb * 1024 * 1024

    bucket = None
    bucket_name = None
    if not args.dry_run:
        bucket, bucket_name = _oss_bucket(public)
        print(
            f"OSS region={public.get('region')} endpoint={public.get('endpoint')} "
            f"bucket={bucket_name} prefix={prefix!r} shard_max={shard_max_mb}MiB",
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
    shard_root = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.files:
        src = src if src.is_absolute() else (ROOT / src)
        export_path = out_dir / src.name
        if args.upload_only or args.reuse_shards:
            if not args.reuse_shards and not export_path.exists():
                raise SystemExit(f"--upload-only but missing {export_path}")
            if not args.reuse_shards:
                print(f"Reusing exported file {export_path}", flush=True)
        else:
            n = _export_messages_jsonl(
                src,
                export_path,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            print(f"Exported {n} messages-only rows → {export_path}", flush=True)

        if args.reuse_shards:
            shards = sorted(shard_root.glob(f"{src.stem}-*.jsonl"))
            if not shards:
                raise SystemExit(f"--reuse-shards but no files match {shard_root}/{src.stem}-*.jsonl")
            print(f"Reusing {len(shards)} shards under {shard_root}", flush=True)
        else:
            shards = _split_jsonl_shards(
                export_path, shard_root, max_bytes=shard_max_bytes
            )

        if args.dry_run:
            continue

        assert bucket is not None
        for shard in shards:
            key = f"{prefix}{shard.name}"
            try:
                _upload_file(
                    bucket,
                    key,
                    shard,
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
                        "Fix RAM permissions, then:\n"
                        "  python scripts/upload.py --reuse-shards\n"
                        f"Need write access to bucket {bucket_name} "
                        f"({public.get('region')}).\n"
                        f"Raw error: {exc}"
                    ) from exc
                raise
            print(f"URI hint: oss://{bucket_name}/{key}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
