#!/usr/bin/env python3
"""Export Muse training / anti-forget eval scalars into one table.

Aggregates **all** TensorBoard runs under ``--log-dir`` (and optionally
``trainer_state.json`` ``log_history`` from train checkpoints) into a single
CSV/TSV/JSONL — every logged step × metric.

Anti-forget **data** source (for context, not this exporter):
  SWE-Zero OpenHands trajectories
    HF:  nvidia/SWE-Zero-openhands-trajectories
    MS:  nv-community/SWE-Zero-openhands-trajectories
  prepared via ``prepare_data.py --anti-forget`` → mix_dir/anti_forget/{train,eval}.jsonl
  Mid-run monitor metric: ``eval_anti_forget_loss`` (held-out eval.jsonl subsample).

Examples:
  # All TB runs under /root/tf-logs → long CSV
  python scripts/export.py --log-dir /root/tf-logs --out /tmp/muse_metrics.csv

  # Wide table (one row per run×step, metrics as columns)
  python scripts/export.py --log-dir /root/tf-logs --wide --out /tmp/muse_wide.csv

  # Also merge latest checkpoint log_history under this train output_dir
  python scripts/export.py --max-length 65536 --choice 1 \\
    --log-dir /root/tf-logs --out /tmp/muse_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trl" / "muse_glimmer_30b_lora.yaml"

_ROLLING_CKPT_RE = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
_EPOCH_END_CKPT_RE = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
_HF_DIGIT_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

_SKIP_HIST = frozenset({"epoch", "step", "total_flos"})


def _resolve(raw: str | Path) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p)


def _resolve_out_dir(*, config: Path, max_length: int, choice: int) -> Path:
    import yaml

    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    base = "outputs/muse_glimmer_wm_mix"
    try:
        cfg = yaml.safe_load(text) or {}
        base = str((cfg.get("train") or {}).get("output_dir") or base)
    except Exception:
        m = re.search(r"(?m)^\s*output_dir:\s*(\S+)", text)
        if m:
            base = m.group(1).strip().strip("'\"")
    out = f"{base}_ml{int(max_length)}_c{int(choice)}"
    return _resolve(out)


def _find_latest_ckpt(out_dir: Path) -> Path | None:
    if not out_dir.is_dir():
        return None
    best: tuple[int, int, int, Path] | None = None
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        epoch = step = None
        kind = 0
        m = _EPOCH_END_CKPT_RE.match(name)
        if m:
            epoch, step, kind = int(m.group(1)), int(m.group(2)), 2
        else:
            m = _ROLLING_CKPT_RE.match(name)
            if m:
                epoch, step, kind = int(m.group(1)), int(m.group(2)), 1
            else:
                m = _HF_DIGIT_CKPT_RE.match(name)
                if m:
                    epoch, step, kind = 0, int(m.group(1)), 0
        if epoch is None or step is None:
            continue
        if not (p / "trainer_state.json").is_file():
            continue
        if not (
            (p / "adapter_model.safetensors").is_file()
            or (p / "pytorch_model_fsdp.bin").is_file()
            or any(p.glob("*.safetensors"))
        ):
            continue
        key = (epoch, step, kind)
        if best is None or key > (best[0], best[1], best[2]):
            best = (epoch, step, kind, p)
    return None if best is None else best[3]


def _normalize_metric(tag: str) -> str:
    """Unify TB tags (tmplog uses train/loss; MuseTB uses bare loss)."""
    t = str(tag).strip().replace("\\", "/")
    if t.startswith("train/"):
        t = t[len("train/") :]
    elif t.startswith("eval/"):
        rest = t[len("eval/") :]
        t = rest if rest.startswith("eval_") else f"eval_{rest}"
    return t


def _rows_from_trainer_state(state_path: Path, *, run: str, source: str) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    hist = state.get("log_history") or []
    if not isinstance(hist, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in hist:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        if step is None:
            continue
        step_i = int(step)
        epoch = entry.get("epoch")
        for k, v in entry.items():
            if k in _SKIP_HIST or k == "step":
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            rows.append(
                {
                    "source": source,
                    "run": run,
                    "step": step_i,
                    "epoch": float(epoch) if isinstance(epoch, (int, float)) else None,
                    "metric": _normalize_metric(k),
                    "value": float(v),
                    "wall_time": None,
                }
            )
    return rows


def _rows_from_tb_run(run_dir: Path) -> list[dict[str, Any]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise SystemExit(
            "Need tensorboard package to read event files "
            f"(pip install tensorboard). Import error: {e!r}"
        ) from e

    # size_guidance: load all scalars (0 = no cap in newer TB; use large int)
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars") or []
    rows: list[dict[str, Any]] = []
    run = run_dir.name
    for tag in tags:
        metric = _normalize_metric(tag)
        for ev in ea.Scalars(tag):
            rows.append(
                {
                    "source": "tensorboard",
                    "run": run,
                    "step": int(ev.step),
                    "epoch": None,
                    "metric": metric,
                    "value": float(ev.value),
                    "wall_time": float(ev.wall_time),
                }
            )
    return rows


def _iter_tb_run_dirs(log_root: Path) -> list[Path]:
    if not log_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(log_root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if any(p.glob("events.out.tfevents.*")):
            out.append(p)
    return out


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep last write for (run, step, metric); prefer tensorboard over trainer_state."""
    rank = {"trainer_state": 0, "tensorboard": 1}
    best: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in rows:
        key = (str(r["run"]), int(r["step"]), str(r["metric"]))
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        if rank.get(str(r["source"]), 0) >= rank.get(str(prev["source"]), 0):
            # Prefer non-null wall_time / epoch when replacing equal rank
            if rank.get(str(r["source"]), 0) == rank.get(str(prev["source"]), 0):
                if r.get("wall_time") is None and prev.get("wall_time") is not None:
                    r = {**r, "wall_time": prev["wall_time"]}
                if r.get("epoch") is None and prev.get("epoch") is not None:
                    r = {**r, "epoch": prev["epoch"]}
            best[key] = r
    return sorted(best.values(), key=lambda x: (x["run"], x["step"], x["metric"]))


def _to_wide(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Pivot long → wide: one row per (run, step)."""
    buckets: dict[tuple[str, int], dict[str, Any]] = {}
    metrics: set[str] = set()
    for r in rows:
        key = (str(r["run"]), int(r["step"]))
        cell = buckets.setdefault(
            key,
            {
                "run": r["run"],
                "step": int(r["step"]),
                "epoch": r.get("epoch"),
                "source": r.get("source"),
                "wall_time": r.get("wall_time"),
            },
        )
        m = str(r["metric"])
        metrics.add(m)
        cell[m] = r["value"]
        if cell.get("epoch") is None and r.get("epoch") is not None:
            cell["epoch"] = r["epoch"]
        if cell.get("wall_time") is None and r.get("wall_time") is not None:
            cell["wall_time"] = r["wall_time"]
    metric_cols = sorted(metrics)
    # Prefer common train/eval names first
    prefer = [
        "loss",
        "grad_norm",
        "learning_rate",
        "entropy",
        "mean_token_accuracy",
        "num_tokens",
        "eval_anti_forget_loss",
        "eval_anti_forget_runtime",
        "eval_anti_forget_samples_per_second",
        "eval_anti_forget_steps_per_second",
    ]
    ordered = [c for c in prefer if c in metrics] + [c for c in metric_cols if c not in prefer]
    header = ["run", "step", "epoch", "wall_time", "source", *ordered]
    wide_rows = [buckets[k] for k in sorted(buckets.keys(), key=lambda t: (t[0], t[1]))]
    return header, wide_rows


def _write_table(
    path: Path,
    *,
    wide: bool,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if wide:
        header, wide_rows = _to_wide(rows)
        if suffix == ".jsonl":
            with path.open("w", encoding="utf-8") as f:
                for r in wide_rows:
                    f.write(json.dumps({h: r.get(h) for h in header}, ensure_ascii=False) + "\n")
        else:
            dialect = "excel-tab" if suffix == ".tsv" else "excel"
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, dialect=dialect, extrasaction="ignore")
                w.writeheader()
                for r in wide_rows:
                    w.writerow({h: r.get(h) for h in header})
        return

    header = ["source", "run", "step", "epoch", "metric", "value", "wall_time"]
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({h: r.get(h) for h in header}, ensure_ascii=False) + "\n")
    else:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, dialect=dialect, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({h: r.get(h) for h in header})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard root (default: LOGGING_DIR / TF_LOGS / /root/tf-logs).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Train output_dir: merge latest ckpt trainer_state.json log_history.",
    )
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--choice", type=int, default=1)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Explicit checkpoint dir (trainer_state.json); skips latest lookup.",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        required=True,
        help="Output path (.csv / .tsv / .jsonl).",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="Pivot to one row per run×step (metrics as columns).",
    )
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional TB run subdirectory names to include (default: all).",
    )
    args = p.parse_args()

    raw_log = (
        str(args.log_dir)
        if args.log_dir is not None
        else (os.environ.get("LOGGING_DIR") or os.environ.get("TF_LOGS") or "/root/tf-logs")
    )
    log_root = Path(raw_log)
    if not log_root.is_absolute():
        log_root = _resolve(log_root)

    all_rows: list[dict[str, Any]] = []

    run_dirs = _iter_tb_run_dirs(log_root)
    if args.runs is not None:
        want = set(args.runs)
        run_dirs = [d for d in run_dirs if d.name in want]
    print(f"[export] tb root={log_root} runs={len(run_dirs)}", flush=True)
    for d in run_dirs:
        part = _rows_from_tb_run(d)
        print(f"[export]   {d.name}: {len(part)} scalar points", flush=True)
        all_rows.extend(part)

    ckpt: Path | None = None
    if args.ckpt is not None:
        ckpt = _resolve(args.ckpt)
    elif args.output_dir is not None or args.max_length is not None:
        if args.output_dir is not None:
            out_dir = _resolve(args.output_dir)
        else:
            out_dir = _resolve_out_dir(
                config=_resolve(args.config),
                max_length=int(args.max_length),
                choice=int(args.choice),
            )
        print(f"[export] output_dir={out_dir}", flush=True)
        ckpt = _find_latest_ckpt(out_dir)
        if ckpt is None:
            print(f"[export] WARN: no complete ckpt under {out_dir}", flush=True)

    if ckpt is not None:
        state_path = ckpt / "trainer_state.json"
        if not state_path.is_file():
            raise SystemExit(f"Missing {state_path}")
        run_name = f"trainer_state:{ckpt.name}"
        part = _rows_from_trainer_state(state_path, run=run_name, source="trainer_state")
        print(f"[export] ckpt={ckpt} log_history points={len(part)}", flush=True)
        all_rows.extend(part)

    if not all_rows:
        raise SystemExit(
            "No scalars found. Check --log-dir (TB event files) and/or "
            "--output-dir / --max-length / --ckpt (trainer_state.json)."
        )

    rows = _dedupe_rows(all_rows)
    out = _resolve(args.out)
    _write_table(out, wide=bool(args.wide), rows=rows)
    n_runs = len({r["run"] for r in rows})
    n_steps = len({(r["run"], r["step"]) for r in rows})
    n_metrics = len({r["metric"] for r in rows})
    print(
        f"[export] wrote {out}  mode={'wide' if args.wide else 'long'}  "
        f"points={len(rows)} runs={n_runs} run×step={n_steps} metrics={n_metrics}",
        flush=True,
    )
    # Hint: where eval lives
    eval_n = sum(1 for r in rows if str(r["metric"]).startswith("eval_anti_forget"))
    if eval_n:
        print(f"[export] eval_anti_forget_* points: {eval_n}", flush=True)
    else:
        print(
            "[export] NOTE: no eval_anti_forget_* yet — those appear only after "
            "post_save eval (same TB run as that training segment).",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
