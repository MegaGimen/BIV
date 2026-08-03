#!/usr/bin/env python3
"""Held-out world-model eval: generate observations and score vs gold.

Requires CUDA when loading the fine-tuned model. Use --dry-run-metrics to
score already-generated predictions JSONL on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biv_wm.metrics import aggregate, score_pair  # noqa: E402


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _iter_eval_rows(path: Path, limit: int | None):
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                break


def dry_run_metrics(pred_path: Path) -> None:
    scores = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scores.append(score_pair(row["prediction"], row["gold"]))
    print(json.dumps(aggregate(scores), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir")
    ap.add_argument("--eval-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "wm_eval" / "preds.jsonl")
    ap.add_argument("--dry-run-metrics", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    if args.dry_run_metrics:
        dry_run_metrics(args.dry_run_metrics)
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for model eval (or pass --dry-run-metrics).")

    from unsloth import FastLanguageModel

    cfg = _load_yaml(args.config)
    mcfg = cfg["model"]
    ecfg = cfg.get("eval", {})
    eval_file = args.eval_file or (ROOT / cfg["data"]["eval_file"])
    adapter = args.adapter or (ROOT / cfg["train"]["output_dir"] / "lora_adapter")
    limit = args.max_samples if args.max_samples is not None else ecfg.get("max_samples")

    max_seq = int(mcfg["max_seq_length"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=max_seq,
        load_in_4bit=bool(mcfg.get("load_in_4bit", False)),
        load_in_16bit=bool(mcfg.get("load_in_16bit", True)),
    )
    FastLanguageModel.for_inference(model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    with args.out.open("w", encoding="utf-8") as fout:
        for row in _iter_eval_rows(eval_file, limit):
            messages = row["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                continue
            gold = messages[-1]["content"]
            prompt_messages = messages[:-1]
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(ecfg.get("max_new_tokens", 1024)),
                    temperature=float(ecfg.get("temperature", 0.0)) or None,
                    do_sample=False,
                )
            gen = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            s = score_pair(gen, gold)
            scores.append(s)
            fout.write(
                json.dumps(
                    {"prediction": gen, "gold": gold, "scores": s},
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = aggregate(scores)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out} and {summary_path}")


if __name__ == "__main__":
    main()
