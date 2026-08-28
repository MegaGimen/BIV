#!/usr/bin/env python3
"""Fetch arXiv HTML and dump to UTF-8 text under refs/papers/. No PDFs."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "papers"

PAPERS: list[tuple[str, str]] = [
    ("2310.04799", "chat-vector"),
    ("2212.04089", "task-arithmetic"),
    ("2501.15065", "tatr-knowledge-conflicts"),
    ("2505.06977", "cat-merging"),
    ("2502.20186", "lata-layer-aware-ta"),
    ("2601.22285", "demystifying-mergeability"),
    ("2306.01708", "ties-merging"),
    ("2311.03099", "dare-merging"),
    ("2301.04104", "dreamerv3"),
    ("2305.16309", "unipi-inverse-dynamics"),
    ("2602.15922", "dreamzero-world-action-model"),
    ("2511.22904", "led-wm"),
    ("2605.12289", "priorzero"),
    ("2506.02918", "dymo"),
    ("2606.02388", "paw-cotraining"),
    ("2605.24517", "echo-terminal-wm"),
    ("2602.05842", "rwml"),
    ("2605.28860", "rl-vs-sft-circuits"),
    ("2305.14992", "rap-reasoning-via-planning"),
    ("2305.11206", "lima"),
    ("2310.14152", "o-lora"),
    ("2601.09684", "ortho-lora-task-conflicts"),
    ("2206.06522", "lst-side-tuning"),
    ("2606.24597", "qwen-agentworld"),
    # Physics / 3D world models: encode laws in structure, not fit appearance
    ("1711.10561", "pinns"),
    ("1906.01563", "hamiltonian-nn"),
    ("2003.04630", "lagrangian-nn"),
    ("1806.07366", "neural-ode"),
    ("1803.10122", "world-models-ha"),
    ("1811.04551", "planet"),
    ("2002.09405", "gns-learn-to-simulate"),
    ("2410.08257", "neuma-residual-physics"),
    ("2605.00412", "physically-native-hamiltonian-wm"),
    ("2605.09586", "deformmaster-physics-neural"),
    ("2311.12198", "physgaussian"),
    ("2501.03575", "cosmos-world-foundation"),
    ("2605.19242", "phyworld"),
    ("2603.03485", "phys4d"),
    ("2604.08503", "phantom-physics-video"),
    ("2608.06799", "psg-jepa-physical-grounding"),
    ("2608.07981", "distill-physical-priors"),
    # Discover laws from observation; compile into anticipatory encoding
    ("1905.11481", "ai-feynman"),
    ("1509.03582", "sindy"),
    ("2005.11212", "symbolic-pregression-video"),
    ("2210.13382", "othello-gpt-world-rep"),
    ("2309.00941", "othello-linear-wm"),
    ("2310.02207", "lm-represent-space-time"),
    ("2201.02177", "grokking"),
    ("2507.21513", "what-is-a-world-model"),
    ("2605.18847", "transformers-linear-wm"),
    ("2607.06401", "world-models-roadmap"),
    ("1707.06203", "i2a-imagination"),
    ("1911.08265", "muzero"),
    ("1912.01603", "dreamer-v1"),
    ("1604.00289", "lake-think-like-people"),
    ("2102.11107", "causal-representation-learning"),
]

UA = "BIV-refs/1.0 (research text mirror; +https://arxiv.org)"
MIN_BYTES = 8000


def fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  miss {url}: {exc}", flush=True)
        return None


def html_to_text(html: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as tmp:
        tmp.write(html)
        tmp.flush()
        proc = subprocess.run(
            ["w3m", "-dump", "-T", "text/html", "-cols", "100", tmp.name],
            check=False,
            capture_output=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[:500])
    return proc.stdout.decode("utf-8", errors="replace")


def pull(arxiv_id: str, slug: str) -> bool:
    out = OUT / f"{arxiv_id}-{slug}.txt"
    if out.is_file() and out.stat().st_size > MIN_BYTES:
        print(f"SKIP {out.name}", flush=True)
        return True
    urls = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    ]
    html = None
    used = ""
    for url in urls:
        print(f"  GET {url}", flush=True)
        blob = fetch(url)
        if blob and len(blob) >= MIN_BYTES and b"<html" in blob[:4000].lower():
            html = blob
            used = url
            break
    if html is None:
        print(f"FAIL {arxiv_id} (no HTML)", flush=True)
        return False
    text = html_to_text(html)
    if len(text.strip()) < 1500:
        print(f"FAIL {arxiv_id} (dump too short: {len(text)} chars)", flush=True)
        return False
    header = (
        f"source: {used}\n"
        f"arxiv: https://arxiv.org/abs/{arxiv_id}\n"
        f"note: HTML dump via w3m; not a PDF.\n"
        f"{'=' * 72}\n\n"
    )
    out.write_text(header + text, encoding="utf-8")
    print(f"OK   {out.name} ({out.stat().st_size} bytes)", flush=True)
    return True


def main() -> int:
    if not shutil_which("w3m"):
        print("need w3m on PATH", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for arxiv_id, slug in PAPERS:
        print(f"== {arxiv_id} {slug}", flush=True)
        if not pull(arxiv_id, slug):
            failed.append(arxiv_id)
    print(f"done. ok={len(PAPERS) - len(failed)} fail={len(failed)}", flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
        return 1
    return 0


def shutil_which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


if __name__ == "__main__":
    sys.exit(main())
