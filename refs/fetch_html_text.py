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
    # Fill theory gaps: identifiability, explicit z, compile T→π
    ("2202.03169", "citris-identifiability"),
    ("2206.06169", "icitris"),
    ("1907.04809", "ivae"),
    ("1501.01332", "invariant-causal-prediction"),
    ("2112.03321", "noether-networks"),
    ("1906.02736", "deepmdp"),
    ("2006.10742", "dbc-bisimulation"),
    ("1907.00953", "slac"),
    ("2206.15477", "denoised-mdp"),
    ("2001.08837", "kg-a2c"),
    ("2209.06356", "mdp-homomorphism-fb"),
    ("1606.05312", "successor-features"),
    ("1906.08253", "mbpo"),
    ("1707.03497", "value-prediction-networks"),
    ("1106.3538", "horde-gvf"),
    # 2026-08 round: close the three gaps (see refs/README.md sections I-O)
    # I. predict-well != learned-the-law
    ("2507.06952", "foundation-model-inductive-bias-probe"),
    ("2406.03689", "eval-implicit-world-model"),
    ("2602.06923", "kepler-to-newton-inductive-biases"),
    ("2502.11831", "intuitive-physics-vjepa"),
    ("2602.05903", "adversarial-seq-verify-implicit-wm"),
    # J. identifiability under unknown multi-node interventions
    ("2406.05937", "crl-unknown-multinode"),
    ("2311.02695", "multinode-interventions-crl"),
    ("2306.00542", "nonparam-crl-unknown-interventions"),
    ("2603.25796", "beyond-identifiability-few-envs"),
    ("2402.00849", "score-based-crl"),
    ("2310.15450", "general-identifiability-crl"),
    ("2505.18374", "shioenv-shell-irreducibility"),
    ("2107.00793", "causal-neural-connection-cht"),
    # K. explicit latent state z in text; policy reads only z
    ("2606.27681", "textual-belief-states-strict-mediation"),
    ("2511.05963", "nextlat-compact-world-models"),
    ("2207.08229", "ac-state-control-endogenous"),
    ("2306.06561", "ifactor-identifiable-factorization"),
    ("2207.05738", "pac-rl-psr"),
    # L. world understanding -> agent: low-rank MDP + transfer bounds + T->pi
    ("2006.10814", "flambe"),
    ("2110.04652", "rep-ucb"),
    ("2208.09515", "speder-spectral-decomposition"),
    ("2512.15036", "spectral-representation-rl"),
    ("2206.05900", "provable-multitask-repr-rl"),
    ("2205.14571", "provable-repr-transfer-rl"),
    ("2210.10464", "power-of-pretraining-generalization-rl"),
    ("2209.14935", "forward-backward-zero-shot-rl"),
    ("2502.10790", "best-features-for-successor-features"),
    ("2411.19418", "proto-successor-measure"),
    ("2212.03319", "understanding-self-predictive-learning"),
    ("2401.08898", "bridging-state-history-representations"),
    ("2307.12933", "mpdp-policy-improvement-distilled"),
    ("1912.02807", "save-amortized-value-estimation"),
    ("2406.11907", "chessbench-amortized-planning"),
    ("2605.08732", "latent-geometry-beyond-search"),
    ("2510.09577", "dyna-mind"),
    ("2602.05327", "proact-search-tree-distill"),
    ("2606.27483", "internalizing-the-future"),
    ("2510.15047", "spa-self-play-agent"),
    ("2606.02372", "comap-coevolving-wm-policy"),
    # M. when auxiliary tasks help vs hurt
    ("2406.17718", "when-does-self-prediction-help"),
    ("2005.00944", "info-transfer-mtl"),
    ("2607.16554", "capacity-redundancy-mtl"),
    ("1812.02224", "adapting-aux-losses-gradient-similarity"),
    # N. evaluation: VoE, counterfactual, probe != cause
    ("2506.09849", "intphys2"),
    ("2605.30346", "yocausal-reverse-surprise"),
    ("2604.12493", "latent-planning-emerges-with-scale"),
    ("2605.07984", "wheres-the-plan-causal-probe"),
    ("2602.16698", "causality-key-for-interpretability"),
    ("2604.06427", "depth-ceiling-latent-planning"),
    ("2311.01460", "implicit-cot-distillation"),
    ("2502.21074", "codi-continuous-cot"),
    ("2509.20317", "sim-cot-step-supervision"),
    ("2602.00449", "do-latent-cot-think-stepwise"),
    # O. remaining holes: free-text actions, active intervention design
    ("2203.08248", "ts3-nonlinear-rl-large-action-spaces"),
    ("2405.16718", "caasl-amortized-intervention-design"),
    ("2006.05690", "active-invariant-causal-prediction"),
    ("2203.02016", "interventions-where-and-how"),
    # P. free-text / large action spaces via action representations
    ("1902.00183", "chandak-action-representations"),
    ("2010.04444", "joint-state-action-embedding"),
    ("1512.07679", "wolpertinger-large-discrete-actions"),
    ("1902.01119", "act2vec-natural-language-of-actions"),
    ("2011.01928", "generalization-to-new-actions"),
    ("2503.08867", "aglo-zero-shot-action-generalization"),
    ("2306.02451", "sale-td7"),
    ("2604.07016", "opsr-predictive-repr-skill-transfer"),
    # Q. programmatic / executable world models in text environments
    ("2605.30880", "patchworld-fidelity-utility-tradeoff"),
    ("2510.12088", "onelife-symbolic-wm-one-life"),
    ("2605.16725", "alice-baba-in-wonderland"),
    ("2602.10480", "nesys-neuro-symbolic-wm"),
    ("2503.20124", "theorycoder"),
    ("2602.00929", "theorycoder-2-learned-abstractions"),
    ("2505.10819", "poe-world"),
    ("2606.16070", "mind-studio-executable-wm"),
    ("2607.01531", "opine-world-ontology-error"),
    ("2605.01293", "nsi-programmatic-skill-induction"),
    ("2503.23145", "codearc-inductive-synthesis"),
    # R. who picks do(a): active intervention, learning-progress curiosity
    ("2007.07853", "awml-progress-curiosity"),
    ("2604.18701", "curiosity-critic-epistemic-vs-aleatoric"),
    ("2412.12098", "maxinforl"),
    ("2507.02639", "pts-be-bayesian-exploration"),
    ("2606.19476", "icl-intrinsic-curiosity-impossibility"),
    # S. law in weights vs in context; diversity threshold for rule generalization
    ("2210.05675", "chan-context-vs-weights-generalization"),
    ("2205.05055", "data-distributional-properties-icl"),
    ("2306.15063", "task-diversity-threshold-icl"),
    ("2412.00104", "differential-learning-kinetics-memorization"),
    ("2512.18634", "shortcut-to-induction-head"),
    ("2410.23042", "toward-understanding-icl-vs-iwl"),
    ("2311.08360", "transient-nature-of-icl"),
    ("2406.00053", "dual-process-weight-forgetting"),
    ("2306.00802", "birth-of-a-transformer"),
    ("2312.03002", "reddy-abrupt-learning-induction-head"),
    ("2604.12151", "distinct-mechanisms-icl-phases"),
    # T. architectural hard limits on state tracking without CoT
    ("2404.08819", "illusion-of-state-ssm"),
    ("2402.12875", "cot-serial-problems"),
    ("2310.07923", "expressive-power-transformers-cot"),
    ("2503.03961", "log-depth-transformers"),
    ("2505.18948", "transformers-with-padding"),
    ("2411.12537", "lrnn-negative-eigenvalues-state-tracking"),
    ("2603.03612", "why-linear-rnns-parallelizable"),
    ("2602.14814", "state-tracking-from-code-repl"),
    ("2412.06464", "gated-delta-networks"),
    ("2502.10297", "deltaproduct-householder"),
    ("2505.17761", "slices-structured-linear-cdes"),
    ("2603.14360", "m2rnn-matrix-valued-states"),
    ("2607.07386", "sparse-delta-memory"),
    # U. objective mismatch: observation CE is provably the wrong loss for control
    ("2204.01464", "vagram-value-gradient-model-learning"),
    # note: Farahmand's Iterative VAML (NeurIPS 2018) has no matching arXiv HTML
    ("2106.14080", "model-advantage-value-aware"),
    ("2011.03506", "value-equivalence-principle"),
    ("2206.02072", "deciding-what-to-model-rate-distortion"),
    ("2010.11876", "error-bounds-imitating-environments"),
    ("2406.16249", "optimal-tightness-simulation-lemma"),
    ("2505.22772", "calibrated-value-aware-model-learning"),
    ("2106.03273", "control-oriented-mbrl-implicit-diff"),
    ("2605.29032", "policy-aware-simulator-learning"),
    ("2409.12799", "central-role-of-loss-function-rl"),
    # W. execution-aware pretraining: the closest empirical precedent (code domain)
    ("2306.07487", "traced-execution-aware-pretraining"),
    ("2406.01006", "semcoder-monologue-reasoning"),
    ("2503.05703", "execution-tuning-dynamic-scratchpad"),
    ("2605.11922", "stepcodereasoner"),
    ("2604.03253", "self-execution-simulation"),
    ("2404.14662", "next-reason-about-execution"),
    ("2305.05383", "codeexecutor"),
    ("2304.12743", "tracefixer"),
    # note: "Towards Effectively Leveraging Execution Traces for Program Repair"
    # has no usable arXiv HTML; see ACL DOI 10.18653/v1/2025.knowledgenlp-1.17
    # 2026-09: collapse / Δz inverse / text JEPA paradox (probe after 32k run)
    ("2410.13232", "web-agents-world-models"),
    ("2607.23531", "jepa-paradox-language"),
    ("2606.31689", "scratchworld-executable-consequences"),
    ("2606.31232", "delta-jepa-latent-difference"),
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
