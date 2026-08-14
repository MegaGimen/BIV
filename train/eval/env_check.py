"""Programmatic env check for Muse agent eval (no GPU)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

TRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from eval.run_harbor import SUITES, harbor_bin  # noqa: E402


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def check_environment() -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "checks": {}}

    try:
        hb = harbor_bin()
        report["checks"]["harbor_bin"] = hb
        venv_py = TRAIN_ROOT / ".venv-eval" / "bin" / "python"
        if venv_py.is_file():
            rc2, out2 = _run(
                [str(venv_py), "-c", "import harbor; print(harbor.__version__)"]
            )
            report["checks"]["harbor_version"] = out2 if rc2 == 0 else f"rc={rc2}"
        else:
            report["checks"]["harbor_version"] = "venv python missing"
            report["ok"] = False
    except SystemExit as e:
        report["ok"] = False
        report["checks"]["harbor_bin"] = str(e)

    rc, out = _run(["docker", "info"])
    docker_ok = rc == 0 and "Server Version" in out
    report["checks"]["docker"] = "ok" if docker_ok else f"fail rc={rc}"
    if not docker_ok:
        report["ok"] = False

    rc, _ = _run(["nvidia-smi", "-L"])
    report["checks"]["gpu"] = "present" if rc == 0 else "none (expected on this app host)"

    venv_py = TRAIN_ROOT / ".venv-eval" / "bin" / "python"
    agents_ok: dict[str, bool] = {}
    if venv_py.is_file():
        code = (
            "from harbor.agents.factory import AgentFactory\n"
            "for n in ['terminus-2','mini-swe-agent']:\n"
            "  try:\n"
            "    AgentFactory.get_agent_class(n); print(n+'=OK')\n"
            "  except Exception as e:\n"
            "    print(n+'=FAIL:'+type(e).__name__)\n"
        )
        rc, out = _run([str(venv_py), "-c", code])
        for line in out.splitlines():
            if "=" in line:
                name, status = line.split("=", 1)
                agents_ok[name] = status.startswith("OK")
    report["checks"]["agents"] = agents_ok
    if not all(agents_ok.values()) or not agents_ok:
        report["ok"] = False

    report["suites"] = {
        sid: {"dataset": s["dataset"], "agent": s["agent"]} for sid, s in SUITES.items()
    }

    usage = shutil.disk_usage(TRAIN_ROOT)
    report["checks"]["disk_free_gb"] = round(usage.free / (1024**3), 1)

    meta = TRAIN_ROOT / "eval" / "meta_reference.json"
    report["checks"]["meta_reference"] = meta.is_file()
    if not meta.is_file():
        report["ok"] = False

    return report


def format_report(report: dict[str, Any]) -> str:
    lines = ["=== Muse agent-eval env check ===", f"ok={report.get('ok')}"]
    for k, v in (report.get("checks") or {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("suites:")
    for sid, info in (report.get("suites") or {}).items():
        lines.append(f"  - {sid}: dataset={info['dataset']} agent={info['agent']}")
    return "\n".join(lines)


if __name__ == "__main__":
    r = check_environment()
    print(format_report(r))
    print(json.dumps(r, indent=2, ensure_ascii=False))
