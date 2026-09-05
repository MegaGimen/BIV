"""Programmatic env check for Muse agent eval (no GPU)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

TRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from eval.run_harbor import SUITES, harbor_bin  # noqa: E402

ALIYUN_E2B_API_URL = "https://api.us-west-1.e2b.fc.aliyuncs.com"
ALIYUN_E2B_DOMAIN = "us-west-1.e2b.fc.aliyuncs.com"


def load_eval_env() -> None:
    """Load ``train/.env`` (gitignored) and fill Aliyun E2B endpoints if needed."""
    env_path = TRAIN_ROOT / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                os.environ.setdefault(key, val)
    if os.environ.get("E2B_API_KEY") and not os.environ.get("E2B_API_URL"):
        os.environ["E2B_API_URL"] = ALIYUN_E2B_API_URL
        os.environ.setdefault("E2B_DOMAIN", ALIYUN_E2B_DOMAIN)


load_eval_env()


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


def check_environment(*, env: str = "e2b") -> dict[str, Any]:
    load_eval_env()
    report: dict[str, Any] = {"ok": True, "checks": {}, "env": env}

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
    if env == "docker" and not docker_ok:
        report["ok"] = False

    if env == "daytona":
        key_ok = bool(os.environ.get("DAYTONA_API_KEY"))
        report["checks"]["DAYTONA_API_KEY"] = "set" if key_ok else "unset"
        if not key_ok:
            report["ok"] = False
        venv_py = TRAIN_ROOT / ".venv-eval" / "bin" / "python"
        if venv_py.is_file():
            rc_d, out_d = _run([str(venv_py), "-c", "import daytona; print('ok')"])
            report["checks"]["daytona_sdk"] = out_d if rc_d == 0 else f"fail {out_d}"
            if rc_d != 0:
                report["ok"] = False
        else:
            report["checks"]["daytona_sdk"] = "venv python missing"
            report["ok"] = False

    if env == "e2b":
        key_ok = bool(os.environ.get("E2B_API_KEY"))
        url = os.environ.get("E2B_API_URL", "")
        domain = os.environ.get("E2B_DOMAIN", "")
        report["checks"]["E2B_API_KEY"] = "set" if key_ok else "unset"
        report["checks"]["E2B_API_URL"] = url or "unset"
        report["checks"]["E2B_DOMAIN"] = domain or "unset"
        if not key_ok or not url or not domain:
            report["ok"] = False
        venv_py = TRAIN_ROOT / ".venv-eval" / "bin" / "python"
        if venv_py.is_file():
            rc_e, out_e = _run(
                [str(venv_py), "-c", "import e2b; print(getattr(e2b, '__version__', 'ok'))"]
            )
            report["checks"]["e2b_sdk"] = out_e if rc_e == 0 else f"fail {out_e}"
            if rc_e != 0:
                report["ok"] = False
        else:
            report["checks"]["e2b_sdk"] = "venv python missing"
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
    r = check_environment(env=os.environ.get("HARBOR_ENV", "e2b"))
    print(format_report(r))
    print(json.dumps(r, indent=2, ensure_ascii=False))
