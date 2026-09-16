#!/usr/bin/env python3
"""CPU smoke: unwrap mix_v2 JSON the same way Enc(h)/Enc(a)/Enc(h,a,o) will.

No GPU, no tokenizer, no 35B. Fixtures match AutoDL mix_v2 keysets
(user = tool+arguments, assistant = output+isError; result is an alias).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biv_wm.strip_json import unwrap_content, unwrap_message, unwrap_messages  # noqa: E402

# Copied from mix_v2 on the AutoDL CPU box (wm_code / wm_os train.jsonl).
WM_CODE_USER = (
    '{"tool": "execute_bash", "arguments": {"command": '
    '"cd /workspace/pandas-dev__pandas__1.0 && python reproduce_issue.py"}}'
)
WM_CODE_VIEW = (
    '{"tool": "str_replace_editor", "arguments": '
    '{"path": "/workspace/pandas-dev__pandas__1.0/README.md", '
    '"command": "view", "view_range": [1, 50]}}'
)
WM_OS_USER = (
    '{"tool": "exec", "arguments": {"command": '
    '"which sag 2>/dev/null; command -v sag 2>/dev/null"}}'
)
WM_OS_WRITE = (
    '{"tool": "write", "arguments": {"path": "/workspace/inventory_audit.md", '
    '"content": "# Valuation Audit"}}'
)
OBS_OK = '{"output": "a.txt", "isError": false}'
OBS_EMPTY = '{"output": "", "isError": false}'
OBS_ERR = '{"output": "This operation was aborted", "isError": true}'
OBS_RESULT_ALIAS = '{"result": "a.txt", "isError": false}'


def test_unwrap_mix_observation() -> None:
    assert unwrap_content(OBS_OK) == "a.txt"
    assert unwrap_content(OBS_EMPTY) == ""
    assert unwrap_content(OBS_ERR) == "error\nThis operation was aborted"
    assert unwrap_content(OBS_RESULT_ALIAS) == "a.txt"
    assert "isError" not in unwrap_content(OBS_OK)
    assert "output" not in unwrap_content(OBS_OK)


def test_unwrap_mix_tool() -> None:
    bash = unwrap_content(WM_CODE_USER)
    assert "execute_bash" in bash
    assert "python reproduce_issue.py" in bash
    assert '"tool"' not in bash
    assert "arguments" not in bash

    view = unwrap_content(WM_CODE_VIEW)
    assert "str_replace_editor" in view
    assert "README.md" in view
    assert "view_range" in view

    os_exec = unwrap_content(WM_OS_USER)
    assert os_exec.startswith("exec")
    assert "which sag" in os_exec

    write = unwrap_content(WM_OS_WRITE)
    assert write.startswith("write")
    assert "/workspace/inventory_audit.md" in write
    assert "# Valuation Audit" in write


def test_unwrap_identity_and_idempotent() -> None:
    plain = "rm a.txt"
    assert unwrap_content(plain) == plain
    twice = unwrap_content(unwrap_content(OBS_OK))
    assert twice == "a.txt"
    assert unwrap_content("") == ""
    assert unwrap_content("{not json") == "{not json"


def test_unwrap_messages_ls_rm_row() -> None:
    """Same mix row the training loop splits: ls then rm a.txt."""
    msgs = [
        {"role": "system", "content": "You are an environment dynamics model."},
        {
            "role": "user",
            "content": json.dumps(
                {"tool": "execute_bash", "arguments": {"command": "ls"}},
                ensure_ascii=False,
            ),
        },
        {"role": "assistant", "content": json.dumps({"output": "a.txt", "isError": False})},
        {
            "role": "user",
            "content": json.dumps(
                {"tool": "execute_bash", "arguments": {"command": "rm a.txt"}},
                ensure_ascii=False,
            ),
        },
        {"role": "assistant", "content": json.dumps({"output": "", "isError": False})},
    ]
    out = unwrap_messages(msgs)
    assert out[0]["content"].startswith("You are")
    assert "ls" in out[1]["content"]
    assert out[2]["content"] == "a.txt"
    assert "rm a.txt" in out[3]["content"]
    assert out[4]["content"] == ""
    for m in out[1:]:
        assert "isError" not in m["content"]
        assert '"tool"' not in m["content"]


def test_unwrap_message_keeps_role() -> None:
    m = unwrap_message({"role": "assistant", "content": OBS_OK})
    assert m["role"] == "assistant"
    assert m["content"] == "a.txt"


def test_sigreg_quadrature_matches_lejepa() -> None:
    """LeJEPA EppsPulley: t in [0, 3], 17 knots, w(t)=exp(-t²/2), trapz with even fold."""
    import math

    t_max, n_points = 3.0, 17
    assert n_points % 2 == 1
    dt = t_max / (n_points - 1)
    t = [i * dt for i in range(n_points)]
    phi = [math.exp(-0.5 * x * x) for x in t]
    weights = [2.0 * dt] * n_points
    weights[0] = dt
    weights[-1] = dt
    quad = [w * p for w, p in zip(weights, phi)]
    assert abs(t[0] - 0.0) < 1e-12
    assert abs(t[-1] - t_max) < 1e-12
    assert abs(phi[0] - 1.0) < 1e-12
    target = math.sqrt(2.0)
    nearest = min(range(n_points), key=lambda i: abs(t[i] - target))
    assert abs(phi[nearest] - math.exp(-0.5 * t[nearest] ** 2)) < 1e-12
    assert quad[0] < quad[1]
    assert all(q > 0 for q in quad)


def test_sigreg_gaussian_below_constant_numpy() -> None:
    """Same Epps–Pulley as sigreg.py, on numpy, no torch: N(0,I) < Dirac/constant."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    t_max, n_points, slices = 3.0, 17, 32
    dt = t_max / (n_points - 1)
    t = np.linspace(0.0, t_max, n_points)
    phi = np.exp(-0.5 * t * t)
    w = np.full(n_points, 2.0 * dt)
    w[0] = dt
    w[-1] = dt
    quad = w * phi

    def sigreg(z: np.ndarray) -> float:
        n, dim = z.shape
        stats = []
        for _ in range(slices):
            a = rng.normal(size=dim)
            a = a / (np.linalg.norm(a) + 1e-12)
            x = z @ a
            xt = np.outer(x, t)
            cos_mean = np.cos(xt).mean(axis=0)
            sin_mean = np.sin(xt).mean(axis=0)
            err = (cos_mean - phi) ** 2 + sin_mean ** 2
            stats.append(float((err * quad).sum() * n))
        return float(sum(stats) / len(stats))

    gauss = rng.normal(size=(256, 16))
    const = np.ones((256, 16))
    loss_g = sigreg(gauss)
    loss_c = sigreg(const)
    assert math.isfinite(loss_g) and math.isfinite(loss_c)
    assert loss_g < loss_c, (loss_g, loss_c)


def main() -> None:
    test_unwrap_mix_observation()
    test_unwrap_mix_tool()
    test_unwrap_identity_and_idempotent()
    test_unwrap_messages_ls_rm_row()
    test_unwrap_message_keeps_role()
    test_sigreg_quadrature_matches_lejepa()
    test_sigreg_gaussian_below_constant_numpy()
    print("ok", flush=True)



if __name__ == "__main__":
    main()
