#!/usr/bin/env python3
"""Offline checks for row-MAV and incomplete-cache detection. No GPU, no hub."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
MERGE = Path(__file__).resolve().parents[2] / "merge"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MERGE) not in sys.path:
    sys.path.insert(0, str(MERGE))

from biv_wm.mav import row_mav  # noqa: E402
from download import checkpoint_ready  # noqa: E402


class _T:
    """Tiny tensor stand-in: abs / mean / detach / float, no torch."""

    def __init__(self, data: list, ndim: int | None = None) -> None:
        self._data = data
        self.ndim = 0 if not isinstance(data, list) else (1 if data and not isinstance(data[0], list) else 2)
        if ndim is not None:
            self.ndim = ndim

    def detach(self) -> "_T":
        return self

    def float(self) -> "_T":
        return self

    def abs(self) -> "_T":
        if self.ndim == 0:
            return _T(abs(self._data), ndim=0)
        if self.ndim == 1:
            return _T([abs(x) for x in self._data], ndim=1)
        return _T([[abs(x) for x in row] for row in self._data], ndim=2)

    def mean(self, dim: int | None = None) -> "_T":
        if self.ndim == 0:
            return _T(float(self._data), ndim=0)
        if self.ndim == 1:
            return _T(sum(self._data) / len(self._data), ndim=0)
        if dim == -1:
            return _T([sum(row) / len(row) for row in self._data], ndim=1)
        flat = [x for row in self._data for x in row]
        return _T(sum(flat) / len(flat), ndim=0)

    def item(self) -> float:
        return float(self._data)


def test_row_mav_matrix() -> None:
    # rows [1, 3] and [5, 7] → row MAVs 2 and 6 → mean 4
    t = _T([[1.0, -3.0], [5.0, -7.0]])
    got = row_mav(t)
    assert abs(got - 4.0) < 1e-9, got


def test_row_mav_vector() -> None:
    t = _T([2.0, -4.0, 6.0], ndim=1)
    got = row_mav(t)
    assert abs(got - 4.0) < 1e-9, got


def test_checkpoint_ready(tmp_path: Path | None = None) -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        d = Path(raw)
        assert checkpoint_ready(d) is False
        (d / "config.json").write_text("{}", encoding="utf-8")
        assert checkpoint_ready(d) is False
        (d / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        assert checkpoint_ready(d) is False
        (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
        assert checkpoint_ready(d) is True


def main() -> None:
    test_row_mav_matrix()
    test_row_mav_vector()
    test_checkpoint_ready()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
