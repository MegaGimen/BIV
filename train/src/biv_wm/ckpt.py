"""Checkpoint names/rotation matching Muse Glimmer (train_muse_trl.py).

Rolling mid-run: ``checkpoint-e{epoch}-s{step}`` (keep newest ``save_total_limit``).
Epoch-end permanent: ``checkpoint-epoch{N}-end-s{step}`` (never rotated).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

ROLLING_RE = re.compile(r"^checkpoint-e(\d+)-s(\d+)$")
EPOCH_END_RE = re.compile(r"^checkpoint-epoch(\d+)-end-s(\d+)$")
HF_DIGIT_RE = re.compile(r"^checkpoint-(\d+)$")
LEGACY_STEP_RE = re.compile(r"^step-(\d+)$")


def rolling_name(epoch: int, step: int) -> str:
    return f"checkpoint-e{int(epoch)}-s{int(step)}"


def epoch_end_name(epoch: int, step: int) -> str:
    """``epoch`` is 1-based completed-epoch index, same as Muse ``round(state.epoch)``."""
    return f"checkpoint-epoch{int(epoch)}-end-s{int(step)}"


def parse_ckpt_name(name: str) -> tuple[int, int, int] | None:
    """``(epoch, step, kind)`` with kind 0=digit/legacy, 1=rolling, 2=epoch-end."""
    m = EPOCH_END_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), 2
    m = ROLLING_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), 1
    m = HF_DIGIT_RE.match(name)
    if m:
        return 0, int(m.group(1)), 0
    m = LEGACY_STEP_RE.match(name)
    if m:
        return 0, int(m.group(1)), 0
    return None


def plain_cpu_tensor(param: Any):
    """Detach a (possibly DTensor) value onto contiguous CPU storage.

    ``torch.save(DTensor)`` round-trips as DTensor. ``load_state_dict`` into a
    plain ``nn.Parameter`` then raises mixed Tensor/DTensor. Always persist and
    reload the materialized full tensor. ``full_tensor`` is a collective: every
    rank that holds the DTensor must call this together.
    """
    t = param.full_tensor() if hasattr(param, "full_tensor") else param
    t = t.detach()
    device = getattr(t, "device", None)
    if getattr(device, "type", "cpu") != "cpu":
        t = t.to("cpu")
    return t.contiguous().clone()


def plain_cpu_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sd.items():
        if hasattr(v, "full_tensor") or hasattr(v, "detach"):
            out[k] = plain_cpu_tensor(v)
        else:
            out[k] = v
    return out


def train_state_name(rank: int) -> str:
    """Per-rank AdamW + scheduler + RNG. Optional: old ckpts omit this file."""
    return f"train_state.rank{int(rank)}.pt"


def capture_rng_state() -> dict[str, Any]:
    """Python / NumPy / torch / CUDA generators. Call on every rank."""
    import random

    payload: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        payload["numpy"] = np.random.get_state()
    except Exception:
        payload["numpy"] = None
    try:
        import torch

        payload["torch"] = torch.get_rng_state()
        payload["cuda"] = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
    except Exception:
        payload["torch"] = None
        payload["cuda"] = None
    return payload


def restore_rng_state(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    import random

    py = payload.get("python")
    if py is not None:
        random.setstate(py)
    ns = payload.get("numpy")
    if ns is not None:
        try:
            import numpy as np

            np.random.set_state(ns)
        except Exception:
            pass
    try:
        import torch

        t = payload.get("torch")
        if t is not None:
            torch.set_rng_state(t)
        cuda = payload.get("cuda")
        if cuda is not None and torch.cuda.is_available():
            n = torch.cuda.device_count()
            if len(cuda) == n:
                torch.cuda.set_rng_state_all(cuda)
            elif len(cuda) >= 1:
                torch.cuda.set_rng_state(cuda[0])
    except Exception:
        pass


def local_cpu_value(v: Any):
    """Shard-local CPU clone. Optimizer DTensors use ``to_local``, never ``full_tensor``."""
    if not (hasattr(v, "detach") and hasattr(v, "device")):
        return v
    t = v.to_local() if hasattr(v, "to_local") else v
    t = t.detach()
    device = getattr(t, "device", None)
    if getattr(device, "type", "cpu") != "cpu":
        t = t.to("cpu")
    if hasattr(t, "contiguous"):
        t = t.contiguous()
    return t.clone() if hasattr(t, "clone") else t


def tree_map_local_cpu(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: tree_map_local_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_map_local_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tree_map_local_cpu(v) for v in obj)
    return local_cpu_value(obj)


def copy_optimizer_state(
    opt,
    saved: dict[str, Any],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> bool:
    """Load AdamW ``state_dict``; on DTensor mismatch copy local shards. False = leave opt as-is."""
    import torch

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    if not isinstance(saved, dict) or "state" not in saved or "param_groups" not in saved:
        _log("AdamW blob missing state/param_groups")
        return False
    try:
        opt.load_state_dict(saved)
        return True
    except Exception as e:
        _log(f"AdamW load_state_dict: {e!r}; copying local shards")
    try:
        id_map: dict[Any, Any] = {}
        for live_g, saved_g in zip(opt.param_groups, saved["param_groups"]):
            for p, pid in zip(live_g["params"], saved_g["params"]):
                id_map[pid] = p
                try:
                    id_map[int(pid)] = p
                except (TypeError, ValueError):
                    pass
                id_map[str(pid)] = p
            for k, v in saved_g.items():
                if k != "params":
                    live_g[k] = v
        for pid, st in saved["state"].items():
            p = id_map.get(pid)
            if p is None:
                try:
                    p = id_map.get(int(pid))
                except (TypeError, ValueError):
                    p = None
            if p is None or not isinstance(st, dict):
                continue
            live = opt.state.setdefault(p, {})
            ref = p.to_local() if hasattr(p, "to_local") else p
            for k, v in st.items():
                if hasattr(v, "detach") and hasattr(v, "shape"):
                    src = v.detach()
                    if k == "step":
                        live[k] = src.to(device=ref.device).clone()
                        continue
                    buf = live.get(k)
                    need = tuple(src.shape)
                    if buf is None or tuple(getattr(buf, "shape", ())) != need:
                        buf = torch.zeros(need, dtype=ref.dtype, device=ref.device)
                        live[k] = buf
                    dst = buf.to_local() if hasattr(buf, "to_local") else buf
                    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
                else:
                    live[k] = v
        return True
    except Exception as e:
        _log(f"AdamW local copy failed: {e!r}")
        return False


def canonical_lora_key(name: str) -> str:
    """Strip FSDP/checkpoint wrappers and PEFT ``.default.`` so save keys match live names."""
    out = name
    for wrap in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."):
        while out.startswith(wrap):
            out = out[len(wrap) :]
        out = out.replace("." + wrap, ".")
    for src, dst in (
        (".lora_A.default.", ".lora_A."),
        (".lora_B.default.", ".lora_B."),
        (".lora_embedding_A.default.", ".lora_embedding_A."),
        (".lora_embedding_B.default.", ".lora_embedding_B."),
    ):
        out = out.replace(src, dst)
    return out


def ckpt_complete(path: Path, *, require_jepa: bool = True) -> bool:
    """trainer_state.json + weights. Mediated Stage 1 also needs jepa.pt and ldad.pt.

    ``train_state.rank{N}.pt`` (AdamW + scheduler + RNG) is optional so older
    weight-only checkpoints still count as complete.
    """
    if not path.is_dir():
        return False
    if not (path / "trainer_state.json").is_file():
        return False
    has_weights = (
        (path / "adapter_model.safetensors").is_file()
        or (path / "pytorch_model_fsdp.bin").is_file()
        or any(path.glob("*.safetensors"))
    )
    if not has_weights:
        return False
    if require_jepa and (
        not (path / "jepa.pt").is_file() or not (path / "ldad.pt").is_file()
    ):
        return False
    return True


def find_latest_ckpt(out_dir: Path, *, require_jepa: bool = True) -> Path | None:
    """Newest complete ckpt. Rank key ``(epoch, step, kind)`` — epoch first, then step.

    Same as Muse / daemon. Rolling ``checkpoint-e{epoch}-s{step}`` uses the 0-based
    epoch index from the training loop; epoch-end uses a 1-based completed epoch,
    so it ranks above that epoch's rolling dirs.
    """
    if not out_dir.is_dir():
        return None
    best: tuple[int, int, int, Path] | None = None
    for p in out_dir.iterdir():
        parsed = parse_ckpt_name(p.name)
        if parsed is None or not ckpt_complete(p, require_jepa=require_jepa):
            continue
        epoch, step, kind = parsed
        key = (epoch, step, kind)
        if best is None or key > (best[0], best[1], best[2]):
            best = (epoch, step, kind, p)
    return None if best is None else best[3]


def rotate_rolling(out: Path, limit: int, *, log: Callable[[str], None] | None = None) -> list[str]:
    """Delete oldest rolling (and leftover digit/step-*) dirs past ``limit``. Epoch-end kept."""
    if limit is None or int(limit) <= 0 or not out.is_dir():
        return []
    rolling: list[tuple[int, Path]] = []
    for p in out.iterdir():
        if not p.is_dir():
            continue
        m = ROLLING_RE.match(p.name)
        if m:
            rolling.append((int(m.group(2)), p))
            continue
        m2 = HF_DIGIT_RE.match(p.name) or LEGACY_STEP_RE.match(p.name)
        if m2:
            rolling.append((int(m2.group(1)), p))
    rolling.sort(key=lambda t: t[0])
    removed: list[str] = []
    while len(rolling) > int(limit):
        _, victim = rolling.pop(0)
        removed.append(victim.name)
        if log is not None:
            log(f"rotate: remove {victim.name}")
        shutil.rmtree(victim, ignore_errors=True)
    return removed


def write_trainer_state(path: Path, *, epoch: int, global_step: int, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    if extra:
        payload.update(extra)
    (path / "trainer_state.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
