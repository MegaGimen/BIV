"""SIGReg: sketched isotropic-Gaussian regularizer (LeJEPA / LeWM).

LeJEPA Definition 2 / eq. (5):

    SIGReg_T(A, {f(x_n)}) = (1/|A|) Σ_{a ∈ A} T({aᵀ f(x_n)}_{n=1..N})

T is the Epps–Pulley statistic against N(0,1). Official code
(`galilai-group/lejepa`, `EppsPulley` + `SlicingUnivariateTest`):

    φ₀(t) = exp(-t²/2)                 # standard-normal characteristic function
    φ̂(t) = (1/N) Σ_j exp(i t X_j)     # empirical CF of one 1-D projection
    T     = N ∫ |φ̂(t) - φ₀(t)|² w(t) dt
    w(t)  = exp(-t²/2)

Quadrature is trapezoid on t ∈ [0, t_max] with ``n_points`` knots
(default t_max=3, 17 knots, odd so Simpson would also work). The
integrand is even, so the official weights double the interior of the
positive half-line (2·Δt) and keep Δt at the endpoints. Directions a
are random unit vectors, redrawn every forward, then T is averaged
over |A| slices (paper default 1024).

This file copies that statistic, not the DDP all-reduce inside the
upstream module: our FSDP2+CP mesh does not replicate the same minibatch
on every rank (CP already gathered the sequence; dp_replicate holds a
different row). CF averaging across ranks would mix two samples.

Stage 1 batch_size is 1, so last-token-only {z} is N=1 and T is
undefined in practice. The cloud here is non-pad last-layer tokens of
Enc(h,a,o), plus the three last-token vectors (z_t, u, z_{t+1}) so the
readout SIGReg is actually pushing on is in the same set.

Torch is imported inside the functions so `import train_jepa` still
works on CPU boxes without a CUDA torch.
"""

from __future__ import annotations

from typing import Any, Iterable


def _quadrature(t_max: float, n_points: int, *, device, dtype):
    import torch

    if n_points < 2 or n_points % 2 == 0:
        raise ValueError(f"n_points must be odd and >= 3, got {n_points}")
    t = torch.linspace(0.0, float(t_max), int(n_points), device=device, dtype=dtype)
    dt = float(t_max) / (int(n_points) - 1)
    weights = torch.full((int(n_points),), 2.0 * dt, device=device, dtype=dtype)
    weights[0] = dt
    weights[-1] = dt
    phi = torch.exp(-0.5 * t.square())
    return t, phi, weights * phi


def epps_pulley(proj: Any, t_max: float = 3.0, n_points: int = 17) -> Any:
    """Epps–Pulley T for already-projected samples.

    ``proj`` is [N, K] (K slices). Returns [K], each entry
    N ∫ |φ̂ - φ₀|² w dt on that slice. Matches official `EppsPulley.forward`
    without the world-size multiplier.
    """
    import torch

    if proj.ndim != 2:
        raise ValueError(f"proj must be [N, K], got {tuple(proj.shape)}")
    n = proj.size(0)
    work = proj.float()
    t, phi, quad_w = _quadrature(t_max, n_points, device=work.device, dtype=torch.float32)
    xt = work.unsqueeze(-1) * t
    cos_mean = torch.cos(xt).mean(dim=0)
    sin_mean = torch.sin(xt).mean(dim=0)
    err = (cos_mean - phi).square() + sin_mean.square()
    return (err * quad_w).sum(dim=-1) * float(n)


def sigreg_loss(
    embeddings: Any,
    *,
    num_slices: int = 1024,
    t_max: float = 3.0,
    n_points: int = 17,
    generator: Any = None,
) -> Any:
    """Mean Epps–Pulley over random unit slices. ``embeddings`` is [N, D]."""
    import torch
    import torch.nn.functional as F

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N, D], got {tuple(embeddings.shape)}")
    n, dim = embeddings.shape
    if n < 2 or dim < 1:
        return embeddings.sum() * 0.0
    slices = max(1, int(num_slices))
    work = embeddings.float()
    directions = torch.randn(
        dim, slices, device=work.device, dtype=work.dtype, generator=generator
    )
    directions = F.normalize(directions, p=2, dim=0)
    stats = epps_pulley(work @ directions, t_max=t_max, n_points=n_points)
    return stats.mean()


def subsample_token_hiddens(
    hidden: Any,
    attention_mask: Any,
    max_tokens: int,
) -> Any:
    """Evenly spaced non-pad last-layer tokens, shape [N, D], grads kept."""
    import torch

    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, S, D], got {tuple(hidden.shape)}")
    flat = hidden[attention_mask.bool()]
    n = int(flat.size(0))
    cap = max(0, int(max_tokens))
    if n == 0 or cap == 0:
        return hidden.new_zeros((0, hidden.size(-1)))
    if n <= cap:
        return flat
    idx = torch.linspace(0, n - 1, cap, device=hidden.device)
    return flat.index_select(0, idx.long())


def sigreg_cloud(
    hidden: Any,
    attention_mask: Any,
    extra: Iterable[Any] | None = None,
    max_tokens: int = 512,
) -> Any:
    """Token subsample from Enc(h,a,o) plus last-token vectors (z_t, u, z_next)."""
    import torch

    parts = [subsample_token_hiddens(hidden, attention_mask, max_tokens)]
    if extra is not None:
        for vec in extra:
            if vec is None:
                continue
            v = vec if vec.ndim == 2 else vec.unsqueeze(0)
            parts.append(v.to(device=hidden.device, dtype=hidden.dtype))
    nonempty = [p for p in parts if p.numel() > 0]
    if not nonempty:
        return hidden.new_zeros((0, hidden.size(-1)))
    return torch.cat(nonempty, dim=0)
