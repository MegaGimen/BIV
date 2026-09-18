"""Qwen3.5 GatedDeltaNet context parallel (Megatron-style all-to-all).

Sequence-sharded activations ``[B, S/cp, F]`` become head-sharded
``[B, S, F/cp]`` so causal conv + ``chunk_gated_delta_rule`` see the full
timeline. Full-attention layers keep torch SDPA-CP. Per-token MoE is local.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh


_PATCHED = "_biv_qwen_gdn_cp_orig_forward"


def _all_to_all_single(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Dim-0 all-to-all that keeps autograd.

    ``dist.all_to_all_single`` writes into a fresh buffer and drops the graph,
    so GDN in-proj / conv / A_log would get no gradient. Prefer functional
    collectives; fall back to ``torch.distributed.nn.functional``.
    """
    x = x.contiguous()
    try:
        import torch.distributed._functional_collectives as funcol

        fn = getattr(funcol, "all_to_all_single_autograd", None) or funcol.all_to_all_single
        y = fn(x, None, None, group)
        wait = getattr(y, "wait", None)
        return wait() if callable(wait) else y
    except Exception:
        from torch.distributed.nn.functional import all_to_all_single as a2a_fn

        out = torch.empty_like(x)
        return a2a_fn(out, x, group=group)


def a2a_seq_to_feat(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``[B, S_local, F]`` → ``[B, S_full, F_local]``."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    if x.size(-1) % world != 0:
        raise ValueError(f"feat {x.size(-1)} not divisible by cp={world}")
    batch, seq_local, feat = x.shape
    feat_local = feat // world
    x = x.reshape(batch, seq_local, world, feat_local).permute(2, 1, 0, 3).contiguous()
    out = _all_to_all_single(x, group)
    return out.permute(2, 0, 1, 3).reshape(batch, world * seq_local, feat_local).contiguous()


def a2a_feat_to_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``[B, S_full, F_local]`` → ``[B, S_local, F]``."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    batch, seq_full, feat_local = x.shape
    if seq_full % world != 0:
        raise ValueError(f"seq {seq_full} not divisible by cp={world}")
    seq_local = seq_full // world
    x = x.reshape(batch, world, seq_local, feat_local).permute(1, 2, 0, 3).contiguous()
    out = _all_to_all_single(x, group)
    return out.permute(2, 1, 0, 3).reshape(batch, seq_local, world * feat_local).contiguous()


def shard_sequence(
    tensor: torch.Tensor, group: dist.ProcessGroup, seq_dim: int = 1
) -> torch.Tensor:
    world = dist.get_world_size(group)
    if world == 1:
        return tensor
    length = tensor.size(seq_dim)
    if length % world != 0:
        raise ValueError(f"seq dim {length} not divisible by cp={world}")
    rank = dist.get_rank(group)
    local = length // world
    slc = [slice(None)] * tensor.ndim
    slc[seq_dim] = slice(rank * local, (rank + 1) * local)
    return tensor[tuple(slc)].contiguous()


def global_position_ids(
    batch: int, seq_full: int, group: dist.ProcessGroup, device: torch.device
) -> torch.Tensor:
    pos = torch.arange(seq_full, device=device).unsqueeze(0).expand(batch, -1)
    return shard_sequence(pos, group, seq_dim=1)


def _plain(t: torch.Tensor) -> torch.Tensor:
    fn = getattr(t, "full_tensor", None)
    return fn() if callable(fn) else t


def _slice_cp(param: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    param = _plain(param)
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    size = param.size(dim)
    if size % world != 0:
        raise ValueError(f"param dim {size} not divisible by cp={world}")
    local = size // world
    slc = [slice(None)] * param.ndim
    slc[dim] = slice(rank * local, (rank + 1) * local)
    return param[tuple(slc)]


def a2a_seq_to_feat_sections(
    x: torch.Tensor, sections: list[int], group: dist.ProcessGroup
) -> torch.Tensor:
    """All-to-all each Q/K/V (or similar) block, then concat. Megatron split_sections."""
    parts = torch.split(x, sections, dim=-1)
    return torch.cat([a2a_seq_to_feat(p, group) for p in parts], dim=-1)


def _slice_sections(
    param: torch.Tensor, sections: list[int], dim: int, group: dist.ProcessGroup
) -> torch.Tensor:
    parts = torch.split(param, sections, dim=dim)
    return torch.cat([_slice_cp(p, dim, group) for p in parts], dim=dim)


def _modeling():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

    return m


def qwen_gdn_forward_cp(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    group: dist.ProcessGroup,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Same recipe as ``Qwen3_5MoeGatedDeltaNet.forward``, with Megatron a2a."""
    m = _modeling()
    world = dist.get_world_size(group)
    hidden_states = m.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, seq_local, _ = hidden_states.shape

    mixed_qkv = module.in_proj_qkv(hidden_states)
    z = module.in_proj_z(hidden_states)
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)

    key_dim = module.key_dim
    value_dim = module.value_dim
    qkv_sections = [key_dim, key_dim, value_dim]
    if world > 1:
        mixed_qkv = a2a_seq_to_feat_sections(mixed_qkv, qkv_sections, group)
        z = a2a_seq_to_feat(z, group)
        b = a2a_seq_to_feat(b, group)
        a = a2a_seq_to_feat(a, group)

    seq_full = mixed_qkv.size(1)
    conv_w = _plain(module.conv1d.weight).squeeze(1)
    conv_b = None if module.conv1d.bias is None else _plain(module.conv1d.bias)
    a_log = _plain(module.A_log)
    dt_bias = _plain(module.dt_bias)
    num_k = module.num_k_heads
    num_v = module.num_v_heads
    if world > 1:
        conv_w = _slice_sections(conv_w, qkv_sections, 0, group)
        if conv_b is not None:
            conv_b = _slice_sections(conv_b, qkv_sections, 0, group)
        a_log = _slice_cp(a_log, 0, group)
        dt_bias = _slice_cp(dt_bias, 0, group)
        num_k = num_k // world
        num_v = num_v // world

    mixed_qkv = mixed_qkv.transpose(1, 2)
    mixed_qkv = m.causal_conv1d_fn(
        mixed_qkv,
        conv_w,
        conv_b,
        activation=module.activation,
    )
    mixed_qkv = mixed_qkv.transpose(1, 2)

    query, key, value = torch.split(
        mixed_qkv, [num_k * module.head_k_dim, num_k * module.head_k_dim, num_v * module.head_v_dim], dim=-1
    )
    query = query.reshape(batch, seq_full, num_k, module.head_k_dim)
    key = key.reshape(batch, seq_full, num_k, module.head_k_dim)
    value = value.reshape(batch, seq_full, num_v, module.head_v_dim)
    z = z.reshape(batch, seq_full, num_v, module.head_v_dim)

    beta = b.sigmoid()
    g = -_plain(a_log).float().exp() * F.softplus(a.float() + _plain(dt_bias))
    if num_v // num_k > 1:
        query = query.repeat_interleave(num_v // num_k, dim=2)
        key = key.repeat_interleave(num_v // num_k, dim=2)

    chunk_fn = getattr(module, "chunk_gated_delta_rule", None) or m.torch_chunk_gated_delta_rule
    core_attn_out, _ = chunk_fn(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.get("cu_seq_lens_q"),
    )
    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z_flat = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z_flat)
    core_attn_out = core_attn_out.reshape(batch, seq_full, -1)
    if world > 1:
        core_attn_out = a2a_feat_to_seq(core_attn_out, group)
    return module.out_proj(core_attn_out)


def patch_gated_delta_net(module: torch.nn.Module, group: dist.ProcessGroup) -> None:
    if getattr(module, _PATCHED, None) is not None:
        return
    orig = module.forward

    def _forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        del cache_params
        return qwen_gdn_forward_cp(
            module, hidden_states, group, attention_mask=attention_mask, **kwargs
        )

    setattr(module, _PATCHED, orig)
    module.forward = _forward


def patch_model_gdn_cp(model: torch.nn.Module, group: dist.ProcessGroup) -> int:
    m = _modeling()
    n = 0
    for mod in model.modules():
        if isinstance(mod, m.Qwen3_5MoeGatedDeltaNet):
            patch_gated_delta_net(mod, group)
            n += 1
    return n


def enable_sdpa_cp(mesh: DeviceMesh) -> None:
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        _cp_options,
        _enable_context_parallel_dispatcher_impl,
    )

    _cp_options.enable_load_balance = False
    _enable_context_parallel_dispatcher_impl(seq_dim=2, mesh=mesh)


def disable_sdpa_cp() -> None:
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        _disable_context_parallel_dispatcher_impl,
    )

    _disable_context_parallel_dispatcher_impl()
