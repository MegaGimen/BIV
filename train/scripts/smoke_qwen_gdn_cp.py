#!/usr/bin/env python3
"""2-GPU Qwen GatedDeltaNet sequence-parallel smoke (random tiny weights).

Uses HuggingFace ``Qwen3_5MoeGatedDeltaNet``, not the hand-written torch.nn
GDN. Sequence is sharded; GDN all-to-alls heads; full-attention uses SDPA-CP.

    torchrun --standalone --nproc_per_node=2 scripts/smoke_qwen_gdn_cp.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def tiny_text_config():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    return Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        full_attention_interval=2,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _gather_seq(h, group):
    import torch
    import torch.distributed as dist

    world = dist.get_world_size(group)
    parts = [torch.empty_like(h) for _ in range(world)]
    dist.all_gather(parts, h.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def main() -> int:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from biv_wm.qwen_gdn_cp import (
        disable_sdpa_cp,
        enable_sdpa_cp,
        global_position_ids,
        patch_model_gdn_cp,
        shard_sequence,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"need 2 ranks, got {world}")

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    cfg = tiny_text_config()
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(cfg).to(device="cuda", dtype=torch.bfloat16)
    ref_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_params = sum(p.numel() for p in model.parameters())
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
    group = mesh.get_group()
    n_gdn = patch_model_gdn_cp(model, group)
    if n_gdn < 1:
        raise RuntimeError("tiny model has 0 Qwen GDN layers")
    enable_sdpa_cp(mesh)

    seq = 128
    tokens = torch.randint(0, cfg.vocab_size, (1, seq), device="cuda")
    dist.broadcast(tokens, src=0)
    mask = torch.ones_like(tokens)
    local_tokens = shard_sequence(tokens, group, seq_dim=1)
    local_mask = shard_sequence(mask, group, seq_dim=1)
    pos = global_position_ids(1, seq, group, tokens.device)

    if rank == 0:
        print(
            f"[qwen-gdn-cp] torch={torch.__version__} params={n_params} gdn_layers={n_gdn} "
            f"seq={seq} local_seq={tuple(local_tokens.shape)}",
            flush=True,
        )

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(local_rank)
    out = model(
        input_ids=local_tokens,
        attention_mask=local_mask,
        position_ids=pos,
        use_cache=False,
        return_dict=True,
    )
    hidden = out.last_hidden_state
    loss = hidden.float().pow(2).mean()
    loss.backward()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024)
    full_h = _gather_seq(hidden.detach(), group)
    print(
        f"[qwen-gdn-cp] rank={rank} loss={loss.item():.6f} peak={peak:.1f}MiB "
        f"sec={elapsed:.3f} hidden={tuple(hidden.shape)}",
        flush=True,
    )
    dist.barrier()

    if rank == 0:
        disable_sdpa_cp()
        ref = Qwen3_5MoeTextModel(cfg).to(device="cuda", dtype=torch.bfloat16)
        ref.load_state_dict(ref_sd)
        ref.eval()
        with torch.no_grad():
            href = ref(
                input_ids=tokens,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        diff = (full_h.float() - href.float()).abs()
        print(
            f"[qwen-gdn-cp] vs unsharded max_abs={diff.max().item():.5f} "
            f"mean_abs={diff.mean().item():.5f}",
            flush=True,
        )
        if diff.max().item() > 0.15:
            raise RuntimeError("Qwen GDN sequence-split hidden diverged from unsharded")
        print("[qwen-gdn-cp] PASS 2-GPU Qwen GDN sequence split", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        raise SystemExit(1)
