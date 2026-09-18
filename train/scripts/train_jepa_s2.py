#!/usr/bin/env python3
"""Stage 2 Step 1: frozen JEPA world + frozen Instruct, train draft/scorer/W.

World: AgentWorld + Stage 1 LoRA + JEPAPred, all frozen.
  z_t = Enc(chat(h)), u* = Enc(chat([a])), zhat* = Pred(z_t, u*).
  Drafts: zhat_k = Pred(z_t, u_k^W) from the draft head (no Enc(o)).

Instruct: own 40 layers frozen. c_t = Enc(chat(h)). Draft(c_t) → K u^W and
one u^A. Scorer softmax-CE over K drafts + real branch. Mouth CE: real
command tokens through W → Instruct lm_head.

  cd train && CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_jepa_s2.sh --jepa-ckpt auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
SRC = TRAIN / "src"
MERGE = ROOT / "merge"
SCRIPTS = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MERGE) not in sys.path:
    sys.path.insert(0, str(MERGE))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_jepa as s1  # noqa: E402
from biv_wm.arch import install_hidden_only_forward, lm_head_module  # noqa: E402
from biv_wm.ckpt import (  # noqa: E402
    epoch_end_name,
    find_latest_ckpt,
    parse_ckpt_name,
    rotate_rolling,
    rolling_name,
    write_trainer_state,
)
from biv_wm.fsdp_ckpt import wrap_fsdp_activation_checkpoint  # noqa: E402
from biv_wm.jepa import JEPAPred  # noqa: E402
from biv_wm.stage2 import DraftHead, MouthW, Scorer, mouth_token_ce, ranking_ce  # noqa: E402
from biv_wm.strip_json import unwrap_message  # noqa: E402
from download import resolve_model  # noqa: E402

DEFAULT_CONFIG = TRAIN / "configs" / "jepa" / "stage2.yaml"


def log(msg: str) -> None:
    print(msg, flush=True)


def freeze_params(module) -> int:
    n = 0
    module.eval()
    for p in module.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    return n


def collate_stage2(batch: list[dict[str, Any]], pad_id: int, pad_multiple: int = 1) -> dict[str, Any]:
    import torch

    out = s1.collate(batch, pad_id, pad_multiple)

    def pad_rows(rows: list[list[int]], fill: int) -> tuple:
        mlen = s1._pad_len(max((len(x) for x in rows), default=1), pad_multiple)
        mlen = max(mlen, 1)
        ids = torch.full((len(rows), mlen), fill, dtype=torch.long)
        mask = torch.zeros((len(rows), mlen), dtype=torch.long)
        for i, row in enumerate(rows):
            if not row:
                continue
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return ids, mask

    ins_ids, ins_mask = pad_rows([ex["instruct_state_ids"] for ex in batch], pad_id)
    ia_ids, ia_mask = pad_rows([ex["instruct_a_label_ids"] for ex in batch], pad_id)
    out["instruct_state_ids"] = ins_ids
    out["instruct_state_mask"] = ins_mask
    out["instruct_a_label_ids"] = ia_ids
    out["instruct_a_label_mask"] = ia_mask
    return out


class Stage2Dataset:
    def __init__(self, rows: list, world_tok, instruct_tok, max_length: int) -> None:
        self.rows = rows
        self.world_tok = world_tok
        self.instruct_tok = instruct_tok
        self.max_length = max_length
        self._fitted: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        hao = self._fitted.get(idx)
        if hao is None:
            hao = s1.fit_hao(self.world_tok, self.rows[idx], self.max_length)
            if hao is None:
                raise RuntimeError(f"row {idx} has no complete (h, a, o) after prefix trim")
            self._fitted[idx] = hao
        h, a, o = hao
        ex = s1.encode_texts(self.world_tok, h, a, o, self.max_length)
        ins_state = s1._fit(
            s1.tokenize_ids(self.instruct_tok, s1.apply_template(self.instruct_tok, list(h))),
            self.max_length,
            keep="prefix",
        )
        a_u = unwrap_message(a)
        ins_a = self.instruct_tok.encode(s1._content(a_u), add_special_tokens=False)
        ex["instruct_state_ids"] = ins_state
        ex["instruct_a_label_ids"] = ins_a
        return ex


def resolve_jepa_ckpt(jepa_dir: Path, raw: str | None) -> Path:
    if raw is None or raw == "auto":
        found = find_latest_ckpt(jepa_dir, require_jepa=True)
        if found is None:
            raise SystemExit(f"no complete Stage 1 checkpoint under {jepa_dir}")
        return found
    pth = Path(raw)
    if not pth.is_absolute():
        pth = s1._resolve(pth)
    if (
        pth.is_dir()
        and (pth / "trainer_state.json").is_file()
        and (pth / "adapter_model.safetensors").is_file()
        and (pth / "jepa.pt").is_file()
    ):
        return pth
    raise SystemExit(f"--jepa-ckpt is not a Stage 1 dir (need adapter + jepa.pt): {pth}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--jepa-ckpt", type=str, default=None, help="Stage 1 dir or 'auto'")
    p.add_argument("--jepa-dir", type=Path, default=None)
    p.add_argument("--world-dir", type=str, default=None)
    p.add_argument("--instruct-dir", type=str, default=None)
    p.add_argument("--mix-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--log-steps", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--cp-size", type=int, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("MERGE_SOURCE", "modelscope"),
    )
    p.add_argument("--logging-dir", type=Path, default=None)
    p.add_argument("--seq-split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-tag", type=str, default="jepa-s2")
    p.add_argument("--resume", nargs="?", const="auto", default=None)
    return p.parse_args()


def save_s2(accelerator, path: Path, *, epoch: int, step: int, extra: dict, draft, scorer, mouth_w) -> None:
    accelerator.wait_for_everyone()
    d_cpu = s1.gather_module_cpu(draft, keep=accelerator.is_main_process)
    s_cpu = s1.gather_module_cpu(scorer, keep=accelerator.is_main_process)
    w_cpu = s1.gather_module_cpu(mouth_w, keep=accelerator.is_main_process)
    path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        import torch

        torch.save(d_cpu, path / "draft.pt")
        torch.save(s_cpu, path / "scorer.pt")
        torch.save(w_cpu, path / "w.pt")
        write_trainer_state(path, epoch=epoch, global_step=step, extra=extra)
        (path / "train_meta.json").write_text(
            json.dumps({"epoch": epoch, "global_step": step, **extra}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        log(f"saved {path}")
    accelerator.wait_for_everyone()


def load_s2_heads(path: Path, draft, scorer, mouth_w, log_fn) -> tuple[int, int]:
    draft.load_state_dict(s1._load_pt(path / "draft.pt"))
    scorer.load_state_dict(s1._load_pt(path / "scorer.pt"))
    mouth_w.load_state_dict(s1._load_pt(path / "w.pt"))
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    log_fn(f"resume Stage 2 heads from {path.name}")
    return int(state.get("epoch") or 0), int(state.get("global_step") or 0)


def s2_ckpt_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "trainer_state.json").is_file()
        and (path / "draft.pt").is_file()
        and (path / "scorer.pt").is_file()
        and (path / "w.pt").is_file()
    )


def find_latest_s2(out_dir: Path) -> Path | None:
    if not out_dir.is_dir():
        return None
    best = None
    for p in out_dir.iterdir():
        parsed = parse_ckpt_name(p.name)
        if parsed is None or not s2_ckpt_complete(p):
            continue
        epoch, step, kind = parsed
        key = (epoch, step, kind)
        if best is None or key > (best[0], best[1], best[2]):
            best = (epoch, step, kind, p)
    return None if best is None else best[3]


def last_hidden(model, ids, mask, cp_size: int, seq_split: bool, last_token: int):
    h = s1.full_hidden(model, ids, mask, cp_size, seq_split=seq_split)
    return s1.gather_at(h, s1.last_token_index(ids, mask, last_token))


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    cfg_path = args.config if args.config.is_absolute() else (TRAIN / args.config)
    if not cfg_path.is_file():
        alt = ROOT / args.config
        if alt.is_file():
            cfg_path = alt
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {args.config}")
    cfg = s1._load_yaml(cfg_path)
    tcfg = cfg.get("train") or {}
    accum = int(args.grad_accum if args.grad_accum is not None else (tcfg.get("grad_accum") or 8))
    cp_size = s1.resolve_cp_size(args.cp_size)
    if cp_size > 1:
        os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
        os.environ.setdefault("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1")
        os.environ.setdefault("PARALLELISM_CONFIG_TP_SIZE", "1")
        os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cp_size)
        os.environ.setdefault("PARALLELISM_CONFIG_CP_BACKEND", "torch")

    accelerator = Accelerator(
        gradient_accumulation_steps=accum, parallelism_config=s1.build_parallelism_config()
    )
    is_main = accelerator.is_main_process

    def rank_log(msg: str) -> None:
        if is_main:
            log(msg)

    cache_dir = s1._resolve(args.cache_dir or "merge/output/cache", ROOT)
    world_dir = resolve_model(
        str(args.world_dir or cfg.get("world_dir") or cfg.get("model_dir")),
        source=args.source,
        cache_dir=cache_dir,
        role="world",
    )
    instruct_dir = resolve_model(
        str(args.instruct_dir or cfg.get("instruct_dir") or "Qwen/Qwen3.5-35B-A3B"),
        source=args.source,
        cache_dir=cache_dir,
        role="instruct",
    )
    sources = list(cfg.get("sources") or ["wm_code", "wm_os"])
    mix_dir = s1.resolve_mix(args.mix_dir or cfg["mix_dir"], sources)
    out_dir = s1._resolve(tcfg.get("output_dir") or "outputs/jepa_stage2")
    jepa_dir = s1._resolve(args.jepa_dir or cfg.get("jepa_dir") or "outputs/jepa_stage1")
    jepa_ckpt = resolve_jepa_ckpt(jepa_dir, args.jepa_ckpt or cfg.get("jepa_ckpt") or "auto")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dtype = torch.bfloat16 if str(tcfg.get("torch_dtype", "bfloat16")).startswith("bf") else torch.float16
    seed = int(tcfg.get("seed") or 42)
    torch.manual_seed(seed)
    max_length = int(args.max_length or tcfg.get("max_length") or 32768)
    pad_multiple = cp_size * 2 if cp_size > 1 else int(tcfg.get("pad_to_multiple_of") or 0)
    attn_impl = "sdpa" if cp_size > 1 else None
    distributed = accelerator.num_processes > 1
    checkpointing = bool(tcfg.get("gradient_checkpointing", True))
    seq_split = bool(args.seq_split)
    n_drafts = int(tcfg.get("n_drafts") or 3)
    last_token = int(tcfg["last_token"] if tcfg.get("last_token") is not None else -3)
    rank_lbd = float(tcfg["rank_lbd"] if tcfg.get("rank_lbd") is not None else 1.0)
    ce_lbd = float(tcfg["ce_lbd"] if tcfg.get("ce_lbd") is not None else 1.0)
    rank_log(f"world={world_dir}")
    rank_log(f"instruct={instruct_dir}")
    rank_log(f"jepa_ckpt={jepa_ckpt}")
    rank_log(f"mix={mix_dir} sources={sources}")
    rank_log(
        f"max_length={max_length} cp_size={cp_size} n_drafts={n_drafts} "
        f"seq_split={seq_split} nproc={accelerator.num_processes}"
    )

    world, world_tok = s1.load_backbone(
        world_dir, dtype, checkpointing, attn_implementation=attn_impl, distributed=distributed
    )
    suffixes = list(tcfg.get("target_modules") or [])
    targets = s1.two_d_lora_targets(world, suffixes)
    world = get_peft_model(
        world,
        LoraConfig(
            r=int(tcfg.get("lora_rank") or 16),
            lora_alpha=int(tcfg.get("lora_alpha") or 32),
            lora_dropout=float(tcfg.get("lora_dropout") or 0.05),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    from safetensors.torch import load_file

    s1.load_lora_into_model(world, load_file(str(jepa_ckpt / "adapter_model.safetensors")), rank_log)
    install_hidden_only_forward(world, detach_head=True)
    hidden_size = s1.backbone_hidden_size(world)
    pred = JEPAPred(hidden_size)
    pred.load_state_dict(s1._load_pt(jepa_ckpt / "jepa.pt"))
    freeze_params(world)
    freeze_params(pred)
    rank_log(f"frozen World+LoRA+Pred from {jepa_ckpt.name}")

    instruct, instruct_tok = s1.load_backbone(
        instruct_dir, dtype, checkpointing, attn_implementation=attn_impl, distributed=distributed
    )
    install_hidden_only_forward(instruct, detach_head=False)
    freeze_params(instruct)
    lm_head = lm_head_module(instruct)
    if lm_head is None:
        raise SystemExit("Stage 2 needs Instruct lm_head attached")
    for p in lm_head.parameters():
        p.requires_grad = False
    rank_log("frozen Instruct backbone + lm_head")

    draft = DraftHead(hidden_size, n_drafts=n_drafts)
    scorer = Scorer(hidden_size)
    mouth_w = MouthW(hidden_size)
    resume_epoch, resume_step = 0, 0
    if args.resume:
        if args.resume == "auto":
            resume_dir = find_latest_s2(out_dir)
            if resume_dir is None:
                raise SystemExit(f"--resume: no Stage 2 checkpoint under {out_dir}")
        else:
            resume_dir = Path(args.resume)
            if not resume_dir.is_absolute():
                resume_dir = s1._resolve(resume_dir)
        resume_epoch, resume_step = load_s2_heads(resume_dir, draft, scorer, mouth_w, rank_log)

    train_rows = s1.load_rows(mix_dir, sources, "train", tcfg.get("max_train_samples"))
    if not train_rows:
        raise SystemExit(f"no mix rows under {mix_dir}")
    rank_log(f"train_rows={len(train_rows)}")
    ds = Stage2Dataset(train_rows, world_tok, instruct_tok, max_length)
    pad_id = instruct_tok.pad_token_id if instruct_tok.pad_token_id is not None else 0
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch_size = int(tcfg.get("batch_size") or 1)
    dp_rank, dp_size = s1.dp_replicate_info(accelerator, cp_size)
    if dp_size > 1:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=s1.ReplicaSampler(len(ds), dp_rank, dp_size, gen),
            collate_fn=lambda b: collate_stage2(b, pad_id, pad_multiple),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
            collate_fn=lambda b: collate_stage2(b, pad_id, pad_multiple),
            num_workers=0,
        )
    s1.assert_equal_loader_len(accelerator, len(loader))

    pred = pred.to(device=accelerator.device, dtype=dtype)
    draft = draft.to(device=accelerator.device, dtype=dtype)
    scorer = scorer.to(device=accelerator.device, dtype=dtype)
    mouth_w = mouth_w.to(device=accelerator.device, dtype=dtype)
    world.add_module("_jepa", pred)
    instruct.add_module("_draft", draft)
    instruct.add_module("_scorer", scorer)
    instruct.add_module("_W", mouth_w)
    opt = torch.optim.AdamW(
        [p for p in instruct.parameters() if p.requires_grad],
        lr=float(tcfg.get("lr") or 1e-4),
        weight_decay=float(tcfg.get("weight_decay") or 0.01),
    )
    world, instruct, opt = accelerator.prepare(world, instruct, opt)
    if distributed and checkpointing:
        n_w = wrap_fsdp_activation_checkpoint(accelerator.unwrap_model(world))
        n_i = wrap_fsdp_activation_checkpoint(accelerator.unwrap_model(instruct))
        rank_log(f"FSDP ckpt wrappers world={n_w} instruct={n_i}")
    un_i = accelerator.unwrap_model(instruct)
    un_w = accelerator.unwrap_model(world)
    draft = un_i._draft
    scorer = un_i._scorer
    mouth_w = un_i._W
    pred = un_w._jepa
    lm_head = lm_head_module(un_i)
    embed = s1.input_embeddings(un_i)

    log_every = int(args.log_steps if args.log_steps is not None else (tcfg.get("log_steps") or 5))
    save_every = int(args.save_steps if args.save_steps is not None else (tcfg.get("save_steps") or 25))
    save_limit = int(tcfg.get("save_total_limit") or 50)
    epochs = int(tcfg.get("num_epochs") or 1)
    max_steps = args.max_steps
    steps_per_epoch = math.ceil(len(loader) / accum)
    planned = steps_per_epoch * epochs
    total_opt = planned if max_steps is None else min(planned, int(max_steps))
    warmup = int(tcfg.get("warmup_steps") or 50)
    warmup = max(0, min(warmup, max(total_opt - 1, 0)))
    for group in opt.param_groups:
        group.setdefault("initial_lr", group["lr"])
    sched = LambdaLR(
        opt,
        s1._warmup_lambda(warmup),
        last_epoch=(resume_step - 1) if resume_step > 0 else -1,
    )
    rank_log(
        f"epochs={epochs} steps_per_epoch≈{steps_per_epoch} accum={accum} "
        f"lr={opt.param_groups[0]['lr']} warmup={warmup} "
        f"rank_lbd={rank_lbd} ce_lbd={ce_lbd} resume_step={resume_step} "
        f"save_steps={save_every} log_steps={log_every}"
    )
    ckpt_extra = {
        "jepa_ckpt": str(jepa_ckpt),
        "world_dir": str(world_dir),
        "instruct_dir": str(instruct_dir),
        "mix_dir": str(mix_dir),
        "max_length": max_length,
        "n_drafts": n_drafts,
        "step": 1,
        "run_tag": run_tag,
        "recipe": "Stage2 step1 draft/scorer/W; frozen JEPA Pred(Enc(h), u)",
    }
    tb_dir = s1.resolve_tb_dir(tcfg, out_dir, args.logging_dir, run_tag)
    writer = s1.open_tb(tb_dir) if is_main else None

    def dump_ckpt(kind: str, epoch_i: int, step_i: int) -> None:
        dest = out_dir / (
            epoch_end_name(epoch_i, step_i) if kind == "epoch-end" else rolling_name(epoch_i, step_i)
        )
        save_s2(
            accelerator,
            dest,
            epoch=epoch_i,
            step=step_i,
            extra={**ckpt_extra, "kind": kind},
            draft=draft,
            scorer=scorer,
            mouth_w=mouth_w,
        )
        if is_main:
            rotate_rolling(out_dir, save_limit, log=lambda m: log(f"[{run_tag}] {m}"))

    if resume_step >= total_opt:
        rank_log(f"resume_step={resume_step} covers total_opt={total_opt}; nothing to do")
        if writer is not None:
            writer.close()
        return

    draft.train()
    scorer.train()
    mouth_w.train()
    world.eval()
    instruct.eval()
    pred.eval()
    step = resume_step
    skip_micro = resume_step * accum
    micro_seen = 0
    opt.zero_grad(set_to_none=True)
    running = run_rank = run_ce = 0.0
    n_loss = 0
    hit_max = False
    last_log_time = time.monotonic()
    max_norm = float(tcfg.get("max_grad_norm") or 1.0)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    pbar = None
    if is_main and tqdm is not None:
        pbar = tqdm(total=total_opt, initial=min(resume_step, total_opt), desc=run_tag, unit="step")

    try:
        for epoch in range(epochs):
            epoch_base = epoch * steps_per_epoch
            epoch_opt = max(0, min(resume_step - epoch_base, steps_per_epoch))
            for batch in loader:
                if micro_seen < skip_micro:
                    micro_seen += 1
                    continue
                micro_seen += 1
                batch = {
                    k: v.to(accelerator.device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                with accelerator.accumulate(instruct):
                    with torch.no_grad():
                        c_t = last_hidden(
                            instruct,
                            batch["instruct_state_ids"],
                            batch["instruct_state_mask"],
                            cp_size,
                            seq_split,
                            last_token,
                        )
                        z_t = last_hidden(
                            world,
                            batch["state_ids"],
                            batch["state_mask"],
                            cp_size,
                            seq_split,
                            last_token,
                        )
                        u_star = last_hidden(
                            world,
                            batch["action_ids"],
                            batch["action_mask"],
                            cp_size,
                            seq_split,
                            last_token,
                        )
                        zhat_star = pred(z_t, u_star)
                    u_w, u_a = draft(c_t)
                    zhats = [pred(z_t, u_w[:, k]) for k in range(n_drafts)]
                    u_all = torch.cat([u_w, u_star.unsqueeze(1)], dim=1)
                    z_all = torch.stack(zhats + [zhat_star], dim=1)
                    bsz, n_cand, dim = u_all.shape
                    scores = scorer(
                        u_all.reshape(bsz * n_cand, dim), z_all.reshape(bsz * n_cand, dim)
                    )
                    scores = scores.view(bsz, n_cand)
                    loss_rank = ranking_ce(scores, n_drafts)
                    loss_ce = mouth_token_ce(
                        u_a,
                        batch["instruct_a_label_ids"],
                        batch["instruct_a_label_mask"],
                        embed,
                        mouth_w,
                        lm_head,
                    )
                    loss = rank_lbd * loss_rank + ce_lbd * loss_ce
                    accelerator.backward(loss)
                    running += float(loss.detach().float().item())
                    run_rank += float(loss_rank.detach().float().item())
                    run_ce += float(loss_ce.detach().float().item())
                    n_loss += 1
                    if accelerator.sync_gradients:
                        clip_params = [p for p in instruct.parameters() if p.requires_grad]
                        accelerator.clip_grad_norm_(clip_params, max_norm)
                        opt.step()
                        sched.step()
                        opt.zero_grad(set_to_none=True)
                        step += 1
                        epoch_opt += 1
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(loss=f"{running / max(n_loss, 1):.4f}", refresh=False)
                        is_last = epoch_opt >= steps_per_epoch
                        if step % log_every == 0:
                            g = s1.merge_group_stats(
                                accelerator, running, run_rank, run_ce, float(n_loss)
                            )
                            running_g, rank_g, ce_g, n_g = g
                            denom = max(n_g, 1.0)
                            eff_step = step * dp_size
                            if is_main:
                                now = time.monotonic()
                                elapsed = max(now - last_log_time, 1e-6)
                                last_log_time = now
                                log(
                                    f"epoch={epoch} step={step} eff_step={eff_step} "
                                    f"loss={running_g / denom:.4f} "
                                    f"rank={rank_g / denom:.4f} ce={ce_g / denom:.4f} "
                                    f"lr={opt.param_groups[0]['lr']:.2e} wall={elapsed:.1f}s"
                                )
                                if writer is not None:
                                    writer.add_scalar("train/loss", running_g / denom, eff_step)
                                    writer.add_scalar("train/loss_rank", rank_g / denom, eff_step)
                                    writer.add_scalar("train/loss_ce", ce_g / denom, eff_step)
                                    writer.add_scalar("train/lr", opt.param_groups[0]["lr"], eff_step)
                            running = run_rank = run_ce = 0.0
                            n_loss = 0
                        if is_last:
                            dump_ckpt("epoch-end", epoch + 1, step)
                        elif step % save_every == 0:
                            dump_ckpt("rolling", epoch, step)
                        if max_steps is not None and step >= max_steps:
                            hit_max = True
                            break
            if hit_max:
                break
    finally:
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.flush()
            writer.close()

    rank_log(f"stop step={step} under {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
