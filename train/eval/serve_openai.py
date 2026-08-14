#!/usr/bin/env python3
"""Local OpenAI-compatible chat server for Muse Glimmer (+ optional PEFT ckpt).

Intended to run on the GPU train host. Harbor / test.py call
http://127.0.0.1:<port>/v1 .

  python -m eval.serve_openai --model-path outputs/models/Muse-Glimmer-30B
  python -m eval.serve_openai --model-path ... --ckpt path/to/checkpoint-e1-s50
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _messages_to_prompt(tokenizer, messages: list[dict[str, Any]]) -> str:
    norm: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or "user"
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
        norm.append({"role": role, "content": content})
    try:
        return tokenizer.apply_chat_template(
            norm, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        lines = [f"{m['role']}: {m['content']}" for m in norm]
        lines.append("assistant:")
        return "\n".join(lines)


def build_app(model, tokenizer, *, served_name: str):
    import threading

    import torch
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Muse local OpenAI shim")
    lock = threading.Lock()

    class ChatMessage(BaseModel):
        role: str
        content: Any = ""

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list[ChatMessage]
        temperature: float | None = 1.0
        top_p: float | None = 0.95
        max_tokens: int | None = 2048
        max_completion_tokens: int | None = None
        stream: bool | None = False
        stop: list[str] | str | None = None

    @app.get("/health")
    def health():
        return {"ok": True, "model": served_name}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        if req.stream:
            raise HTTPException(400, "stream=false only in this shim")
        max_new = int(req.max_completion_tokens or req.max_tokens or 2048)
        msgs = [m.model_dump() for m in req.messages]
        prompt = _messages_to_prompt(tokenizer, msgs)
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": True,
            "temperature": float(req.temperature if req.temperature is not None else 1.0),
            "top_p": float(req.top_p if req.top_p is not None else 0.95),
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        stop = req.stop
        if isinstance(stop, str):
            stop = [stop]

        with lock:
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)

        in_len = int(inputs["input_ids"].shape[-1])
        new_tokens = out[0][in_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s, 1)[0]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": in_len,
                "completion_tokens": int(new_tokens.numel()),
                "total_tokens": in_len + int(new_tokens.numel()),
            },
        }

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--served-name", type=str, default="Muse-Glimmer-30B")
    args = p.parse_args(argv)

    from eval.load_muse import load_muse_for_infer

    model, tokenizer = load_muse_for_infer(
        args.model_path, ckpt=args.ckpt, dtype=args.dtype
    )
    app = build_app(model, tokenizer, served_name=args.served_name)

    import uvicorn

    print(
        f"[serve] OpenAI shim http://{args.host}:{args.port}/v1 "
        f"served_name={args.served_name} ckpt={args.ckpt}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
