"""Per-request provider credentials (from the browser) for Agent A/B."""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name
from nanobot.utils.llm_runtime import LLMRuntime

DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONTEXT_WINDOW = 128_000

_creds: contextvars.ContextVar[ProviderCreds | None] = contextvars.ContextVar(
    "cartesian_provider_creds", default=None
)


@dataclass(frozen=True)
class ProviderCreds:
    api_key: str
    api_base: str
    model: str = DEFAULT_MODEL


def normalize_api_base(api_base: str | None) -> str:
    base = (api_base or "").strip() or DEFAULT_API_BASE
    return base.rstrip("/")


def resolve_creds(
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> ProviderCreds:
    """Prefer request fields; fall back to env / defaults (operator installs)."""
    key = (api_key or "").strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base = normalize_api_base(api_base)
    mdl = (model or "").strip() or DEFAULT_MODEL
    if not key:
        raise ValueError(
            "Missing API key. Open Settings and save your provider API key "
            "(stored in this browser), or set DEEPSEEK_API_KEY on the server."
        )
    return ProviderCreds(api_key=key, api_base=base, model=mdl)


def set_creds(creds: ProviderCreds | None) -> contextvars.Token:
    return _creds.set(creds)


def reset_creds(token: contextvars.Token) -> None:
    _creds.reset(token)


def get_creds() -> ProviderCreds | None:
    return _creds.get()


def get_creds_or_env() -> ProviderCreds:
    current = get_creds()
    if current is not None:
        return current
    return resolve_creds()


def build_agent_a_runtime(creds: ProviderCreds) -> LLMRuntime:
    """Build an LLMRuntime with nanobot's OpenAI-compatible DeepSeek provider."""
    spec = find_by_name("deepseek")
    provider = OpenAICompatProvider(
        api_key=creds.api_key,
        api_base=creds.api_base,
        default_model=creds.model,
        spec=spec,
    )
    # Match cartesian.json Agent A generation defaults.
    from nanobot.providers.base import GenerationSettings

    provider.generation = GenerationSettings(
        temperature=0.2,
        max_tokens=8192,
        reasoning_effort=None,
    )
    return LLMRuntime.capture(
        provider,
        creds.model,
        context_window_tokens=DEFAULT_CONTEXT_WINDOW,
        model_preset="agentA",
        snapshot_signature=("cartesian-browser", creds.api_base, creds.model),
    )


def chat_completions_url(api_base: str) -> str:
    """DeepSeek / OpenAI-compat chat completions endpoint."""
    base = normalize_api_base(api_base)
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/chat/completions"
