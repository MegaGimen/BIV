"""Harbor subprocess hook loaded via PYTHONPATH (see eval/run_harbor.py).

Terminus 2 starts the agent tty with ``tmux new-session -e KEY=value``.
That flag exists only in tmux 3.2+. TB 2.1 ``qemu-alpine-ssh`` ships Debian
bullseye's tmux 3.1c, Harbor sees ``tmux -V`` succeed and skips the upgrade,
then ``new-session -e`` exits 1 (``unknown option -- e``). Compose merges
stderr into stdout so the trial becomes ``Failed to start tmux session.
Error: None`` — infra, not the model.

Inject extra_env through the inner shell instead of ``-e``, which 3.1c accepts.
"""

from __future__ import annotations

import shlex


def _patch_terminus_tmux() -> None:
    try:
        from harbor.agents.terminus_2.tmux_session import TmuxSession
    except ImportError:
        return
    if getattr(TmuxSession, "_biv_tmux_e_compat", False):
        return

    def _tmux_start_session(self) -> str:
        exports = "".join(
            f"export {shlex.quote(str(key))}={shlex.quote(str(value))}; "
            for key, value in self._extra_env.items()
        )
        inner = f"{exports}exec bash --login" if exports else "bash --login"
        return (
            f"export TERM=xterm-256color && "
            f"export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} "
            f"-d -s {self._session_name} {shlex.quote(inner)} \\; "
            f"pipe-pane -t {self._session_name} "
            f"'cat > {self._logging_path}'"
        )

    TmuxSession._tmux_start_session = property(_tmux_start_session)

    _orig_start = TmuxSession.start

    async def start(self):
        try:
            await _orig_start(self)
        except RuntimeError as exc:
            msg = str(exc)
            if "Failed to start tmux session" in msg and "Error: None" in msg:
                raise RuntimeError(
                    "Failed to start tmux session. Error: None "
                    "(docker compose merged stderr into stdout; "
                    "likely old tmux rejected new-session -e)"
                ) from exc
            raise

    TmuxSession.start = start  # type: ignore[method-assign]
    TmuxSession._biv_tmux_e_compat = True


def _patch_litellm_max_tokens_clamp() -> None:
    """Never send max_tokens >= the vLLM window; that 400s even a tiny prompt."""
    try:
        from harbor.llms.lite_llm import LiteLLM
    except ImportError:
        return
    if getattr(LiteLLM, "_biv_max_tokens_clamp", False):
        return

    _orig_call = LiteLLM.call

    async def call(self, *args, **kwargs):
        info = getattr(self, "_model_info", None) or {}
        window = info.get("max_input_tokens") or info.get("max_tokens")
        output_cap = info.get("max_output_tokens")
        requested = kwargs.get("max_tokens")
        if (
            isinstance(requested, int)
            and isinstance(window, int)
            and window > 0
            and requested >= window
        ):
            kwargs["max_tokens"] = (
                int(output_cap)
                if isinstance(output_cap, int) and 0 < output_cap < window
                else max(1, window // 4)
            )
        return await _orig_call(self, *args, **kwargs)

    LiteLLM.call = call  # type: ignore[method-assign]
    LiteLLM._biv_max_tokens_clamp = True


_ALIYUN_OFFICIAL_REGISTRY = "fc-e2b-registry."
_ALIYUN_DEST_MISSING = (
    "Aliyun envd inject needs a pushable dest registry. "
    "TB images (alexgshaw/*) have no /.fce2b; default conversion fails with "
    "'envd inject failed', Harbor retries the same alias and you only see "
    "409 CREATE_FAILED. Open ACR EE in the same region as E2B_DOMAIN, then "
    "set E2B_TEMPLATE_DEST_IMAGE_REF / E2B_TEMPLATE_DEST_USERNAME / "
    "E2B_TEMPLATE_DEST_PASSWORD in train/.env. Dest ref may use {src} or a "
    "repo without tag (tag becomes the source image)."
)


def _template_base_image(template) -> str:
    inner = getattr(template, "_template", None) or template
    return str(getattr(inner, "_base_image", "") or "")


def _dest_image_ref(dest: str, source: str) -> str:
    safe = (
        source.replace("/", "_").replace(":", "-").replace("@", "-")[:80]
        or "image"
    )
    if "{src}" in dest:
        return dest.replace("{src}", safe)
    last = dest.rsplit("/", 1)[-1]
    if ":" not in last:
        return f"{dest}:{safe}"
    return dest


def _patch_aliyun_e2b_template_headers() -> None:
    """Aliyun: official registry works as-is; TB images need builder→ACR EE."""
    import os

    url = os.environ.get("E2B_API_URL", "")
    if "e2b.fc.aliyuncs.com" not in url:
        return
    try:
        from e2b import AsyncTemplate, default_build_logger
        from e2b.exceptions import BuildException
    except ImportError:
        return
    if getattr(AsyncTemplate, "_biv_aliyun_headers", False):
        return

    _orig = AsyncTemplate.build

    async def build(template, *args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        source = _template_base_image(template)
        official = _ALIYUN_OFFICIAL_REGISTRY in source
        user = os.environ.get("E2B_TEMPLATE_SOURCE_USERNAME", "").strip()
        password = os.environ.get("E2B_TEMPLATE_SOURCE_PASSWORD", "").strip()
        dest = os.environ.get("E2B_TEMPLATE_DEST_IMAGE_REF", "").strip()
        dest_user = os.environ.get("E2B_TEMPLATE_DEST_USERNAME", "").strip()
        dest_password = os.environ.get("E2B_TEMPLATE_DEST_PASSWORD", "").strip()
        if not official and not (dest and dest_user and dest_password):
            raise BuildException(_ALIYUN_DEST_MISSING)
        if dest and dest_user and dest_password and not official:
            headers.setdefault("X-E2B-Template-Build-Mode", "builder")
            headers.setdefault(
                "X-E2B-Template-Dest-Image-Ref",
                _dest_image_ref(dest, source),
            )
            headers.setdefault("X-E2B-Template-Dest-Username", dest_user)
            headers.setdefault("X-E2B-Template-Dest-Password", dest_password)
        if user and password:
            headers.setdefault("X-E2B-Template-Source-Username", user)
            headers.setdefault("X-E2B-Template-Source-Password", password)
        if headers:
            kwargs["headers"] = headers
        if kwargs.get("on_build_logs") is None:
            kwargs["on_build_logs"] = default_build_logger()
        # Aliyun returns 400 "force is not supported".
        kwargs["skip_cache"] = False
        return await _orig(template, *args, **kwargs)

    AsyncTemplate.build = staticmethod(build)  # type: ignore[method-assign]
    AsyncTemplate._biv_aliyun_headers = True


def _patch_harbor_aliyun_template_retry() -> None:
    """Harbor retries the same alias; Aliyun CREATE_FAILED forbids a second build."""
    import os

    if "e2b.fc.aliyuncs.com" not in os.environ.get("E2B_API_URL", ""):
        return
    try:
        from harbor.environments.e2b import E2BEnvironment
    except ImportError:
        return
    if getattr(E2BEnvironment, "_biv_aliyun_no_retry", False):
        return
    wrapped = E2BEnvironment._create_template
    inner = getattr(wrapped, "__wrapped__", wrapped)

    async def _create_template(self):
        return await inner(self)

    E2BEnvironment._create_template = _create_template  # type: ignore[method-assign]
    E2BEnvironment._biv_aliyun_no_retry = True


def _read_smart_timeout_mult() -> float:
    import json
    import os
    from pathlib import Path

    raw = os.environ.get("BIV_SMART_TIMEOUT_PATH", "").strip()
    if not raw:
        return 1.0
    path = Path(raw)
    if not path.is_file():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        extra = float(data.get("multiplier") or 1.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1.0
    if extra <= 0:
        return 1.0
    return extra


def _patch_smart_agent_timeout() -> None:
    """Apply live 40 tok/s timeout scale to trials that start after we measure v."""
    try:
        from harbor.trial.trial import Trial
    except ImportError:
        return
    if getattr(Trial, "_biv_smart_timeout", False):
        return

    orig = Trial._compute_agent_timeout_sec

    def _compute(self):
        base = orig(self)
        if base is None:
            return None
        extra = _read_smart_timeout_mult()
        if extra == 1.0:
            return base
        return float(base) * extra

    Trial._compute_agent_timeout_sec = _compute  # type: ignore[method-assign]
    Trial._biv_smart_timeout = True


_patch_terminus_tmux()
_patch_litellm_max_tokens_clamp()
_patch_aliyun_e2b_template_headers()
_patch_harbor_aliyun_template_retry()
_patch_smart_agent_timeout()
