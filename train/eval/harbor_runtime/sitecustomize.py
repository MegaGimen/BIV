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


_patch_terminus_tmux()
_patch_litellm_max_tokens_clamp()
