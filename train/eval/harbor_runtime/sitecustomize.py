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


_patch_terminus_tmux()
