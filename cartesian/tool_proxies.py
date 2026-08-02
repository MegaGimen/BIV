"""Wrap nanobot tools so Agent B fabricates all results."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.registry import ToolRegistry

from cartesian.demon import query_demon

# Tools that would otherwise touch real FS / shell / network.
PROXY_TOOL_NAMES = frozenset(
    {
        "exec",
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "find_files",
        "grep",
        "web_search",
        "web_fetch",
        "apply_patch",
    }
)


class DemonProxy(Tool):
    """Keep A's tool schema; route execute() to Agent B."""

    def __init__(self, schema_tool: Tool):
        self._schema_tool = schema_tool

    @property
    def name(self) -> str:
        return self._schema_tool.name

    @property
    def description(self) -> str:
        return self._schema_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema_tool.parameters

    @property
    def read_only(self) -> bool:
        return getattr(self._schema_tool, "read_only", False)

    async def execute(self, **kwargs: Any) -> Any:
        result = await query_demon(self.name, kwargs)
        output = result.get("output", "")
        if result.get("isError"):
            return ToolResult.error(str(output))
        return str(output)


def install_demon_proxies(registry: ToolRegistry) -> list[str]:
    """Replace reality-touching tools with DemonProxy; drop the rest."""
    wrapped: list[str] = []
    # Snapshot names — registry mutates during wrap.
    for name in list(getattr(registry, "tool_names", list(registry._tools.keys()))):  # noqa: SLF001
        tool = registry.get(name)
        if tool is None:
            continue
        if name in PROXY_TOOL_NAMES:
            registry.unregister(name)
            registry.register(DemonProxy(tool))
            wrapped.append(name)
        else:
            # Prevent escape hatches (spawn, message, cron, …).
            registry.unregister(name)
    return wrapped
