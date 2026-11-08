from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from yuni_agent.runtime import AgentRuntime
from yuni_agent.tools.errors import ToolExecutionError, ToolTimeoutError
from yuni_agent.tools.models import ToolResult, ToolRisk
from yuni_agent.tools.permissions import PathPolicy

ArgsT = TypeVar("ArgsT", bound=BaseModel)

_SCHEMA_DROPPED_KEYS = {"default", "minLength", "maxLength", "maxItems"}


@dataclass(slots=True)
class ToolContext:
    """Per-call state handed to tool handlers."""

    runtime: AgentRuntime
    conversation_id: int
    user_message_id: int | None
    paths: PathPolicy


@dataclass(frozen=True, slots=True)
class ToolPreview:
    summary: str
    target_paths: list[str] = field(default_factory=list)
    diff_preview: str | None = None
    # None means "derive from risk": reads run directly, everything else asks.
    requires_confirmation: bool | None = None


PrepareHandler = Callable[[ToolContext, Any], ToolPreview]
RunHandler = Callable[[ToolContext, Any], Any]


@dataclass(frozen=True, slots=True)
class RegisteredTool(Generic[ArgsT]):
    """Declarative tool spec: schema, validation, risk, preview and execution.

    ``prepare`` validates the request against current state and builds what the
    user sees in the confirmation dialog. It must not change anything.
    ``run`` performs the action. Synchronous handlers (``blocking=True``) are
    moved to a worker thread so file and database I/O never stall the event
    loop; async handlers are awaited directly.
    """

    name: str
    description: str
    arguments: type[ArgsT]
    risk: ToolRisk
    prepare: PrepareHandler
    run: RunHandler
    invalid_message: str = "工具参数无效"
    blocking: bool = True
    # Seconds; None disables the limit (used where a handler holds a lock).
    timeout: float | None = 30.0
    enabled: Callable[[AgentRuntime], bool] = lambda _runtime: True
    hidden_schema_fields: tuple[str, ...] = ()
    # Attempts allowed per user turn (failed attempts count); None means unlimited.
    max_calls_per_turn: int | None = None
    group: str = "general"
    summary: str = ""
    not_for: str = ""
    requires_prior_reply: bool = False
    requires_followup_reply: bool = False
    redacted_fields: tuple[str, ...] = ()

    def definition(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.schema(),
        }}

    def schema(self) -> dict[str, Any]:
        def compact(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: compact(item) for key, item in value.items()
                        if key not in _SCHEMA_DROPPED_KEYS
                        and not (key == "title" and isinstance(item, str))}
            if isinstance(value, list):
                return [compact(item) for item in value]
            return value

        schema = compact(self.arguments.model_json_schema())
        for name in self.hidden_schema_fields:
            schema.get("properties", {}).pop(name, None)
            if name in schema.get("required", []):
                schema["required"].remove(name)
        return schema

    def parse(self, arguments: dict[str, Any]) -> ArgsT:
        try:
            return self.arguments.model_validate(arguments)
        except ValidationError as exc:
            fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors() if error["loc"]})
            suffix = f"（{'、'.join(fields)}）" if fields else ""
            raise ToolExecutionError(f"{self.invalid_message}{suffix}") from exc

    def requires_confirmation(self, preview: ToolPreview) -> bool:
        if preview.requires_confirmation is not None:
            return preview.requires_confirmation
        return self.risk != ToolRisk.READ

    async def _call_with_timeout(
        self, handler: Callable[[ToolContext, ArgsT], Any], context: ToolContext, arguments: ArgsT
    ) -> Any:
        async def call() -> Any:
            if self.blocking:
                return await asyncio.to_thread(handler, context, arguments)
            result = handler(context, arguments)
            return await result if isinstance(result, Awaitable) else result

        if self.timeout is None:
            return await call()
        try:
            return await asyncio.wait_for(call(), self.timeout)
        except TimeoutError as exc:
            # A worker thread cannot be killed; it finishes in the background
            # and its result is discarded.
            raise ToolTimeoutError(f"工具执行超时（{self.timeout:g} 秒）") from exc

    async def build_preview(self, context: ToolContext, arguments: ArgsT) -> ToolPreview:
        return await self._call_with_timeout(self.prepare, context, arguments)

    async def invoke(self, context: ToolContext, arguments: ArgsT) -> ToolResult:
        return await self._call_with_timeout(self.run, context, arguments)


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool] | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolExecutionError("不支持的工具")
        return tool

    def list(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def enabled_tools(self, runtime: AgentRuntime | None = None) -> list[RegisteredTool]:
        return [tool for tool in self._tools.values() if runtime is None or tool.enabled(runtime)]

    def definitions(
        self, runtime: AgentRuntime | None = None, *, names: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        allowed = set(names) if names is not None else None
        return [
            tool.definition()
            for tool in self.enabled_tools(runtime)
            if allowed is None or tool.name in allowed
        ]

    def call_limits(self) -> dict[str, int]:
        return {tool.name: tool.max_calls_per_turn for tool in self._tools.values()
                if tool.max_calls_per_turn is not None}

    def prior_reply_tools(self) -> set[str]:
        return {tool.name for tool in self._tools.values() if tool.requires_prior_reply}

    def followup_reply_tools(self) -> set[str]:
        return {tool.name for tool in self._tools.values() if tool.requires_followup_reply}

    def redacted_fields_for(self, name: str) -> set[str]:
        tool = self._tools.get(name)
        return set(tool.redacted_fields) if tool is not None else set()
