"""Tool execution orchestration: persistence, confirmation and events.

Tool behavior itself lives in ``builtin.py`` as ``RegisteredTool`` specs; this
module only drives the shared lifecycle every tool goes through:

    validate -> preview -> (confirm) -> run -> record result
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from yuni_agent.api.models import iso_utc
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import ToolCall, utc_now
from yuni_agent.tools.builtin import MAX_TEXT_BYTES, build_registry, memory_source
from yuni_agent.tools.errors import ToolExecutionError, ToolTimeoutError
from yuni_agent.tools.models import ToolRequest, ToolResult
from yuni_agent.tools.permissions import PathPolicy
from yuni_agent.tools.registry import ToolContext, ToolRegistry
from yuni_agent.tools.router import build_request_tools_definition, resolve_tool_request

__all__ = [
    "EmitEvent", "LocalToolExecutor", "MAX_TEXT_BYTES", "TOOL_DEFINITIONS", "TOOL_REGISTRY", "ToolExecutionError",
]

EmitEvent = Callable[[str, dict[str, Any]], None]

TOOL_REGISTRY: ToolRegistry = build_registry()
# Full, unfiltered schema list; kept for callers that build planner prompts directly.
TOOL_DEFINITIONS: list[dict[str, Any]] = TOOL_REGISTRY.definitions()

_REDACTED_ARGUMENTS = {"content", "old_text", "new_text", "source_quote", "request_quote", "prompt"}


class LocalToolExecutor:
    def __init__(self, runtime: AgentRuntime, registry: ToolRegistry | None = None) -> None:
        self.runtime = runtime
        self.registry = registry or TOOL_REGISTRY

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return self.registry.definitions(self.runtime)

    @property
    def call_limits(self) -> dict[str, int]:
        return self.registry.call_limits()

    @property
    def prior_reply_tools(self) -> set[str]:
        return self.registry.prior_reply_tools()

    @property
    def followup_reply_tools(self) -> set[str]:
        return self.registry.followup_reply_tools()

    def planner_definitions(
        self,
        activated_tools: Iterable[str] = (),
        exhausted_tools: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        exhausted = set(exhausted_tools)
        active_names = [name for name in activated_tools if name not in exhausted]
        mounted = self.registry.definitions(self.runtime, names=active_names)
        router_def = build_request_tools_definition(
            self.registry,
            self.runtime,
            activated_tools=activated_tools,
            exhausted_tools=exhausted,
        )
        return [*mounted, router_def] if router_def is not None else mounted

    def resolve_tool_request(
        self,
        arguments: dict[str, Any],
        activated_tools: Iterable[str] = (),
        exhausted_tools: Iterable[str] = (),
    ) -> tuple[list[str], str]:
        return resolve_tool_request(
            self.registry,
            self.runtime,
            arguments,
            activated_tools=activated_tools,
            exhausted_tools=exhausted_tools,
        )

    def _context(self, conversation_id: int, user_message_id: int | None) -> ToolContext:
        return ToolContext(runtime=self.runtime, conversation_id=conversation_id,
                           user_message_id=user_message_id, paths=PathPolicy(self.runtime))

    async def execute(
        self,
        request: ToolRequest,
        *,
        model_call_id: str,
        conversation_id: int,
        assistant_message_id: int,
        emit: EmitEvent,
        user_message_id: int | None = None,
    ) -> ToolResult:
        # Validation and preview errors propagate as ToolExecutionError before any
        # ToolCall row exists; the planner reports them to the model as bad arguments.
        tool = self.registry.require(request.name)
        arguments = tool.parse(request.arguments)
        context = self._context(conversation_id, user_message_id)
        call_id = str(uuid.uuid4())
        try:
            preview = await tool.build_preview(context, arguments)
        except ToolTimeoutError as exc:
            with self.runtime.require_database().session() as session:
                session.add(ToolCall(
                    id=call_id,
                    conversation_id=conversation_id,
                    assistant_message_id=assistant_message_id,
                    tool_name=tool.name,
                    risk=tool.risk.value,
                    status="failed",
                    target_paths=[],
                    arguments_json=_sanitized_arguments(request.arguments, tool.redacted_fields),
                    result_summary=tool.name,
                    error_message=str(exc)[:500],
                    completed_at=utc_now(),
                ))
            emit("tool_started", {"toolCall": self._dto(call_id)})
            emit("tool_completed", {"toolCall": self._dto(call_id)})
            return ToolResult(content=f"工具执行失败：{exc}", is_error=True, details={"modelCallId": model_call_id})
        confirmation_required = tool.requires_confirmation(preview)

        with self.runtime.require_database().session() as session:
            session.add(ToolCall(
                id=call_id,
                conversation_id=conversation_id,
                assistant_message_id=assistant_message_id,
                tool_name=tool.name,
                risk=tool.risk.value,
                status="awaiting_confirmation" if confirmation_required else "running",
                target_paths=preview.target_paths,
                arguments_json=_sanitized_arguments(request.arguments, tool.redacted_fields),
                diff_preview=preview.diff_preview,
                result_summary=preview.summary,
            ))

        if confirmation_required:
            emit("tool_confirmation_required", {"toolCall": self._dto(call_id)})
            try:
                decision = await self.runtime.wait_for_tool_decision(call_id)
            except TimeoutError:
                decision = "reject"
            except asyncio.CancelledError:
                self._finish(call_id, "failed", None, "聊天已中断")
                raise
            if decision != "approve":
                self._finish(call_id, "rejected", "用户拒绝执行")
                emit("tool_completed", {"toolCall": self._dto(call_id)})
                return ToolResult(content="用户拒绝了这次操作，未更改任何内容。")
            self._set_status(call_id, "running")
            context = self._context(conversation_id, user_message_id)

        emit("tool_started", {"toolCall": self._dto(call_id)})
        try:
            result = await tool.invoke(context, arguments)
            self._finish(call_id, "succeeded", result.content[:500])
        except asyncio.CancelledError:
            self._finish(call_id, "failed", None, "聊天已中断")
            raise
        except Exception as exc:
            self._finish(call_id, "failed", None, str(exc)[:500])
            result = ToolResult(content=f"工具执行失败：{exc}", is_error=True)
        emit("tool_completed", {"toolCall": self._dto(call_id)})
        return result.model_copy(update={"details": {**result.details, "modelCallId": model_call_id}})

    def _memory_source(
        self, request: ToolRequest, conversation_id: int, user_message_id: int | None,
    ) -> tuple[int, int]:
        """Compatibility shim used by tests: check save_memory quotes against chat history."""
        arguments = self.registry.require("save_memory").parse(request.arguments)
        return memory_source(self._context(conversation_id, user_message_id), arguments)

    def _dto(self, call_id: str) -> dict[str, Any]:
        with self.runtime.require_database().session() as session:
            item = session.get(ToolCall, call_id)
            assert item is not None
            return {
                "id": item.id,
                "toolName": item.tool_name,
                "risk": item.risk,
                "status": item.status,
                "summary": item.result_summary or item.tool_name,
                "targetPaths": item.target_paths,
                "diffPreview": item.diff_preview,
                "errorMessage": item.error_message,
                "createdAt": iso_utc(item.created_at),
                "completedAt": iso_utc(item.completed_at) if item.completed_at else None,
            }

    def _set_status(self, call_id: str, status: str) -> None:
        with self.runtime.require_database().session() as session:
            item = session.get(ToolCall, call_id)
            assert item is not None
            item.status = status

    def _finish(
        self, call_id: str, status: str, summary: str | None, error: str | None = None
    ) -> None:
        with self.runtime.require_database().session() as session:
            item = session.get(ToolCall, call_id)
            assert item is not None
            item.status = status
            item.completed_at = utc_now()
            if summary:
                item.result_summary = summary
            item.error_message = error


def _sanitized_arguments(
    arguments: dict[str, Any], redacted_fields: Iterable[str] = ()
) -> dict[str, Any]:
    redacted = _REDACTED_ARGUMENTS | set(redacted_fields)
    result: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in redacted:
            text = str(value)
            result[key] = {"length": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        else:
            result[key] = value
    return result
