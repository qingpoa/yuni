from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from yuni_agent.agent.chat_agent import CompanionChatAgent
from yuni_agent.agent.events import AgentEvent
from yuni_agent.agent.planner import REPLY_TOOL, STOP_TOOL, Planner, build_reply_tool
from yuni_agent.agent.state import AgentState, ChatState
from yuni_agent.context.builder import ContextBuilder
from yuni_agent.context.compaction import ConversationCompactor
from yuni_agent.context.models import ContextRequest
from yuni_agent.model.client import ModelClient
from yuni_agent.model.voice import voice_context_content
from yuni_agent.tools.service import LocalToolExecutor

logger = logging.getLogger(__name__)


class AgentGraph(Protocol):
    """Port implemented by the future LangGraph workflow."""

    async def stream(self, state: AgentState) -> AsyncIterator[AgentEvent]: ...


@dataclass(slots=True)
class ChatGraphContext:
    context_builder: ContextBuilder
    model_client: ModelClient | None
    tool_executor: LocalToolExecutor | None = None
    compactor: ConversationCompactor | None = None
    expression_selector: object | None = None
    sticker_selector: object | None = None
    voice_selector: object | None = None
    deliver_reply: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None


class LangGraphChat:
    def __init__(self) -> None:
        workflow = StateGraph(ChatState, context_schema=ChatGraphContext)
        workflow.add_node("context", self._context)
        workflow.add_node("memory_then_compact", self._memory_then_compact)
        planner = Planner()
        workflow.add_node("planner", planner.plan)
        workflow.add_node("replyer", self._reply)
        workflow.add_node("tool", planner.execute)
        workflow.add_edge(START, "context")
        workflow.add_conditional_edges(
            "context",
            self._route_after_context,
            {"compact": "memory_then_compact", "ready": "planner"},
        )
        workflow.add_edge("memory_then_compact", "planner")
        workflow.add_conditional_edges(
            "planner", self._route_after_tools, {"execute": "tool", "reply": "replyer", "stop": END}
        )
        workflow.add_edge("tool", "planner")
        workflow.add_edge("replyer", "planner")
        self.graph = workflow.compile(name="yuni-chat")

    async def stream(
        self, state: ChatState, context: ChatGraphContext
    ) -> AsyncIterator[AgentEvent]:
        # The graph keeps planning while the HTTP consumer delivers a reply.
        # Its bounded action counts also bound this queue. Disconnect/cancel joins
        # the producer so tools cannot continue as an orphaned background task.
        queue: asyncio.Queue[AgentEvent | BaseException | None] = asyncio.Queue()

        async def produce():
            try:
                async for event in self._stream_graph(state, context):
                    queue.put_nowait(event)
            except Exception as exc:
                queue.put_nowait(exc)
            except asyncio.CancelledError as exc:
                queue.put_nowait(exc)
                raise
            finally:
                queue.put_nowait(None)

        producer = asyncio.create_task(produce(), name="chat-planner-loop")
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer

    async def _stream_graph(
        self, state: ChatState, context: ChatGraphContext
    ) -> AsyncIterator[AgentEvent]:
        final_state: ChatState = state
        async for mode, chunk in self.graph.astream(
            state,
            context=context,
            stream_mode=["custom", "values"],
            config={"recursion_limit": 32},
        ):
            if mode == "custom":
                yield AgentEvent(type=chunk["event"], data=chunk["data"])
            elif mode == "values":
                final_state = chunk
        yield AgentEvent(
            type="completed",
            data={
                "content": final_state.get("response_text", ""),
                "replyParts": final_state.get("reply_parts") or [final_state.get("response_text", "")],
                "sticker": final_state.get("sticker"),
                "voice": final_state.get("voice"),
                "image": final_state.get("image"),
                "messageType": final_state.get("message_type", "normal"),
                "helpResources": final_state.get("help_resources", []),
                "memoryIds": final_state.get("memory_ids", []),
                "stopReason": final_state.get("stop_reason"),
                "replyCount": final_state.get("reply_count", 0),
            },
        )

    @staticmethod
    def _initial_planner_tools(tool_executor: Any, voice_selector: Any = None) -> list[dict[str, Any]]:
        voice_ready = bool(
            voice_selector is not None
            and getattr(getattr(voice_selector, "runtime", None), "voice_generation_ready", False)
        )
        reply_tool = build_reply_tool(voice_enabled=voice_ready) if voice_selector is not None else REPLY_TOOL
        if tool_executor is None:
            return [STOP_TOOL, reply_tool]
        if hasattr(tool_executor, "planner_definitions"):
            business = tool_executor.planner_definitions([], set())
        else:
            business = tool_executor.definitions
        return [*business, STOP_TOOL, reply_tool]

    async def _context(
        self, state: ChatState, runtime: Runtime[ChatGraphContext]
    ) -> ChatState:
        request = ContextRequest(
            conversation_id=state["conversation_id"],
            current_user_message_id=state["user_message_id"],
            current_input=state["current_input"],
            history_message_ids=state.get("history_message_ids"),
            planner_tools=self._initial_planner_tools(
                runtime.context.tool_executor, getattr(runtime.context, "voice_selector", None)
            ),
        )
        context = await runtime.context.context_builder.build(request)
        return {
            "messages": context.messages,
            "tool_messages": context.tool_messages,
            "tool_results": [],
            "tool_note": None,
            "reply_plan": None,
            "tool_iterations": 0,
            "activated_tools": [],
            "executed_tool_keys": [],
            "executed_tool_names": [],
            "pending_image": None,
            "image": None,
            "reply_count": 0,
            "planner_steps": 0,
            "stop_reason": None,
            "published_replies": [],
            "memory_ids": context.memory_ids,
            "compaction_required": context.compaction_required,
        }

    @staticmethod
    def _route_after_context(state: ChatState) -> str:
        return "compact" if state.get("compaction_required") else "ready"

    async def _memory_then_compact(
        self, state: ChatState, runtime: Runtime[ChatGraphContext]
    ) -> ChatState:
        compactor = runtime.context.compactor
        client = runtime.context.model_client
        if compactor is not None and client is not None:
            try:
                compacted = await compactor.compact(
                    state["conversation_id"],
                    client,
                    reason="request",
                    current_user_message_id=state["user_message_id"],
                    force=True,
                )
                if compacted:
                    context = await runtime.context.context_builder.build(
                        ContextRequest(
                            conversation_id=state["conversation_id"],
                            current_user_message_id=state["user_message_id"],
                            current_input=state["current_input"],
                            history_message_ids=state.get("history_message_ids"),
                            planner_tools=self._initial_planner_tools(
                                runtime.context.tool_executor, getattr(runtime.context, "voice_selector", None)
                            ),
                        )
                    )
                    return {
                        "messages": context.messages,
                        "tool_messages": context.tool_messages,
                        "memory_ids": context.memory_ids,
                        "compaction_required": False,
                    }
            except Exception:
                # ContextBuilder already returns a bounded fallback projection. A failed
                # compaction must not make a chat request unavailable.
                logger.exception("Request-time compaction failed")
        return {"compaction_required": False}

    @staticmethod
    def _route_after_tools(state: ChatState) -> str:
        if state.get("stop_reason") is not None:
            return "stop"
        if state.get("pending_tool_call"):
            return "execute"
        if state.get("reply_plan"):
            return "reply"
        raise RuntimeError("planner_action_missing")

    async def _reply(self, state: ChatState, runtime: Runtime[ChatGraphContext]) -> ChatState:
        result = await CompanionChatAgent().generate(state, runtime)
        image = state.get("pending_image")
        voice = None if image else result.get("voice")
        payload = {"content": result["response_text"], "replyParts": result["reply_parts"],
                   "sticker": None if (image or voice) else result.get("sticker"),
                   "image": image,
                   "voice": voice,
                   "messageType": result["message_type"], "helpResources": result["help_resources"]}
        # Persist and enqueue this reply before planning again. stream_writer queues an
        # independent event; the next model request never waits for UI part animations.
        receipt = await runtime.context.deliver_reply(payload) if runtime.context.deliver_reply else {
            "status": "generated", "delivery": "in_memory", "assistantMessageId": state.get("assistant_message_id"),
        }
        published = {**payload, **receipt}
        runtime.stream_writer({"event": "reply_completed", "data": published})
        observation = {"toolName": "reply", **published,
                       "note": "这是实际生成的正文和服务端交付状态，不代表用户已读；不要重复发送。"}
        return {**result,
                "sticker": payload["sticker"],
                "voice": voice,
                "image": published.get("image", image),
                "pending_image": None,
                "messages": [*state["messages"], {"role": "assistant", "content": voice_context_content(result["response_text"], voice)}],
                "tool_messages": [*state["tool_messages"], {"role": "tool", "tool_call_id": state["reply_call_id"],
                                                            "content": json.dumps(observation, ensure_ascii=False)}],
                "reply_plan": None, "reply_count": state.get("reply_count", 0) + 1,
                "published_replies": [*state.get("published_replies", []), published]}

