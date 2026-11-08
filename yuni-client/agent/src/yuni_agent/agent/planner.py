from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Annotated, Any, Literal

from langgraph.runtime import Runtime
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from yuni_agent.agent.state import ChatState
from yuni_agent.context.budget import (
    TOOL_LOOP_MAX_ACTIONS,
    TOOL_RESULT_MAX_TOKENS,
    ContextBudget,
    estimate_tokens,
)
from yuni_agent.context.planner_builder import bounded_text, build_planner_context
from yuni_agent.model.client import ModelClientError
from yuni_agent.tools.models import ToolRequest, ToolResult
from yuni_agent.tools.service import ToolExecutionError

IntentText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


# 字段语义参考 MaiBot ff0fe425 的 reply schema；必填性沿用 Yuni 的已确认契约。
class ExpressionIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    focus: str = Field(default="", max_length=500, description="本次表达主要应贴合的目标消息、片段或话题。")
    reply_act: str = Field(default="", max_length=500, description="本次回复要完成的动作。")
    scene: str = Field(default="", max_length=500, description="当前表达场景。")
    tone: str = Field(default="", max_length=500, description="期望语气。")
    prefer: list[IntentText] = Field(default_factory=list, max_length=12, description="优先考虑的表达类型或话术倾向。")
    avoid: list[IntentText] = Field(default_factory=list, max_length=12, description="需要避免的表达类型、误判方向或不该注入的具体结论。")

    @model_validator(mode="after")
    def meaningful(self) -> ExpressionIntent:
        if not any(value.strip() for value in (self.focus, self.reply_act, self.scene, self.tone)) and not (self.prefer or self.avoid):
            raise ValueError("expression intent is empty")
        return self


class ReplyAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reply_reference: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=6000)] = Field(
        description="提供本次回复的上下文关键信息与回复方向，包括当前聊天状态、人物关系、事实信息和回忆信息；不是回复正文或操作成功证明。",
    )
    expression_intent: ExpressionIntent = Field(
        description="供 Replyer 使用的结构化表达意图，说明本次回复的语气、场景或话术倾向，不是回复正文。只填写相关维度，整体至少一项有内容。",
    )
    reply_style: Literal["自然", "简短", "展开"] = Field(
        default="自然", description="控制本次回复的篇幅和表达方式。自然不追加篇幅要求；简短偏口语、省略；展开允许较详细的评论和说明。",
    )
    use_voice: bool = Field(
        default=False,
        description="结合上下文判断本次回复是否以语音消息发送。当用户明确或委婉想听声音（如‘再说一遍’‘没听清’‘哄哄我’‘叫我一声’‘跟我说晚安’），或当前处于撒娇、亲昵、哄睡、早晚安、情绪安抚等适合用声音表达的时刻设为 true；普通文字闲聊、长篇说明或工具结果汇报设为 false。",
    )


def build_reply_tool(*, voice_enabled: bool = True) -> dict[str, Any]:
    schema = ReplyAction.model_json_schema()
    if not voice_enabled and "properties" in schema:
        schema = {
            **schema,
            "properties": {k: v for k, v in schema["properties"].items() if k != "use_voice"},
        }
    return {
        "type": "function",
        "function": {
            "name": "reply",
            "description": "请求角色向用户发送一次回复，发送后返回实际正文和交付回执，再继续规划。提供事实/背景依据和自由表达意图，不直接写角色台词，不代替真实工具结果。",
            "parameters": schema,
        },
    }


REPLY_TOOL = build_reply_tool(voice_enabled=True)


class StopAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(default="本轮互动已完成", max_length=500)


STOP_TOOL = {"type": "function", "function": {
    "name": "stop", "description": "结束当前轮次，不向用户输出新消息。已回复且无必要操作或补充时调用；不能用来代替尚未发送的正常回应。",
    "parameters": StopAction.model_json_schema(),
}}

MAX_BUSINESS_ACTIONS = TOOL_LOOP_MAX_ACTIONS
MAX_REPLIES = 3
MAX_PLANNER_STEPS = 10

logger = logging.getLogger(__name__)


class Planner:
    async def plan(self, state: ChatState, runtime: Runtime[Any]) -> ChatState:
        started_at = time.perf_counter()
        client, executor = runtime.context.model_client, runtime.context.tool_executor
        if client is None:
            raise RuntimeError("model_not_configured")
        if not hasattr(client, "stream_chat_with_tools"):
            raise ModelClientError("规划模型不支持工具调用")
        messages = list(state.get("tool_messages") or build_planner_context(
            [], state["current_input"], allowed_roots=[],
        ))
        activated = list(state.get("activated_tools", []))
        planner_steps = state.get("planner_steps", 0)
        reply_limited = state.get("reply_count", 0) >= MAX_REPLIES
        exhausted = self._exhausted_tools(state, executor, reply_limited)
        prior_reply_tools: set[str] = getattr(executor, "prior_reply_tools", None) or set()
        supports_routing = (
            executor is not None
            and hasattr(executor, "planner_definitions")
            and hasattr(executor, "resolve_tool_request")
        )
        all_enabled_tools = [
            item
            for item in (executor.definitions if executor is not None else [])
            if item["function"]["name"] not in exhausted
        ]
        enabled_business_names = {item["function"]["name"] for item in all_enabled_tools}
        limited = state.get("tool_iterations", 0) >= MAX_BUSINESS_ACTIONS
        pending_image = bool(state.get("pending_image"))
        feedback = "本轮业务操作次数达到上限，不能继续业务操作，可 reply 或 stop。" if limited else state.get("tool_note")
        if reply_limited:
            feedback = "本轮回复次数达到上限，不能继续 reply；仅完成必要剩余操作或 stop。"
        if pending_image and not reply_limited:
            feedback = "已生成的图片尚未发给用户，下一步需要调用 reply 把图片和说明一起发出。"
        budget = getattr(runtime.context.context_builder, "budget", ContextBudget()).input_tokens

        voice_selector = getattr(runtime.context, "voice_selector", None)
        voice_ready = bool(
            voice_selector is not None
            and getattr(getattr(voice_selector, "runtime", None), "voice_generation_ready", False)
        )
        reply_tool = build_reply_tool(voice_enabled=voice_ready)
        while True:
            if planner_steps >= MAX_PLANNER_STEPS:
                return {
                    "pending_tool_call": None,
                    "reply_plan": None,
                    "stop_reason": "服务端决策次数上限",
                    "planner_steps": planner_steps,
                    "activated_tools": activated,
                    "tool_messages": messages,
                }
            business_tools = (
                executor.planner_definitions(activated, exhausted)
                if supports_routing
                else all_enabled_tools
            )
            definitions = [*([] if limited else business_tools), STOP_TOOL, *([] if reply_limited else [reply_tool])]
            routed_in_step = False
            for _ in range(3):
                request_messages = messages
                if feedback:
                    request_messages = [*messages, {"role": "system", "content": "服务端规划反馈（不是角色台词）：" + feedback}]
                size = sum(estimate_tokens(json.dumps(item, ensure_ascii=False)) + 8 for item in request_messages)
                size += estimate_tokens(json.dumps(definitions, ensure_ascii=False))
                # Drop only old real-dialogue pairs, never the current input or tool trace.
                current_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), -1)
                while size > budget:
                    index = next((i for i, item in enumerate(messages[:current_index]) if item["role"] != "system"), None)
                    if index is None:
                        break
                    removed = messages.pop(index)
                    current_index -= 1
                    if removed["role"] == "user" and index < current_index and messages[index]["role"] == "assistant":
                        messages.pop(index)
                        current_index -= 1
                    request_messages = [*messages, *([{"role": "system", "content": "服务端规划反馈（不是角色台词）：" + feedback}] if feedback else [])]
                    size = sum(estimate_tokens(json.dumps(item, ensure_ascii=False)) + 8 for item in request_messages) + estimate_tokens(json.dumps(definitions, ensure_ascii=False))
                if size > budget:
                    raise ModelClientError("规划上下文过长，请缩短请求")
                calls, chunks, text_size = [], [], 0
                invalid_call = False
                async for event in client.stream_chat_with_tools(request_messages, definitions):
                    if event.type == "text":
                        text_size += len(event.text)
                        if text_size > 64000:
                            raise ModelClientError("规划输出过长，请重试")
                        chunks.append(event.text)
                    elif event.type == "tool_call" and event.tool_call is not None:
                        calls.append(event)
                        if len(calls) > 16:
                            raise ModelClientError("规划动作过多，请重试")
                    elif event.type == "invalid_tool_call":
                        invalid_call = True
                feedback = "每次必须调用且只调用一个有效工具；需要向用户说话时 reply，互动完成时 stop，不能只输出文本。"
                if invalid_call or len(calls) != 1:
                    continue
                event = calls[0]
                call = event.tool_call
                if call.name == "stop":
                    if pending_image and not reply_limited:
                        feedback = "图片已生成但尚未发送，不能 stop；请先调用 reply 把图片发给用户。"
                        continue
                    try:
                        stop = StopAction.model_validate(call.arguments)
                    except ValidationError:
                        feedback = "stop 参数只能包含长度不超过 500 的字符串 reason。"
                        continue
                    step = planner_steps + 1
                    logger.info(
                        "[Flow 1/4 Planner] step=%s action=stop elapsed_ms=%.1f",
                        step, (time.perf_counter() - started_at) * 1000,
                    )
                    return {
                        "pending_tool_call": None,
                        "reply_plan": None,
                        "stop_reason": stop.reason,
                        "planner_steps": step,
                        "activated_tools": activated,
                        "tool_messages": messages,
                    }
                if call.name == "reply":
                    if reply_limited:
                        feedback = "本轮回复次数达到上限，请调用其他可用动作或 stop。"
                        continue
                    try:
                        reply = ReplyAction.model_validate(call.arguments)
                    except ValidationError:
                        feedback = "reply 需要非空 reply_reference 和有效 expression_intent；至少填写一个实际表达意图，reply_style 仅自然/简短/展开。"
                        continue
                    step = planner_steps + 1
                    intent_fields = [k for k, v in reply.expression_intent.model_dump().items() if (v if isinstance(v, list) else v.strip())]
                    logger.info(
                        "[Flow 1/4 Planner] step=%s action=reply style=%s use_voice=%s intent_dims=%s elapsed_ms=%.1f",
                        step, reply.reply_style, reply.use_voice, intent_fields, (time.perf_counter() - started_at) * 1000,
                    )
                    return {
                        "pending_tool_call": None,
                        "reply_plan": reply.model_dump(),
                        "reply_call_id": call.id,
                        "tool_messages": [*request_messages, self._assistant_call(call, chunks, event.reasoning_content)],
                        "planner_steps": step,
                        "activated_tools": activated,
                        "tool_note": state.get("tool_note") or ("本轮操作次数达到上限，未执行剩余操作。" if limited else None),
                    }
                if call.name == "request_tools":
                    if not supports_routing:
                        feedback = "调用了未提供的工具。请只调用本轮 schema 中的工具。"
                        continue
                    if limited:
                        feedback = "本轮业务操作次数达到上限，只能调用 reply 或 stop。"
                        continue
                    try:
                        new_activated, receipt = executor.resolve_tool_request(call.arguments, activated, exhausted)
                    except ToolExecutionError as exc:
                        feedback = str(exc)
                        continue
                    activated = new_activated
                    planner_steps += 1
                    assistant_message = self._assistant_call(call, chunks, event.reasoning_content)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {
                                "toolName": "request_tools",
                                "status": "mounted",
                                "activatedTools": activated,
                                "content": receipt,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    messages = [*messages, assistant_message, tool_message]
                    feedback = None
                    routed_in_step = True
                    logger.info(
                        "[Flow 1/4 Planner] step=%s action=request_tools activated=%s elapsed_ms=%.1f",
                        planner_steps, activated, (time.perf_counter() - started_at) * 1000,
                    )
                    break
                if call.name in exhausted:
                    feedback = f"{call.name} 本轮可用次数已用完，请依据已有结果 reply 或 stop。"
                    continue
                if call.name not in enabled_business_names:
                    feedback = "调用了未提供的工具。请只调用本轮 schema 中的工具。"
                    continue
                if call.name in prior_reply_tools and state.get("reply_count", 0) == 0:
                    feedback = f"调用 {call.name} 前必须先 reply 一句自然的话（如“等我拍给你看看～”）安抚等待，再调用 {call.name}。"
                    continue
                if limited:
                    feedback = "本轮业务操作次数达到上限，只能调用 reply 或 stop。"
                    continue
                key = self._call_key(call.name, call.arguments)
                if key in state.get("executed_tool_keys", []):
                    feedback = "同一操作本轮已处理，未重复执行；请依据真实结果调用 reply 或选择其他必要操作。"
                    messages = [*messages, {"role": "system", "content": feedback}]
                    continue
                if call.name not in activated:
                    activated = [*activated, call.name]
                assistant_message = self._assistant_call(call, chunks, event.reasoning_content)
                step = planner_steps + 1
                logger.info(
                    "[Flow 1/4 Planner] step=%s action=tool:%s elapsed_ms=%.1f",
                    step, call.name, (time.perf_counter() - started_at) * 1000,
                )
                return {
                    "tool_messages": [*request_messages, assistant_message],
                    "pending_tool_call": call.model_dump(),
                    "tool_iterations": state.get("tool_iterations", 0) + 1,
                    "planner_steps": step,
                    "activated_tools": activated,
                }
            if not routed_in_step:
                raise ModelClientError("规划未返回有效动作，请重试")

    async def execute(self, state: ChatState, runtime: Runtime[Any]) -> ChatState:
        executor, pending = runtime.context.tool_executor, state.get("pending_tool_call")
        if executor is None or pending is None:
            raise RuntimeError("tool_executor_unavailable")
        execution_status: str | None = None

        def emit(event: str, data: dict[str, Any]) -> None:
            nonlocal execution_status
            if event == "tool_completed":
                execution_status = data.get("toolCall", data).get("status")
            runtime.stream_writer({"event": event, "data": data})

        try:
            result = await executor.execute(
                ToolRequest(name=pending["name"], arguments=pending["arguments"]), model_call_id=pending["id"],
                conversation_id=state["conversation_id"], assistant_message_id=state["assistant_message_id"],
                emit=emit, user_message_id=state["user_message_id"],
            )
        except ToolExecutionError as exc:
            result = ToolResult(content=f"工具参数无效：{exc}", is_error=True)
        content = bounded_text(result.content, TOOL_RESULT_MAX_TOKENS)
        observation = {"toolName": pending["name"], "isError": result.is_error,
                       "status": execution_status or ("failed" if result.is_error else "returned"), "content": content}
        update: ChatState = {
            "tool_messages": [*state["tool_messages"], {"role": "tool", "tool_call_id": pending["id"], "content": json.dumps(observation, ensure_ascii=False)}],
            "tool_results": [*state.get("tool_results", []), observation],
            "executed_tool_keys": [*state.get("executed_tool_keys", []), self._call_key(pending["name"], pending["arguments"])],
            "executed_tool_names": [*state.get("executed_tool_names", []), pending["name"]],
            "pending_tool_call": None,
        }
        image = result.details.get("image")
        if not result.is_error and isinstance(image, dict):
            update["pending_image"] = image
        return update

    @staticmethod
    def _exhausted_tools(state: ChatState, executor: Any, reply_limited: bool) -> set[str]:
        limits: dict[str, int] = getattr(executor, "call_limits", None) or {}
        used = state.get("executed_tool_names", [])
        exhausted = {name for name, limit in limits.items() if used.count(name) >= limit}
        if reply_limited:
            # Tools whose product must be delivered via a follow-up reply are withdrawn when replies are exhausted.
            followup_tools: set[str] | None = getattr(executor, "followup_reply_tools", None)
            exhausted.update(followup_tools if followup_tools is not None else {"generate_image"})
        return exhausted

    @staticmethod
    def _assistant_call(call, chunks, reasoning):
        message = {"role": "assistant", "content": "".join(chunks) or None, "tool_calls": [{
            "id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
        }]}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return message

    @staticmethod
    def _call_key(name: str, arguments: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps([name, arguments], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
