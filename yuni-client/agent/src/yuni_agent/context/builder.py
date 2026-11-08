from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select

from yuni_agent.context.budget import ContextBudget, estimate_tokens
from yuni_agent.context.models import BuiltContext, ContextRequest
from yuni_agent.context.planner_builder import build_planner_context
from yuni_agent.memory.retriever import SqliteMemoryRetriever
from yuni_agent.model.reply import REPLY_OUTPUT_INSTRUCTIONS
from yuni_agent.model.voice import strip_voice_markers, voice_capability_prompt
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import (
    AllowedRoot,
    ChatMessage,
    Companion,
    Conversation,
    LocalProfile,
)


class ContextBuilder(Protocol):
    """Builds the bounded model projection from persisted conversation state."""

    async def build(self, request: ContextRequest) -> BuiltContext: ...


class SqliteContextBuilder:
    def __init__(self, runtime: AgentRuntime, budget: ContextBudget | None = None) -> None:
        self.runtime = runtime
        self.budget = budget or ContextBudget()
        self.retriever = SqliteMemoryRetriever(runtime)

    async def build(self, request: ContextRequest) -> BuiltContext:
        with self.runtime.require_database().session() as session:
            conversation = session.get(Conversation, request.conversation_id)
            if conversation is None:
                raise ValueError("conversation does not exist")
            companion = session.get(Companion, conversation.companion_id)
            if companion is None:
                raise ValueError("companion does not exist")
            history = list(
                session.scalars(
                    select(ChatMessage)
                    .where(
                        ChatMessage.conversation_id == request.conversation_id,
                        ChatMessage.status == "succeeded",
                        (ChatMessage.id < request.current_user_message_id if request.history_message_ids is None
                         else ChatMessage.id.in_(request.history_message_ids)),
                        ChatMessage.id > (conversation.summary_until_message_id or 0),
                    )
                    .order_by(ChatMessage.created_at, ChatMessage.id)
                )
            )
            summary = conversation.summary.strip()
            companion_id = companion.id
            profile = session.get(LocalProfile, 1)
            system_prompt = _system_prompt(
                companion,
                user_nickname=profile.nickname if profile is not None else "",
            )
            allowed_roots = list(
                session.scalars(
                    select(AllowedRoot.canonical_path).where(AllowedRoot.status == "active")
                )
            )

        recent_context = [
            re.sub(r"\[(?:发送了?|系统已自动配发)[^\]]*表情包[^\]]*\]", "", item.content or "").strip()
            for item in history[-3:]
            if item.role == "user"
        ]
        current_clean = re.sub(r"\[发送了表情包；[^\]]*\]", "", request.current_input).strip()
        memory_query = " ".join(part for part in [*recent_context, current_clean] if part).strip()[-300:]
        memories = await self.retriever.retrieve(
            companion_id,
            memory_query or "[表情包]",
            limit=6,
            token_budget=self.budget.memory_max_tokens,
        )
        now = datetime.now().astimezone()
        timeline_lines = _format_recent_timeline(history, now)
        time_content = f"此刻：{now.isoformat()}。"
        if timeline_lines:
            time_content += (
                "\n近期消息时间线（注意结合消息距今的真实时间差与现实常识判断进程："
                "现实中的行动、位移或状态变化需要相应的物理时间；短时间内的连续对话通常意味着之前的打算尚未实施或尚未完成，切勿跳过现实耗时直接预设后续状态已发生）：\n"
                + "\n".join(f"- {line}" for line in timeline_lines)
            )
        fixed: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": REPLY_OUTPUT_INSTRUCTIONS},
        ]
        if self.runtime.voice_generation_ready:
            fixed.append({"role": "system", "content": voice_capability_prompt(True)})
        fixed.append(
            {
                "role": "system",
                "content": time_content,
            }
        )
        if summary:
            fixed.append(
                {
                    "role": "system",
                    "content": f"之前的对话背景：\n{summary}",
                }
            )
        if memories:
            labels = {"planned": "尚未发生的计划", "in_progress": "进行中", "completed": "已完成", "cancelled": "已取消"}
            lines = []
            for item in memories:
                label = (labels.get(item.event_status, "经历或计划") if item.kind == "event" else
                         "近期提到的状态，过期不表示已恢复" if item.kind == "ongoing_state" else
                         "过往事实" if item.status == "historical" else "当前事实")
                moment = item.time_text or (item.valid_from.isoformat() if item.valid_from else "时间未明确")
                line = f"- [{label} · {moment}] {item.content}"
                if item.previous_facts and item.status == "active":
                    line += f"（已过期的过往事实：{'；'.join(item.previous_facts)}）"
                lines.append(line)
            fixed.append(
                {
                    "role": "system",
                    "content": "你记得的相关事情，仅在相关时自然用到；当前用户原话优先于旧记忆，不把计划当完成、不推断近况已恢复，不复述内部字段：\n"
                    + "\n".join(lines),
                }
            )

        current = {"role": "user", "content": request.current_input}
        selected: list[ChatMessage] = []
        tool_tokens = _message_tokens(request.tool_messages)
        used = _message_tokens(fixed) + _message_tokens([current]) + tool_tokens
        history_tokens = sum(estimate_tokens(item.context_content) + 8 for item in history)
        # Both Planner and Replyer later receive this turn's tool results; keep
        # room for them so they never push out history mid-turn.
        loop_reserve = self.budget.tool_loop_reserve if request.planner_tools else 0
        projected_tokens = used + history_tokens + loop_reserve
        recent_budget = max(0, self.budget.input_tokens - used - loop_reserve)
        recent_used = 0
        for item in reversed(history):
            cost = estimate_tokens(item.context_content) + 8
            if selected and recent_used + cost > recent_budget:
                break
            if not selected and cost > max(0, recent_budget):
                continue
            selected.append(item)
            recent_used += cost
        selected.reverse()
        recent_messages = [
            {"role": item.role, "content": item.context_content if item.role != "assistant" or item.image or item.voice else strip_voice_markers(item.content)}
            for item in selected
        ]
        messages = [*fixed, *recent_messages, current, *request.tool_messages]
        schema_tokens = estimate_tokens(json.dumps(request.planner_tools, ensure_ascii=False)) if request.planner_tools else 0
        planner_messages = build_planner_context(
            [{"role": item.role, "content": item.context_content} for item in history],
            request.current_input, allowed_roots=allowed_roots, summary=summary,
            memory_enabled=self.runtime.memory_generation_enabled,
            character_profile=system_prompt,
            memories=[{key: value for key, value in item.model_dump(mode="json", exclude_none=True).items()
                       if key in {"id", "kind", "domain", "status", "content", "qualifier", "event_status", "time_text", "valid_from", "valid_until", "fresh_until", "time_meta", "time_certainty", "previous_facts"}
                       and value != "" and value != {} and value != []} for item in memories],
            input_token_budget=max(0, self.budget.input_tokens - schema_tokens - loop_reserve),
            recent_timeline=timeline_lines,
            now_iso=now.isoformat(),
            default_workspace=str(self.runtime.config.workspace_path),
            voice_reply_available=self.runtime.voice_generation_ready,
        )
        planner_fixed = [item for item in planner_messages if item["role"] == "system"]
        planner_history_tokens = sum(estimate_tokens(json.dumps({"role": item.role, "content": item.context_content}, ensure_ascii=False)) + 8 for item in history)
        planner_fixed_tokens = sum(estimate_tokens(json.dumps(item, ensure_ascii=False)) + 8 for item in [*planner_fixed, current])
        projected_tokens = max(projected_tokens, planner_fixed_tokens + planner_history_tokens + schema_tokens + loop_reserve)
        return BuiltContext(
            messages=messages,
            tool_messages=planner_messages,
            estimated_tokens=_message_tokens(messages),
            projected_tokens=projected_tokens,
            compaction_required=projected_tokens > self.budget.input_tokens,
            summary_used=bool(summary),
            memory_ids=[item.id for item in memories],
            recent_message_ids=[item.id for item in selected],
        )


def _message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(str(message.get("content", ""))) + 8 for message in messages)


def _format_recent_timeline(items: list[ChatMessage], now: datetime) -> list[str]:
    lines: list[str] = []
    for item in items[-6:]:
        ts = getattr(item, "created_at", None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        local_ts = ts.astimezone(now.tzinfo)
        delta_sec = max(0, int((now - local_ts).total_seconds()))
        if delta_sec < 60:
            ago = f"距今{delta_sec}秒"
        elif delta_sec < 3600:
            ago = f"距今{delta_sec // 60}分{delta_sec % 60:02d}秒"
        elif delta_sec < 86400:
            ago = f"距今{delta_sec // 3600}小时{(delta_sec % 3600) // 60}分"
        else:
            ago = f"距今{delta_sec // 86400}天"
        sticker = getattr(item, "sticker", None)
        if isinstance(sticker, dict):
            emotions = [str(e).strip() for e in (sticker.get("emotions") or []) if str(e).strip()]
            tag = f"[表情包：{'、'.join(emotions)}]" if emotions else "[表情包]"
            text_part = " ".join((item.content or "").split())[:16]
            snippet = f"{text_part} {tag}".strip() if text_part else tag
        else:
            snippet = " ".join((item.context_content or "").split())[:24]
        lines.append(f"{local_ts.strftime('%H:%M:%S')} ({ago}) {item.role}: {snippet}")
    return lines


def _system_prompt(companion: Companion, *, user_nickname: str = "") -> str:
    intimacy = {
        "reserved": "克制",
        "gentle_natural": "自然亲近",
        "warm_close": "较亲密",
    }.get(companion.intimacy_level, "自然亲近")
    relationship = {"male": "男朋友", "female": "女朋友"}.get(companion.gender, "伴侣")
    nickname = companion.user_nickname or user_nickname or "你"
    boundary = (companion.boundary_text or "").strip() or "未额外设置"
    description = (companion.custom_description or "").strip()
    return (
        "以下是你的身份，以及与用户的关系，请以第一人称对话。\n\n"
        f"姓名：{companion.name}\n关系身份：{relationship}\n对用户的称呼：{nickname}\n"
        f"亲密表达范围：{intimacy}\n用户明确边界：{boundary}\n\n"
        f"<人物资料>\n{description}\n</人物资料>\n\n"
        "人物资料决定你的性格、措辞、说话习惯和相处方式。\n"
        "其中的性格与互动描述用于理解人物，结合当前对话选择合适的反应，"
        "不需要每轮展示所有特点，也不把每种互动机械地执行成固定步骤。\n"
        "未规定的地方自然接话，不主动补全人物背景。\n\n"
        "一次回应可以自然结束，不必承担让每轮对话继续下去的任务。\n"
        "人物可以主动、好奇、撒娇或表达感受，是否这样做取决于人物性格和当下内容。\n\n"
        "涉及真实身份、现实能力和执行结果时如实说明；未完成的操作不能说已完成。\n"
        "尊重用户，不操控、贬低或施压，不泄露凭据；危险和危机话题认真回应。\n\n"
    )
