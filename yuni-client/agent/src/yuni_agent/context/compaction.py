from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import func, select

from yuni_agent.context.budget import ContextBudget, estimate_tokens
from yuni_agent.memory.worker import MemoryMaintenanceWorker
from yuni_agent.model.client import ModelClient, ModelClientError
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import ChatMessage, ChatTurn, Conversation

logger = logging.getLogger(__name__)


class ConversationCompactor(Protocol):
    """Compacts persisted conversation context without deleting raw messages."""

    async def compact(
        self,
        conversation_id: int,
        client: ModelClient,
        *,
        reason: str,
        current_user_message_id: int | None = None,
        force: bool = False,
    ) -> bool: ...


@dataclass(slots=True)
class CompactionPreparation:
    conversation_id: int
    previous_summary: str
    messages: list[ChatMessage]
    compact_messages: list[ChatMessage]
    tokens_before: int

    @property
    def compact_until_message_id(self) -> int:
        return self.compact_messages[-1].id


class SqliteConversationCompactor:
    def __init__(
        self,
        runtime: AgentRuntime,
        budget: ContextBudget | None = None,
        memory_worker: MemoryMaintenanceWorker | None = None,
    ) -> None:
        self.runtime = runtime
        self.budget = budget or ContextBudget()
        self.memory_worker = memory_worker or MemoryMaintenanceWorker(runtime)

    async def compact(
        self,
        conversation_id: int,
        client: ModelClient,
        *,
        reason: str,
        current_user_message_id: int | None = None,
        force: bool = False,
    ) -> bool:
        async with self.runtime.maintenance_lock:
            preparation = self._prepare(conversation_id, current_user_message_id)
            if preparation is None:
                return False
            threshold = (
                self.budget.background_compaction_tokens
                if reason == "background"
                else self.budget.input_tokens
            )
            if not force and preparation.tokens_before <= threshold:
                return False

            # Memory extraction is attempted before the summary boundary advances. A failed
            # extraction remains retryable because compact never deletes the source messages.
            await self.memory_worker.process_through_locked(
                conversation_id,
                preparation.compact_until_message_id,
                client,
            )
            summary = await self._generate_summary(client, preparation)
            with self.runtime.require_database().session() as session:
                conversation = session.get(Conversation, conversation_id)
                if conversation is None:
                    return False
                conversation.summary = summary
                conversation.summary_until_message_id = preparation.compact_until_message_id
            logger.info(
                "Compacted conversation %s through message %s (%s)",
                conversation_id,
                preparation.compact_until_message_id,
                reason,
            )
            return True

    def _prepare(
        self, conversation_id: int, current_user_message_id: int | None
    ) -> CompactionPreparation | None:
        with self.runtime.require_database().session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return None
            conditions = [
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.status == "succeeded",
                ChatMessage.id > (conversation.summary_until_message_id or 0),
            ]
            if current_user_message_id is not None:
                conditions.append(ChatMessage.id < current_user_message_id)
            first_unhandled = session.scalar(select(func.min(ChatMessage.id)).join(
                ChatTurn, ChatTurn.user_message_id == ChatMessage.id,
            ).where(ChatMessage.conversation_id == conversation_id, ChatTurn.status.in_(["queued", "running"])))
            if first_unhandled is not None:
                conditions.append(ChatMessage.id < first_unhandled)
            messages = list(
                session.scalars(
                    select(ChatMessage)
                    .where(*conditions)
                    # Summary progress is an ID boundary. Delivery timestamps can
                    # move an old placeholder after queued users; select a
                    # contiguous ID prefix so no uncompressed reply is skipped.
                    .order_by(ChatMessage.id)
                )
            )
            for message in messages:
                session.expunge(message)
            previous_summary = conversation.summary.strip()

        cut_index = _find_cut_index(messages, self.budget.recent_context_tokens)
        if cut_index <= 0:
            return None
        compact_messages = messages[:cut_index]
        tokens_before = estimate_tokens(previous_summary) + sum(
            estimate_tokens(item.context_content) + 8 for item in messages
        )
        return CompactionPreparation(
            conversation_id=conversation_id,
            previous_summary=previous_summary,
            messages=messages,
            compact_messages=compact_messages,
            tokens_before=tokens_before,
        )

    async def _generate_summary(
        self, client: ModelClient, preparation: CompactionPreparation
    ) -> str:
        result = await client.complete_json(_summary_prompt(preparation))
        summary = str(result.get("summary", "")).strip()
        if not summary:
            raise ModelClientError("摘要模型未返回 summary")
        if estimate_tokens(summary) > self.budget.summary_max_tokens:
            shortened = await client.complete_json(
                _shorten_summary_prompt(summary, self.budget.summary_max_tokens)
            )
            summary = str(shortened.get("summary", "")).strip()
        if not summary:
            raise ModelClientError("摘要压缩失败")
        return _truncate_to_tokens(summary, self.budget.summary_max_tokens)


class ContextCompactionWorker:
    """Proactively compacts long, idle conversations in the background."""

    def __init__(
        self,
        runtime: AgentRuntime,
        idle_seconds: int = 300,
        budget: ContextBudget | None = None,
    ) -> None:
        self.runtime = runtime
        self.idle_seconds = idle_seconds
        self.memory_worker = MemoryMaintenanceWorker(runtime)
        self.compactor = SqliteConversationCompactor(
            runtime, budget=budget, memory_worker=self.memory_worker
        )

    async def run_forever(self) -> None:
        while True:
            try:
                worked = await self.run_once()
                await asyncio.sleep(5 if worked else 30)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Context compaction loop failed")
                await asyncio.sleep(30)

    async def run_once(self, model_client: ModelClient | None = None) -> bool:
        client = model_client or self.memory_worker.configured_client()
        if client is None:
            return False
        for candidate in self._idle_conversations():
            if await self.compactor.compact(candidate, client, reason="background"):
                return True
        return False

    def _idle_conversations(self) -> list[int]:
        cutoff = datetime.now(UTC) - timedelta(seconds=self.idle_seconds)
        with self.runtime.require_database().session() as session:
            rows = session.execute(
                select(Conversation.id, func.max(ChatMessage.created_at))
                .join(ChatMessage, ChatMessage.conversation_id == Conversation.id)
                .where(ChatMessage.status == "succeeded")
                .group_by(Conversation.id)
                .order_by(func.max(ChatMessage.created_at))
            ).all()
        candidates: list[int] = []
        for conversation_id, last_message_at in rows:
            if last_message_at is None:
                continue
            if last_message_at.tzinfo is None:
                last_message_at = last_message_at.replace(tzinfo=UTC)
            if last_message_at <= cutoff:
                candidates.append(int(conversation_id))
        return candidates


def _find_cut_index(messages: list[ChatMessage], keep_recent_tokens: int) -> int:
    if len(messages) < 2:
        return 0
    used = 0
    boundary = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        used += estimate_tokens(messages[index].context_content) + 8
        boundary = index
        if used >= keep_recent_tokens:
            break
    while boundary > 0 and messages[boundary].role != "user":
        boundary -= 1
    return boundary


def _summary_prompt(preparation: CompactionPreparation) -> list[dict[str, str]]:
    transcript = "\n".join(
        f"[{item.role}#{item.id}] {item.context_content}" for item in preparation.compact_messages
    )
    return [
        {
            "role": "system",
            "content": (
                "你负责压缩虚拟伴侣的会话上下文，不负责生成长期记忆。"
                "把旧摘要和本次移出近期窗口的对话重写成一份新摘要，返回 JSON："
                "{\"summary\":\"...\"}。摘要按以下结构组织：当前关系状态、最近事件与时间线、"
                "用户近期情绪、持续话题、伴侣已经表达的承诺或立场、尚未结束的话题。"
                "不要罗列稳定人物档案，不要把助手猜测写成用户事实；"
                "助手自述的离线活动、地点和动作也不能写成已发生的事实；"
                "用户发送的表情包仅代表情绪象征或顺手活跃气氛，不能把表情包画面中的动作、物品或场景写成用户的真实活动或处境。"
            ),
        },
        {
            "role": "user",
            "content": f"旧摘要：\n{preparation.previous_summary or '无'}\n\n新增对话：\n{transcript}",
        },
    ]


def _shorten_summary_prompt(summary: str, max_tokens: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"把会话摘要压缩到不超过约 {max_tokens} token，保留未完成话题、近期情绪、"
                "关系变化和必要时间线。返回 JSON：{\"summary\":\"...\"}。"
            ),
        },
        {"role": "user", "content": summary},
    ]


def _truncate_to_tokens(value: str, max_tokens: int) -> str:
    if estimate_tokens(value) <= max_tokens:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(value[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return value[:low].rstrip()
