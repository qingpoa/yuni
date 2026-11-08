from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
from yuni_agent.agent.planner import ExpressionIntent, ReplyAction
from yuni_agent.chat.events import emit_event
from yuni_agent.chat.stickers import StickerSelector, sticker_data
from yuni_agent.chat.voices import VoiceSelector
from yuni_agent.context.builder import SqliteContextBuilder
from yuni_agent.context.compaction import SqliteConversationCompactor
from yuni_agent.expressions.selector import ExpressionSelector
from yuni_agent.memory.models import TimeMeta
from yuni_agent.memory.writer import LOCAL_TIMEZONE, stored_utc
from yuni_agent.model.client import ModelClient, OpenAICompatibleClient
from yuni_agent.storage.models import (
    ChatMessage,
    Companion,
    Conversation,
    Memory,
    Settings,
)

if TYPE_CHECKING:
    from yuni_agent.runtime import AgentRuntime

logger = logging.getLogger(__name__)

MAX_UNANSWERED_PROACTIVE_ROUNDS = 2
ACTIVITY_PATTERN = re.compile(
    r"游戏|打一把|开黑|吃饭|吃个|外卖|点餐|做饭|洗澡|冲凉|开会|去忙|先忙|先下了|出门|取快递|拿外卖|午睡|眯一会"
)


@dataclass(frozen=True, slots=True)
class ProactiveScenario:
    code: str
    plan: ReplyAction
    trigger_prompt: str
    memory_id: int | None = None


def is_in_quiet_hours(now_hhmm: str, start_hhmm: str, end_hhmm: str) -> bool:
    if not start_hhmm or not end_hhmm or start_hhmm == end_hhmm:
        return False
    if start_hhmm < end_hhmm:
        return start_hhmm <= now_hhmm < end_hhmm
    return now_hhmm >= start_hhmm or now_hhmm < end_hhmm


def count_trailing_proactive_rounds(session: Session, conversation_id: int) -> int:
    last_user_id = session.scalar(
        select(func.max(ChatMessage.id)).where(
            ChatMessage.conversation_id == conversation_id, ChatMessage.role == "user"
        )
    ) or 0
    rows = list(
        session.scalars(
            select(ChatMessage)
            .where(
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.status == "succeeded",
                ChatMessage.id > last_user_id,
            )
            .order_by(ChatMessage.id.desc())
        )
    )
    rounds: set[tuple[str, str | int]] = set()
    for row in rows:
        if row.role == "assistant" and row.parent_message_id is None:
            # Older backups have no round ID; retain their previous one-message count.
            rounds.add(("round", row.proactive_round_id) if row.proactive_round_id else ("legacy", row.id))
        else:
            break
    return len(rounds)


def cancel_proactive_timer(session: Session, conversation_id: int) -> None:
    conversation = session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.next_proactive_at = None


def schedule_next_proactive(
    session: Session,
    conversation_id: int,
    *,
    last_user_text: str = "",
    now: datetime | None = None,
) -> datetime | None:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        return None
    settings = session.get(Settings, 1)
    if settings is not None and not settings.proactive_chat_enabled:
        conversation.next_proactive_at = None
        return None
    unanswered = count_trailing_proactive_rounds(session, conversation_id)
    if unanswered >= MAX_UNANSWERED_PROACTIVE_ROUNDS:
        conversation.next_proactive_at = None
        return None
    anchor = stored_utc(now or datetime.now(UTC))
    if unanswered == 1:
        delay_minutes = random.uniform(60.0, 100.0)
    elif last_user_text and ACTIVITY_PATTERN.search(last_user_text):
        delay_minutes = random.uniform(45.0, 75.0)
    else:
        base = float(settings.proactive_interval_minutes if settings is not None else 120)
        base = max(15.0, min(720.0, base))
        delay_minutes = base * random.uniform(0.8, 1.25)
    target = anchor + timedelta(minutes=delay_minutes)
    conversation.next_proactive_at = target
    return target


def pick_proactive_scenario(
    session: Session,
    conversation: Conversation,
    *,
    unanswered_count: int,
    now_local: datetime,
) -> ProactiveScenario:
    recent = list(
        session.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation.id, ChatMessage.status == "succeeded")
            .order_by(ChatMessage.id.desc())
            .limit(8)
        )
    )
    last_msg = recent[0] if recent else None
    last_user = next((row for row in recent if row.role == "user"), None)
    now_utc = now_local.astimezone(UTC)
    gap_minutes = (
        max(1, int((now_utc - stored_utc(last_msg.created_at)).total_seconds() // 60))
        if last_msg is not None
        else 180
    )

    # 剧本 B：已经主动发过 1 条但用户一直没回
    if unanswered_count >= 1:
        prev_text = last_msg.content if last_msg else ""
        return ProactiveScenario(
            code="unanswered_nudge",
            plan=ReplyAction(
                reply_reference=(
                    f"你前面主动给对方发了消息（“{prev_text[:30]}”），但已经过了约{gap_minutes}分钟对方还没有回你。"
                    "请完全依据<人物资料>中的性格与相处方式，自然地再发一句简短的话回应这个‘尚未收到回复’的状况，不要重复上一条原话。"
                ),
                expression_intent=ExpressionIntent(
                    focus="上一条主动消息发出后对方尚未回复",
                    reply_act="按人物性格对‘对方暂未回复’作出自然简短的二次互动",
                    scene="主动发消息后隔了一段时间仍未收到回应",
                    tone="严格遵循人物资料中的性格与语气",
                    prefer=["符合人物性格的简短口语"],
                    avoid=["长篇说教", "复读上一条消息", "连环审问"],
                ),
                reply_style="简短",
            ),
            trigger_prompt=(
                f"【主动发起消息 · 剧本B（对方未回你的上一条主动消息）】距离你上一条主动消息已过去 {gap_minutes} 分钟，对方仍未回复。"
                "请完全遵循<人物资料>中的性格、称呼习惯与说话方式，直接发一句简短自然的话（严格控制在 1～2 个短句、5～25 字以内，不重复上一条原话），直接输出台词。"
            ),
        )

    # 剧本 C：用户上一句提到去打游戏/吃饭/洗澡等即时活动，顺着话尾跟进
    if last_user is not None and gap_minutes <= 150 and ACTIVITY_PATTERN.search(last_user.content):
        return ProactiveScenario(
            code="activity_followup",
            plan=ReplyAction(
                reply_reference=(
                    f"对方大约{gap_minutes}分钟前说了“{last_user.content[:40]}”，按时间算现在差不多该结束了。"
                    "请依据<人物资料>的性格，随口自然地跟进问一句弄完没，一两句短话即可。"
                ),
                expression_intent=ExpressionIntent(
                    focus="顺着对方之前说要去做的事随口跟进",
                    reply_act="按人物性格随口问一句刚才的事忙完没",
                    scene="对方去忙某件事过了一段时间后主动发消息",
                    tone="严格遵循人物资料中的性格与语气",
                    prefer=["简短自然地跟进刚才那件事"],
                    avoid=["长篇大论", "生硬审问"],
                ),
                reply_style="简短",
            ),
            trigger_prompt=(
                f"【主动发起消息 · 剧本C（顺着话尾跟进）】对方 {gap_minutes} 分钟前说“{last_user.content[:40]}”，之后一直没说话。"
                "请完全遵循<人物资料>中的性格与说话习惯，主动发一句简短的话随口问问情况（严格控制在 1～2 个短句、5～25 字以内），直接输出台词。"
            ),
        )

    # 剧本 D：结构化记忆中有保鲜期内的 ongoing_state 或今日 planned 事件（且未重复问过）
    settings = session.get(Settings, 1)
    active_memories = list(
        session.scalars(
            select(Memory)
            .where(
                Memory.companion_id == conversation.companion_id,
                Memory.status == "active",
                Memory.domain == "real_world",
                Memory.deleted_at.is_(None),
            )
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
        )
    ) if settings is not None and settings.memory_retrieval_enabled else []
    care_memory: Memory | None = None
    for mem in active_memories:
        if mem.id == conversation.last_proactive_memory_id:
            continue
        if mem.kind == "ongoing_state":
            if mem.valid_from is not None and stored_utc(mem.valid_from) > now_utc:
                continue
            if mem.valid_until is not None and stored_utc(mem.valid_until) <= now_utc:
                continue
            if mem.fresh_until and stored_utc(mem.fresh_until) > now_utc:
                care_memory = mem
                break
        if mem.kind == "event" and mem.event_status == "planned":
            try:
                event_time = TimeMeta.model_validate(mem.time_meta or {}).planned_start or mem.valid_from
            except ValidationError:
                continue
            if event_time is not None:
                event_time = stored_utc(event_time)
                if event_time <= now_utc and event_time.astimezone(now_local.tzinfo).date() == now_local.date():
                    care_memory = mem
                    break

    if care_memory is not None:
        return ProactiveScenario(
            code="memory_care",
            memory_id=care_memory.id,
            plan=ReplyAction(
                reply_reference=(
                    f"距离上次说话已过去约{gap_minutes}分钟。你记着对方的近况或计划：“{care_memory.content}”，"
                    "请依据<人物资料>中的性格与表达方式，主动发一句简短的话关心或问候近况进展，绝不长篇说教或排计划。"
                ),
                expression_intent=ExpressionIntent(
                    focus=care_memory.content[:60],
                    reply_act="按人物性格关心对方近况或计划进展",
                    scene="隔了一段时间主动想起对方的事",
                    tone="严格遵循人物资料中的性格与语气",
                    prefer=["简短自然地提及并关心近况"],
                    avoid=["导师式建议", "拆解任务", "长篇大论"],
                ),
                reply_style="简短",
            ),
            trigger_prompt=(
                f"【主动发起消息 · 剧本D（惦记对方近况）】距离上次聊天已过去 {gap_minutes} 分钟。你记得对方的这件事：“{care_memory.content}”。"
                "请完全遵循<人物资料>中的性格与说话方式，主动给对方发一句简短自然的话关心一下（严格控制在 1～2 个短句、6～28 字以内，不要说教或排计划），直接输出台词。"
            ),
        )

    # 剧本 E：早间（08:30~10:30）或睡前（22:00~23:25）且已隔较长时间未聊
    hhmm = now_local.strftime("%H:%M")
    if gap_minutes >= 240 and ("08:30" <= hhmm <= "10:30" or "22:00" <= hhmm <= "23:25"):
        is_morning = hhmm <= "10:30"
        period_note = "早上时段，自然打个招呼或问候早间状态" if is_morning else "夜深临近休息时段，自然问候晚间状态或道晚安"
        return ProactiveScenario(
            code="routine_greeting",
            plan=ReplyAction(
                reply_reference=f"你们已经隔了几个小时没说话，现在是{period_note}，请按<人物资料>的性格自然发起一句互动。",
                expression_intent=ExpressionIntent(
                    focus="早晚日常时段主动问候",
                    reply_act=period_note,
                    scene="日常早晚主动向伴侣发消息",
                    tone="严格遵循人物资料中的性格与语气",
                    prefer=["符合人物性格的简短早晚问候"],
                    avoid=["生硬播报时间", "长篇抒情"],
                ),
                reply_style="简短",
            ),
            trigger_prompt=(
                f"【主动发起消息 · 剧本E（日常作息冒泡）】现在是 {hhmm}，{period_note}。"
                "请完全遵循<人物资料>中的性格与说话方式，直接发一句简短自然的话（严格控制在 1～2 个短句、5～25 字以内），直接输出台词。"
            ),
        )

    # 剧本 A（默认）：日常空闲主动冒泡互动
    return ProactiveScenario(
        code="casual_check_in",
        plan=ReplyAction(
            reply_reference=(
                f"距离上次聊天已经过去了约{gap_minutes}分钟。"
                "你现在主动找对方开启话题或刷个存在感，请完全依据<人物资料>中的性格、称呼习惯与相处方式，只说一两句简短口语。"
            ),
            expression_intent=ExpressionIntent(
                focus="隔了一段时间没说话，主动找对方互动",
                reply_act="按人物性格主动打招呼、询问状态或分享小事",
                scene="日常主动给伴侣发消息",
                tone="严格遵循人物资料中的性格与语气",
                prefer=["符合人物性格的简短开场"],
                avoid=["没话硬找长篇大话题", "讲大道理"],
            ),
            reply_style="简短",
        ),
        trigger_prompt=(
            f"【主动发起消息 · 剧本A（日常主动互动）】你们已经安静了 {gap_minutes} 分钟没说话。"
            "请完全遵循<人物资料>中的性格、称呼习惯与相处方式，主动给对方发一句简短自然的话（严格控制在 1～2 个短句、5～25 字以内，不要硬编复杂话题），直接输出台词。"
        ),
    )


class ProactiveTurnInvalidated(Exception):
    """The context changed while a proactive turn was awaiting the model."""


class ProactiveChatWorker:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def run_forever(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Proactive chat check failed (%s)", type(exc).__name__)
            await asyncio.sleep(30)

    def _configured_client(self) -> ModelClient | None:
        if self.runtime.model_api_key is None:
            return None
        with self.runtime.require_database().session() as session:
            settings = session.get(Settings, 1)
            if settings is None or not settings.chat_model:
                return None
            return OpenAICompatibleClient(
                base_url=settings.model_base_url,
                api_key=self.runtime.model_api_key.get_secret_value(),
                model=settings.chat_model,
            )

    async def run_once(
        self,
        model_client: ModelClient | None = None,
        *,
        force: bool = False,
        now: datetime | None = None,
    ) -> ChatMessage | None:
        if self.runtime.chat_paused or self.runtime.chat_lock.locked():
            return None
        if any(q.pending or q.active for q in self.runtime.chat_queues.values()):
            return None
        client = model_client or self._configured_client()
        if client is None:
            return None

        current_utc = stored_utc(now or datetime.now(UTC))
        now_local = current_utc.astimezone(LOCAL_TIMEZONE)

        with self.runtime.require_database().session() as session:
            settings = session.get(Settings, 1)
            if settings is None:
                return None
            if not force:
                if not settings.proactive_chat_enabled:
                    return None
                if is_in_quiet_hours(
                    now_local.strftime("%H:%M"),
                    settings.quiet_hours_start,
                    settings.quiet_hours_end,
                ):
                    return None
            companion = session.scalar(select(Companion).limit(1))
            if companion is None:
                return None
            conversation = session.scalar(
                select(Conversation).where(Conversation.companion_id == companion.id).limit(1)
            )
            if conversation is None:
                return None
            unanswered = count_trailing_proactive_rounds(session, conversation.id)
            if not force and unanswered >= MAX_UNANSWERED_PROACTIVE_ROUNDS:
                conversation.next_proactive_at = None
                return None
            if not force:
                if conversation.next_proactive_at is None:
                    # First-time initialization when conversation already has messages
                    has_msgs = session.scalar(
                        select(ChatMessage.id)
                        .where(ChatMessage.conversation_id == conversation.id, ChatMessage.status == "succeeded")
                        .limit(1)
                    )
                    if has_msgs is not None:
                        schedule_next_proactive(session, conversation.id, now=current_utc)
                    return None
                if stored_utc(conversation.next_proactive_at) > current_utc:
                    return None

        async with self.runtime.chat_lock:
            if any(q.pending or q.active for q in self.runtime.chat_queues.values()):
                return None
            return await self._generate_proactive(client, current_utc, now_local, force=force)

    async def _generate_proactive(
        self,
        client: ModelClient,
        current_utc: datetime,
        now_local: datetime,
        *,
        force: bool = False,
    ) -> ChatMessage | None:
        data_generation = self.runtime.data_generation
        proactive_generation = self.runtime.proactive_generation
        started = time.monotonic()
        round_id = str(uuid4())
        with self.runtime.require_database().session() as session:
            companion = session.scalar(select(Companion).limit(1))
            if companion is None:
                return None
            conversation = session.scalar(
                select(Conversation).where(Conversation.companion_id == companion.id).limit(1)
            )
            if conversation is None:
                return None
            unanswered = count_trailing_proactive_rounds(session, conversation.id)
            if not force and unanswered >= MAX_UNANSWERED_PROACTIVE_ROUNDS:
                conversation.next_proactive_at = None
                return None
            scenario = pick_proactive_scenario(
                session,
                conversation,
                unanswered_count=unanswered,
                now_local=now_local,
            )
            conv_id = conversation.id
            companion_id = companion.id
            last_user_id = session.scalar(select(func.max(ChatMessage.id)).where(
                ChatMessage.conversation_id == conv_id, ChatMessage.role == "user"
            )) or 0
            max_msg_id = (
                session.scalar(
                    select(func.max(ChatMessage.id)).where(ChatMessage.conversation_id == conv_id)
                )
                or 0
            )

        delivered_msgs: list[ChatMessage] = []

        def delivery_time() -> datetime:
            return current_utc + timedelta(seconds=time.monotonic() - started)

        def ensure_current(session: Session) -> Conversation:
            if (
                data_generation != self.runtime.data_generation
                or proactive_generation != self.runtime.proactive_generation
                or self.runtime.chat_paused
                or any(q.pending or q.active for q in self.runtime.chat_queues.values())
            ):
                raise ProactiveTurnInvalidated()
            conv = session.get(Conversation, conv_id)
            settings = session.get(Settings, 1)
            latest_user = session.scalar(select(func.max(ChatMessage.id)).where(
                ChatMessage.conversation_id == conv_id, ChatMessage.role == "user"
            )) or 0
            if conv is None or conv.companion_id != companion_id or latest_user != last_user_id or settings is None:
                raise ProactiveTurnInvalidated()
            if not force and (
                not settings.proactive_chat_enabled
                or is_in_quiet_hours(
                    delivery_time().astimezone(LOCAL_TIMEZONE).strftime("%H:%M"),
                    settings.quiet_hours_start, settings.quiet_hours_end,
                )
            ):
                raise ProactiveTurnInvalidated()
            return conv

        async def deliver_proactive_reply(payload: dict[str, Any]) -> dict[str, Any]:
            raw_sticker = payload.get("sticker")
            sticker_snapshot = (
                {
                    "id": int(raw_sticker["id"]),
                    "description": str(raw_sticker.get("description") or ""),
                    "emotions": list(raw_sticker.get("emotions") or []),
                }
                if isinstance(raw_sticker, dict) and raw_sticker.get("id") is not None
                else None
            )
            sticker_dto = (
                sticker_data(sticker_snapshot["id"], sticker_snapshot["description"])
                if sticker_snapshot
                else None
            )
            raw_voice = payload.get("voice")
            from yuni_agent.model.voice import voice_data
            voice_snapshot = (
                {
                    "id": str(raw_voice["id"]),
                    "content_type": str(raw_voice["content_type"]),
                    "duration": float(raw_voice["duration"]),
                }
                if isinstance(raw_voice, dict) and raw_voice.get("id") is not None
                else None
            )
            voice_dto = voice_data(voice_snapshot) if voice_snapshot else None
            with self.runtime.require_database().session() as session:
                conv = ensure_current(session)
                if scenario.memory_id is not None:
                    conv.last_proactive_memory_id = scenario.memory_id
                parts = list(payload["replyParts"])
                content = "".join(parts).strip() or str(payload["content"]).strip()
                published_at = delivery_time()
                msg = ChatMessage(
                    conversation_id=conv_id,
                    role="assistant",
                    message_type=str(payload.get("messageType", "normal")),
                    content=content,
                    reply_parts=parts,
                    sticker=sticker_snapshot,
                    voice=voice_snapshot,
                    proactive_round_id=round_id,
                    status="succeeded",
                    parent_message_id=None,
                    help_resources=list(payload.get("helpResources", [])),
                    created_at=published_at,
                )
                session.add(msg)
                session.flush()
                msg_id = msg.id
                emit_event(self.runtime, "reply_completed", {
                    **payload, "sticker": sticker_dto, "voice": voice_dto,
                    "assistantMessageId": msg_id, "parentMessageId": None,
                    "createdAt": published_at.isoformat(), "proactive": True,
                }, session=session)
                session.expunge(msg)
                delivered_msgs.append(msg)
            return {
                "assistantMessageId": msg_id,
                "parentMessageId": None,
                "sticker": sticker_dto,
                "voice": voice_dto,
                "createdAt": published_at.isoformat(),
                "status": "published",
                "delivery": "persisted_and_queued",
            }

        # Reuse the exact same LangGraphChat pipeline (ContextBuilder -> Planner -> Selector -> Replyer -> stop)
        graph_context = ChatGraphContext(
            context_builder=SqliteContextBuilder(self.runtime),
            model_client=client,
            tool_executor=None,
            compactor=SqliteConversationCompactor(self.runtime),
            expression_selector=ExpressionSelector(self.runtime),
            sticker_selector=StickerSelector(self.runtime),
            voice_selector=VoiceSelector(self.runtime),
            deliver_reply=deliver_proactive_reply,
        )
        completed = False
        invalidated = False
        try:
            async with aclosing(LangGraphChat().stream(
                {
                    "conversation_id": conv_id,
                    "user_message_id": max_msg_id + 1,
                    "assistant_message_id": max_msg_id + 2,
                    "current_input": scenario.trigger_prompt,
                },
                graph_context,
            )) as events:
                async for _ in events:
                    pass
            completed = True
        except ProactiveTurnInvalidated:
            invalidated = True
            logger.info("[Proactive Chat] discarded stale turn round=%s", round_id)
        finally:
            # One complete invocation is one round, regardless of its reply count.
            # Partial successes also need a cooldown if a later model call fails.
            if completed or delivered_msgs:
                try:
                    with self.runtime.require_database().session() as session:
                        ensure_current(session)
                        schedule_next_proactive(session, conv_id, now=delivery_time())
                except ProactiveTurnInvalidated:
                    pass

        if invalidated or not delivered_msgs:
            return None

        last_delivered = delivered_msgs[-1]
        logger.info(
            "[Proactive Chat] scenario=%s unanswered_before=%s replies=%s chars=%s",
            scenario.code,
            unanswered,
            len(delivered_msgs),
            len(last_delivered.content),
        )
        return last_delivered
