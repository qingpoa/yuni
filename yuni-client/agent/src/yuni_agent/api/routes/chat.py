from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import Field
from sqlalchemy import and_, func, or_, select

from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
from yuni_agent.agent.proactive import (
    ProactiveChatWorker,
    cancel_proactive_timer,
    schedule_next_proactive,
)
from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.chat.events import emit_event
from yuni_agent.chat.queue import ConversationQueue
from yuni_agent.chat.stickers import StickerSelector, sticker_data, sticker_path
from yuni_agent.chat.voices import VoiceSelector
from yuni_agent.common.http import fail
from yuni_agent.context.builder import SqliteContextBuilder
from yuni_agent.context.compaction import SqliteConversationCompactor
from yuni_agent.expressions.selector import ExpressionSelector
from yuni_agent.memory.scheduling import enqueue_memory_job
from yuni_agent.model.client import ModelClientError, OpenAICompatibleClient
from yuni_agent.model.image import EXTENSIONS as IMAGE_EXTENSIONS
from yuni_agent.model.image import IMAGE_ID_RE, image_data, image_path
from yuni_agent.model.voice import EXTENSIONS as VOICE_EXTENSIONS
from yuni_agent.model.voice import VOICE_ID_RE, voice_data, voice_path
from yuni_agent.runtime import AgentRuntime, RuntimeStatus
from yuni_agent.storage.models import (
    ChatMessage,
    ChatTurn,
    Companion,
    Conversation,
    Settings,
    Sticker,
)
from yuni_agent.tools.service import LocalToolExecutor

router = APIRouter(prefix="/api/chat", tags=["chat"])
chat_graph = LangGraphChat()


class SendMessageInput(CamelModel):
    content: str = Field(default="", max_length=8000)
    client_message_id: str = Field(min_length=8, max_length=64)
    sticker_id: int | None = Field(default=None, gt=0, strict=True)


class ChatMessageData(CamelModel):
    local_id: str
    id: int
    client_message_id: str | None
    role: str
    content: str
    reply_parts: list[str]
    sticker: dict[str, Any] | None = None
    image: dict[str, Any] | None = None
    voice: dict[str, Any] | None = None
    status: str
    message_type: str
    help_resources: list[dict[str, str]]
    parent_message_id: int | None
    error_code: int | None
    error_message: str | None
    created_at: str


class ChatPage(CamelModel):
    items: list[ChatMessageData]
    next_cursor: str | None
    proactive_cursor: int


class ProactivePage(CamelModel):
    items: list[ChatMessageData]
    cursor: int


class MessageAccepted(CamelModel):
    message: ChatMessageData
    turn_status: str


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"event: {event}\ndata: {payload}\n\n"


def _to_data(item: ChatMessage) -> ChatMessageData:
    return ChatMessageData(
        local_id=f"server-{item.id}",
        id=item.id,
        client_message_id=item.client_message_id,
        role=item.role,
        content=item.content,
        reply_parts=item.reply_parts or ([item.content] if item.role == "assistant" and item.content else []),
        sticker=sticker_data(item.sticker["id"], item.sticker["description"]) if item.sticker else None,
        image=image_data(item.image) if item.image else None,
        voice=voice_data(item.voice) if item.voice else None,
        status=item.status,
        message_type=item.message_type,
        help_resources=item.help_resources or [],
        parent_message_id=item.parent_message_id,
        error_code=item.error_code,
        error_message=item.error_message,
        created_at=iso_utc(item.created_at),
    )


def _encode_cursor(item: ChatMessage) -> str:
    value = json.dumps([iso_utc(item.created_at), item.id], separators=(",", ":"))
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[datetime, int]:
    try:
        padded = value + "=" * (-len(value) % 4)
        created_at, message_id = json.loads(base64.urlsafe_b64decode(padded).decode())
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")), int(message_id)
    except (ValueError, TypeError, json.JSONDecodeError):
        fail(status.HTTP_400_BAD_REQUEST, 40000, "分页游标无效")


def _current_conversation(runtime: AgentRuntime) -> tuple[Companion, Conversation]:
    with runtime.require_database().session() as session:
        companion = session.scalar(select(Companion).limit(1))
        if companion is None:
            fail(status.HTTP_409_CONFLICT, 40102, "请先创建伴侣")
        conversation = session.scalar(
            select(Conversation).where(Conversation.companion_id == companion.id).limit(1)
        )
        if conversation is None:
            conversation = Conversation(companion_id=companion.id)
            session.add(conversation)
            session.flush()
        session.expunge(companion)
        session.expunge(conversation)
        return companion, conversation


@router.get("/stickers", response_model=ApiResponse[list[dict[str, Any]]])
def available_stickers(runtime: Annotated[AgentRuntime, Depends(require_session)]):
    with runtime.require_database().session() as session:
        items = []
        for item in session.scalars(select(Sticker).where(Sticker.enabled.is_(True)).order_by(Sticker.id)):
            try:
                if sticker_path(runtime.config.data_dir, item).is_file():
                    items.append(sticker_data(item.id, item.description))
            except ValueError:
                continue
    return ApiResponse(data=items)


@router.get("/stickers/{sticker_id}/content")
def sticker_content(sticker_id: int, runtime: Annotated[AgentRuntime, Depends(require_session)]):
    with runtime.require_database().session() as session:
        item = session.get(Sticker, sticker_id)
        if item is None:
            fail(404, 40301, "表情不存在")
        try:
            path = sticker_path(runtime.config.data_dir, item)
        except ValueError:
            fail(404, 40301, "表情素材不存在")
        if not path.is_file():
            fail(404, 40301, "表情素材不存在")
        return FileResponse(path, media_type=item.content_type, headers={
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        })


@router.get("/images/{image_id}")
def generated_image_content(image_id: str, runtime: Annotated[AgentRuntime, Depends(require_session)]):
    if not IMAGE_ID_RE.fullmatch(image_id):
        fail(404, 40301, "图片不存在")
    for content_type in IMAGE_EXTENSIONS:
        path = image_path(runtime.config.data_dir, image_id, content_type)
        if path.is_file():
            return FileResponse(path, media_type=content_type, headers={
                "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
            })
    fail(404, 40301, "图片不存在")


@router.get("/voices/{voice_id}")
def generated_voice_content(voice_id: str, runtime: Annotated[AgentRuntime, Depends(require_session)]):
    if not VOICE_ID_RE.fullmatch(voice_id):
        fail(404, 40301, "语音不存在")
    for content_type in VOICE_EXTENSIONS:
        try:
            path = voice_path(runtime.config.data_dir, voice_id, content_type)
        except ValueError:
            continue
        if path.is_file():
            return FileResponse(path, media_type=content_type, headers={
                "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
            })
    fail(404, 40301, "语音不存在")


@router.get("/messages", response_model=ApiResponse[ChatPage])
def history(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
    cursor: str = "",
    limit: int = Query(default=30, ge=1, le=100),
) -> ApiResponse[ChatPage]:
    _, conversation = _current_conversation(runtime)
    conditions = [
        ChatMessage.conversation_id == conversation.id,
        ChatMessage.status != "streaming",
    ]
    if cursor:
        cursor_time, cursor_id = _decode_cursor(cursor)
        conditions.append(
            or_(
                ChatMessage.created_at < cursor_time,
                and_(ChatMessage.created_at == cursor_time, ChatMessage.id < cursor_id),
            )
        )
    with runtime.require_database().session() as session:
        # Capture the live watermark before reading history. A concurrent insert
        # can then be replayed, but can never be skipped by a newer watermark.
        proactive_cursor = session.scalar(select(func.max(ChatMessage.id)).where(
            ChatMessage.conversation_id == conversation.id,
            ChatMessage.role == "assistant",
            ChatMessage.parent_message_id.is_(None),
            ChatMessage.status == "succeeded",
        )) or 0
        rows = list(
            session.scalars(
                select(ChatMessage)
                .where(*conditions)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit + 1)
            )
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        rows.reverse()
        items = [_to_data(item) for item in rows]
        next_cursor = _encode_cursor(rows[0]) if has_more and rows else None
    return ApiResponse(data=ChatPage(items=items, next_cursor=next_cursor, proactive_cursor=proactive_cursor))


def _replay(runtime: AgentRuntime, user_message: ChatMessage) -> AsyncIterator[str] | None:
    with runtime.require_database().session() as session:
        turn = session.get(ChatTurn, user_message.id)
        parent_id = (turn.batch_user_message_id if turn else None) or user_message.id
        assistants = list(session.scalars(
            select(ChatMessage)
            .where(
                ChatMessage.parent_message_id == parent_id,
                ChatMessage.role == "assistant",
                ChatMessage.status == "succeeded",
            )
            .order_by(ChatMessage.id)
        ))
        if not assistants and (turn is None or turn.status != "completed"):
            return None
        replies = [_to_data(item) for item in assistants]
        modern = turn is not None
        batch_ids = list(session.scalars(select(ChatTurn.user_message_id).where(ChatTurn.batch_user_message_id == parent_id).order_by(ChatTurn.user_message_id)))

    async def generate() -> AsyncIterator[str]:
        yield _sse("message_created", {"messageId": user_message.id})
        if len(batch_ids) > 1:
            yield _sse("batch_started", {"userMessageIds": batch_ids, "parentMessageId": parent_id})
        last = None
        for item in replies:
            last = {"assistantMessageId": item.id, "content": item.content,
                    "replyParts": item.reply_parts, "sticker": item.sticker, "image": item.image,
                    "voice": item.voice,
                    "messageType": item.message_type,
                    "helpResources": item.help_resources, "parentMessageId": parent_id,
                    "createdAt": item.created_at}
            if modern or len(replies) > 1:
                yield _sse("reply_completed", {**last, "status": "published", "delivery": "persisted_and_queued"})
        # A partially delivered turn is replayed, never re-executed: its earlier
        # tools may already have changed files or created reminders.
        yield _sse("completed", last or {"replyCount": 0})

    return generate()


async def _generate(
    runtime: AgentRuntime,
    companion: Companion,
    conversation: Conversation,
    user_message: ChatMessage,
    *,
    batch_messages: list[ChatMessage] | None = None,
    history_message_ids: list[int] | None = None,
) -> AsyncIterator[str]:
    async with runtime.chat_lock:
        replay = _replay(runtime, user_message)
        if replay is not None:
            async for frame in replay:
                yield frame
            return

        yield _sse("message_created", {"messageId": user_message.id})
        batch = batch_messages or [user_message]
        batch_ids = [item.id for item in batch]
        if len(batch) > 1:
            emit_event(runtime, "batch_started", {"userMessageIds": batch_ids, "parentMessageId": user_message.id})
            yield _sse("batch_started", {"userMessageIds": batch_ids, "parentMessageId": user_message.id})
        current_input = user_message.context_content if len(batch) == 1 else (
            "本轮用户连续发送的消息，按发送顺序一起理解；每条保留真实ID、时间和原文：\n"
            + json.dumps([{"messageId": item.id, "createdAt": iso_utc(item.created_at), "content": item.context_content} for item in batch], ensure_ascii=False)
        )
        assistant_id: int | None = None
        with runtime.require_database().session() as session:
            for message_id in batch_ids:
                session.merge(ChatTurn(user_message_id=message_id, status="running", batch_user_message_id=user_message.id))
            assistant = ChatMessage(
                conversation_id=conversation.id,
                role="assistant",
                message_type="normal",
                content="",
                status="streaming",
                parent_message_id=user_message.id,
                help_resources=[],
            )
            session.add(assistant)
            session.flush()
            assistant_id = assistant.id

        try:
            published: list[dict[str, Any]] = []

            async def deliver_reply(payload: dict[str, Any]) -> dict[str, Any]:
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
                raw_image = payload.get("image")
                image_snapshot = (
                    {"id": str(raw_image["id"]), "content_type": str(raw_image["content_type"]),
                     "description": str(raw_image.get("description") or "")[:500]}
                    if isinstance(raw_image, dict) and raw_image.get("id") else None
                )
                image_dto = image_data(image_snapshot) if image_snapshot else None
                raw_voice = payload.get("voice") if image_snapshot is None else None
                voice_snapshot = (
                    {"id": str(raw_voice["id"]), "content_type": str(raw_voice["content_type"]),
                     "duration": float(raw_voice["duration"])}
                    if isinstance(raw_voice, dict) and raw_voice.get("id") else None
                )
                voice_dto = voice_data(voice_snapshot) if voice_snapshot else None
                if voice_snapshot:
                    sticker_snapshot = sticker_dto = None
                with runtime.require_database().session() as session:
                    assistant = session.get(ChatMessage, assistant_id) if not published else ChatMessage(
                        conversation_id=conversation.id, role="assistant", status="streaming",
                        parent_message_id=user_message.id, content="", help_resources=[],
                    )
                    assert assistant is not None
                    session.add(assistant)
                    assistant.content = str(payload["content"])
                    assistant.reply_parts = list(payload["replyParts"])
                    assistant.sticker = sticker_snapshot
                    assistant.image = image_snapshot
                    assistant.voice = voice_snapshot
                    if voice_snapshot:
                        assistant.reply_parts = [assistant.content]
                    assistant.status = "succeeded"
                    assistant.created_at = datetime.now(UTC)
                    assistant.message_type = str(payload.get("messageType", "normal"))
                    assistant.help_resources = list(payload.get("helpResources", []))
                    session.flush()
                    message_id = assistant.id
                    created_at = iso_utc(assistant.created_at)
                    if runtime.memory_generation_enabled:
                        enqueue_memory_job(session, conversation.id, user_message.id)
                    emit_event(runtime, "reply_completed", {
                        **payload, "sticker": sticker_dto, "image": image_dto, "voice": voice_dto,
                        "assistantMessageId": message_id, "parentMessageId": user_message.id,
                        "userMessageIds": batch_ids, "createdAt": created_at, "proactive": False,
                    }, session=session)
                receipt = {"assistantMessageId": message_id, "parentMessageId": user_message.id,
                           "sticker": sticker_dto, "image": image_dto, "voice": voice_dto,
                           "createdAt": created_at, "status": "published", "delivery": "persisted_and_queued"}
                published.append({**payload, **receipt})
                return receipt

            with runtime.require_database().session() as session:
                settings = session.get(Settings, 1)
                assert settings is not None
                model_client = (
                    OpenAICompatibleClient(
                        base_url=settings.model_base_url,
                        api_key=runtime.model_api_key.get_secret_value(),
                        model=settings.chat_model,
                    )
                    if runtime.model_api_key is not None
                    else None
                )
            graph_context = ChatGraphContext(
                context_builder=SqliteContextBuilder(runtime),
                model_client=model_client,
                tool_executor=LocalToolExecutor(runtime),
                compactor=SqliteConversationCompactor(runtime),
                expression_selector=ExpressionSelector(runtime),
                sticker_selector=StickerSelector(runtime),
                voice_selector=VoiceSelector(runtime),
                deliver_reply=deliver_reply,
            )
            result: dict[str, Any] | None = None
            async with aclosing(chat_graph.stream(
                {
                    "conversation_id": conversation.id,
                    "user_message_id": user_message.id,
                    "assistant_message_id": assistant_id,
                    "current_input": current_input,
                    "history_message_ids": history_message_ids,
                },
                graph_context,
            )) as events:
                async for event in events:
                    if event.type.value == "delta":
                        emit_event(runtime, "delta", {**event.data, "parentMessageId": user_message.id})
                        yield _sse("delta", event.data)
                    elif event.type.value in {
                        "reply_completed",
                        "tool_confirmation_required",
                        "tool_started",
                        "tool_completed",
                    }:
                        if event.type.value != "reply_completed":
                            emit_event(runtime, event.type.value, {**event.data, "parentMessageId": user_message.id})
                        yield _sse(event.type.value, event.data)
                    elif event.type.value == "completed":
                        result = event.data
            if result is None:
                raise ModelClientError("规划流未完成")
            last = published[-1] if published else {"replyCount": 0}
            terminal = {key: value for key, value in last.items() if key not in {"status", "delivery"}}
            with runtime.require_database().session() as session:
                for message_id in batch_ids:
                    session.merge(ChatTurn(user_message_id=message_id, status="completed", batch_user_message_id=user_message.id))
                if not published:
                    placeholder = session.get(ChatMessage, assistant_id)
                    if placeholder is not None:
                        session.delete(placeholder)
                if runtime.memory_generation_enabled:
                    enqueue_memory_job(session, conversation.id, user_message.id)
                schedule_next_proactive(session, conversation.id, last_user_text=user_message.content)
                emit_event(runtime, "completed", {
                    **terminal, "parentMessageId": user_message.id, "userMessageIds": batch_ids,
                }, session=session)
            # Legacy terminal payload retained for old histories/clients. New clients
            # already displayed reply_completed and deduplicate by message id.
            yield _sse("completed", terminal)
        except (asyncio.CancelledError, GeneratorExit):
            with runtime.require_database().session() as session:
                for message_id in batch_ids:
                    turn = session.get(ChatTurn, message_id)
                    if turn is not None and turn.status != "completed":
                        turn.status = "interrupted"
                assistant = session.get(ChatMessage, assistant_id)
                if assistant is not None and assistant.status != "succeeded":
                    assistant.status = "failed"
                    assistant.error_code = 40304
                    assistant.error_message = "回复已中断"
            raise
        except Exception as exc:
            if str(exc) == "model_not_configured":
                error_code, message = 40701, "请先配置模型"
            elif isinstance(exc, ModelClientError) and str(exc) == "模型凭据无效":
                error_code, message = 40702, str(exc)
            else:
                error_code = 40303
                message = str(exc) if isinstance(exc, ModelClientError) else "回复生成失败，请稍后重试"
            with runtime.require_database().session() as session:
                for message_id in batch_ids:
                    session.merge(ChatTurn(user_message_id=message_id, status="failed", batch_user_message_id=user_message.id))
                assistant = session.get(ChatMessage, assistant_id)
                if assistant is not None and assistant.status != "succeeded":
                    assistant.status = "failed"
                    assistant.error_code = error_code
                    assistant.error_message = message
                emit_event(runtime, "error", {"code": error_code, "message": message,
                    "parentMessageId": user_message.id, "userMessageIds": batch_ids}, session=session)
            yield _sse("error", {"code": error_code, "message": message})


def _stream_response(generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _produce_batch(runtime: AgentRuntime, conversation_id: int, message_ids: list[int]) -> AsyncIterator[str]:
    with runtime.require_database().session() as session:
        conversation = session.get(Conversation, conversation_id)
        assert conversation is not None
        companion = session.get(Companion, conversation.companion_id)
        batch = list(session.scalars(select(ChatMessage).where(ChatMessage.id.in_(message_ids)).order_by(ChatMessage.id)))
        # Snapshot explicit history IDs before the first await. In particular, an
        # earlier turn's late reply can have a larger ID than the queued users.
        queued = select(ChatTurn.user_message_id).where(ChatTurn.status.in_(["queued", "running"]))
        history_ids = list(session.scalars(select(ChatMessage.id).where(
            ChatMessage.conversation_id == conversation_id, ChatMessage.status == "succeeded",
            ChatMessage.id.not_in(message_ids), ChatMessage.id.not_in(queued),
        ).order_by(ChatMessage.id)))
    if not batch:
        return
    async with aclosing(_generate(runtime, companion, conversation, batch[-1], batch_messages=batch, history_message_ids=history_ids)) as frames:
        async for frame in frames:
            # Each HTTP subscriber gets its own acknowledgement from _queued_generate.
            if not frame.startswith("event: message_created\n"):
                yield frame


def _queued_generate(runtime: AgentRuntime, conversation: Conversation, user_message: ChatMessage, *, detached: bool = False) -> AsyncIterator[str] | None:
    queue = runtime.chat_queues.get(conversation.id)
    members = [user_message.id]
    # Check before joining: completed/partially published turns are never run twice.
    if queue is None or user_message.id not in queue.entries:
        replay = _replay(runtime, user_message)
        if replay is not None:
            return None if detached else replay
        # A retry of any member retries the original whole batch, with the same
        # source IDs. No successful reply => no published content to replay.
        with runtime.require_database().session() as session:
            turn = session.get(ChatTurn, user_message.id)
            anchor = turn.batch_user_message_id if turn else None
            members = list(session.scalars(select(ChatTurn.user_message_id).where(ChatTurn.batch_user_message_id == anchor))) if anchor else [user_message.id]
            for message_id in members:
                item = session.get(ChatTurn, message_id)
                if item is None:
                    session.add(ChatTurn(user_message_id=message_id, status="queued"))
                else:
                    item.status = "queued"
            # Accepted inputs survive an abandoned pending stream/restart. The
            # next live request drains them too; no unattended tool work runs.
            queued_ids = list(session.scalars(select(ChatTurn.user_message_id).join(
                ChatMessage, ChatMessage.id == ChatTurn.user_message_id,
            ).where(ChatMessage.conversation_id == conversation.id, ChatTurn.status == "queued")))
            members = sorted(set(members) | set(queued_ids))
            emit_event(runtime, "turn_queued", {"userMessageIds": members, "parentMessageId": user_message.id}, session=session)
    if queue is None:
        queue = ConversationQueue(lambda ids: _produce_batch(runtime, conversation.id, ids))
        runtime.chat_queues[conversation.id] = queue
    if detached:
        queue.enqueue(user_message.id, members)
        return None
    return queue.stream(user_message.id, members, initial_frame=_sse("message_created", {"messageId": user_message.id}))


def _accept_message(
    body: SendMessageInput,
    runtime: AgentRuntime,
) -> tuple[Conversation, ChatMessage]:
    if runtime.chat_paused or runtime.status == RuntimeStatus.STOPPING:
        fail(status.HTTP_503_SERVICE_UNAVAILABLE, 40304, "聊天正在停止，请稍后重试")
    content = body.content.strip()
    if not content and body.sticker_id is None:
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40301, "消息不能为空")
    companion, conversation = _current_conversation(runtime)
    with runtime.require_database().session() as session:
        existing = session.scalar(
            select(ChatMessage).where(ChatMessage.client_message_id == body.client_message_id)
        )
        if existing is None:
            sticker = None
            if body.sticker_id is not None:
                asset = session.get(Sticker, body.sticker_id)
                if asset is None or not asset.enabled:
                    fail(422, 40301, "表情不存在或已停用")
                try:
                    path = sticker_path(runtime.config.data_dir, asset)
                except ValueError:
                    fail(422, 40301, "表情素材不存在")
                if not path.is_file():
                    fail(422, 40301, "表情素材不存在")
                # Snapshot what the companion needs to read the sticker; usage stays out.
                sticker = {"id": asset.id, "description": asset.description, "emotions": list(asset.emotions or [])}
            existing = ChatMessage(
                conversation_id=conversation.id,
                client_message_id=body.client_message_id,
                role="user",
                message_type="normal",
                content=content,
                sticker=sticker,
                status="succeeded",
                help_resources=[],
            )
            session.add(existing)
            session.flush()
            session.add(ChatTurn(user_message_id=existing.id, status="queued"))
            cancel_proactive_timer(session, conversation.id)
            emit_event(runtime, "user_message_created", {"message": _to_data(existing).model_dump(mode="json", by_alias=True)}, session=session)
        elif existing.content != content or (existing.sticker or {}).get("id") != body.sticker_id:
            fail(status.HTTP_409_CONFLICT, 40000, "clientMessageId 已用于其他消息")
        session.expunge(existing)
    return conversation, existing


def _accepted(runtime: AgentRuntime, message: ChatMessage) -> ApiResponse[MessageAccepted]:
    with runtime.require_database().session() as session:
        turn = session.get(ChatTurn, message.id)
        return ApiResponse(data=MessageAccepted(message=_to_data(message), turn_status=turn.status if turn else "completed"))


@router.post("/messages", response_model=ApiResponse[MessageAccepted], status_code=202)
async def submit_message(
    body: SendMessageInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[MessageAccepted]:
    conversation, existing = _accept_message(body, runtime)
    _queued_generate(runtime, conversation, existing, detached=True)
    return _accepted(runtime, existing)


@router.post("/messages/{user_message_id}/submit-retry", response_model=ApiResponse[MessageAccepted], status_code=202)
async def submit_retry(
    user_message_id: int,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[MessageAccepted]:
    if runtime.chat_paused or runtime.status == RuntimeStatus.STOPPING:
        fail(status.HTTP_503_SERVICE_UNAVAILABLE, 40304, "聊天正在停止，请稍后重试")
    _, conversation = _current_conversation(runtime)
    with runtime.require_database().session() as session:
        message = session.get(ChatMessage, user_message_id)
        if message is None or message.role != "user" or message.conversation_id != conversation.id:
            fail(status.HTTP_404_NOT_FOUND, 40000, "消息不存在")
        session.expunge(message)
    _queued_generate(runtime, conversation, message, detached=True)
    return _accepted(runtime, message)


@router.post("/messages/stream")
async def stream_message(
    body: SendMessageInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> StreamingResponse:
    conversation, existing = _accept_message(body, runtime)
    return _stream_response(_queued_generate(runtime, conversation, existing))


@router.get("/proactive", response_model=ApiResponse[ProactivePage])
def poll_proactive_messages(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
    after_id: int | None = Query(default=None, alias="afterId", ge=0),
) -> ApiResponse[ProactivePage]:
    _, conversation = _current_conversation(runtime)
    with runtime.require_database().session() as session:
        if after_id is None:
            # Establish a live cursor without notifying about historical messages.
            latest = session.scalar(select(func.max(ChatMessage.id)).where(
                ChatMessage.conversation_id == conversation.id,
                ChatMessage.role == "assistant",
                ChatMessage.parent_message_id.is_(None),
                ChatMessage.status == "succeeded",
            )) or 0
            return ApiResponse(data=ProactivePage(items=[], cursor=latest))
        rows = list(
            session.scalars(
                select(ChatMessage)
                .where(
                    ChatMessage.conversation_id == conversation.id,
                    ChatMessage.id > after_id,
                    ChatMessage.role == "assistant",
                    ChatMessage.parent_message_id.is_(None),
                    ChatMessage.status == "succeeded",
                )
                .order_by(ChatMessage.id)
                .limit(10)
            )
        )
        items = [_to_data(row) for row in rows]
    return ApiResponse(data=ProactivePage(items=items, cursor=rows[-1].id if rows else after_id))


@router.post("/proactive/trigger", response_model=ApiResponse[ChatMessageData])
async def trigger_proactive_message(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[ChatMessageData]:
    if runtime.chat_paused or runtime.status == RuntimeStatus.STOPPING:
        fail(status.HTTP_503_SERVICE_UNAVAILABLE, 40304, "聊天正在停止，请稍后重试")
    if runtime.model_api_key is None:
        fail(status.HTTP_409_CONFLICT, 40701, "请先配置模型")
    msg = await ProactiveChatWorker(runtime).run_once(force=True)
    if msg is None:
        fail(status.HTTP_409_CONFLICT, 40304, "当前正在对话中或暂未满足触发条件，请稍后再试")
    return ApiResponse(data=_to_data(msg))


@router.post("/messages/{user_message_id}/retry")
async def retry_message(
    user_message_id: int,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> StreamingResponse:
    if runtime.chat_paused or runtime.status == RuntimeStatus.STOPPING:
        fail(status.HTTP_503_SERVICE_UNAVAILABLE, 40304, "聊天正在停止，请稍后重试")
    companion, conversation = _current_conversation(runtime)
    with runtime.require_database().session() as session:
        user_message = session.get(ChatMessage, user_message_id)
        if (
            user_message is None
            or user_message.role != "user"
            or user_message.conversation_id != conversation.id
        ):
            fail(status.HTTP_404_NOT_FOUND, 40303, "找不到需要重试的消息")
        session.expunge(user_message)
    return _stream_response(_queued_generate(runtime, conversation, user_message))
