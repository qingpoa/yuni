from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from planner_helpers import reply_delivered, reply_event, stop_event
from pydantic import SecretStr
from sqlalchemy import select

from yuni_agent.api.app import create_app
from yuni_agent.api.routes import chat
from yuni_agent.api.routes.data import clear_data
from yuni_agent.api.routes.events import event_stream
from yuni_agent.api.routes.reminders import mark_delivered
from yuni_agent.chat.events import ReminderEventsWorker, emit_event
from yuni_agent.chat.queue import ConversationQueue
from yuni_agent.config import StartupConfig
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import (
    ChatMessage,
    ChatTurn,
    ClientEvent,
    Companion,
    Conversation,
    Reminder,
    Settings,
)


@pytest_asyncio.fixture
async def event_env(tmp_path):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    runtime.model_api_key = SecretStr("fake-key")
    with runtime.require_database().session() as session:
        settings = session.get(Settings, 1)
        settings.chat_model = "fake-model"
        settings.proactive_chat_enabled = False
        companion = Companion(name="测试伴侣", gender="female", user_nickname="用户")
        session.add(companion)
        session.flush()
        conversation = Conversation(companion_id=companion.id)
        session.add(conversation)
        session.flush()
        conv_id = conversation.id
    try:
        yield runtime, conv_id
    finally:
        await runtime.close_chat_queues()
        runtime.close_storage()


def read_frame(frame):
    lines = frame.splitlines()
    event_id = next((int(line[4:].strip()) for line in lines if line.startswith("id:")), None)
    event_type = next(line[7:].strip() for line in lines if line.startswith("event:"))
    payload = json.loads(next(line[6:].strip() for line in lines if line.startswith("data:")))
    return event_id, event_type, payload


@pytest.mark.asyncio
async def test_http_accepts_without_event_listener_and_disconnect_does_not_cancel(event_env, monkeypatch):
    runtime, conv_id = event_env

    class Model:
        calls = 0
        started, release = asyncio.Event(), asyncio.Event()

        async def stream_chat_with_tools(self, messages, tools):
            yield stop_event() if reply_delivered(messages) else reply_event()

        async def stream_chat(self, messages):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            yield "后台独立生成完成了。"

    model = Model()
    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **_: model)
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test", headers=headers) as client:
        assert (await client.get("/api/events", headers={"X-Yuni-Session": "invalid"})).status_code == 401
        body = {"content": "后台继续处理", "clientMessageId": "detached-delivery-test"}
        response = await client.post("/api/chat/messages", json=body)
        assert response.status_code == 202
        user_id = response.json()["data"]["message"]["id"]
        await asyncio.wait_for(model.started.wait(), 5)
        queue = runtime.chat_queues[conv_id]
        task = queue.task
        # Subscribe and disconnect while the model is blocked.
        events = event_stream(runtime, 0)
        ready = read_frame(await anext(events))
        assert user_id in ready[2]["activeUserMessageIds"]
        await events.aclose()
        duplicate = await client.post("/api/chat/messages", json=body)
        assert duplicate.json()["data"]["message"]["id"] == user_id
        assert model.calls == 1 and task is queue.task and not task.done()
        model.release.set()
        await asyncio.wait_for(asyncio.shield(task), 5)
        with runtime.require_database().session() as session:
            assert session.get(ChatTurn, user_id).status == "completed"
            replies = list(session.scalars(select(ChatMessage).where(ChatMessage.role == "assistant")))
            assert len(replies) == 1 and replies[0].content == "后台独立生成完成了。"
            outbox = session.scalar(select(ClientEvent).where(ClientEvent.event_type == "reply_completed"))
            assert outbox.payload["assistantMessageId"] == replies[0].id
        # Reconnect from the old cursor: every missed durable event is replayed.
        async with aclosing(event_stream(runtime, 0)) as replay:
            assert read_frame(await anext(replay))[1] == "ready"
            received = []
            while not received or received[-1][1] != "completed":
                received.append(read_frame(await asyncio.wait_for(anext(replay), 5)))
        ids = [item[0] for item in received]
        assert ids == sorted(set(ids))
        assert [item[1] for item in received] == ["user_message_created", "turn_queued", "reply_completed", "completed"]
        async with aclosing(event_stream(runtime, ids[-1])) as resumed:
            await anext(resumed)
            newer_id = emit_event(runtime, "reminder_due", {"reminder": {"id": 99}})
            assert read_frame(await anext(resumed))[0] == newer_id
        assert model.calls == 1


@pytest.mark.asyncio
async def test_detached_queue_preserves_five_second_merge_window():
    now, batches = [0.0], []
    async def produce(ids):
        batches.append(ids)
        yield "done"
    queue = ConversationQueue(produce, clock=lambda: now[0])
    async def tick(value):
        now[0] = value
        queue._pending_changed.set()
        for _ in range(10): await asyncio.sleep(0)
    try:
        queue.enqueue(1)
        await tick(4)
        queue.enqueue(2)
        await tick(8.9)
        assert batches == []
        # Duplicate submissions must not postpone the same batch.
        queue.enqueue(2)
        await tick(9)
        assert batches == [[1, 2]]
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_due_reminder_is_recorded_once_and_restored_on_initial_connection(event_env):
    runtime, _ = event_env
    with runtime.require_database().session() as session:
        item = Reminder(title="该休息了", note="起来走走", remind_at=datetime.now(UTC) - timedelta(seconds=1))
        session.add(item)
        session.flush()
        reminder_id = item.id
    worker = ReminderEventsWorker(runtime)
    worker.run_once()
    worker.run_once()
    with runtime.require_database().session() as session:
        events = list(session.scalars(select(ClientEvent).where(ClientEvent.event_type == "reminder_due")))
        assert len(events) == 1
    async with aclosing(event_stream(runtime, None)) as stream:
        _, _, snapshot = read_frame(await anext(stream))
        assert snapshot["reminders"][0]["id"] == reminder_id
    mark_delivered(reminder_id, runtime)
    mark_delivered(reminder_id, runtime)  # Multiple clients/ack retries are idempotent.
    async with aclosing(event_stream(runtime, None)) as stream:
        assert read_frame(await anext(stream))[2]["reminders"] == []


@pytest.mark.asyncio
async def test_data_reset_removes_old_events_but_keeps_ids_monotonic(event_env):
    runtime, _ = event_env
    old = emit_event(runtime, "reply_completed", {"content": "旧数据"})
    async with aclosing(event_stream(runtime, old)) as stream:
        await anext(stream)
        await clear_data(runtime)
        event_id, event_type, payload = read_frame(await anext(stream))
        assert event_id > old and event_type == "state_reset" and payload["reason"] == "cleared"
    with runtime.require_database().session() as session:
        assert [item.event_type for item in session.scalars(select(ClientEvent))] == ["state_reset"]


@pytest.mark.asyncio
async def test_reply_and_its_event_rollback_together(event_env):
    runtime, conv_id = event_env
    with pytest.raises(RuntimeError):
        with runtime.require_database().session() as session:
            message = ChatMessage(conversation_id=conv_id, role="assistant", status="succeeded", content="不能只保存半边")
            session.add(message)
            session.flush()
            emit_event(runtime, "reply_completed", {"assistantMessageId": message.id}, session=session)
            raise RuntimeError("模拟事务失败")
    with runtime.require_database().session() as session:
        assert list(session.scalars(select(ChatMessage))) == []
        assert list(session.scalars(select(ClientEvent))) == []
