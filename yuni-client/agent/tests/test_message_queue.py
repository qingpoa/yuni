from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import aclosing
from datetime import UTC, datetime, timedelta

import pytest
from planner_helpers import reply_event, stop_event
from pydantic import SecretStr
from sqlalchemy import select
from test_business_api import make_client, parse_sse, partner_payload
from test_chat_pipeline import initialized_runtime, seed_conversation

from yuni_agent.api.routes import chat
from yuni_agent.chat.queue import ConversationQueue
from yuni_agent.context.budget import ContextBudget
from yuni_agent.context.compaction import SqliteConversationCompactor
from yuni_agent.model.client import ModelClientError
from yuni_agent.storage.migrations import upgrade_database
from yuni_agent.storage.models import ChatMessage, ChatTurn, Conversation
from yuni_agent.tools.models import ToolRequest
from yuni_agent.tools.service import LocalToolExecutor, ToolExecutionError


async def collect(stream):
    async with aclosing(stream):
        return [frame async for frame in stream]


class MergeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def move_to(self, value, *queues):
        self.now = value
        for queue in queues:
            queue._pending_changed.set()
        for _ in range(10):
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_five_second_silence_resets_for_new_inputs_but_not_duplicates():
    clock, batches = MergeClock(), []
    async def produce(ids):
        batches.append(ids)
        yield 'finished'
    queue = ConversationQueue(produce, clock=clock)
    assert queue.quiet_seconds == 5
    first = queue.stream(1, initial_frame='ack')
    try:
        assert await anext(first) == 'ack'
        await clock.move_to(4, queue)
        assert batches == []
        second = queue.stream(2, initial_frame='ack')
        assert await anext(second) == 'ack'
        await clock.move_to(8, queue)
        assert batches == []  # More than five seconds since the first input.
        duplicate = queue.stream(2, initial_frame='ack')
        assert await anext(duplicate) == 'ack'
        await clock.move_to(8.999, queue)
        assert batches == []
        await clock.move_to(9, queue)
        assert batches == [[1, 2]]
        assert await collect(first) == await collect(second) == await collect(duplicate) == ['finished']
    finally:
        await queue.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('busy_seconds', [2, 7])
async def test_next_batch_waits_only_remaining_silence_and_keeps_current_turn(busy_seconds):
    clock, batches = MergeClock(), []
    release = asyncio.Event()
    async def produce(ids):
        batches.append(ids)
        if ids == [1]:
            await release.wait()
        yield 'finished'
    queue = ConversationQueue(produce, clock=clock)
    first = queue.stream(1, initial_frame='ack')
    try:
        assert await anext(first) == 'ack'
        await clock.move_to(5, queue)
        assert batches == [[1]]
        second = queue.stream(2, initial_frame='ack')
        assert await anext(second) == 'ack'
        await clock.move_to(5 + busy_seconds, queue)
        assert batches == [[1]]  # New inputs never interrupt the active batch.
        release.set()
        assert await asyncio.wait_for(collect(first), 1) == ['finished']
        await clock.move_to(clock.now, queue)
        if busy_seconds == 7:
            assert batches == [[1], [2]]  # Deadline passed while busy; no new wait.
        else:
            assert batches == [[1]]
            await clock.move_to(9, queue)
            third = queue.stream(3, initial_frame='ack')
            assert await anext(third) == 'ack'
            await clock.move_to(10, queue)
            assert batches == [[1]]  # The third input reset the next deadline.
            await clock.move_to(14, queue)
            assert batches == [[1], [2, 3]]
            assert await collect(third) == ['finished']
        assert await collect(second) == ['finished']
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_waiting_batch_without_listeners_stays_pending_and_reconnect_does_not_reset():
    clock, batches = MergeClock(), []
    async def produce(ids):
        batches.append(ids)
        yield 'finished'
    queue = ConversationQueue(produce, clock=clock)
    first = queue.stream(1, initial_frame='ack')
    try:
        assert await anext(first) == 'ack'
        await first.aclose()
        await clock.move_to(6, queue)
        assert batches == [] and queue.task is None and list(queue.pending) == [1]
        reconnected = queue.stream(1, initial_frame='ack')
        assert await anext(reconnected) == 'ack'
        await clock.move_to(6, queue)
        assert batches == [[1]]
        assert await collect(reconnected) == ['finished']
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_close_cancels_silence_wait_and_ends_subscribers_without_generation():
    clock, batches = MergeClock(), []
    async def produce(ids):
        batches.append(ids)
        yield 'finished'
    queue = ConversationQueue(produce, clock=clock)
    stream = queue.stream(1, initial_frame='ack')
    assert await anext(stream) == 'ack'
    await clock.move_to(1, queue)
    await queue.close()
    assert await collect(stream) == []
    assert batches == [] and queue.task is None and not queue.pending


@pytest.mark.asyncio
async def test_silence_deadlines_are_independent_between_conversations():
    clock, batches = MergeClock(), []
    async def produce(ids):
        batches.append(ids)
        yield 'finished'
    first_queue, second_queue = ConversationQueue(produce, clock=clock), ConversationQueue(produce, clock=clock)
    first = first_queue.stream(1, initial_frame='ack')
    try:
        assert await anext(first) == 'ack'
        await clock.move_to(4, first_queue)
        second = second_queue.stream(2, initial_frame='ack')
        assert await anext(second) == 'ack'
        await clock.move_to(5, first_queue, second_queue)
        assert batches == [[1]]
        await clock.move_to(9, second_queue)
        assert batches == [[1], [2]]
        assert await collect(first) == await collect(second) == ['finished']
    finally:
        await first_queue.close()
        await second_queue.close()


@pytest.mark.asyncio
async def test_batches_snapshot_inputs_keep_late_replies_and_replay_any_member(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    _, conversation_id, first_id = seed_conversation(runtime)
    runtime.model_api_key = SecretStr("isolated")
    runtime.memory_generation_enabled = runtime.memory_retrieval_enabled = False
    started = [asyncio.Event() for _ in range(3)]
    release = [asyncio.Event() for _ in range(3)]
    requests, reply_requests = [], []

    class Model:
        def __init__(self, **kwargs):
            self.turn = len(requests)
            requests.append([])
            self.calls = self.replies = 0

        async def stream_chat_with_tools(self, messages, tools):
            requests[self.turn].append(messages)
            self.calls += 1
            if self.turn == 0:
                if self.calls == 1:
                    yield reply_event()
                elif self.calls == 2:
                    started[0].set()
                    await release[0].wait()
                    yield reply_event()
                else:
                    yield stop_event()
            elif self.calls == 1:
                started[self.turn].set()
                await release[self.turn].wait()
                yield reply_event()
            else:
                yield stop_event()

        async def stream_chat(self, messages):
            self.replies += 1
            reply_requests.append((self.turn, messages))
            yield f"实际回复{self.turn + 1}.{self.replies}。"

    monkeypatch.setattr(chat, "OpenAICompatibleClient", Model)

    async def send(content, key):
        response = await chat.stream_message(chat.SendMessageInput(content=content, client_message_id=key), runtime)
        return asyncio.create_task(collect(response.body_iterator))

    try:
        with runtime.require_database().session() as session:
            conversation, user = session.get(Conversation, conversation_id), session.get(ChatMessage, first_id)
        first = asyncio.create_task(collect(chat._queued_generate(runtime, conversation, user)))
        await asyncio.wait_for(started[0].wait(), 3)
        second = await send("刚下班", "queued-second")
        third = await send("还没吃饭", "queued-third")
        duplicate = await send("刚下班", "queued-second")
        # Both have already persisted/queued, even before HTTP consumers pull.
        assert len(runtime.chat_queues[conversation_id].pending) == 2
        release[0].set()
        await asyncio.wait_for(started[1].wait(), 3)
        fourth = await send("明天想休息", "queued-fourth")
        text = str(requests[1][0])
        assert "刚下班" in text and "还没吃饭" in text
        assert "明天想休息" not in text and "实际回复1.2。" in text
        assert "刚下班" not in str(requests[0]) and "还没吃饭" not in str(requests[0])
        release[1].set()
        await asyncio.wait_for(started[2].wait(), 3)
        assert "实际回复2.1。" in str(requests[2][0]) and "明天想休息" in str(requests[2][0])
        release[2].set()
        results = await asyncio.wait_for(asyncio.gather(first, second, third, duplicate, fourth), 5)
        assert len(requests) == 3
        second_frames, third_frames = parse_sse("".join(results[1])), parse_sse("".join(results[2]))
        assert second_frames[1:] == third_frames[1:] == parse_sse("".join(results[3]))[1:]
        batch_event = next(data for name, data in second_frames if name == "batch_started")
        ids = batch_event["userMessageIds"]
        assert len(ids) == 2 and ids == sorted(ids)
        with runtime.require_database().session() as session:
            members = [session.get(ChatTurn, message_id) for message_id in ids]
            assert all(item.status == "completed" and item.batch_user_message_id == ids[-1] for item in members)
            late = session.scalar(select(ChatMessage).where(ChatMessage.content == "实际回复1.2。"))
            assert late.id > ids[-1]
            member = session.get(ChatMessage, ids[0])
        replay = parse_sse("".join(await collect(chat._queued_generate(runtime, conversation, member))))
        assert replay == second_frames and len(requests) == 3
        second_current = [item["content"] for turn, messages in reply_requests if turn == 1 for item in messages if item["role"] == "user"][-1]
        assert "刚下班" in second_current and "还没吃饭" in second_current and "明天想休息" not in second_current
    finally:
        await runtime.close_chat_queues()
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_failed_or_disconnected_batch_does_not_drop_queued_messages(tmp_path, monkeypatch, cancel_first):
    runtime = initialized_runtime(tmp_path)
    _, conversation_id, first_id = seed_conversation(runtime)
    runtime.model_api_key = SecretStr("isolated")
    runtime.memory_generation_enabled = runtime.memory_retrieval_enabled = False
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Model:
        def __init__(self, **kwargs):
            self.number = len(calls)
            calls.append(self)
            self.plans = 0
        async def stream_chat_with_tools(self, messages, tools):
            self.plans += 1
            if self.number == 0:
                started.set()
                await release.wait()
                raise ModelClientError("隔离失败")
            yield reply_event() if self.plans == 1 else stop_event()
        async def stream_chat(self, messages):
            yield "下一批正常完成。"
    monkeypatch.setattr(chat, "OpenAICompatibleClient", Model)
    try:
        with runtime.require_database().session() as session:
            conversation, user = session.get(Conversation, conversation_id), session.get(ChatMessage, first_id)
        first_stream = chat._queued_generate(runtime, conversation, user)
        assert "message_created" in await anext(first_stream)
        await asyncio.wait_for(started.wait(), 3)
        response = await chat.stream_message(chat.SendMessageInput(content="排队的内容", client_message_id="after-failed-turn"), runtime)
        second = asyncio.create_task(collect(response.body_iterator))
        if cancel_first:
            await first_stream.aclose()
        else:
            release.set()
            assert "event: error" in "".join(await collect(first_stream))
        frames = await asyncio.wait_for(second, 5)
        assert "下一批正常完成" in "".join(frames) and "completed" in frames[-1]
        assert len(calls) == 2
        with runtime.require_database().session() as session:
            assert session.get(ChatTurn, first_id).status == ("interrupted" if cancel_first else "failed")
    finally:
        await runtime.close_chat_queues()
        runtime.close_storage()


@pytest.mark.asyncio
async def test_one_subscriber_disconnect_does_not_cancel_shared_batch():
    ready, release = asyncio.Event(), asyncio.Event()
    batches = []
    async def produce(ids):
        batches.append(ids)
        yield "started"
        ready.set()
        await release.wait()
        yield "finished"
    queue = ConversationQueue(produce, quiet_seconds=0)
    first, second = queue.stream(1), queue.stream(2)
    try:
        assert await anext(first) == "started"
        assert await anext(second) == "started"
        duplicate = queue.stream(2)
        assert await anext(duplicate) == "started"
        await first.aclose()
        release.set()
        assert await collect(second) == ["finished"]
        assert await collect(duplicate) == ["finished"]
        assert batches == [[1, 2]]
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_pending_message_recovers_on_next_live_input_after_restart(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    _, conversation_id, first_id = seed_conversation(runtime)
    with runtime.require_database().session() as session:
        session.get(ChatMessage, first_id).content = "上次已经发送但还没处理"
        session.add(ChatTurn(user_message_id=first_id, status="queued"))
    runtime.close_storage()
    runtime = initialized_runtime(tmp_path)
    runtime.model_api_key = SecretStr("isolated")
    runtime.memory_generation_enabled = runtime.memory_retrieval_enabled = False
    requests = []
    class Model:
        def __init__(self, **kwargs):
            pass
        async def stream_chat_with_tools(self, messages, tools):
            requests.append(messages)
            yield stop_event()
    monkeypatch.setattr(chat, "OpenAICompatibleClient", Model)
    try:
        response = await chat.stream_message(chat.SendMessageInput(content="这次接着说", client_message_id="restart-next-live"), runtime)
        events = parse_sse("".join(await collect(response.body_iterator)))
        batch = next(data for name, data in events if name == "batch_started")
        assert first_id in batch["userMessageIds"] and len(batch["userMessageIds"]) == 2
        assert "上次已经发送但还没处理" in str(requests[0]) and "这次接着说" in str(requests[0])
        with runtime.require_database().session() as session:
            assert session.get(ChatTurn, first_id).status == "completed"
    finally:
        await runtime.close_chat_queues()
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_current", [False, True])
async def test_pending_without_listeners_waits_for_next_live_request(cancel_current):
    started, release = asyncio.Event(), asyncio.Event()
    batches = []
    async def produce(ids):
        batches.append(ids)
        started.set()
        await release.wait()
        yield "completed"
    queue = ConversationQueue(produce, quiet_seconds=0)
    first = queue.stream(1, initial_frame="ack")
    try:
        assert await anext(first) == "ack"
        await started.wait()
        pending = queue.stream(2, initial_frame="ack")
        assert await anext(pending) == "ack"
        await pending.aclose()
        if cancel_current:
            await first.aclose()
        else:
            release.set()
            assert await collect(first) == ["completed"]
        assert queue.task is None and list(queue.pending) == [2]
        release.set()
        live = queue.stream(3)
        assert await collect(live) == ["completed"]
        assert batches == [[1], [2, 3]]
    finally:
        await queue.close()


def test_save_memory_accepts_request_from_any_batch_member_but_not_future(tmp_path):
    runtime = initialized_runtime(tmp_path)
    _, conversation_id, first_id = seed_conversation(runtime)
    try:
        with runtime.require_database().session() as session:
            first = session.get(ChatMessage, first_id)
            first.content = "帮我记住这个偏好"
            second = ChatMessage(conversation_id=conversation_id, role="user", content="我不吃香菜", status="succeeded")
            future = ChatMessage(conversation_id=conversation_id, role="user", content="把下一批保存下来", status="succeeded")
            session.add_all([second, future]); session.flush()
            root_id = second.id
            session.add_all([ChatTurn(user_message_id=first_id, batch_user_message_id=root_id, status="running"), ChatTurn(user_message_id=root_id, batch_user_message_id=root_id, status="running"), ChatTurn(user_message_id=future.id, status="queued")])
        arguments = dict(type="user_preference", subject_key="food.cilantro", title="香菜", content="用户不吃香菜", request_quote="帮我记住这个偏好", source_quote="我不吃香菜", sensitivity="normal")
        executor = LocalToolExecutor(runtime)
        assert executor._memory_source(ToolRequest(name="save_memory", arguments=arguments), conversation_id, root_id)[1] == root_id
        arguments["request_quote"] = "把下一批保存下来"
        with pytest.raises(ToolExecutionError, match="本轮用户"):
            executor._memory_source(ToolRequest(name="save_memory", arguments=arguments), conversation_id, root_id)
        prepared = SqliteConversationCompactor(runtime, budget=ContextBudget(recent_context_tokens=1))._prepare(conversation_id, None)
        assert prepared is None or all(item.id < first_id for item in prepared.messages)
    finally:
        runtime.close_storage()


def test_batch_migration_preserves_previous_turns(tmp_path):
    path = tmp_path / "previous.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY);INSERT INTO chat_messages VALUES (1);CREATE TABLE chat_turns (user_message_id INTEGER PRIMARY KEY, status VARCHAR(20));INSERT INTO chat_turns VALUES (1, 'completed');CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY);INSERT INTO alembic_version VALUES ('0007_memory_vectors_chat_turns');")
    upgrade_database(path)
    upgrade_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM chat_turns").fetchone() == (1, "completed", None)


def test_compaction_boundary_never_skips_a_late_delivered_placeholder(tmp_path):
    runtime = initialized_runtime(tmp_path)
    _, conversation_id, _ = seed_conversation(runtime)
    try:
        with runtime.require_database().session() as session:
            late = ChatMessage(conversation_id=conversation_id, role="assistant", content="占位后延迟交付", status="succeeded", created_at=datetime.now(UTC) + timedelta(minutes=1))
            session.add(late); session.flush()
            session.add_all([ChatMessage(conversation_id=conversation_id, role=role, content="已处理的内容", status="succeeded") for role in ["user", "assistant", "user"]])
        prepared = SqliteConversationCompactor(runtime, budget=ContextBudget(recent_context_tokens=1))._prepare(conversation_id, None)
        assert prepared is not None
        compacted = {item.id for item in prepared.compact_messages}
        assert all(item.id in compacted for item in prepared.messages if item.id <= prepared.compact_until_message_id)
    finally:
        runtime.close_storage()


def test_backup_preserves_batch_replay_and_old_backups_still_import(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        with runtime.require_database().session() as session:
            conversation_id = session.scalar(select(Conversation.id))
            users = [ChatMessage(conversation_id=conversation_id, role="user", content=f"消息{i}", status="succeeded") for i in range(2)]
            session.add_all(users); session.flush()
            ids = [item.id for item in users]
            session.add_all([ChatTurn(user_message_id=message_id, batch_user_message_id=ids[-1], status="completed") for message_id in ids])
            session.add(ChatMessage(conversation_id=conversation_id, role="assistant", content="合并回复。", status="succeeded", parent_message_id=ids[-1]))
        backup = client.get("/api/data/export", headers=headers).json()
        assert len(backup["tables"]["chatTurns"]) == 2
        assert client.post("/api/data/import", headers=headers, files={"file": ("batch.json", json.dumps(backup).encode())}).status_code == 200
        replay = client.post(f"/api/chat/messages/{ids[0]}/retry", headers=headers)
        assert "合并回复" in replay.text and parse_sse(replay.text)[-1][0] == "completed"
        backup["tables"].pop("chatTurns")
        assert client.post("/api/data/import", headers=headers, files={"file": ("old.json", json.dumps(backup).encode())}).status_code == 200


def test_history_excludes_streaming_placeholder_on_route_remount(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        with runtime.require_database().session() as session:
            conversation_id = session.scalar(select(Conversation.id))
            user_msg = ChatMessage(conversation_id=conversation_id, role="user", content="这点小事夸夸夸", status="succeeded")
            session.add(user_msg)
            session.flush()
            streaming_placeholder = ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content="",
                status="streaming",
                parent_message_id=user_msg.id,
            )
            session.add(streaming_placeholder)
        items = client.get("/api/chat/messages", headers=headers).json()["data"]["items"]
        assert [item["role"] for item in items] == ["user"]
        assert all(item["status"] != "streaming" for item in items)
