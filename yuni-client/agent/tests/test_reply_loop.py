from __future__ import annotations

import asyncio
import json

import pytest
from planner_helpers import reply_call, reply_event, stop_call, stop_event
from pydantic import SecretStr
from sqlalchemy import select
from test_agent_separation import Builder, Tools
from test_chat_pipeline import initialized_runtime, seed_conversation

from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
from yuni_agent.api.routes.chat import _generate, _replay
from yuni_agent.model.client import ModelClientError, ModelStreamEvent, ModelToolCall
from yuni_agent.storage.models import ChatMessage, ChatTurn, Companion, Conversation


class Scripted:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []
        self.replies = 0

    async def stream_chat_with_tools(self, messages, tools):
        self.requests.append((messages, tools))
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        yield ModelStreamEvent(type="tool_call", tool_call=action)

    async def stream_chat(self, messages):
        self.replies += 1
        if self.replies == 2:
            assert any(item["role"] == "assistant" and item["content"] == "第1次实际回复。" for item in messages)
        yield f"第{self.replies}次实际回复。"


@pytest.mark.asyncio
async def test_reply_receipt_then_tool_then_second_reply_and_stop():
    tools = Tools()
    model = Scripted([reply_call(), ModelToolCall(id="r", name="read_text_file"), reply_call("根据新工具结果补充"), stop_call()])
    deliveries = []
    async def deliver(payload):
        deliveries.append(payload)
        return {"assistantMessageId": 10 + len(deliveries), "status": "published", "delivery": "persisted_and_queued"}
    events = [event async for event in LangGraphChat().stream(
        {"conversation_id": 1, "user_message_id": 2, "assistant_message_id": 3, "current_input": "帮我看看"},
        ChatGraphContext(Builder(), model, tools, deliver_reply=deliver))]
    assert [event.type.value for event in events] == ["reply_completed", "tool_completed", "reply_completed", "completed"]
    receipt = json.loads(model.requests[1][0][-1]["content"])
    assert receipt["content"] == "第1次实际回复。" and receipt["assistantMessageId"] == 11 and receipt["status"] == "published"
    assert "不代表用户已读" in receipt["note"]
    assert tools.calls == 1 and len(deliveries) == 2 and len(model.requests) == 4


@pytest.mark.asyncio
async def test_reply_and_decision_limits_stop_unbounded_model():
    class Endless(Scripted):
        async def stream_chat_with_tools(self, messages, tools):
            self.requests.append((messages, tools))
            names = {item["function"]["name"] for item in tools}
            yield reply_event() if "reply" in names else stop_event()
    model = Endless([])
    events = [event async for event in LangGraphChat().stream(
        {"conversation_id": 1, "user_message_id": 2, "assistant_message_id": 3, "current_input": "你好"}, ChatGraphContext(Builder(), model))]
    assert model.replies == 3 and len(model.requests) == 4 and events[-1].data["replyCount"] == 3
    assert {item["function"]["name"] for item in model.requests[-1][1]} == {"stop"}


@pytest.mark.asyncio
async def test_api_publishes_before_followup_planner_finishes_and_preserves_cancel(tmp_path, monkeypatch):
    from yuni_agent.api.routes import chat
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, user_id = seed_conversation(runtime)
    runtime.model_api_key = SecretStr("isolated")
    runtime.memory_generation_enabled = False
    continued, release = asyncio.Event(), asyncio.Event()
    class Model:
        calls = 0
        def __init__(self, **kwargs):
            pass
        async def stream_chat_with_tools(self, messages, tools):
            Model.calls += 1
            if Model.calls == 1:
                yield reply_event()
            else:
                continued.set()
                await release.wait()
                yield stop_event()
        async def stream_chat(self, messages):
            yield "已经直接发出的回复。"
    monkeypatch.setattr(chat, "OpenAICompatibleClient", Model)
    try:
        with runtime.require_database().session() as session:
            companion, conversation, user = session.get(Companion, companion_id), session.get(Conversation, conversation_id), session.get(ChatMessage, user_id)
        stream = _generate(runtime, companion, conversation, user)
        assert "message_created" in await anext(stream)
        reply = await asyncio.wait_for(anext(stream), 3)
        assert "reply_completed" in reply and "已经直接发出" in reply
        # Do not request another frame yet: planning must be independent of the
        # transport consumer pulling the next event.
        await asyncio.wait_for(continued.wait(), 3)
        with runtime.require_database().session() as session:
            saved = session.scalar(select(ChatMessage).where(ChatMessage.parent_message_id == user_id, ChatMessage.role == "assistant"))
            assert saved.status == "succeeded" and saved.content == "已经直接发出的回复。"
            assistant_id = saved.id
        # Close while the API is paused at its yielded frame, as a transport
        # disconnect can do. It must close the nested graph producer as well.
        await stream.aclose()
        assert not [task for task in asyncio.all_tasks() if task.get_name() == "chat-planner-loop" and not task.done()]
        with runtime.require_database().session() as session:
            assert session.get(ChatMessage, assistant_id).status == "succeeded"
            assert session.get(ChatTurn, user_id).status == "interrupted"
        replay = "".join([frame async for frame in _replay(runtime, user)])
        assert "已经直接发出" in replay and "reply_completed" in replay and Model.calls == 2
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_later_failure_does_not_overwrite_delivered_reply(tmp_path, monkeypatch):
    from yuni_agent.api.routes import chat
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, user_id = seed_conversation(runtime)
    runtime.model_api_key = SecretStr("isolated")
    runtime.memory_generation_enabled = False
    model = Scripted([reply_call(), ModelClientError("后续规划失败")])
    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: model)
    try:
        with runtime.require_database().session() as session:
            companion, conversation, user = session.get(Companion, companion_id), session.get(Conversation, conversation_id), session.get(ChatMessage, user_id)
        frames = [frame async for frame in _generate(runtime, companion, conversation, user)]
        assert "reply_completed" in frames[-2] and "event: error" in frames[-1]
        with runtime.require_database().session() as session:
            saved = session.scalar(select(ChatMessage).where(ChatMessage.parent_message_id == user_id))
            assert saved.status == "succeeded" and saved.content == "第1次实际回复。"
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_times", [0, 2])
async def test_api_stops_and_replays_entire_round_without_regeneration(tmp_path, monkeypatch, reply_times):
    from yuni_agent.api.routes import chat
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, user_id = seed_conversation(runtime)
    runtime.model_api_key = SecretStr("isolated")
    runtime.memory_generation_enabled = False
    model = Scripted([*[reply_call() for _ in range(reply_times)], stop_call()])
    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: model)
    try:
        with runtime.require_database().session() as session:
            companion, conversation, user = session.get(Companion, companion_id), session.get(Conversation, conversation_id), session.get(ChatMessage, user_id)
        frames = [frame async for frame in _generate(runtime, companion, conversation, user)]
        assert sum("event: reply_completed" in frame for frame in frames) == reply_times
        assert "event: completed" in frames[-1]
        replay = [frame async for frame in _generate(runtime, companion, conversation, user)]
        assert replay == frames and len(model.requests) == reply_times + 1
        with runtime.require_database().session() as session:
            assert session.get(ChatTurn, user_id).status == "completed"
            replies = list(session.scalars(select(ChatMessage).where(ChatMessage.parent_message_id == user_id)))
            assert len(replies) == reply_times and all(item.status == "succeeded" for item in replies)
    finally:
        runtime.close_storage()
