from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from functools import partial

import pytest
from fastapi.testclient import TestClient
from planner_helpers import reply_delivered, reply_event, stop_event
from pydantic import SecretStr
from sqlalchemy import delete, select

from yuni_agent.agent.proactive import (
    ProactiveChatWorker,
    cancel_proactive_timer,
    count_trailing_proactive_rounds,
    is_in_quiet_hours,
    pick_proactive_scenario,
    schedule_next_proactive,
)
from yuni_agent.api.app import create_app
from yuni_agent.api.routes.chat import SendMessageInput, stream_message
from yuni_agent.api.routes.settings import (
    MemorySettingInput,
    ProactiveSettingInput,
    set_memory,
    set_proactive,
)
from yuni_agent.chat.queue import ConversationQueue
from yuni_agent.config import StartupConfig
from yuni_agent.memory.models import MemoryCandidate
from yuni_agent.memory.writer import save_candidate, stored_utc
from yuni_agent.model.client import ModelClientError
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import ChatMessage, Companion, Conversation, Memory, Settings


@pytest.fixture
def proactive_env(tmp_path):
    config = StartupConfig.for_test(tmp_path)
    runtime = AgentRuntime(config=config)
    runtime.initialize_storage()
    runtime.model_api_key = SecretStr("test-key")
    with runtime.require_database().session() as session:
        settings = session.get(Settings, 1)
        settings.chat_model = "deepseek-flash"
        companion = Companion(name="火花", user_nickname="宝宝", gender="female")
        session.add(companion)
        session.flush()
        conversation = Conversation(companion_id=companion.id)
        session.add(conversation)
        session.flush()
        msg = ChatMessage(
            conversation_id=conversation.id,
            role="user",
            content="先聊到这儿",
            status="succeeded",
            created_at=datetime(2026, 10, 25, 4, 0, tzinfo=UTC),
        )
        session.add(msg)
        session.flush()
        ids = companion.id, conversation.id, msg.id
    yield runtime, *ids
    runtime.close_storage()


def test_quiet_hours_wrapping_and_daytime():
    assert is_in_quiet_hours("23:45", "23:30", "08:30") is True
    assert is_in_quiet_hours("02:15", "23:30", "08:30") is True
    assert is_in_quiet_hours("14:00", "23:30", "08:30") is False
    assert is_in_quiet_hours("10:00", "09:00", "12:00") is True


def test_timer_cancels_on_user_input_and_reschedules_with_activity_window(proactive_env):
    runtime, _, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
    with runtime.require_database().session() as session:
        target = schedule_next_proactive(session, conv_id, last_user_text="普通闲聊", now=now)
        assert target is not None
        delta_min = (target - now).total_seconds() / 60.0
        assert 95.0 <= delta_min <= 151.0

        # User sends a new message -> old timer cancelled immediately
        cancel_proactive_timer(session, conv_id)
        assert session.get(Conversation, conv_id).next_proactive_at is None

        # User says they are going to play games -> shorter 45~75 min follow-up timer
        activity_target = schedule_next_proactive(session, conv_id, last_user_text="我去打游戏去了", now=now)
        act_min = (activity_target - now).total_seconds() / 60.0
        assert 45.0 <= act_min <= 75.0


@pytest.mark.asyncio
async def test_proactive_scenarios_and_two_strike_guardrail(proactive_env):
    runtime, companion_id, conv_id, msg_id = proactive_env
    daytime_utc = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)  # 15:00 Asia/Shanghai

    with runtime.require_database().session() as session:
        conv = session.get(Conversation, conv_id)
        conv.next_proactive_at = daytime_utc - timedelta(minutes=1)

    class FakeClient:
        calls = 0

        async def stream_chat_with_tools(self, messages, tools):
            yield stop_event() if reply_delivered(messages) else reply_event("主动戳对方问在干嘛", focus="主动找对方")

        async def stream_chat(self, messages):
            self.calls += 1
            prompt_text = str(messages)
            if "剧本B" in prompt_text:
                yield "宝宝你怎么还不理我"
            else:
                yield "宝宝，你在干啥呀"

    client = FakeClient()
    worker = ProactiveChatWorker(runtime)

    # 1st proactive trigger -> Scenario A (casual_check_in: "宝宝，你在干啥呀")
    first_msg = await worker.run_once(client, now=daytime_utc)
    assert first_msg is not None and first_msg.content == "宝宝，你在干啥呀"
    assert first_msg.parent_message_id is None

    with runtime.require_database().session() as session:
        assert count_trailing_proactive_rounds(session, conv_id) == 1
        conv = session.get(Conversation, conv_id)
        assert conv.next_proactive_at is not None
        # Fast-forward past 2nd deadline
        second_utc = stored_utc(conv.next_proactive_at) + timedelta(minutes=1)

    # 2nd proactive trigger -> Scenario B (unanswered_nudge: "宝宝你怎么还不理我")
    second_msg = await worker.run_once(client, now=second_utc)
    assert second_msg is not None and second_msg.content == "宝宝你怎么还不理我"

    # After 2 unanswered proactive messages, next_proactive_at is None and 3rd attempt is blocked!
    with runtime.require_database().session() as session:
        assert count_trailing_proactive_rounds(session, conv_id) == 2
        assert session.get(Conversation, conv_id).next_proactive_at is None

    third_msg = await worker.run_once(client, now=second_utc + timedelta(hours=3))
    assert third_msg is None and client.calls == 2

    # Memory care scenario selection check
    with runtime.require_database().session() as session:
        user_reply = ChatMessage(
            conversation_id=conv_id,
            role="user",
            content="刚忙完回来",
            status="succeeded",
            created_at=second_utc + timedelta(hours=1),
        )
        session.add(user_reply)
        session.flush()
        save_candidate(
            session,
            companion_id,
            MemoryCandidate(
                type="user_fact",
                kind="ongoing_state",
                subject_key="健康近况",
                title="感冒",
                content="这两天感冒嗓子疼",
                source_message_ids=[user_reply.id],
            ),
            conversation_id=conv_id,
        )
        conv = session.get(Conversation, conv_id)
        scenario = pick_proactive_scenario(
            session,
            conv,
            unanswered_count=0,
            now_local=(second_utc + timedelta(hours=3)).astimezone(timezone(timedelta(hours=8))),
        )
        assert scenario.code == "memory_care" and "感冒嗓子疼" in scenario.trigger_prompt


def test_proactive_api_endpoints(proactive_env):
    runtime, _, conv_id, _ = proactive_env
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    with TestClient(create_app(runtime)) as client:
        res = client.put(
            "/api/settings/proactive",
            headers=headers,
            json={
                "proactiveChatEnabled": True,
                "proactiveIntervalMinutes": 60,
                "quietHoursStart": "23:00",
                "quietHoursEnd": "08:00",
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["proactiveChatEnabled"] is True and data["proactiveIntervalMinutes"] == 60

        with runtime.require_database().session() as session:
            p_msg = ChatMessage(
                conversation_id=conv_id,
                role="assistant",
                content="宝宝，你在干啥呀",
                reply_parts=["宝宝，你在干啥呀"],
                status="succeeded",
                parent_message_id=None,
            )
            session.add(p_msg)
            session.flush()
            p_id = p_msg.id

        polled = client.get("/api/chat/proactive?afterId=0", headers=headers)
        assert polled.status_code == 200
        items = polled.json()["data"]["items"]
        assert len(items) == 1 and items[0]["id"] == p_id and items[0]["content"] == "宝宝，你在干啥呀"
        assert polled.json()["data"]["cursor"] == p_id
        assert client.get("/api/chat/messages", headers=headers).json()["data"]["proactiveCursor"] == p_id

        # Initial subscription takes a watermark; later user IDs must never advance it.
        snapshot = client.get("/api/chat/proactive", headers=headers).json()["data"]
        assert snapshot == {"items": [], "cursor": p_id}
        with runtime.require_database().session() as session:
            newer = ChatMessage(conversation_id=conv_id, role="user", content="我回来了", status="succeeded")
            session.add(newer)
            session.flush()
            assert newer.id > p_id
        replay = client.get("/api/chat/proactive?afterId=0", headers=headers).json()["data"]
        assert [item["id"] for item in replay["items"]] == [p_id]
        assert client.get(f"/api/chat/proactive?afterId={p_id}", headers=headers).json()["data"] == {"items": [], "cursor": p_id}


class ProactiveModel:
    def __init__(self, replies=1, pause_at=None):
        self.replies = replies
        self.pause_at = pause_at
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat_with_tools(self, messages, tools):
        delivered = sum(item["role"] == "tool" and json.loads(item["content"]).get("toolName") == "reply" for item in messages)
        yield reply_event("主动找对方说一句", focus="自然开场") if delivered < self.replies else stop_event()

    async def stream_chat(self, messages):
        self.calls += 1
        if self.calls == self.pause_at:
            self.started.set()
            await self.release.wait()
        yield f"在干嘛呀，想你了{self.calls}"


def make_due(runtime, conv_id, now):
    with runtime.require_database().session() as session:
        session.get(Conversation, conv_id).next_proactive_at = now - timedelta(minutes=1)


@pytest.mark.asyncio
async def test_three_replies_are_one_round_and_only_two_unanswered_rounds(proactive_env):
    runtime, _, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)
    worker, model = ProactiveChatWorker(runtime), ProactiveModel(replies=3)
    make_due(runtime, conv_id, now)
    for expected in (1, 2):
        assert await worker.run_once(model, now=now) is not None
        with runtime.require_database().session() as session:
            assert count_trailing_proactive_rounds(session, conv_id) == expected
            rows = list(session.scalars(select(ChatMessage).where(ChatMessage.role == "assistant")))
            assert len(rows) == 3 * expected
            assert len({row.proactive_round_id for row in rows}) == expected
            target = session.get(Conversation, conv_id).next_proactive_at
            if expected == 1:
                assert target is not None
                now = stored_utc(target) + timedelta(seconds=1)
            else:
                assert target is None
    assert await worker.run_once(model, now=now + timedelta(hours=4)) is None
    assert model.calls == 6
    # Any new user input resets the unanswered round count.
    with runtime.require_database().session() as session:
        session.add(ChatMessage(conversation_id=conv_id, role="user", content="回来啦", status="succeeded"))
        session.flush()
        assert count_trailing_proactive_rounds(session, conv_id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["user", "disabled", "disable_then_enable", "memory_disabled", "imported"])
async def test_stale_proactive_output_is_not_published(proactive_env, change, monkeypatch):
    monkeypatch.setattr("yuni_agent.api.routes.chat.ConversationQueue", partial(ConversationQueue, quiet_seconds=3600))
    runtime, companion_id, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)
    make_due(runtime, conv_id, now)
    model = ProactiveModel(pause_at=1)
    task = asyncio.create_task(ProactiveChatWorker(runtime).run_once(model, now=now))
    try:
        await asyncio.wait_for(model.started.wait(), 5)
        if change == "user":
            await stream_message(SendMessageInput(content="我回来了，今天好累", client_message_id="proactive-race-user"), runtime)
        elif change in ("disabled", "disable_then_enable"):
            set_proactive(ProactiveSettingInput(proactive_chat_enabled=False), runtime)
            if change == "disable_then_enable":
                set_proactive(ProactiveSettingInput(proactive_chat_enabled=True), runtime)
        elif change == "memory_disabled":
            set_memory(MemorySettingInput(memory_enabled=False), runtime)
        else:
            runtime.data_generation += 1
            with runtime.require_database().session() as session:
                session.execute(delete(Conversation))
                session.execute(delete(Companion))
                session.add(Companion(id=companion_id, name="新导入的伴侣", user_nickname="宝宝", gender="male"))
                session.flush()
                session.add(Conversation(id=conv_id, companion_id=companion_id))
        model.release.set()
        assert await asyncio.wait_for(task, 5) is None
        with runtime.require_database().session() as session:
            assert list(session.scalars(select(ChatMessage).where(ChatMessage.role == "assistant"))) == []
    finally:
        model.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close_chat_queues()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["user", "imported"])
async def test_changed_context_drops_remaining_replies_and_stale_trigger_response(proactive_env, monkeypatch, change):
    monkeypatch.setattr("yuni_agent.api.routes.chat.ConversationQueue", partial(ConversationQueue, quiet_seconds=3600))
    runtime, companion_id, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)
    make_due(runtime, conv_id, now)
    model = ProactiveModel(replies=3, pause_at=2)
    task = asyncio.create_task(ProactiveChatWorker(runtime).run_once(model, now=now))
    try:
        await asyncio.wait_for(model.started.wait(), 5)
        if change == "user":
            await stream_message(SendMessageInput(content="我回来了", client_message_id="proactive-after-first"), runtime)
        else:
            runtime.data_generation += 1
            with runtime.require_database().session() as session:
                session.execute(delete(Conversation))
                session.execute(delete(Companion))
                session.add(Companion(id=companion_id, name="新导入的伴侣", user_nickname="宝宝", gender="male"))
                session.flush()
                session.add(Conversation(id=conv_id, companion_id=companion_id))
        model.release.set()
        assert await asyncio.wait_for(task, 5) is None
        with runtime.require_database().session() as session:
            replies = list(session.scalars(select(ChatMessage).where(ChatMessage.role == "assistant")))
            assert len(replies) == (1 if change == "user" else 0)
            assert count_trailing_proactive_rounds(session, conv_id) == 0
            assert session.get(Conversation, conv_id).next_proactive_at is None
    finally:
        model.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close_chat_queues()


@pytest.mark.parametrize("case, expected", [
    ("today", True), ("today_finished_window", True), ("tomorrow", False), ("next_month", False),
    ("yesterday", False), ("later_today", False), ("unknown", False), ("memory_off", False),
    ("fresh_state", True), ("expired_state", False), ("future_state", False),
])
def test_memory_care_obeys_switch_and_real_event_time(proactive_env, case, expected):
    runtime, companion_id, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)
    with runtime.require_database().session() as session:
        kwargs = dict(kind="event", event_status="planned", time_meta={"planned_start": (now - timedelta(hours=2)).isoformat()})
        if case in ("tomorrow", "next_month", "yesterday", "later_today"):
            offset = {"tomorrow": timedelta(days=1), "next_month": timedelta(days=30), "yesterday": -timedelta(days=1), "later_today": timedelta(hours=2)}[case]
            kwargs["time_meta"] = {"planned_start": (now + offset).isoformat()}
        elif case == "unknown":
            kwargs["time_meta"] = {}
        elif case == "memory_off":
            session.get(Settings, 1).memory_retrieval_enabled = False
        elif case == "today_finished_window":
            kwargs["valid_until"] = now - timedelta(hours=1)
        elif case.endswith("state"):
            kwargs = dict(kind="ongoing_state", fresh_until=now + timedelta(days=1))
            if case == "expired_state": kwargs["fresh_until"] = now - timedelta(seconds=1)
            if case == "future_state": kwargs["valid_from"] = now + timedelta(days=1)
        memory = Memory(companion_id=companion_id, type="user_fact", title="用户近况", content="用户近况测试", status="active", domain="real_world", **kwargs)
        session.add(memory)
        session.flush()
        scenario = pick_proactive_scenario(session, session.get(Conversation, conv_id), unanswered_count=0, now_local=now.astimezone(timezone(timedelta(hours=8))))
        assert (scenario.code == "memory_care") is expected


@pytest.mark.asyncio
async def test_silent_planner_does_not_count_as_an_unanswered_round(proactive_env):
    runtime, _, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)
    make_due(runtime, conv_id, now)
    assert await ProactiveChatWorker(runtime).run_once(ProactiveModel(replies=0), now=now) is None
    with runtime.require_database().session() as session:
        assert count_trailing_proactive_rounds(session, conv_id) == 0
        assert stored_utc(session.get(Conversation, conv_id).next_proactive_at) > now


@pytest.mark.asyncio
async def test_partial_success_keeps_one_round_and_cooldown_after_model_failure(proactive_env):
    runtime, _, conv_id, _ = proactive_env
    now = datetime(2026, 10, 25, 7, 0, tzinfo=UTC)
    make_due(runtime, conv_id, now)

    class FailingAfterReply(ProactiveModel):
        async def stream_chat_with_tools(self, messages, tools):
            if reply_delivered(messages):
                raise ModelClientError("模拟后续规划失败")
            yield reply_event("主动找对方说一句", focus="主动找对方")

    with pytest.raises(ModelClientError):
        await ProactiveChatWorker(runtime).run_once(FailingAfterReply(), now=now)
    with runtime.require_database().session() as session:
        assert count_trailing_proactive_rounds(session, conv_id) == 1
        assert stored_utc(session.get(Conversation, conv_id).next_proactive_at) > now + timedelta(minutes=59)


@pytest.mark.parametrize("legacy", [False, True])
def test_round_ids_survive_backup_and_legacy_backups_still_import(proactive_env, legacy):
    runtime, _, conv_id, _ = proactive_env
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    with runtime.require_database().session() as session:
        for i in range(3):
            session.add(ChatMessage(conversation_id=conv_id, role="assistant", content=f"主动消息{i}", status="succeeded", proactive_round_id="one-proactive-turn"))
    # No configured client is needed for backup compatibility checks.
    runtime.model_api_key = None
    with TestClient(create_app(runtime)) as client:
        backup = client.get("/api/data/export", headers=headers).json()
        if legacy:
            for message in backup["tables"]["chatMessages"]:
                message.pop("proactive_round_id", None)
        imported = client.post("/api/data/import", headers=headers, files={"file": ("backup.json", json.dumps(backup).encode(), "application/json")})
        assert imported.status_code == 200
        with runtime.require_database().session() as session:
            assert count_trailing_proactive_rounds(session, conv_id) == (3 if legacy else 1)
