import asyncio
import base64
import hashlib
import json
from contextlib import aclosing

import httpx
import pytest
import pytest_asyncio
from planner_helpers import reply_delivered, reply_event, stop_event
from pydantic import SecretStr
from sqlalchemy import select

from yuni_agent.api.app import create_app
from yuni_agent.api.routes import chat
from yuni_agent.api.routes.events import event_stream
from yuni_agent.config import StartupConfig
from yuni_agent.context.builder import SqliteContextBuilder
from yuni_agent.context.models import ContextRequest
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import (
    ChatMessage,
    Companion,
    Conversation,
    Memory,
    Settings,
    Sticker,
)

GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


@pytest_asyncio.fixture
async def sticker_env(tmp_path, monkeypatch):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    runtime.model_api_key = SecretStr("fake-test-key")
    runtime.memory_generation_enabled = runtime.memory_retrieval_enabled = False
    with runtime.require_database().session() as session:
        settings = session.get(Settings, 1)
        settings.chat_model = "fake-model"
        settings.memory_generation_enabled = settings.memory_retrieval_enabled = False
        settings.proactive_chat_enabled = False
        companion = Companion(name="测试伴侣", user_nickname="用户")
        session.add(companion)
        session.flush()
        session.add(Conversation(companion_id=companion.id))
    original_queue = chat._queued_generate
    monkeypatch.setattr(chat, "_queued_generate", lambda *args, **kwargs: None)
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test", headers=headers) as client:
        yield runtime, client, original_queue
    await runtime.close_chat_queues()
    runtime.close_storage()


def add_sticker(runtime, *, enabled=True, missing=False, outside=False):
    path = runtime.config.data_dir / ("outside.gif" if outside else "stickers/fixture.gif")
    if not missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(GIF)
    with runtime.require_database().session() as session:
        item = Sticker(description="开心庆祝的小猫", emotions=["开心"], usage="对方分享好消息时", relative_path=path.relative_to(runtime.config.data_dir).as_posix(),
                       content_type="image/gif", size_bytes=len(GIF), sha256=hashlib.sha256(GIF).hexdigest(), enabled=enabled)
        session.add(item)
        session.flush()
        return item.id


@pytest.mark.asyncio
async def test_empty_catalog_and_invalid_requests(sticker_env):
    _, client, _ = sticker_env
    assert (await client.get("/api/chat/stickers")).json()["data"] == []
    assert (await client.get("/api/chat/stickers", headers={"X-Yuni-Session": "wrong"})).status_code == 401
    assert (await client.post("/api/chat/stickers")).status_code == 405
    for body in ({"content": ""}, {"stickerId": 123}, {"stickerId": -1}, {"stickerId": "1"}):
        response = await client.post("/api/chat/messages", json={**body, "clientMessageId": "invalid-sticker-request"})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_catalog_filters_and_message_deduplication(sticker_env):
    runtime, client, _ = sticker_env
    valid = add_sticker(runtime)
    disabled = add_sticker(runtime, enabled=False)
    outside = add_sticker(runtime, outside=True)
    missing = add_sticker(runtime, missing=True)
    # Enabled metadata without an existing asset must not appear in the picker.
    with runtime.require_database().session() as session:
        session.get(Sticker, missing).relative_path = "stickers/absent.gif"
    items = (await client.get("/api/chat/stickers")).json()["data"]
    assert [item["id"] for item in items] == [valid]
    image = await client.get(items[0]["imageUrl"])
    assert image.content == GIF and image.headers["content-type"] == "image/gif"
    assert (await client.get(items[0]["imageUrl"], headers={"X-Yuni-Session": "wrong"})).status_code == 401
    for item_id in (disabled, missing, outside):
        assert (await client.post("/api/chat/messages", json={"stickerId": item_id, "clientMessageId": f"unavailable-sticker-{item_id}"})).status_code == 422
    body = {"stickerId": valid, "clientMessageId": "same-sticker-message"}
    first = await client.post("/api/chat/messages", json=body)
    second = await client.post("/api/chat/messages", json=body)
    assert first.status_code == second.status_code == 202
    assert first.json()["data"]["message"]["id"] == second.json()["data"]["message"]["id"]
    assert (await client.post("/api/chat/messages", json={**body, "content": "另一个草稿"})).status_code == 409
    assert (await client.post("/api/chat/messages", json={"content": "庆祝一下", "stickerId": valid, "clientMessageId": "text-with-sticker-message"})).status_code == 202
    history = (await client.get("/api/chat/messages")).json()["data"]["items"]
    assert len(history) == 2 and history[0]["sticker"] == items[0] and history[0]["content"] == ""
    async with aclosing(event_stream(runtime, 0)) as stream:
        ready = await anext(stream)
        assert '"sticker"' in ready and '开心庆祝的小猫' in ready
        event = await anext(stream)
        assert "user_message_created" in event and '"sticker"' in event


@pytest.mark.asyncio
async def test_sticker_only_uses_existing_graph_and_context(sticker_env, monkeypatch):
    runtime, client, original_queue = sticker_env
    item_id = add_sticker(runtime)
    seen = []
    replyer_seen = []

    class Model:
        async def stream_chat_with_tools(self, messages, tools):
            seen.append(json.dumps(messages, ensure_ascii=False))
            yield stop_event() if reply_delivered(messages) else reply_event()

        async def stream_chat(self, messages):
            replyer_seen.append(json.dumps(messages, ensure_ascii=False))
            yield "一起庆祝一下！"

    monkeypatch.setattr(chat, "_queued_generate", original_queue)
    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: Model())
    response = await client.post("/api/chat/messages", json={"stickerId": item_id, "clientMessageId": "sticker-graph-request"})
    assert response.status_code == 202
    await asyncio.wait_for(asyncio.shield(next(iter(runtime.chat_queues.values())).task), 5)
    assert seen and "开心庆祝的小猫" in seen[0] and "素材描述" in seen[0]
    assert "图案借喻" in seen[0] and "表情包为情绪象征" in seen[0]
    assert replyer_seen and "【表情包理解与发送约束】" in replyer_seen[0]
    with runtime.require_database().session() as session:
        assert session.scalar(select(ChatMessage).where(ChatMessage.role == "assistant")).content == "一起庆祝一下！"

    runtime.memory_retrieval_enabled = True
    with runtime.require_database().session() as session:
        companion_id = session.scalar(select(Companion.id))
        conversation_id = session.scalar(select(Conversation.id))
        session.add(Memory(
            companion_id=companion_id, type="user_fact", kind="fact", domain="real_world",
            subject_key="pet.cat", title="养小猫", content="用户家里养了一只小猫",
        ))
        sticker_only_msg = ChatMessage(
            conversation_id=conversation_id, role="user", content="",
            sticker={"id": item_id, "description": "开心庆祝的小猫"}, status="succeeded",
        )
        text_with_sticker_msg = ChatMessage(
            conversation_id=conversation_id, role="user", content="我家小猫今天很乖",
            sticker={"id": item_id, "description": "开心庆祝的小猫"}, status="succeeded",
        )
        session.add_all([sticker_only_msg, text_with_sticker_msg])
        session.flush()
        sticker_only_id, sticker_only_input = sticker_only_msg.id, sticker_only_msg.context_content
        text_msg_id, text_msg_input = text_with_sticker_msg.id, text_with_sticker_msg.context_content

    builder = SqliteContextBuilder(runtime)
    built_sticker_only = await builder.build(ContextRequest(
        conversation_id=conversation_id, current_user_message_id=sticker_only_id, current_input=sticker_only_input,
    ))
    assert built_sticker_only.memory_ids == []

    built_with_text = await builder.build(ContextRequest(
        conversation_id=conversation_id, current_user_message_id=text_msg_id, current_input=text_msg_input,
    ))
    assert len(built_with_text.memory_ids) == 1
    timeline_msg = next(item["content"] for item in built_with_text.messages if item["role"] == "system" and "近期消息时间线" in item["content"])
    assert "[表情包：开心]" in timeline_msg and "[表情包]" in timeline_msg
    assert "素材描述：" not in timeline_msg


@pytest.mark.asyncio
async def test_sticker_backup_roundtrip_and_corruption(sticker_env):
    runtime, client, _ = sticker_env
    item_id = add_sticker(runtime)
    await client.post("/api/chat/messages", json={"stickerId": item_id, "clientMessageId": "backup-sticker-request"})
    backup = (await client.get("/api/data/export")).json()
    broken = {**backup, "stickerPayloads": {str(item_id): "broken"}}
    assert (await client.post("/api/data/import", files={"file": ("bad.json", json.dumps(broken), "application/json")})).status_code == 422
    assert (await client.get(f"/api/chat/stickers/{item_id}/content")).content == GIF
    assert (await client.delete("/api/data")).status_code == 200
    assert (await client.get("/api/chat/stickers")).json()["data"] == []
    restored = await client.post("/api/data/import", files={"file": ("backup.json", json.dumps(backup), "application/json")})
    assert restored.status_code == 200
    assert (await client.get(f"/api/chat/stickers/{item_id}/content")).content == GIF
    assert (await client.get("/api/chat/messages")).json()["data"]["items"][0]["sticker"]["id"] == item_id
    with runtime.require_database().session() as session:
        assert session.get(Sticker, item_id).emotions == ["开心"]

    # Backups written before 0014 have no emotions/usage and must still restore.
    legacy = json.loads(json.dumps(backup))
    for row in legacy["tables"]["stickers"]:
        row.pop("emotions"), row.pop("usage")
    for row in legacy["tables"]["chatMessages"]:
        if row.get("sticker"):
            row["sticker"].pop("emotions", None)
    restored = await client.post("/api/data/import", files={"file": ("legacy.json", json.dumps(legacy), "application/json")})
    assert restored.status_code == 200
    with runtime.require_database().session() as session:
        assert session.get(Sticker, item_id).emotions == [] and session.get(Sticker, item_id).usage == ""


def test_context_includes_emotions_but_not_usage():
    message = ChatMessage(role="user", content="", status="succeeded",
                          sticker={"id": 1, "description": "手写字：知道了，你厉害", "emotions": ["调侃", "敷衍"]})
    assert "表达情绪：调侃、敷衍；素材描述：手写字：知道了，你厉害" in message.context_content
    assistant_msg = ChatMessage(role="assistant", content="哼，知道就好！", status="succeeded",
                                sticker={"id": 1, "description": "得意的小猫", "emotions": ["得意"]})
    assert "[系统已自动配发表情包（仅作上下文记录，回复时严禁手写此标记）；表达情绪：得意；素材描述：得意的小猫]" in assistant_msg.context_content
    legacy = ChatMessage(role="user", content="", status="succeeded", sticker={"id": 1, "description": "小猫"})
    assert "表达情绪" not in legacy.context_content and "小猫" in legacy.context_content


@pytest.mark.asyncio
async def test_assistant_selects_and_delivers_sticker_with_cooldown(sticker_env, monkeypatch):
    runtime, client, original_queue = sticker_env
    s1 = add_sticker(runtime)
    path2 = runtime.config.data_dir / "stickers/fixture2.gif"
    path2.write_bytes(GIF + b"\x00")
    with runtime.require_database().session() as session:
        item2 = Sticker(
            description="得意坏笑的柴犬", emotions=["调侃", "得意"], usage="朋友吹牛时打趣对方",
            relative_path=path2.relative_to(runtime.config.data_dir).as_posix(),
            content_type="image/gif", size_bytes=len(GIF) + 1,
            sha256=hashlib.sha256(GIF + b"\x00").hexdigest(), enabled=True,
        )
        session.add(item2)
        session.flush()
        s2 = item2.id

    json_calls = []

    class Model:
        def for_internal_json(self):
            return self

        async def stream_chat_with_tools(self, messages, tools):
            yield stop_event() if reply_delivered(messages) else reply_event()

        async def stream_chat(self, messages):
            yield "太棒啦，给你鼓掌！"

        async def complete_json(self, messages):
            json_calls.append(messages)
            return {"sticker_id": s1}

    monkeypatch.setattr(chat, "_queued_generate", original_queue)
    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: Model())
    orig_init = chat.ConversationQueue.__init__
    monkeypatch.setattr(chat.ConversationQueue, "__init__", lambda self, produce, **kw: orig_init(self, produce, quiet_seconds=0.01))

    # Turn 1: user sends plain text -> assistant selects sticker s1
    r1 = await client.post("/api/chat/messages", json={"content": "我今天拿了第一名", "clientMessageId": "assist-sticker-turn-1"})
    assert r1.status_code == 202
    await asyncio.wait_for(asyncio.shield(next(iter(runtime.chat_queues.values())).task), 5)
    assert len(json_calls) == 1
    history = (await client.get("/api/chat/messages")).json()["data"]["items"]
    assist_1 = next(item for item in history if item["role"] == "assistant")
    assert assist_1["sticker"] == {"id": s1, "description": "开心庆祝的小猫", "imageUrl": f"/api/chat/stickers/{s1}/content"}

    # Turn 2: user sends plain text immediately after assistant sent a sticker -> cooldown skips sticker selection
    r2 = await client.post("/api/chat/messages", json={"content": "晚上去吃大餐", "clientMessageId": "assist-sticker-turn-2"})
    assert r2.status_code == 202
    await asyncio.wait_for(asyncio.shield(next(iter(runtime.chat_queues.values())).task), 5)
    assert len(json_calls) == 1  # selector skipped due to 2-message cooldown
    history2 = (await client.get("/api/chat/messages")).json()["data"]["items"]
    assist_rows = [item for item in history2 if item["role"] == "assistant"]
    assert len(assist_rows) == 2 and assist_rows[-1]["sticker"] is None

    # Turn 3: user explicitly asks assistant to change/send a sticker (plain text) -> bypasses cooldown,
    # strips any hallucinated [附带了表情包...] tag from text, and attaches a real sticker (excluding s1)
    class ModelExplicitAsk(Model):
        async def stream_chat(self, messages):
            yield "那我可得挑张像样的。这张呢，本小姐自己看着还挺得意的。\n[附带了表情包；表达情绪：得意、俏皮；素材描述：卡通猫翘着尾巴]"

        async def complete_json(self, messages):
            json_calls.append(messages)
            payload = json.loads(messages[-1]["content"])
            candidate_ids = [c["id"] for c in payload["candidates"]]
            assert s1 not in candidate_ids and s2 in candidate_ids
            assert "不得返回 null" in messages[0]["content"]
            return {"sticker_id": None}  # Even if model returns None, explicit intent falls back to top candidate

    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: ModelExplicitAsk())
    r3 = await client.post("/api/chat/messages", json={"content": "那你换一张，我看看你能换什么", "clientMessageId": "assist-sticker-turn-3"})
    assert r3.status_code == 202
    await asyncio.wait_for(asyncio.shield(next(iter(runtime.chat_queues.values())).task), 5)
    assert len(json_calls) == 2
    history3 = (await client.get("/api/chat/messages")).json()["data"]["items"]
    assist_rows3 = [item for item in history3 if item["role"] == "assistant"]
    assert "[附带了表情包" not in assist_rows3[-1]["content"]
    assert all("[附带了表情包" not in part for part in assist_rows3[-1]["replyParts"])
    assert assist_rows3[-1]["sticker"]["id"] == s2

