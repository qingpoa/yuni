import asyncio
import base64
import hashlib
import json

import httpx
import pytest
import pytest_asyncio
from planner_helpers import reply_event, stop_event
from pydantic import SecretStr

from yuni_agent.api.app import create_app
from yuni_agent.api.routes import chat
from yuni_agent.config import StartupConfig
from yuni_agent.model import image as image_module
from yuni_agent.model.auth import ImageAuth
from yuni_agent.model.client import ModelStreamEvent, ModelToolCall
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import ChatMessage, Companion, Conversation, Settings, ToolCall
from yuni_agent.tools.service import LocalToolExecutor

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_ID = hashlib.sha256(PNG).hexdigest()
CAT = "一只在窗台晒太阳的橘猫，水彩风格"


def image_call(call_id="img-1", prompt=CAT):
    return ModelStreamEvent(type="tool_call", tool_call=ModelToolCall(
        id=call_id, name="generate_image", arguments={"prompt": prompt, "aspect_ratio": "1:1"}))


def last_tool_name(messages):
    # Planner feedback is appended as a system message after the last tool result.
    rest = [item for item in messages if item.get("role") != "system"]
    if not rest or rest[-1].get("role") != "tool":
        return None
    return json.loads(rest[-1]["content"]).get("toolName")


@pytest_asyncio.fixture
async def image_env(tmp_path, monkeypatch):
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
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        return PNG

    monkeypatch.setattr("yuni_agent.tools.builtin.request_image", fake_generate)
    orig_init = chat.ConversationQueue.__init__
    monkeypatch.setattr(chat.ConversationQueue, "__init__",
                        lambda self, produce, **kw: orig_init(self, produce, quiet_seconds=0.01))
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(runtime)),
                                 base_url="http://test", headers=headers) as client:
        yield runtime, client, calls
    await runtime.close_chat_queues()
    runtime.close_storage()


def enable_images(runtime, **overrides):
    runtime.save_image_auth(ImageAuth(enabled=True, url="https://img.example.com/v1", model="flux",
                                      **overrides), "image-key")


async def send(client, runtime, text, client_id):
    response = await client.post("/api/chat/messages", json={"content": text, "clientMessageId": client_id})
    assert response.status_code == 202
    await asyncio.wait_for(asyncio.shield(next(iter(runtime.chat_queues.values())).task), 5)
    return [item for item in (await client.get("/api/chat/messages")).json()["data"]["items"]
            if item["role"] == "assistant"]


def use_model(monkeypatch, planner, reply_text="画好啦，给你看看～", seen_tools=None):
    class Model:
        def for_internal_json(self):
            return self

        async def stream_chat_with_tools(self, messages, tools):
            if seen_tools is not None:
                seen_tools.append([item["function"]["name"] for item in tools])
            yield planner(messages, tools)

        async def stream_chat(self, messages):
            yield reply_text

        async def complete_json(self, messages):
            return {"sticker_id": None}

    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: Model())


def _replied(messages):
    return any(m.get("role") == "tool" and '"toolName": "reply"' in (m.get("content") or "") for m in messages)


def _image_attempted(messages):
    return any(m.get("role") == "tool" and '"toolName": "generate_image"' in (m.get("content") or "")
               for m in messages)


def image_then_reply(messages, tools):
    # Enforced order: reply first (soothe the wait), then generate once, then deliver, then stop.
    if not _replied(messages):
        return reply_event("等我拍给你看看～")
    if not _image_attempted(messages):
        return image_call()
    name = last_tool_name(messages)
    return stop_event() if name == "reply" else reply_event("图片已生成，介绍一下画面")


def test_tool_hidden_until_configured_and_enabled(tmp_path):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    try:
        def names():
            return [item["function"]["name"] for item in LocalToolExecutor(runtime).definitions]
        assert "generate_image" not in names()
        runtime.save_image_auth(ImageAuth(enabled=False, url="https://img.example.com", model="flux"), "k")
        assert "generate_image" not in names()
        runtime.save_image_auth(ImageAuth(enabled=True, url="https://img.example.com", model="flux"), None)
        assert "generate_image" in names()
        runtime.clear_image_key()
        assert "generate_image" not in names()
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_settings_api_never_returns_key(image_env):
    runtime, client, _ = image_env
    data = (await client.get("/api/settings")).json()["data"]["image"]
    assert data == {"enabled": False, "baseUrl": "", "modelName": "", "defaultSize": "1024x1024",
                    "stylePrefix": "", "configured": False}
    response = await client.put("/api/settings/image", json={
        "enabled": True, "baseUrl": "https://img.example.com/", "modelName": "flux",
        "apiKey": "secret-image-key", "defaultSize": "768x1024", "stylePrefix": "水彩",
    })
    assert response.status_code == 200
    assert "secret-image-key" not in response.text
    image = response.json()["data"]["image"]
    assert image["configured"] and image["baseUrl"] == "https://img.example.com"
    assert runtime.image_generation_ready
    assert json.loads(runtime.image_auth_path.read_text(encoding="utf-8"))["apikey"] == "secret-image-key"
    # Blank key keeps it for the same URL, but a new URL needs a new key.
    same = await client.put("/api/settings/image", json={
        "enabled": True, "baseUrl": "https://img.example.com", "modelName": "flux2"})
    assert same.json()["data"]["image"]["configured"]
    moved = await client.put("/api/settings/image", json={
        "enabled": True, "baseUrl": "https://other.example.com", "modelName": "flux"})
    assert moved.status_code == 400
    missing = await client.put("/api/settings/image", json={"enabled": True, "baseUrl": "", "modelName": ""})
    assert missing.status_code == 400
    cleared = await client.delete("/api/settings/image/key")
    assert not cleared.json()["data"]["image"]["configured"]
    assert not runtime.image_generation_ready
    # Same URL as main model with blank key reuses main model key automatically.
    reused = await client.put("/api/settings/image", json={
        "enabled": True, "baseUrl": "https://api.openai.com/v1", "modelName": "codex/gpt-image-2"})
    assert reused.status_code == 200 and reused.json()["data"]["image"]["configured"]
    assert runtime.image_generation_ready


@pytest.mark.asyncio
async def test_generated_image_attaches_to_next_reply(image_env, monkeypatch):
    runtime, client, calls = image_env
    enable_images(runtime, style_prefix="吉卜力风格")
    use_model(monkeypatch, image_then_reply, reply_text="画好啦[已发出生成的图片：橘猫]")
    assistants = await send(client, runtime, "帮我画一只晒太阳的橘猫", "img-turn-001")

    assert calls[0]["prompt"].startswith("吉卜力风格") and calls[0]["size"] == "1024x1024"
    assert len(assistants) == 2  # reply-first soothing line, then the image-bearing reply
    reply = assistants[-1]
    assert reply["image"] == {"id": PNG_ID, "url": f"/api/chat/images/{PNG_ID}", "description": CAT}
    assert reply["sticker"] is None
    assert assistants[0]["image"] is None  # the pre-generation reply carries no image
    assert "[已发出生成的图片" not in reply["content"]
    with runtime.require_database().session() as session:
        message = session.get(ChatMessage, reply["id"])
        assert "系统已发出生成的图片" in message.context_content
        audit = session.query(ToolCall).one()
        assert audit.tool_name == "generate_image" and audit.risk == "write" and audit.status == "succeeded"
        assert audit.arguments_json["prompt"]["length"] > 0  # prompt text is not stored
    content = await client.get(f"/api/chat/images/{PNG_ID}")
    assert content.status_code == 200 and content.content == PNG
    assert content.headers["content-type"] == "image/png"

    # Follow-up turn: both Planner and Replyer history must include the generated image context.
    from yuni_agent.context.builder import SqliteContextBuilder
    from yuni_agent.context.models import ContextRequest
    with runtime.require_database().session() as session:
        follow_up = ChatMessage(
            conversation_id=message.conversation_id, role="user", content="刚才画的猫是什么颜色？", status="succeeded"
        )
        session.add(follow_up)
        session.flush()
        follow_up_id, conversation_id = follow_up.id, message.conversation_id
    built = await SqliteContextBuilder(runtime).build(ContextRequest(
        conversation_id=conversation_id, current_user_message_id=follow_up_id, current_input="刚才画的猫是什么颜色？",
    ))
    replyer_assistant_msgs = [m["content"] for m in built.messages if m["role"] == "assistant"]
    planner_assistant_msgs = [m["content"] for m in built.tool_messages if m["role"] == "assistant"]
    assert any("系统已发出生成的图片" in m and CAT in m for m in replyer_assistant_msgs)
    assert any("系统已发出生成的图片" in m and CAT in m for m in planner_assistant_msgs)


@pytest.mark.asyncio
@pytest.mark.parametrize("hallucinated_only", [
    "[已发出生成的图片：橘猫]",
    "[图片：一只晒太阳的猫]",
    "[附带了表情包；表达情绪：开心；素材描述：小猫]",
])
async def test_tag_only_reply_does_not_fallback_to_raw_tag(hallucinated_only):
    from types import SimpleNamespace

    from yuni_agent.agent.chat_agent import CompanionChatAgent
    from yuni_agent.model.client import ModelClientError

    class TagOnlyClient:
        async def stream_chat(self, _messages):
            yield hallucinated_only

    state = {
        "messages": [{"role": "user", "content": "画只猫"}],
        "reply_plan": {
            "reply_reference": "回应用户",
            "expression_intent": {"focus": "画猫"},
            "reply_style": "自然",
        },
    }
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            model_client=TagOnlyClient(),
            context_builder=None,
            expression_selector=None,
            sticker_selector=None,
        )
    )
    with pytest.raises(ModelClientError):
        await CompanionChatAgent().generate(state, runtime)


@pytest.mark.asyncio
async def test_reply_before_image_attaches_to_later_reply(image_env, monkeypatch):
    runtime, client, _ = image_env
    enable_images(runtime)
    steps = iter([reply_event(), image_call(), reply_event(), stop_event()])
    use_model(monkeypatch, lambda messages, tools: next(steps))
    assistants = await send(client, runtime, "画一张海边日落壁纸", "img-turn-002")
    assert [item["image"] is not None for item in assistants] == [False, True]


@pytest.mark.asyncio
async def test_stop_with_unsent_image_is_redirected_to_reply(image_env, monkeypatch):
    runtime, client, _ = image_env
    enable_images(runtime)
    feedback = []

    def planner(messages, tools):
        newest = messages[-1]
        if newest["role"] == "system" and "尚未发送" in str(newest.get("content") or ""):
            feedback.append(True)
            return reply_event()
        if not _replied(messages):
            return reply_event("等我拍给你看看～")
        if not _image_attempted(messages):
            return image_call()
        return stop_event()

    use_model(monkeypatch, planner)
    assistants = await send(client, runtime, "画只小狗", "img-turn-003")
    assert feedback and len(assistants) == 2 and assistants[-1]["image"]["id"] == PNG_ID


@pytest.mark.asyncio
async def test_one_image_per_turn(image_env, monkeypatch):
    runtime, client, calls = image_env
    enable_images(runtime)
    seen = []

    def planner(messages, tools):
        if not _replied(messages):
            return reply_event("等我拍给你看看～")
        names = [item["function"]["name"] for item in tools]
        if not _image_attempted(messages):
            if "generate_image" not in names:
                return ModelStreamEvent(type="tool_call", tool_call=ModelToolCall(
                    id="req-img", name="request_tools",
                    arguments={"groups": ["image"], "intent": "画两张图"},
                ))
            return image_call("img-a")
        if "generate_image" in names:  # try a second image; must be withdrawn after one
            return image_call("img-b", prompt="第二张")
        return stop_event()

    use_model(monkeypatch, planner, seen_tools=seen)
    await send(client, runtime, "画两张图", "img-turn-004")
    assert len(calls) == 1  # only one image generated this turn
    assert any("generate_image" in round_tools for round_tools in seen)
    assert "generate_image" not in seen[-1]


@pytest.mark.asyncio
async def test_failed_generation_attaches_nothing(image_env, monkeypatch):
    runtime, client, _ = image_env
    enable_images(runtime)

    async def broken(**kwargs):
        raise image_module.ImageGenerationError("生图请求失败（HTTP 500）")

    monkeypatch.setattr("yuni_agent.tools.builtin.request_image", broken)
    use_model(monkeypatch, image_then_reply)
    assistants = await send(client, runtime, "画只猫", "img-turn-005")
    assert len(assistants) == 2 and all(item["image"] is None for item in assistants)


@pytest.mark.asyncio
async def test_image_endpoint_rejects_bad_ids(image_env):
    _, client, _ = image_env
    for value in ["..%2F..%2Fyuni.db", "abc", "g" * 64, PNG_ID]:
        assert (await client.get(f"/api/chat/images/{value}")).status_code == 404


def test_store_image_validates_content(tmp_path):
    with pytest.raises(image_module.ImageGenerationError):
        image_module.store_image(tmp_path, b"<html>not an image</html>", "x")
    with pytest.raises(image_module.ImageGenerationError):
        image_module.store_image(tmp_path, b"", "x")
    stored = image_module.store_image(tmp_path, PNG, "猫" * 600)
    assert stored["id"] == PNG_ID and stored["content_type"] == "image/png" and len(stored["description"]) == 500
    assert (tmp_path / "generated-images" / f"{PNG_ID}.png").read_bytes() == PNG


@pytest.mark.asyncio
async def test_client_handles_b64_url_and_limits(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/images/generations"):
            prompt = json.loads(request.content)["prompt"]
            if prompt == "b64":
                return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
            if prompt == "url":
                return httpx.Response(200, json={"data": [{"url": "https://cdn.example.com/a.png"}]})
            if prompt == "bad-scheme":
                return httpx.Response(200, json={"data": [{"url": "file:///etc/passwd"}]})
            if prompt == "auth":
                return httpx.Response(401, json={})
            return httpx.Response(200, json={"data": []})
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=PNG)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr(image_module.httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))
    common = {"base_url": "https://img.example.com", "api_key": "k", "model": "flux", "size": "1024x1024"}

    assert await image_module.generate_image(prompt="b64", **common) == PNG
    assert str(requests[0].url) == "https://img.example.com/v1/images/generations"
    assert requests[0].headers["authorization"] == "Bearer k"
    assert await image_module.generate_image(prompt="url", **common) == PNG
    for prompt, message in [("bad-scheme", "地址无效"), ("auth", "鉴权失败"), ("empty", "格式无效")]:
        with pytest.raises(image_module.ImageGenerationError, match=message):
            await image_module.generate_image(prompt=prompt, **common)

    monkeypatch.setattr(image_module, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(image_module.ImageGenerationError, match="15 MB"):
        await image_module.generate_image(prompt="url", **common)


@pytest.mark.asyncio
async def test_backup_roundtrip_and_tamper(image_env, monkeypatch):
    runtime, client, _ = image_env
    enable_images(runtime)
    use_model(monkeypatch, image_then_reply)
    await send(client, runtime, "画只猫", "img-turn-006")
    exported = (await client.get("/api/data/export")).json()
    assert set(exported["imagePayloads"]) == {PNG_ID}
    assert "image-key" not in json.dumps(exported)

    tampered = json.loads(json.dumps(exported))
    tampered["imagePayloads"][PNG_ID] = base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode()
    bad = await client.post("/api/data/import", files={"file": ("b.json", json.dumps(tampered), "application/json")})
    assert bad.status_code == 422

    ok = await client.post("/api/data/import", files={"file": ("b.json", json.dumps(exported), "application/json")})
    assert ok.status_code == 200
    assert (runtime.config.data_dir / "generated-images" / f"{PNG_ID}.png").read_bytes() == PNG
    assert runtime.image_auth is None  # import resets local credentials like the other model configs
    history = (await client.get("/api/chat/messages")).json()["data"]["items"]
    assert any(item["image"] and item["image"]["id"] == PNG_ID for item in history)


@pytest.mark.asyncio
async def test_clear_data_removes_images_and_config(image_env):
    runtime, client, _ = image_env
    enable_images(runtime)
    image_module.store_image(runtime.config.data_dir, PNG, "x")
    assert (await client.delete("/api/data")).status_code == 200
    assert not any((runtime.config.data_dir / "generated-images").iterdir())
    assert not runtime.image_auth_path.exists() and runtime.image_auth is None


def _set_uploaded_avatar(runtime, data=PNG, content_type="image/png"):
    from sqlalchemy import select

    from yuni_agent.storage.models import AvatarFile
    rel = "avatars/companion-avatar.png"
    path = runtime.config.data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    with runtime.require_database().session() as session:
        avatar = AvatarFile(relative_path=rel, content_type=content_type, size_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest())
        session.add(avatar)
        session.flush()
        companion = session.scalar(select(Companion).limit(1))
        companion.avatar_type = "uploaded"
        companion.avatar_file_id = avatar.id


def companion_photo_call(prompt="你在咖啡馆的自拍"):
    return ModelStreamEvent(type="tool_call", tool_call=ModelToolCall(
        id="selfie-1", name="generate_image", arguments={"mode": "companion_photo", "prompt": prompt}))


def selfie_planner(messages, tools):
    # reply-first, then one companion_photo, then a delivering reply, then stop
    if not _replied(messages):
        return reply_event("等我拍给你看看～")
    if not _image_attempted(messages):
        return companion_photo_call()
    name = last_tool_name(messages)
    return stop_event() if name == "reply" else reply_event("拍好啦，给你看")


@pytest.mark.asyncio
async def test_companion_photo_uses_avatar_reference(image_env, monkeypatch):
    runtime, client, _ = image_env
    enable_images(runtime)
    _set_uploaded_avatar(runtime)
    edit_calls, gen_calls = [], []

    async def fake_edit(**kwargs):
        edit_calls.append(kwargs)
        return PNG

    async def fake_gen(**kwargs):
        gen_calls.append(kwargs)
        return PNG

    monkeypatch.setattr("yuni_agent.tools.builtin.edit_image", fake_edit)
    monkeypatch.setattr("yuni_agent.tools.builtin.request_image", fake_gen)
    use_model(monkeypatch, selfie_planner)
    assistants = await send(client, runtime, "给我看看你现在的样子", "selfie-turn-1")
    # Image-to-image with the avatar bytes, not text-to-image.
    assert len(edit_calls) == 1 and not gen_calls
    assert edit_calls[0]["image_bytes"] == PNG and edit_calls[0]["image_type"] == "image/png"
    assert "同一人" in edit_calls[0]["prompt"]
    assert len(assistants) == 2 and assistants[-1]["image"]["id"] == PNG_ID


@pytest.mark.asyncio
async def test_companion_photo_falls_back_without_uploaded_avatar(image_env, monkeypatch):
    runtime, client, calls = image_env  # calls tracks request_image (text-to-image)
    enable_images(runtime)  # companion keeps the default preset avatar
    monkeypatch.setattr("yuni_agent.tools.builtin.edit_image",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("edit must not be called")))
    use_model(monkeypatch, selfie_planner)
    assistants = await send(client, runtime, "拍张照给我", "selfie-turn-2")
    # Preset avatar has no backend pixels -> text-to-image fallback, image still delivered.
    assert len(calls) == 1
    assert len(assistants) == 2 and assistants[-1]["image"]["id"] == PNG_ID


@pytest.mark.asyncio
async def test_creative_mode_never_uses_avatar(image_env, monkeypatch):
    runtime, client, calls = image_env
    enable_images(runtime)
    _set_uploaded_avatar(runtime)
    monkeypatch.setattr("yuni_agent.tools.builtin.edit_image",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("creative must not edit")))
    use_model(monkeypatch, image_then_reply)  # image_call() is mode=creative
    assistants = await send(client, runtime, "画一只橘猫", "creative-turn-1")
    assert len(calls) == 1 and assistants[-1]["image"]["id"] == PNG_ID


@pytest.mark.asyncio
async def test_edit_image_sends_multipart_and_decodes(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["ctype"] = request.headers.get("content-type", "")
        seen["auth"] = request.headers.get("authorization", "")
        seen["body"] = request.content
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr(image_module.httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))
    out = await image_module.edit_image(base_url="https://img.example.com", api_key="k", model="gpt-image",
                                        prompt="same person selfie", image_bytes=PNG, image_type="image/png",
                                        size="1024x1024")
    assert out == PNG
    assert seen["url"] == "https://img.example.com/v1/images/edits"
    assert seen["auth"] == "Bearer k"
    assert "multipart/form-data" in seen["ctype"]
    assert b"same person selfie" in seen["body"] and PNG in seen["body"]


@pytest.mark.asyncio
async def test_edit_image_propagates_errors(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr(image_module.httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))
    with pytest.raises(image_module.ImageGenerationError, match="鉴权失败"):
        await image_module.edit_image(base_url="https://img.example.com", api_key="k", model="m",
                                      prompt="x", image_bytes=PNG, image_type="image/png", size="1024x1024")
