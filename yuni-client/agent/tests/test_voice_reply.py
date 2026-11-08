import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import pytest_asyncio
from planner_helpers import reply_event, stop_event
from pydantic import SecretStr
from sqlalchemy import select

from yuni_agent.agent.planner import ReplyAction
from yuni_agent.agent.proactive import ProactiveChatWorker
from yuni_agent.api.app import create_app
from yuni_agent.api.routes import chat
from yuni_agent.api.routes.events import event_stream
from yuni_agent.chat import voices as voices_module
from yuni_agent.chat.voices import VoiceSelector, clean_text_for_tts
from yuni_agent.config import StartupConfig
from yuni_agent.context.builder import SqliteContextBuilder
from yuni_agent.context.models import ContextRequest
from yuni_agent.model import voice as voice_module
from yuni_agent.model.auth import VoiceAuth, load_voice_auth, save_voice_auth
from yuni_agent.model.voice import (
    VoiceGenerationError,
    estimate_voice_duration,
    sniff_voice_type,
    store_voice,
    synthesize_speech,
)
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import ChatMessage, ClientEvent, Companion, Conversation, Settings


def test_voice_sniffing():
    # Simple check for ID3 tag
    assert sniff_voice_type(b"ID3" + b"X" * 10) == "audio/mpeg"
    assert sniff_voice_type(b"OggS" + b"X" * 10) == "audio/ogg"
    assert sniff_voice_type(b"RIFF" + b"X" * 4 + b"WAVE") == "audio/wav"

def test_voice_duration():
    assert estimate_voice_duration(b"X" * 16000, "audio/mpeg", "") == 1.0
    assert estimate_voice_duration(b"X" * 100, "audio/mpeg", "你好") == 1.0

def test_voice_auth(tmp_path: Path):
    auth_file = tmp_path / "voice-auth.json"
    auth = VoiceAuth(enabled=True, url="http://localhost:8000", model="test", apikey="secret", voice="alloy")
    save_voice_auth(auth_file, auth)
    loaded = load_voice_auth(auth_file)
    assert loaded is not None
    assert loaded.enabled == True
    assert loaded.apikey.get_secret_value() == "secret"

def test_store_voice(tmp_path: Path):
    data_dir = tmp_path
    (data_dir / "generated-voices").mkdir()
    content = b"ID3" + b"X" * 100
    res = store_voice(data_dir, content, "hello")
    assert "id" in res
    assert res["content_type"] == "audio/mpeg"


AUDIO = b"ID3" + b"test-audio" * 10


def mock_voice_http(monkeypatch, handler):
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        voice_module.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", [
    "https://api.minimax.cn", "https://api.minimax.cn/v1/",
    "https://api.minimaxi.com/v1", "https://api.minimax.io/v1",
    "https://api-bj.minimaxi.com/v1",
])
async def test_minimax_native_request_and_hex_audio(monkeypatch, base_url):
    def handle(request):
        assert request.url.path == "/v1/t2a_v2"
        assert request.headers["Authorization"] == "Bearer test-voice-key"
        body = json.loads(request.content)
        assert body["model"] == "speech-2.8-turbo"
        assert body["text"] == "你好呀。"
        assert body["voice_setting"]["voice_id"] == "female-shaonv"
        assert body["stream"] is False
        assert body["output_format"] == "hex"
        assert body["audio_setting"]["format"] == "mp3"
        assert "input" not in body
        return httpx.Response(200, json={
            "base_resp": {"status_code": 0},
            "data": {"status": 2, "audio": AUDIO.hex()},
        })

    mock_voice_http(monkeypatch, handle)
    audio = await synthesize_speech(
        base_url=base_url, api_key="test-voice-key", model="speech-2.8-turbo",
        voice="female-shaonv", text="你好呀。",
    )
    assert audio == AUDIO


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", [
    "https://tts.example.com/v1", "https://api.minimax.cn.example.com/v1",
])
async def test_other_hosts_keep_openai_protocol(monkeypatch, base_url):
    def handle(request):
        assert request.url.path == "/v1/audio/speech"
        assert json.loads(request.content) == {
            "model": "tts-test", "input": "你好呀。", "voice": "alloy",
            "response_format": "mp3",
        }
        return httpx.Response(200, content=AUDIO)

    mock_voice_http(monkeypatch, handle)
    assert await synthesize_speech(
        base_url=base_url, api_key="test-voice-key", model="tts-test",
        voice="alloy", text="你好呀。",
    ) == AUDIO


@pytest.mark.asyncio
@pytest.mark.parametrize(("payload", "message"), [
    ({"base_resp": {"status_code": 1008, "status_msg": "private-secret"}}, "余额不足"),
    ({"base_resp": {"status_code": 1004, "status_msg": "private-secret"}}, "鉴权失败"),
    ({"base_resp": {"status_code": 2013, "status_msg": "private-secret"}}, "参数无效"),
    ({"base_resp": {"status_code": 9999, "status_msg": "private-secret"}}, "错误码 9999"),
    ({"data": {"status": 2, "audio": AUDIO.hex()}}, "状态码"),
    ({"base_resp": {"status_code": 0}, "data": None}, "完整音频"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 1, "audio": AUDIO.hex()}}, "完整音频"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": ""}}, "为空"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "xyz"}}, "编码无效"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": b"bad".hex()}}, "未知格式"),
    ([], "返回格式无效"),
])
async def test_minimax_rejects_invalid_results_without_exposing_provider_text(
    monkeypatch, payload, message,
):
    mock_voice_http(monkeypatch, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(VoiceGenerationError, match=message) as error:
        await synthesize_speech(
            base_url="https://api.minimax.cn/v1", api_key="test-voice-key",
            model="speech-2.8-turbo", voice="female-shaonv", text="你好呀。",
        )
    assert "private-secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 403, 500])
async def test_voice_http_failures_are_safe(monkeypatch, status):
    mock_voice_http(monkeypatch, lambda request: httpx.Response(status, text="private-secret"))
    with pytest.raises(VoiceGenerationError) as error:
        await synthesize_speech(
            base_url="https://api.minimax.cn/v1", api_key="test-voice-key",
            model="speech-2.8-turbo", voice="female-shaonv", text="你好呀。",
        )
    assert "private-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_voice_timeout(monkeypatch):
    def timeout(request):
        raise httpx.ReadTimeout("private-secret", request=request)

    mock_voice_http(monkeypatch, timeout)
    with pytest.raises(VoiceGenerationError, match="超时"):
        await synthesize_speech(
            base_url="https://api.minimax.cn/v1", api_key="test-voice-key",
            model="speech-2.8-turbo", voice="female-shaonv", text="你好呀。",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"not-json", b"X" * 65569], ids=["invalid-json", "oversized"])
async def test_minimax_rejects_invalid_json_and_oversized_response(monkeypatch, content):
    monkeypatch.setattr(voice_module, "MAX_VOICE_BYTES", 16)
    mock_voice_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    with pytest.raises(VoiceGenerationError, match="返回格式无效|过大"):
        await synthesize_speech(
            base_url="https://api.minimax.cn/v1", api_key="test-voice-key",
            model="speech-2.8-turbo", voice="female-shaonv", text="你好呀。",
        )


@pytest.mark.asyncio
async def test_voice_preview_uses_saved_key_and_returns_playable_audio(tmp_path, monkeypatch):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://test", headers=headers,
        ) as client:
            saved = await client.put("/api/settings/voice", json={
                "enabled": True, "baseUrl": "https://api.minimax.cn/v1",
                "modelName": "speech-2.8-turbo", "voice": "female-shaonv",
                "triggerMode": "explicit_only", "apiKey": "test-voice-key",
            })
            assert saved.status_code == 200
            assert saved.json()["data"]["voice"]["configured"] is True
            assert "test-voice-key" not in saved.text

            def handle(request):
                assert request.headers["Authorization"] == "Bearer test-voice-key"
                return httpx.Response(200, json={
                    "base_resp": {"status_code": 0},
                    "data": {"status": 2, "audio": AUDIO.hex()},
                })

            mock_voice_http(monkeypatch, handle)
            preview = await client.post("/api/settings/voice/preview", json={
                "baseUrl": "https://api.minimax.cn/v1", "modelName": "speech-2.8-turbo",
                "voice": "female-shaonv", "text": "你好呀。",
            })
            assert preview.status_code == 200
            assert preview.content == AUDIO
            assert preview.headers["content-type"] == "audio/mpeg"
            assert preview.headers["cache-control"] == "private, no-store"
            assert "test-voice-key" not in preview.text
    finally:
        runtime.close_storage()


@pytest_asyncio.fixture
async def voice_env(tmp_path, monkeypatch):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    runtime.model_api_key = SecretStr("fake-chat-key")
    runtime.memory_generation_enabled = runtime.memory_retrieval_enabled = False
    runtime.save_voice_auth(VoiceAuth(
        enabled=True, url="https://api.minimax.cn/v1", model="speech-2.8-turbo",
        voice="female-shaonv", trigger_mode="explicit_only",
    ), "fake-voice-key")
    with runtime.require_database().session() as session:
        settings = session.get(Settings, 1)
        settings.chat_model = "fake-model"
        settings.memory_generation_enabled = settings.memory_retrieval_enabled = False
        settings.proactive_chat_enabled = False
        companion = Companion(name="测试伴侣", user_nickname="用户")
        session.add(companion)
        session.flush()
        conversation = Conversation(companion_id=companion.id)
        session.add(conversation)
        session.flush()
        conversation_id = conversation.id
    calls, seen = [], []

    async def synthesize(**kwargs):
        calls.append(kwargs)
        return AUDIO

    class Model:
        def for_internal_json(self):
            return self

        async def stream_chat_with_tools(self, messages, tools):
            delivered = any(
                item.get("role") == "tool" and '"toolName": "reply"' in item.get("content", "")
                for item in messages
            )
            yield stop_event() if delivered else reply_event("根据当前用户要求自然回应。")

        async def stream_chat(self, messages):
            seen.append(messages)
            yield "今天辛苦啦。（轻笑）\n早点休息，我陪着你。😊"

        async def complete_json(self, messages):
            return {"sticker_id": None}

    monkeypatch.setattr(voices_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(chat, "OpenAICompatibleClient", lambda **kwargs: Model())
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test", headers=headers,
        ) as client:
            yield runtime, client, conversation_id, calls, seen, Model
    finally:
        await runtime.close_chat_queues()
        runtime.close_storage()


async def send_voice_message(env, text="发条语音给我吧", client_id="voice-message-001"):
    runtime, client, *_ = env
    response = await client.post("/api/chat/messages", json={"content": text, "clientMessageId": client_id})
    assert response.status_code == 202
    queue = next(iter(runtime.chat_queues.values()))
    await asyncio.wait_for(asyncio.shield(queue.task), timeout=5)
    messages = (await client.get("/api/chat/messages")).json()["data"]["items"]
    return response.json()["data"]["message"], [item for item in messages if item["role"] == "assistant"]


@pytest.mark.asyncio
async def test_voice_chat_delivery_history_events_and_restart(voice_env):
    runtime, client, conv_id, calls, seen, _ = voice_env
    user, replies = await send_voice_message(voice_env)
    assert len(replies) == len(calls) == 1
    reply = replies[0]
    assert reply["voice"] and reply["voice"]["url"].startswith("/api/chat/voices/")
    assert reply["replyParts"] == [reply["content"]]
    assert "轻笑" not in reply["content"] and "😊" not in reply["content"]
    assert "。" in calls[0]["text"] and "，" in calls[0]["text"]
    assert reply["image"] is None and reply["sticker"] is None
    assert any("语音消息约束" in item["content"] for item in seen[0])
    audio = await client.get(reply["voice"]["url"])
    assert audio.status_code == 200 and audio.content == AUDIO
    assert (await client.get(reply["voice"]["url"], headers={"X-Yuni-Session": "wrong"})).status_code == 401
    assert (await client.get("/api/chat/voices/invalid")).status_code == 404
    with runtime.require_database().session() as session:
        stored = session.get(ChatMessage, reply["id"])
        assert stored.voice and "以语音形式发送" in stored.context_content
        events = list(session.scalars(select(ClientEvent).where(ClientEvent.event_type.in_(["reply_completed", "completed"]))))
        assert len(events) == 2
        assert all(event.payload["voice"] == reply["voice"] for event in events)
        original_user = session.get(ChatMessage, user["id"])
        session.expunge(original_user)
    replay = "".join([frame async for frame in chat._replay(runtime, original_user)])
    assert reply["voice"]["id"] in replay
    context = await SqliteContextBuilder(runtime).build(ContextRequest(
        conversation_id=conv_id, current_user_message_id=reply["id"] + 1,
        current_input="你刚才的声音很好听。",
    ))
    for messages in (context.messages, context.tool_messages):
        assert any("以语音形式发送" in item["content"] for item in messages)
    runtime.close_storage()
    runtime.initialize_storage()
    restored = (await client.get("/api/chat/messages")).json()["data"]["items"]
    assert next(item for item in restored if item["id"] == reply["id"])["voice"] == reply["voice"]
    assert (await client.get(reply["voice"]["url"])).content == AUDIO
    stream = event_stream(runtime, None)
    try:
        ready = await anext(stream)
        assert reply["voice"]["id"] in ready
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_voice_failure_falls_back_to_text_and_no_voice_context(voice_env, monkeypatch):
    runtime, _, conv_id, *_ = voice_env

    async def fail(**kwargs):
        raise VoiceGenerationError("fake failure")

    monkeypatch.setattr(voices_module, "synthesize_speech", fail)
    _, replies = await send_voice_message(voice_env)
    assert len(replies) == 1 and replies[0]["voice"] is None
    assert replies[0]["status"] == "succeeded" and replies[0]["content"]
    with runtime.require_database().session() as session:
        assert "以语音形式发送" not in session.get(ChatMessage, replies[0]["id"]).context_content
    assert not (runtime.config.data_dir / "generated-voices").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["今天过得怎么样", "不要发语音，用文字就好", "不用语音，打字就好"])
async def test_text_and_negative_requests_do_not_synthesize(voice_env, text):
    _, _, _, calls, *_ = voice_env
    _, replies = await send_voice_message(voice_env, text)
    assert replies[0]["voice"] is None and calls == []


@pytest.mark.asyncio
async def test_voice_choice_happens_once_per_reply(voice_env, monkeypatch):
    choose = Mock(return_value=True)
    monkeypatch.setattr(VoiceSelector, "should_reply_with_voice", choose)
    _, replies = await send_voice_message(voice_env)
    assert replies[0]["voice"] is not None
    assert choose.call_count == 1


@pytest.mark.asyncio
async def test_voice_selector_modes_cooldown_and_image_exclusion(voice_env):
    runtime, _, conv_id, *_ = voice_env
    plan = ReplyAction(reply_reference="正常聊天", expression_intent={"focus": "对方", "scene": "晚安"})
    rng = Mock()
    rng.random.return_value = 0.0
    selector = VoiceSelector(runtime, rng=rng)
    state = {"conversation_id": conv_id, "current_input": "晚安"}
    assert selector.should_reply_with_voice(state, plan) is False
    runtime.voice_auth = runtime.voice_auth.model_copy(update={"trigger_mode": "occasional"})
    assert selector.should_reply_with_voice(state, plan) is True
    with runtime.require_database().session() as session:
        session.add(ChatMessage(conversation_id=conv_id, role="assistant", content="晚安", status="succeeded",
                                voice={"id": "a" * 64, "content_type": "audio/mpeg", "duration": 2}))
    assert selector.should_reply_with_voice(state, plan) is False
    assert selector.should_reply_with_voice({**state, "current_input": "用语音说晚安"}, plan) is True
    assert selector.should_reply_with_voice({**state, "current_input": "发条语音", "pending_image": {"id": "img"}}, plan) is False
    assert selector.should_reply_with_voice({**state, "current_input": "发条语音", "reply_count": 1}, plan) is False
    runtime.voice_auth = runtime.voice_auth.model_copy(update={"trigger_mode": "always"})
    assert selector.should_reply_with_voice(state, plan) is True
    assert selector.should_reply_with_voice({**state, "current_input": "别发语音"}, plan) is False


@pytest.mark.asyncio
async def test_voice_stale_result_is_discarded(voice_env, monkeypatch):
    runtime, _, conv_id, *_ = voice_env

    async def stale(**kwargs):
        runtime.data_generation += 1
        return AUDIO

    monkeypatch.setattr(voices_module, "synthesize_speech", stale)
    assert await VoiceSelector(runtime).synthesize_for_reply({"conversation_id": conv_id}, "你好。") is None
    assert not (runtime.config.data_dir / "generated-voices").exists()


@pytest.mark.asyncio
async def test_proactive_voice_uses_same_persisted_attachment(voice_env):
    runtime, client, _, calls, _, model = voice_env
    runtime.voice_auth = runtime.voice_auth.model_copy(update={"trigger_mode": "always"})
    message = await ProactiveChatWorker(runtime).run_once(model_client=model(), force=True)
    assert message is not None and message.voice and message.proactive_round_id
    assert len(calls) == 1
    assert message.reply_parts == [message.content]
    assert message.sticker is None and "以语音形式发送" in message.context_content
    assert (await client.get(f"/api/chat/voices/{message.voice['id']}")).content == AUDIO


def test_voice_transcript_cleaning_keeps_chinese_punctuation():
    text = "[已发语音：测试]（轻笑）今天开心吗？*挥手* 我很好，早点休息！😊 QAQ"
    assert clean_text_for_tts(text) == "今天开心吗？ 我很好，早点休息！"
    assert clean_text_for_tts("[已发语音]你好呀。") == "你好呀。"


@pytest.mark.asyncio
async def test_voice_backup_roundtrip_and_credential_cleanup(voice_env):
    runtime, client, *_ = voice_env
    _, replies = await send_voice_message(voice_env)
    reply = replies[0]
    backup = (await client.get("/api/data/export")).json()
    assert backup["voicePayloads"][reply["voice"]["id"]] == base64.b64encode(AUDIO).decode()
    assert "fake-voice-key" not in json.dumps(backup)
    restored = await client.post("/api/data/import", files={"file": ("backup.json", json.dumps(backup), "application/json")})
    assert restored.status_code == 200
    assert runtime.voice_auth is None and not runtime.voice_auth_path.exists()
    assert (await client.get(reply["voice"]["url"])).content == AUDIO
    runtime.save_voice_auth(VoiceAuth(enabled=True, url="https://api.minimax.cn/v1", model="speech-2.8-turbo"), "cleanup-key")
    assert (await client.delete("/api/data")).status_code == 200
    assert runtime.voice_auth is None and not runtime.voice_auth_path.exists()
    assert (await client.get(reply["voice"]["url"])).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "发句语音说你想我了",
    "你发个语音说你想我了呗",
    "给我来段语音",
    "回条语音呗",
    "用语音啊",
    "用语音",
    "那就用语音吧。",
    "请用语音回复我",
    "你用语音和我说晚安",
    "亲口跟我说声晚安",
])
async def test_screenshot_and_short_requests_produce_real_voice_attachment(voice_env, text):
    _, _, _, calls, *_ = voice_env
    _, replies = await send_voice_message(voice_env, text)
    assert len(replies) == 1 and replies[0]["voice"] is not None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_planner_voice_intent_assists_uncommon_user_voice_phrasing(voice_env):
    runtime, _, conv_id, *_ = voice_env
    selector = VoiceSelector(runtime)
    plan = ReplyAction(
        reply_reference="按用户要求用语音回应想念",
        expression_intent={"focus": "对方", "reply_act": "发送语音消息回应"},
    )
    assert selector.should_reply_with_voice(
        {"conversation_id": conv_id, "current_input": "能不能把这句转成语音呀"},
        plan,
    ) is True
    # Without user mentioning voice/sound, explicit_only must not trigger from plan text alone.
    assert selector.should_reply_with_voice(
        {"conversation_id": conv_id, "current_input": "今天好累呀"},
        plan,
    ) is False
    # When Planner explicitly sets use_voice=True from context (e.g. "再说一遍嘛" / "哄哄我"), it triggers.
    contextual_plan = ReplyAction(
        reply_reference="结合上下文用语音再说一遍想念",
        expression_intent={"focus": "对方", "reply_act": "撒娇回应"},
        use_voice=True,
    )
    assert selector.should_reply_with_voice(
        {"conversation_id": conv_id, "current_input": "没听清，再说一遍嘛"},
        contextual_plan,
    ) is True
    assert selector.should_reply_with_voice(
        {"conversation_id": conv_id, "current_input": "别发语音，打字说"},
        contextual_plan,
    ) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "语音功能怎么用", "你刚才用语音了吗", "不要用语音啊", "不用语音回复", "别发句语音", "刚才发条语音了吗",
])
async def test_voice_questions_and_negation_do_not_request_audio(voice_env, text):
    _, _, _, calls, *_ = voice_env
    _, replies = await send_voice_message(voice_env, text)
    assert replies[0]["voice"] is None and not calls


@pytest.mark.asyncio
async def test_planner_and_replyer_receive_current_voice_capability(voice_env):
    from yuni_agent.agent.planner import build_reply_tool

    runtime, _, conv_id, *_ = voice_env
    for ready in (True, False):
        runtime.voice_auth = runtime.voice_auth.model_copy(update={"enabled": ready})
        context = await SqliteContextBuilder(runtime).build(ContextRequest(
            conversation_id=conv_id, current_user_message_id=1, current_input="你能发语音吗？",
        ))
        for messages in (context.messages, context.tool_messages):
            system = "\n".join(item["content"] for item in messages if item["role"] == "system")
            if ready:
                assert "本机语音合成已就绪" in system
            else:
                assert "本机语音合成已就绪" not in system
                assert "【当前语音能力】" not in system
        runtime_data = next(item["content"] for item in context.tool_messages if item["content"].startswith("本轮可用环境数据："))
        assert json.loads(runtime_data.split("：", 1)[1])["voiceReplyAvailable"] is ready
        reply_schema = build_reply_tool(voice_enabled=ready)["function"]["parameters"]["properties"]
        assert ("use_voice" in reply_schema) is ready


@pytest.mark.asyncio
async def test_selected_voice_directive_follows_conflicting_planner_reference(voice_env, monkeypatch):
    _, _, _, _, seen, model = voice_env

    async def plan(self, messages, tools):
        delivered = any(item.get("role") == "tool" and '"toolName": "reply"' in item.get("content", "") for item in messages)
        yield stop_event() if delivered else reply_event("错误的旧说法：只能打字，语音发不出来。")

    monkeypatch.setattr(model, "stream_chat_with_tools", plan)
    _, replies = await send_voice_message(voice_env, "用语音啊")
    assert replies[0]["voice"] is not None
    messages = seen[0]
    reference = next(i for i, item in enumerate(messages) if "【回复信息参考】" in item.get("content", ""))
    directive = next(i for i, item in enumerate(messages) if "【语音消息约束】" in item.get("content", ""))
    assert directive > reference
    assert "本轮发送形式以此状态为准" in messages[directive]["content"]


@pytest.mark.parametrize("marker", [
    "[系统记录：本条回复以语音形式发送，时长约 4.2 秒；不代表用户已播放；此标记仅供上下文理解，回复时不要手写]",
    "【系统记录：本条回复以语音形式发送，\n时长约 4.2 秒】",
    "[已发语音：我想你了]",
])
def test_voice_transport_notes_are_removed_from_visible_text(marker):
    assert clean_text_for_tts("我想你了。\n" + marker) == "我想你了。"


@pytest.mark.asyncio
async def test_model_copied_transport_note_is_not_published_or_spoken(voice_env, monkeypatch):
    _, _, _, calls, _, model = voice_env

    async def speak(self, messages):
        yield "我想你了。\n[系统记录：本条回复以语音形式发送，时长约 4.2 秒；不代表用户已播放；此标记仅供上下文理解，回复时不要手写]"

    monkeypatch.setattr(model, "stream_chat", speak)
    _, replies = await send_voice_message(voice_env, "用语音啊")
    assert replies[0]["voice"] is not None
    assert replies[0]["content"] == calls[0]["text"] == "我想你了。"


@pytest.mark.asyncio
async def test_transport_note_only_output_is_repaired_once(voice_env, monkeypatch):
    _, _, _, calls, _, model = voice_env
    attempts = []

    async def speak(self, messages):
        attempts.append(messages)
        yield "[系统记录：本条回复以语音形式发送，时长约4秒]" if len(attempts) == 1 else "我想你了。"

    monkeypatch.setattr(model, "stream_chat", speak)
    _, replies = await send_voice_message(voice_env, "用语音啊")
    assert len(attempts) == 2 and len(calls) == 1
    assert replies[0]["content"] == "我想你了。" and replies[0]["voice"] is not None


@pytest.mark.asyncio
async def test_historical_leaked_note_is_excluded_from_plain_assistant_context(voice_env):
    runtime, _, conv_id, *_ = voice_env
    with runtime.require_database().session() as session:
        message = ChatMessage(conversation_id=conv_id, role="assistant", status="succeeded", content="我想你了。\n[系统记录：本条回复以语音形式发送，时长约4秒]")
        session.add(message)
        session.flush()
        next_id = message.id + 1
    context = await SqliteContextBuilder(runtime).build(ContextRequest(
        conversation_id=conv_id, current_user_message_id=next_id, current_input="用语音啊",
    ))
    for messages in (context.messages, context.tool_messages):
        histories = [item["content"] for item in messages if item["role"] == "assistant"]
        assert histories == ["我想你了。"]
