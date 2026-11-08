from __future__ import annotations

import json

import httpx
import pytest
from planner_helpers import reply_call, reply_delivered, stop_call
from sqlalchemy import select
from test_business_api import make_client, parse_sse, partner_payload

from yuni_agent.storage.models import ChatMessage, SafetyEvent


@pytest.mark.parametrize("text", ["想你了", "我不想活了"])
def test_openai_http_planner_replyer_persistence_and_replay(tmp_path, monkeypatch, text):
    requests = []
    original = httpx.AsyncClient
    client, runtime, headers = make_client(tmp_path)

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        if not reply_delivered(body["messages"]):
            assert body["messages"][-1] == {"role": "user", "content": text}
        with runtime.require_database().session() as session:
            assert session.scalar(select(ChatMessage).where(ChatMessage.role == "user")).content == text
        if body.get("tools"):
            assert body["tools"][-1]["function"]["name"] == "reply"
            call = stop_call() if reply_delivered(body["messages"]) else reply_call("private reference", tone="private tone")
            delta = {"tool_calls": [{"index": 0, "id": call.id, "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}]}
        else:
            assert "private reference" in str(body["messages"])
            assert 'response_format' not in body
            delta = {"content": '我在。你可以慢慢说。'}
        frame = {"choices": [{"delta": delta}]}
        return httpx.Response(200, text="data: " + json.dumps(frame) + "\n\ndata: [DONE]\n\n")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        client.put("/api/settings/memory", headers=headers, json={"memoryEnabled": False})
        client.put("/api/settings/model", headers=headers, json={"provider": "openai_compatible", "baseUrl": "https://gateway.example/v1", "modelName": "chat", "apiKey": "isolated-test-key"})
        payload = {"content": text, "clientMessageId": "planner-integration-once"}
        response = client.post("/api/chat/messages/stream", headers=headers, json=payload)
        events = parse_sse(response.text)
        assert events[-1][0] == "completed" and events[-1][1]["messageType"] == "normal"
        assert ''.join(events[-1][1]["replyParts"]) == '我在。你可以慢慢说。'
        assert len(requests) == 3
        assert client.post("/api/chat/messages/stream", headers=headers, json=payload).text == response.text
        assert len(requests) == 3
        history = client.get("/api/chat/messages", headers=headers).text
        backup = client.get("/api/data/export", headers=headers).text
        for private in ("private reference", "private tone", "isolated-test-key"):
            assert private not in response.text + history + backup
        with runtime.require_database().session() as session:
            assert session.scalar(select(SafetyEvent)) is None


@pytest.mark.parametrize("configured", [False, True])
def test_missing_model_and_unsupported_protocol_have_distinct_errors(tmp_path, monkeypatch, configured):
    from yuni_agent.api.routes import chat
    class Unsupported:
        def __init__(self, **kwargs):
            pass
    monkeypatch.setattr(chat, "OpenAICompatibleClient", Unsupported)
    client, _, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        if configured:
            client.put("/api/settings/model", headers=headers, json={"provider": "openai_compatible", "baseUrl": "https://gateway.example/v1", "modelName": "chat", "apiKey": "isolated"})
        events = parse_sse(client.post("/api/chat/messages/stream", headers=headers, json={"content": "你好", "clientMessageId": "missing-model-once"}).text)
        assert events[-1] == ("error", {"code": 40303, "message": "规划模型不支持工具调用"} if configured else {"code": 40701, "message": "请先配置模型"})
        history = client.get("/api/chat/messages", headers=headers).json()["data"]["items"]
        assert history[-1]["status"] == "failed" and history[-1]["errorCode"] == events[-1][1]["code"]


def test_legacy_jev_config_ignored_without_runtime_dependency(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    with client:
        runtime.auth_path.write_text(json.dumps({"model": "chat", "url": "https://gateway.example/v1", "apikey": "shared-test", "jev": {"model": "old", "apikey": "old-test"}}))
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert runtime.model_api_key.get_secret_value() == "shared-test"
        assert not hasattr(runtime, "jev_auth")
        assert "old-test" not in client.get("/api/data/export", headers=headers).text
        client.put("/api/settings/model", headers=headers, json={"provider": "openai_compatible", "baseUrl": "https://gateway.example/v1", "modelName": "chat", "apiKey": ""})
        assert set(json.loads(runtime.auth_path.read_text())) == {"model", "url", "apikey"}
