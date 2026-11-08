from __future__ import annotations

import json

import httpx
import pytest

from yuni_agent.model.catalog import list_models
from yuni_agent.model.client import ModelClientError, OpenAICompatibleClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "model", "enable_thinking"),
    [
        ("https://api.deepseek.com", "deepseek-flash", True),
        ("https://api.deepseek.com/v1/", "deepseek-flash", True),
        ("https://api.example.com/v1", "deepseek/deepseek-flash", True),
        ("http://gateway.example.test:8080/v1", "deepseek/deepseek-flash", True),
        ("https://api.example.com/v1", "other-model", False),
        ("https://api.deepseek.com.example.com", "other-model", False),
        ("http://192.0.2.10:8080/v1", "other-model", False),
    ],
)
async def test_provider_options_and_responses(monkeypatch, base_url, model, enable_thinking):
    requests: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{base_url.rstrip('/')}/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        if body["stream"]:
            frame = {"choices": [{"delta": {"content": "hello"}}]}
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"memories": []}'}}],
        })

    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    client = OpenAICompatibleClient(base_url=base_url, api_key="test-only", model=model)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]
    events = [event async for event in client.stream_chat_with_tools(messages, tools)]
    assert [event.text for event in events] == ["hello"]
    assert await client.complete_json(messages) == {"memories": []}
    assert requests[0]["tools"] == tools
    assert requests[0]["tool_choice"] == "auto"
    assert "response_format" not in requests[0]
    assert requests[1]["response_format"] == {"type": "json_object"}
    assert [chunk async for chunk in client.stream_chat(messages)] == ["hello"]
    assert "response_format" not in requests[2]
    assert "tools" not in requests[2]
    for body in requests:
        if enable_thinking:
            assert body["thinking"] == {"type": "enabled"}
        else:
            assert "thinking" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
async def test_thinking_stream_keeps_reasoning_out_of_reply(monkeypatch, with_tools):
    original = httpx.AsyncClient
    def handle(request):
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "enabled"}
        deltas = [{"reasoning_content": "internal "}, {"reasoning_content": "reasoning"}]
        if with_tools:
            deltas.extend([
                {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "read_text_file", "arguments": '{"path":'}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": '"D:/allowed/a.txt"}'}}]},
            ])
        else:
            deltas.append({"content": '{"replies":["hello"]}'})
        frames = ["data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n" for delta in deltas]
        return httpx.Response(200, text="".join(frames) + "data: [DONE]\n\n")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    client = OpenAICompatibleClient(base_url="https://gateway.example/v1", api_key="test", model="deepseek/deepseek-flash")
    messages = [{"role": "user", "content": "hello"}]
    if with_tools:
        events = [event async for event in client.stream_chat_with_tools(messages, [{"type": "function", "function": {"name": "read_text_file"}}])]
        assert len(events) == 1
        assert events[0].tool_call.arguments == {"path": "D:/allowed/a.txt"}
        assert events[0].reasoning_content == "internal reasoning"
        assert events[0].text == ""
    else:
        assert [text async for text in client.stream_chat(messages)] == ['{"replies":["hello"]}']


@pytest.mark.asyncio
async def test_tool_reasoning_has_size_limit(monkeypatch):
    original = httpx.AsyncClient
    frame = {"choices": [{"delta": {"reasoning_content": "x" * 64001}}]}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="data: " + json.dumps(frame) + "\n\n")), **kwargs))
    client = OpenAICompatibleClient(base_url="https://gateway.example/v1", api_key="test", model="deepseek/deepseek-flash")
    with pytest.raises(ModelClientError, match="工具思考内容过长"):
        _ = [event async for event in client.stream_chat_with_tools([{"role": "user", "content": "hello"}], [{"type": "function"}])]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ['{broken', '[]', 'null'])
async def test_malformed_tool_arguments_remain_visible_to_whole_action_validation(monkeypatch, arguments):
    original = httpx.AsyncClient
    delta = {"tool_calls": [
        {"index": 0, "id": "valid", "function": {"name": "read_text_file", "arguments": '{"path":"test"}'}},
        {"index": 1, "id": "invalid", "function": {"name": "reply", "arguments": arguments}},
    ]}
    frame = "data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n"
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=frame)), **kwargs))
    client = OpenAICompatibleClient(base_url="https://gateway.example/v1", api_key="test", model="chat")
    events = [event async for event in client.stream_chat_with_tools([], [{"type": "function"}])]
    assert [event.type for event in events] == ["tool_call", "invalid_tool_call"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("payload", "expected"), [
    ({"data": [{"id": "z-model"}, {"id": "chat"}, {"id": "chat"}]}, ["chat", "z-model"]),
    ({"models": [{"name": "jev-preview"}, {"name": "jev-latest"}]}, ["jev-latest", "jev-preview"]),
    (["chat", "jev-latest"], ["chat", "jev-latest"]),
    ({"data": []}, []),
])
async def test_model_catalog_formats(monkeypatch, payload, expected):
    original = httpx.AsyncClient
    def handle(request):
        assert str(request.url) == "https://gateway.example/v1/model"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=payload)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    assert await list_models("https://gateway.example/v1/", "test-key") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 405])
async def test_catalog_falls_back_to_plural_only_when_path_not_supported(monkeypatch, status):
    original, paths = httpx.AsyncClient, []
    def handle(request):
        paths.append(request.url.path)
        return httpx.Response(status) if request.url.path.endswith('/model') else httpx.Response(200, json={"models": [{"name": "jev-latest"}]})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    assert await list_models("https://gateway.example", "test-key") == ["jev-latest"]
    assert paths == ["/v1/model", "/v1/models"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_catalog_errors_are_safe_and_do_not_fallback(monkeypatch, status):
    original, paths = httpx.AsyncClient, []
    def handle(request):
        paths.append(request.url.path)
        return httpx.Response(status, text="private-test-key")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(ModelClientError) as exc:
        await list_models("https://gateway.example", "test-key")
    assert "private-test-key" not in str(exc.value)
    assert paths == ["/v1/model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"data": {}}, {"data": [{}]}, {"models": [1]}, {"data": [" "]}])
async def test_catalog_rejects_invalid_data(monkeypatch, payload):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)), **kwargs))
    with pytest.raises(ModelClientError, match="格式无效"):
        await list_models("https://gateway.example", "test-key")


@pytest.mark.asyncio
async def test_catalog_timeout(monkeypatch):
    original = httpx.AsyncClient
    def handle(request):
        raise httpx.ReadTimeout("private", request=request)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(ModelClientError, match="超时"):
        await list_models("https://gateway.example", "test-key")
