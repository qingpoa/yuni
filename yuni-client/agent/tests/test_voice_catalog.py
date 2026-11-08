from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from yuni_agent.api.app import create_app
from yuni_agent.config import StartupConfig
from yuni_agent.model import voice_catalog
from yuni_agent.model.auth import VoiceAuth
from yuni_agent.model.voice_catalog import VoiceCatalogError, list_voices
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import Settings

CATALOG_URL = "/api/settings/voice/voices"
BASE_URL = "https://api.minimax.cn/v1"
KEY = "private-catalog-key"
PAYLOAD = {
    "system_voice": [
        {
            "voice_id": "Chinese (Mandarin)_Reliable_Executive",
            "voice_name": "沉稳高管",
            "description": ["A provider description that must not be returned"],
        },
        {"voice_id": "female-shaonv"},
    ],
    "voice_cloning": [{"voice_id": "my-cloned-voice", "name": "我的复刻声音"}],
    "voice_generation": [{"voice_id": "ttv-voice-2025082011321125-2uEN0X1S"}],
    "base_resp": {"status_code": 0, "status_msg": "success"},
}
ENTRIES = [
    {"id": "Chinese (Mandarin)_Reliable_Executive", "name": "沉稳高管", "kind": "system"},
    {"id": "female-shaonv", "name": "female-shaonv", "kind": "system"},
    {"id": "my-cloned-voice", "name": "我的复刻声音", "kind": "cloned"},
    {
        "id": "ttv-voice-2025082011321125-2uEN0X1S",
        "name": "ttv-voice-2025082011321125-2uEN0X1S",
        "kind": "generated",
    },
]


def mock_catalog_http(monkeypatch, handler):
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        voice_catalog.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest_asyncio.fixture
async def catalog_env(tmp_path):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    headers = {"X-Yuni-Session": runtime.config.session_token.get_secret_value()}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test",
        ) as client:
            yield runtime, client, headers
    finally:
        runtime.close_storage()


def configure_keys(runtime, *, voice_url=BASE_URL, main_url=BASE_URL):
    runtime.voice_auth = VoiceAuth(url=voice_url, apikey=KEY)
    runtime.model_api_key = SecretStr("private-main-key")
    with runtime.require_database().session() as session:
        settings = session.get(Settings, 1)
        settings.model_base_url = main_url


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", [
    "https://api.minimax.cn", "https://api.minimax.cn/v1/",
    "https://api.minimaxi.com/v1", "https://api.minimax.io/v1",
    "https://api-bj.minimaxi.com/v1", "https://api.minimax.cn:443/v1",
])
async def test_official_catalog_request_and_documented_voice_groups(monkeypatch, base_url):
    def handle(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/get_voice"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {"voice_type": "all"}
        assert request.extensions["timeout"]["connect"] == 5.0
        assert request.extensions["timeout"]["read"] == 15.0
        return httpx.Response(200, json=PAYLOAD)

    mock_catalog_http(monkeypatch, handle)
    assert await list_voices(base_url, KEY) == ENTRIES


@pytest.mark.asyncio
async def test_catalog_sanitizes_entries_deduplicates_and_bounds_names(monkeypatch):
    payload = {
        "base_resp": {"status_code": 0},
        "system_voice": [
            {"voice_id": " named ", "voice_name": " \n温柔\u202e女声\t "},
            {"voice_id": "fallback", "voice_name": {}, "name": "备用名称"},
            {"voice_id": "long-name", "voice_name": "声" * 201},
            {"voice_id": "private-name", "voice_name": KEY, "description": KEY},
            {"voice_id": "other-name", "voice_name": "", "name": "名称"},
            {"voice_id": "named", "voice_name": "duplicate"},
            None, "invalid", {"voice_id": None}, {"voice_id": 123},
            {"voice_id": " "}, {"voice_id": "x" * 121},
            {"voice_id": "bad\nvoice"}, {"voice_id": "bad\u202evoice"},
            {"voice_id": KEY},
        ],
        "voice_cloning": [{"voice_id": "clone", "voice_name": "复刻音色"}],
        "voice_generation": [{"voice_id": "generated", "name": "设计音色"}],
        "unrelated": {"api_key": KEY},
    }
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json=payload))
    entries = await list_voices(BASE_URL, KEY)
    assert entries == [
        {"id": "named", "name": "温柔女声", "kind": "system"},
        {"id": "fallback", "name": "备用名称", "kind": "system"},
        {"id": "long-name", "name": "声" * 200, "kind": "system"},
        {"id": "private-name", "name": "private-name", "kind": "system"},
        {"id": "other-name", "name": "名称", "kind": "system"},
        {"id": "clone", "name": "复刻音色", "kind": "cloned"},
        {"id": "generated", "name": "设计音色", "kind": "generated"},
    ]
    assert KEY not in json.dumps(entries)


@pytest.mark.asyncio
async def test_catalog_does_not_return_truncated_credential_echoes(monkeypatch):
    long_key = "private-" + "x" * 300
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json={
        "base_resp": {"status_code": 0},
        "system_voice": [{"voice_id": "safe-id", "voice_name": long_key}],
    }))
    assert await list_voices(BASE_URL, long_key) == [
        {"id": "safe-id", "name": "safe-id", "kind": "system"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", ["", " ", "x" * 1001, "secret\nkey", "secret\u4e2d"])
async def test_catalog_rejects_invalid_keys_without_network(monkeypatch, api_key):
    def unexpected(request):
        pytest.fail("Invalid key must not make a provider request")

    mock_catalog_http(monkeypatch, unexpected)
    with pytest.raises(VoiceCatalogError, match="API Key 格式无效"):
        await list_voices(BASE_URL, api_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    [],
    {"system_voice": []},
    {"base_resp": {"status_code": True}, "system_voice": []},
    {"base_resp": {"status_code": "0"}, "system_voice": []},
    {"base_resp": {"status_code": 0}},
    {"base_resp": {"status_code": 0}, "system_voice": {}},
    {"base_resp": {"status_code": 0}, "system_voice": [], "voice_cloning": None},
    {"base_resp": {"status_code": 0}, "system_voice": [], "voice_generation": "invalid"},
])
async def test_catalog_rejects_undocumented_or_invalid_shapes(monkeypatch, payload):
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(VoiceCatalogError, match="格式无效|状态码"):
        await list_voices(BASE_URL, KEY)


@pytest.mark.asyncio
async def test_catalog_accepts_empty_documented_categories(monkeypatch):
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json={
        "base_resp": {"status_code": 0}, "system_voice": [],
        "voice_cloning": [], "voice_generation": [],
    }))
    assert await list_voices(BASE_URL, KEY) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "message"), [
    (1001, "超时"), (1002, "频繁"), (1004, "鉴权失败"), (1008, "余额不足"),
    (1039, "限流"), (2013, "参数无效"), (2049, "鉴权失败"), (9999, "获取 MiniMax 音色列表失败"),
])
async def test_provider_errors_are_local_messages(monkeypatch, code, message):
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json={
        "base_resp": {"status_code": code, "status_msg": f"provider-secret {KEY}"},
    }))
    with pytest.raises(VoiceCatalogError, match=message) as error:
        await list_voices(BASE_URL, KEY)
    assert KEY not in str(error.value)
    assert "provider-secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
async def test_http_failures_are_sanitized_and_redirects_are_not_followed(monkeypatch, status):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(
            status, text=f"provider-secret {KEY}",
            headers={"Location": "https://untrusted.example/steal"},
        )

    mock_catalog_http(monkeypatch, handle)
    with pytest.raises(VoiceCatalogError) as error:
        await list_voices(BASE_URL, KEY)
    assert len(calls) == 1
    assert KEY not in str(error.value)
    assert "provider-secret" not in str(error.value)
    if status in (401, 403):
        assert "鉴权失败" in str(error.value)
    else:
        assert f"HTTP {status}" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("error_type", "message"), [
    (httpx.ReadTimeout, "超时"), (httpx.ConnectError, "连接失败"),
])
async def test_transport_failures_are_sanitized(monkeypatch, error_type, message):
    def handle(request):
        raise error_type(f"provider-secret {KEY}", request=request)

    mock_catalog_http(monkeypatch, handle)
    with pytest.raises(VoiceCatalogError, match=message) as error:
        await list_voices(BASE_URL, KEY)
    assert KEY not in str(error.value)
    assert "provider-secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"not-json", b"\xff", b"[" * 1100])
async def test_catalog_rejects_invalid_json_without_raw_text(monkeypatch, content):
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    with pytest.raises(VoiceCatalogError, match="返回格式无效"):
        await list_voices(BASE_URL, KEY)


@pytest.mark.asyncio
async def test_catalog_bounds_response_bytes(monkeypatch):
    monkeypatch.setattr(voice_catalog, "MAX_CATALOG_BYTES", 20)
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 21))
    with pytest.raises(VoiceCatalogError, match="数据过大"):
        await list_voices(BASE_URL, KEY)


@pytest.mark.asyncio
async def test_catalog_stops_streaming_at_response_limit(monkeypatch):
    read_chunks = []

    class CatalogStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in (b"x" * 10, b"x" * 11, b"must-not-be-read"):
                read_chunks.append(chunk)
                yield chunk

    monkeypatch.setattr(voice_catalog, "MAX_CATALOG_BYTES", 20)
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, stream=CatalogStream()))
    with pytest.raises(VoiceCatalogError, match="数据过大"):
        await list_voices(BASE_URL, KEY)
    assert len(read_chunks) == 2


@pytest.mark.asyncio
async def test_catalog_bounds_total_rows_across_voice_kinds(monkeypatch):
    monkeypatch.setattr(voice_catalog, "MAX_CATALOG_VOICES", 3)
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json=PAYLOAD))
    with pytest.raises(VoiceCatalogError, match="数据过多"):
        await list_voices(BASE_URL, KEY)


INVALID_URLS = [
    "", "not-a-url", "http://api.minimax.cn/v1", "https://tts.example.com/v1",
    "https://api.minimax.cn.example.com/v1", "https://api.minimax.cn./v1",
    "https://api.minimax.cn:444/v1", "https://api.minimax.cn:invalid/v1",
    "https://user:password@api.minimax.cn/v1", "https://api.minimax.cn/v1?key=secret",
    "https://api.minimax.cn/v1#fragment", "https://api.minimax.cn/proxy/v1",
    "https://api.minimax.cn/v1/get_voice", "https://api.minimax.cn/\nv1",
    "https://[broken/v1",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", INVALID_URLS)
async def test_catalog_helper_validates_urls_before_network(monkeypatch, base_url):
    def unexpected(request):
        pytest.fail("Invalid URL must not make a provider request")

    mock_catalog_http(monkeypatch, unexpected)
    with pytest.raises(VoiceCatalogError, match="MiniMax 官方 HTTPS 地址"):
        await list_voices(base_url, KEY)


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", INVALID_URLS)
async def test_catalog_route_rejects_invalid_urls(catalog_env, monkeypatch, base_url):
    _, client, headers = catalog_env

    def unexpected(request):
        pytest.fail("Invalid URL must not make a provider request")

    mock_catalog_http(monkeypatch, unexpected)
    response = await client.post(CATALOG_URL, headers=headers, json={
        "baseUrl": base_url, "apiKey": KEY,
    })
    assert response.status_code == 422
    assert response.json()["code"] == 40000
    assert KEY not in response.text
    assert "password" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("supplied_key", [None, "", "   "])
async def test_route_reuses_saved_voice_key_and_returns_exact_contract(
    catalog_env, monkeypatch, supplied_key,
):
    runtime, client, headers = catalog_env
    configure_keys(runtime)

    def handle(request):
        assert request.url.path == "/v1/get_voice"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, json=PAYLOAD)

    mock_catalog_http(monkeypatch, handle)
    body = {"baseUrl": f" {BASE_URL}/ "}
    if supplied_key is not None:
        body["apiKey"] = supplied_key
    response = await client.post(CATALOG_URL, headers=headers, json=body)
    assert response.status_code == 200
    assert response.json() == {"code": 200, "message": "success", "data": ENTRIES}
    assert response.headers["cache-control"] == "private, no-store"
    assert KEY not in response.text
    assert "private-main-key" not in response.text
    assert runtime.voice_auth.apikey.get_secret_value() == KEY


@pytest.mark.asyncio
async def test_route_reuses_main_key_only_for_matching_normalized_url(catalog_env, monkeypatch):
    runtime, client, headers = catalog_env
    configure_keys(runtime, voice_url="https://api.minimax.io/v1", main_url=f"{BASE_URL}/")

    def handle(request):
        assert request.headers["Authorization"] == "Bearer private-main-key"
        return httpx.Response(200, json=PAYLOAD)

    mock_catalog_http(monkeypatch, handle)
    response = await client.post(CATALOG_URL, headers=headers, json={"baseUrl": BASE_URL})
    assert response.status_code == 200
    assert response.json()["data"] == ENTRIES
    assert "private-main-key" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_url", ["https://api.minimax.io/v1", "https://api.minimax.cn"])
async def test_route_does_not_reuse_saved_keys_for_changed_host_or_path(
    catalog_env, monkeypatch, changed_url,
):
    runtime, client, headers = catalog_env
    configure_keys(runtime)

    def unexpected(request):
        pytest.fail("Changed URL without an explicit key must not make a provider request")

    mock_catalog_http(monkeypatch, unexpected)
    response = await client.post(CATALOG_URL, headers=headers, json={"baseUrl": changed_url})
    assert response.status_code == 400
    assert response.json()["code"] == 40000
    assert "API Key" in response.json()["message"]
    assert KEY not in response.text
    assert "private-main-key" not in response.text


@pytest.mark.asyncio
async def test_route_allows_explicit_key_for_changed_official_host(catalog_env, monkeypatch):
    runtime, client, headers = catalog_env
    configure_keys(runtime)

    def handle(request):
        assert request.url.host == "api.minimax.io"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        return httpx.Response(200, json=PAYLOAD)

    mock_catalog_http(monkeypatch, handle)
    response = await client.post(CATALOG_URL, headers=headers, json={
        "baseUrl": "https://api.minimax.io/v1", "apiKey": " explicit-key ",
    })
    assert response.status_code == 200
    assert runtime.voice_auth.apikey.get_secret_value() == KEY
    assert "explicit-key" not in response.text


@pytest.mark.asyncio
async def test_route_requires_key_when_none_is_configured(catalog_env, monkeypatch):
    _, client, headers = catalog_env

    def unexpected(request):
        pytest.fail("Missing key must not make a provider request")

    mock_catalog_http(monkeypatch, unexpected)
    response = await client.post(CATALOG_URL, headers=headers, json={"baseUrl": BASE_URL})
    assert response.status_code == 400
    assert response.json()["code"] == 40000


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"X-Yuni-Session": "incorrect-token"}])
async def test_route_requires_local_session_before_provider_call(catalog_env, monkeypatch, headers):
    _, client, _ = catalog_env

    def unexpected(request):
        pytest.fail("Unauthenticated request must not make a provider request")

    mock_catalog_http(monkeypatch, unexpected)
    response = await client.post(CATALOG_URL, headers=headers, json={
        "baseUrl": BASE_URL, "apiKey": KEY,
    })
    assert response.status_code == 401
    assert response.json()["code"] == 40100
    assert KEY not in response.text


@pytest.mark.asyncio
async def test_route_provider_error_is_sanitized_api_response(catalog_env, monkeypatch):
    _, client, headers = catalog_env
    mock_catalog_http(monkeypatch, lambda request: httpx.Response(200, json={
        "base_resp": {"status_code": 1004, "status_msg": f"provider-secret {KEY}"},
    }))
    response = await client.post(CATALOG_URL, headers=headers, json={
        "baseUrl": BASE_URL, "apiKey": KEY,
    })
    assert response.status_code == 502
    assert response.json() == {
        "code": 40303, "message": "MiniMax 音色列表鉴权失败，请检查 API Key", "data": None,
    }
    assert KEY not in response.text
    assert "provider-secret" not in response.text
