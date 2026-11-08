from __future__ import annotations

import json
import unicodedata
from typing import Literal, TypedDict
from urllib.parse import urlsplit

import httpx

from yuni_agent.model.auth import normalize_model_url
from yuni_agent.model.catalog import v1_endpoint
from yuni_agent.model.voice import MINIMAX_HOSTS

MAX_CATALOG_BYTES = 1024 * 1024
MAX_CATALOG_VOICES = 2000
MAX_VOICE_ID_LENGTH = 120
MAX_VOICE_NAME_LENGTH = 200
VoiceKind = Literal["system", "cloned", "generated"]
VOICE_GROUPS: tuple[tuple[str, VoiceKind], ...] = (
    ("system_voice", "system"),
    ("voice_cloning", "cloned"),
    ("voice_generation", "generated"),
)


class VoiceCatalogEntry(TypedDict):
    id: str
    name: str
    kind: VoiceKind


class VoiceCatalogError(Exception):
    pass


def normalize_voice_catalog_url(value: str) -> str:
    """Accept only the documented native API on official MiniMax HTTPS hosts."""
    try:
        if any(unicodedata.category(char).startswith("C") for char in value):
            raise ValueError
        normalized = normalize_model_url(value)
        parts = urlsplit(normalized)
        if (
            parts.scheme != "https"
            or parts.hostname not in MINIMAX_HOSTS
            or parts.port not in (None, 443)
            or parts.query
            or parts.fragment
            or parts.path not in ("", "/v1")
        ):
            raise ValueError
    except ValueError:
        raise ValueError("音色列表仅支持 MiniMax 官方 HTTPS 地址（根地址或 /v1）") from None
    return normalized


def _catalog_entries(payload: object, api_key: str) -> list[VoiceCatalogEntry]:
    if not isinstance(payload, dict):
        raise VoiceCatalogError("MiniMax 音色列表返回格式无效")
    status = payload.get("base_resp")
    if not isinstance(status, dict) or type(status.get("status_code")) is not int:
        raise VoiceCatalogError("MiniMax 音色列表缺少有效状态码")
    code = status["status_code"]
    if code != 0:
        # Provider messages may echo request data or credentials. Return only local text.
        messages = {
            1001: "获取 MiniMax 音色列表超时，请重试",
            1002: "MiniMax 音色列表请求过于频繁，请稍后重试",
            1004: "MiniMax 音色列表鉴权失败，请检查 API Key",
            1008: "MiniMax 账户余额不足，请充值后重试",
            1039: "MiniMax 音色列表触发限流，请稍后重试",
            2013: "MiniMax 音色列表请求参数无效",
            2049: "MiniMax 音色列表鉴权失败，请检查 API Key",
        }
        raise VoiceCatalogError(messages.get(code, "获取 MiniMax 音色列表失败"))
    if not any(field in payload for field, _ in VOICE_GROUPS):
        raise VoiceCatalogError("MiniMax 音色列表返回格式无效")

    result: list[VoiceCatalogEntry] = []
    seen: set[str] = set()
    row_count = 0
    for field, kind in VOICE_GROUPS:
        rows = payload.get(field, [])
        if not isinstance(rows, list):
            raise VoiceCatalogError("MiniMax 音色列表返回格式无效")
        row_count += len(rows)
        if row_count > MAX_CATALOG_VOICES:
            raise VoiceCatalogError("MiniMax 音色列表数据过多")
        for row in rows:
            if not isinstance(row, dict):
                continue
            voice_id = row.get("voice_id")
            if not isinstance(voice_id, str):
                continue
            voice_id = voice_id.strip()
            if (
                not 1 <= len(voice_id) <= MAX_VOICE_ID_LENGTH
                or any(unicodedata.category(char).startswith("C") for char in voice_id)
                or api_key in voice_id
                or voice_id in seen
            ):
                continue
            name = ""
            for candidate in (row.get("voice_name"), row.get("name")):
                if isinstance(candidate, str):
                    name = "".join(
                        char for char in candidate
                        if not unicodedata.category(char).startswith("C")
                    ).strip()
                    if api_key in name:
                        name = ""
                    if name:
                        name = name[:MAX_VOICE_NAME_LENGTH]
                        break
            seen.add(voice_id)
            result.append({"id": voice_id, "name": name or voice_id, "kind": kind})
    return result


async def list_voices(
    base_url: str, api_key: str, *, timeout: float = 15.0,
) -> list[VoiceCatalogEntry]:
    """List MiniMax account voices without synthesizing or activating any voice."""
    try:
        base_url = normalize_voice_catalog_url(base_url)
    except ValueError as exc:
        raise VoiceCatalogError(str(exc)) from None
    api_key = api_key.strip()
    if not api_key or len(api_key) > 1000 or any(not 33 <= ord(char) <= 126 for char in api_key):
        raise VoiceCatalogError("API Key 格式无效")
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)), follow_redirects=False,
        ) as client, client.stream(
            "POST", v1_endpoint(base_url, "get_voice"),
            headers={"Authorization": f"Bearer {api_key}"},
            json={"voice_type": "all"},
        ) as response:
            if response.status_code in (401, 403):
                raise VoiceCatalogError("MiniMax 音色列表鉴权失败，请检查 API Key")
            if response.status_code != 200:
                raise VoiceCatalogError(f"获取 MiniMax 音色列表失败（HTTP {response.status_code}）")
            content = bytearray()
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > MAX_CATALOG_BYTES:
                    raise VoiceCatalogError("MiniMax 音色列表数据过大")
                content.extend(chunk)
        try:
            payload = json.loads(content)
        except (ValueError, RecursionError):
            raise VoiceCatalogError("MiniMax 音色列表返回格式无效") from None
        return _catalog_entries(payload, api_key)
    except httpx.TimeoutException:
        raise VoiceCatalogError("获取 MiniMax 音色列表超时，请重试") from None
    except httpx.HTTPError:
        raise VoiceCatalogError("MiniMax 音色列表连接失败，请重试") from None
