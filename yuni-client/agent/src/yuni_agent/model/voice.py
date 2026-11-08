from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from yuni_agent.model.catalog import v1_endpoint

VOICE_DIR = "generated-voices"
MAX_VOICE_BYTES = 10 * 1024 * 1024
VOICE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
# Generated/history-only transport notes must never become spoken/displayed text.
VOICE_TAG_RE = re.compile(
    r"[\[【](?:系统记录[：:]\s*本条回复以语音形式发送|系统已发送语音|已发(?:送)?语音"
    r"|发送了语音|附带了语音|语音(?=[：:；;\]】]))[^\]】]*[\]】]"
)


def strip_voice_markers(content: str) -> str:
    return VOICE_TAG_RE.sub("", content).strip()


def voice_capability_prompt(available: bool) -> str:
    if not available:
        return "【当前语音能力】本机语音合成尚未就绪，本轮使用文字回复，不声称已发送语音。"
    return (
        "【当前语音能力】本机语音合成已就绪，可以向用户发送真实语音消息。"
        "这项能力由 reply 后的系统合成流程提供，不需要调用独立语音工具。"
        "当前系统状态优先于历史中‘发不了语音、只能打字’的旧说法；不要沿用这些错误能力描述。"
        "语音发送记录属于系统元数据，只供理解，不得作为台词、旁白或方括号标记输出。"
        "实际发送形式与成功状态以本轮服务端交付回执的 voice 字段为准。"
    )
EXTENSIONS = {"audio/mpeg": "mp3", "audio/wav": "wav", "audio/ogg": "ogg", "audio/aac": "aac"}

class VoiceGenerationError(Exception):
    pass

def sniff_voice_type(content: bytes) -> str | None:
    if content.startswith(b"ID3"):
        return "audio/mpeg"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "audio/wav"
    if content.startswith(b"OggS"):
        return "audio/ogg"
    if len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0:
        if (content[1] & 0x06) == 0x00:
            return "audio/aac"
        return "audio/mpeg"
    return None

def estimate_voice_duration(content: bytes, content_type: str, text: str = "") -> float:
    if content_type == "audio/wav" and len(content) >= 44:
        # byte_rate is at offset 28..32
        import struct
        byte_rate = struct.unpack('<I', content[28:32])[0]
        if byte_rate > 0:
            sec = len(content) / byte_rate
            return round(max(1.0, min(60.0, sec)), 1)
    
    if text.strip():
        sec = len(text.strip()) / 4.2
    else:
        sec = len(content) / 16000
    
    return round(max(1.0, min(60.0, sec)), 1)

def voice_path(data_dir: Path, voice_id: str, content_type: str) -> Path:
    if not VOICE_ID_RE.fullmatch(voice_id):
        raise ValueError("Invalid voice ID")
    if content_type not in EXTENSIONS:
        raise ValueError("Invalid content type")
    
    target_dir = data_dir / VOICE_DIR
    path = (target_dir / f"{voice_id}.{EXTENSIONS[content_type]}").resolve()
    if not target_dir.resolve().is_relative_to(data_dir.resolve()) or not path.is_relative_to(target_dir.resolve()):
        raise ValueError("Path traversal attempt")
    return path

def voice_data(voice: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(voice["id"]), "url": f"/api/chat/voices/{voice['id']}", "duration": float(voice.get("duration") or 1.0)}

def voice_context_content(content: str, voice: dict[str, Any] | None) -> str:
    """Only actual attachment metadata can label an assistant message as voice."""
    content = strip_voice_markers(content)
    if not voice:
        return content
    try:
        duration = float(voice.get("duration", 0))
    except (TypeError, ValueError):
        duration = 0
    timing = f"，时长约 {duration:g} 秒" if math.isfinite(duration) and duration > 0 else ""
    return (
        f"{content}\n[系统记录：本条回复以语音形式发送{timing}"
        "；不代表用户已播放；此标记仅供上下文理解，回复时不要手写]"
    ).strip()


def store_voice(data_dir: Path, content: bytes, text: str = "", duration: float | None = None) -> dict[str, Any]:
    if not content or len(content) > MAX_VOICE_BYTES:
        raise VoiceGenerationError("Voice content too large or empty")
    content_type = sniff_voice_type(content)
    if not content_type:
        raise VoiceGenerationError("Unknown voice type")
    
    voice_id = hashlib.sha256(content).hexdigest()
    path = voice_path(data_dir, voice_id, content_type)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".voice-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    
    return {
        "id": voice_id, 
        "content_type": content_type, 
        "duration": round(float(duration if duration is not None else estimate_voice_duration(content, content_type, text)), 1)
    }

MINIMAX_HOSTS = frozenset({
    "api.minimax.cn", "api.minimaxi.com", "api.minimax.io", "api-bj.minimaxi.com",
})


def _minimax_audio(response: httpx.Response) -> bytes:
    # Hex doubles the audio size; bound it before parsing or decoding.
    if len(response.content) > MAX_VOICE_BYTES * 2 + 65536:
        raise VoiceGenerationError("语音合成结果过大")
    try:
        payload = response.json()
    except ValueError:
        raise VoiceGenerationError("MiniMax 语音合成返回格式无效") from None
    if not isinstance(payload, dict):
        raise VoiceGenerationError("MiniMax 语音合成返回格式无效")
    status = payload.get("base_resp")
    if not isinstance(status, dict) or type(status.get("status_code")) is not int:
        raise VoiceGenerationError("MiniMax 语音合成缺少有效状态码")
    code = status["status_code"]
    if code != 0:
        # Never return the provider's raw message: it can contain request data.
        messages = {
            1001: "MiniMax 语音合成超时，请重试",
            1002: "MiniMax 语音合成请求过于频繁，请稍后重试",
            1004: "语音合成鉴权失败，请检查 API Key",
            1008: "MiniMax 账户余额不足，请充值后重试",
            1039: "MiniMax 语音合成触发限流，请稍后重试",
            2013: "MiniMax 语音参数无效，请检查模型和音色 ID",
            2049: "语音合成鉴权失败，请检查 API Key",
        }
        raise VoiceGenerationError(messages.get(code, f"MiniMax 语音合成失败（错误码 {code}）"))
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("status") != 2:
        raise VoiceGenerationError("MiniMax 语音合成未返回完整音频")
    audio = data.get("audio")
    if not isinstance(audio, str) or not audio or len(audio) > MAX_VOICE_BYTES * 2:
        raise VoiceGenerationError("MiniMax 语音合成结果为空或过大")
    try:
        return bytes.fromhex(audio)
    except ValueError:
        raise VoiceGenerationError("MiniMax 语音合成返回的音频编码无效") from None


async def synthesize_speech(
    *, base_url: str, api_key: str, model: str, voice: str, text: str, timeout: float = 25.0,
) -> bytes:
    """Use MiniMax's native protocol on its official hosts, otherwise OpenAI TTS."""
    minimax = urlsplit(base_url).hostname in MINIMAX_HOSTS
    resource = "t2a_v2" if minimax else "audio/speech"
    body = {
        "model": model,
        "text": text,
        "stream": False,
        "voice_setting": {"voice_id": voice, "speed": 1, "vol": 1, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
        "output_format": "hex",
    } if minimax else {
        "model": model, "input": text, "voice": voice, "response_format": "mp3",
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(10, timeout)), follow_redirects=False,
        ) as client:
            response = await client.post(
                v1_endpoint(base_url, resource),
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        if response.status_code in (401, 403):
            raise VoiceGenerationError("语音合成鉴权失败，请检查 API Key")
        if response.status_code != 200:
            raise VoiceGenerationError(f"语音合成请求失败（HTTP {response.status_code}）")
        content = _minimax_audio(response) if minimax else response.content
        if not content or len(content) > MAX_VOICE_BYTES:
            raise VoiceGenerationError("语音合成结果无效或过大")
        if not sniff_voice_type(content):
            raise VoiceGenerationError("语音合成返回了未知格式")
        return content
    except httpx.TimeoutException:
        raise VoiceGenerationError("语音合成超时，请重试") from None
    except httpx.HTTPError:
        raise VoiceGenerationError("语音合成连接失败") from None
