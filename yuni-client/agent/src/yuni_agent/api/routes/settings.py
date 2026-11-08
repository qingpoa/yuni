from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import Field, field_validator

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.common.http import fail
from yuni_agent.model.auth import AuthFileError, ImageAuth, normalize_model_url
from yuni_agent.model.catalog import list_models
from yuni_agent.model.client import ModelClientError
from yuni_agent.model.voice_catalog import (
    MAX_VOICE_ID_LENGTH,
    MAX_VOICE_NAME_LENGTH,
    VoiceCatalogError,
    VoiceKind,
    list_voices,
    normalize_voice_catalog_url,
)
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import Settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ModelSummary(CamelModel):
    provider: str
    base_url: str
    model_name: str
    configured: bool


class ImageModelSummary(CamelModel):
    enabled: bool
    base_url: str
    model_name: str
    default_size: str
    style_prefix: str
    configured: bool


class VoiceModelSummary(CamelModel):
    enabled: bool
    base_url: str
    model_name: str
    voice: str
    trigger_mode: Literal["explicit_only", "occasional", "always"]
    configured: bool


class SettingsData(CamelModel):
    memory_enabled: bool
    memory_generation_enabled: bool
    memory_retrieval_enabled: bool
    proactive_chat_enabled: bool
    proactive_interval_minutes: int
    quiet_hours_start: str
    quiet_hours_end: str
    privacy_version: str
    model: ModelSummary
    embedding: ModelSummary
    image: ImageModelSummary
    voice: VoiceModelSummary


class ImageSettingInput(CamelModel):
    enabled: bool
    base_url: str = Field(default="", max_length=500)
    model_name: str = Field(default="", max_length=120)
    api_key: str | None = Field(default=None, max_length=1000)
    default_size: Literal["1024x1024", "768x1024", "1024x768"] = "1024x1024"
    style_prefix: str = Field(default="", max_length=300)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return normalize_model_url(value) if value.strip() else ""


class VoiceSettingInput(CamelModel):
    enabled: bool
    base_url: str = Field(default="", max_length=500)
    model_name: str = Field(default="", max_length=120)
    api_key: str | None = Field(default=None, max_length=1000)
    voice: str = Field(default="alloy", max_length=120)
    trigger_mode: Literal["explicit_only", "occasional", "always"] = "occasional"


class VoiceListInput(CamelModel):
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str | None = Field(default=None, max_length=1000)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return normalize_voice_catalog_url(value)


class VoiceCatalogSummary(CamelModel):
    id: str = Field(min_length=1, max_length=MAX_VOICE_ID_LENGTH)
    name: str = Field(min_length=1, max_length=MAX_VOICE_NAME_LENGTH)
    kind: VoiceKind


class VoicePreviewInput(CamelModel):
    base_url: str = Field(max_length=500)
    model_name: str = Field(max_length=120)
    api_key: str | None = Field(default=None, max_length=1000)
    voice: str = Field(max_length=120)
    text: str = Field(default="你好呀，现在能听见我的声音吗？", min_length=1, max_length=200)


class MemorySettingInput(CamelModel):
    memory_enabled: bool


class ProactiveSettingInput(CamelModel):
    proactive_chat_enabled: bool
    proactive_interval_minutes: int = Field(default=120, ge=15, le=720)
    quiet_hours_start: str = Field(default="23:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    quiet_hours_end: str = Field(default="08:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class ModelListInput(CamelModel):
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    api_key: str | None = Field(default=None, max_length=1000)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        return normalize_model_url(value) if value is not None else None


class ModelSettingInput(CamelModel):
    provider: Literal["openai_compatible", "openai"]
    base_url: str = Field(min_length=1, max_length=500)
    model_name: str = Field(min_length=1, max_length=120)
    api_key: str | None = Field(default=None, max_length=1000)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return normalize_model_url(value)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("模型名不能为空")
        return value


def _to_data(runtime: AgentRuntime, item: Settings) -> SettingsData:
    return SettingsData(
        memory_enabled=item.memory_generation_enabled and item.memory_retrieval_enabled,
        memory_generation_enabled=item.memory_generation_enabled,
        memory_retrieval_enabled=item.memory_retrieval_enabled,
        proactive_chat_enabled=item.proactive_chat_enabled,
        proactive_interval_minutes=item.proactive_interval_minutes,
        quiet_hours_start=item.quiet_hours_start,
        quiet_hours_end=item.quiet_hours_end,
        privacy_version=item.privacy_version,
        model=ModelSummary(
            provider=item.model_provider,
            base_url=item.model_base_url,
            model_name=item.chat_model,
            configured=runtime.model_api_key is not None,
        ),
        embedding=ModelSummary(provider="openai_compatible",
            base_url=runtime.embedding_auth.url if runtime.embedding_auth else "",
            model_name=runtime.embedding_auth.model if runtime.embedding_auth else "",
            configured=bool(runtime.embedding_auth and runtime.embedding_auth.apikey.get_secret_value())),
        image=_image_summary(runtime),
        voice=_voice_summary(runtime),
    )


def _image_summary(runtime: AgentRuntime) -> ImageModelSummary:
    auth = runtime.image_auth or ImageAuth()
    return ImageModelSummary(
        enabled=auth.enabled, base_url=auth.url, model_name=auth.model, default_size=auth.default_size,
        style_prefix=auth.style_prefix,
        configured=bool(auth.url and auth.model and auth.apikey.get_secret_value()),
    )


def _voice_summary(runtime: AgentRuntime) -> VoiceModelSummary:
    from yuni_agent.model.auth import VoiceAuth
    auth = runtime.voice_auth or VoiceAuth()
    return VoiceModelSummary(
        enabled=auth.enabled, base_url=auth.url, model_name=auth.model, voice=auth.voice,
        trigger_mode=auth.trigger_mode,
        configured=bool(auth.url and auth.model and auth.voice and auth.apikey.get_secret_value()),
    )


def _load(runtime: AgentRuntime) -> Settings:
    with runtime.require_database().session() as session:
        item = session.get(Settings, 1)
        if item is None:
            item = Settings(id=1)
            session.add(item)
            session.flush()
        session.expunge(item)
        return item


@router.get("", response_model=ApiResponse[SettingsData])
def get_settings(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[SettingsData]:
    return ApiResponse(data=_to_data(runtime, _load(runtime)))


@router.put("/memory", response_model=ApiResponse[SettingsData])
def set_memory(
    body: MemorySettingInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[SettingsData]:
    runtime.proactive_generation += 1
    with runtime.require_database().session() as session:
        item = session.get(Settings, 1)
        assert item is not None
        item.memory_generation_enabled = body.memory_enabled
        item.memory_retrieval_enabled = body.memory_enabled
        session.flush()
        result = _to_data(runtime, item)
    runtime.memory_generation_enabled = body.memory_enabled
    runtime.memory_retrieval_enabled = body.memory_enabled
    return ApiResponse(data=result)


@router.put("/proactive", response_model=ApiResponse[SettingsData])
def set_proactive(
    body: ProactiveSettingInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[SettingsData]:
    runtime.proactive_generation += 1
    from sqlalchemy import select

    from yuni_agent.agent.proactive import cancel_proactive_timer, schedule_next_proactive
    from yuni_agent.storage.models import Conversation
    with runtime.require_database().session() as session:
        item = session.get(Settings, 1)
        assert item is not None
        item.proactive_chat_enabled = body.proactive_chat_enabled
        item.proactive_interval_minutes = body.proactive_interval_minutes
        item.quiet_hours_start = body.quiet_hours_start
        item.quiet_hours_end = body.quiet_hours_end
        session.flush()
        for conv in session.scalars(select(Conversation)):
            if body.proactive_chat_enabled:
                schedule_next_proactive(session, conv.id)
            else:
                cancel_proactive_timer(session, conv.id)
        result = _to_data(runtime, item)
    return ApiResponse(data=result)


@router.put("/model", response_model=ApiResponse[SettingsData])
def set_model(
    body: ModelSettingInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[SettingsData]:
    with runtime.require_database().session() as session:
        item = session.get(Settings, 1)
        assert item is not None
        item.model_provider = body.provider
        item.model_base_url = body.base_url
        item.chat_model = body.model_name.strip()
        runtime.save_model_auth(item.chat_model, item.model_base_url, body.api_key)
        session.flush()
        result = _to_data(runtime, item)
    return ApiResponse(data=result)


@router.delete("/model/key", response_model=ApiResponse[SettingsData])
def clear_model_key(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[SettingsData]:
    item = _load(runtime)
    if item.chat_model:
        runtime.save_model_auth(item.chat_model, item.model_base_url, "", preserve_empty=False)
    else:
        runtime.remove_model_auth()
    return ApiResponse(data=_to_data(runtime, item))


@router.post("/models", response_model=ApiResponse[list[str]])
async def get_models(
    body: ModelListInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[list[str]]:
    settings = _load(runtime)
    url = body.base_url or settings.model_base_url
    key = (body.api_key or "").strip()
    if not key and runtime.image_auth and runtime.image_auth.url and runtime.image_auth.apikey.get_secret_value():
        if normalize_model_url(url) == normalize_model_url(runtime.image_auth.url):
            key = runtime.image_auth.apikey.get_secret_value()
    if not key and runtime.voice_auth and runtime.voice_auth.url and runtime.voice_auth.apikey.get_secret_value():
        if normalize_model_url(url) == normalize_model_url(runtime.voice_auth.url):
            key = runtime.voice_auth.apikey.get_secret_value()
    if not key and runtime.model_api_key is not None:
        if normalize_model_url(url) != normalize_model_url(settings.model_base_url):
            fail(400, 40000, "地址已更改，请填写该地址的 API Key 后获取列表")
        key = runtime.model_api_key.get_secret_value()
    if not key:
        fail(400, 40000, "请先填写或保存 API Key")
    try:
        return ApiResponse(data=await list_models(url, key))
    except ModelClientError as exc:
        fail(502, 40303, str(exc))


@router.put("/embedding", response_model=ApiResponse[SettingsData])
def set_embedding(body: ModelSettingInput, runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[SettingsData]:
    settings = _load(runtime)
    api_key = body.api_key
    prev = runtime.embedding_auth
    has_prev_key_for_url = bool(prev and prev.url == body.base_url and prev.apikey.get_secret_value())
    if not (api_key or "").strip() and not has_prev_key_for_url and body.base_url and runtime.model_api_key is not None:
        if body.base_url == normalize_model_url(settings.model_base_url):
            api_key = runtime.model_api_key.get_secret_value()
    try:
        runtime.save_embedding_auth(body.model_name, body.base_url, api_key)
    except ValueError:
        fail(400, 40000, "更换 Embedding 地址时请填写新 Key")
    return ApiResponse(data=_to_data(runtime, settings))


@router.delete("/embedding/key", response_model=ApiResponse[SettingsData])
def clear_embedding_key(runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[SettingsData]:
    with runtime.embedding_lock:
        auth = runtime.embedding_auth
        if auth:
            from pydantic import SecretStr

            from yuni_agent.model.auth import save_auth
            cleared = auth.model_copy(update={"apikey": SecretStr("")})
            save_auth(runtime.embedding_auth_path, cleared)
            runtime.embedding_auth = cleared
            runtime.embedding_generation += 1
    return ApiResponse(data=_to_data(runtime, _load(runtime)))


@router.put("/image", response_model=ApiResponse[SettingsData])
def set_image(body: ImageSettingInput, runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[SettingsData]:
    if body.enabled and (not body.base_url or not body.model_name.strip()):
        fail(400, 40000, "启用生图需要填写 Base URL 和模型名")
    settings = _load(runtime)
    auth = ImageAuth(enabled=body.enabled, url=body.base_url, model=body.model_name,
                     default_size=body.default_size, style_prefix=body.style_prefix)
    api_key = body.api_key
    prev = runtime.image_auth
    has_prev_key_for_url = bool(prev and prev.url == auth.url and prev.apikey.get_secret_value())
    if not (api_key or "").strip() and not has_prev_key_for_url and auth.url and runtime.model_api_key is not None:
        if auth.url == normalize_model_url(settings.model_base_url):
            api_key = runtime.model_api_key.get_secret_value()
    try:
        runtime.save_image_auth(auth, api_key)
    except ValueError:
        fail(400, 40000, "更换生图地址时请填写新 Key")
    except AuthFileError:
        fail(500, 40000, "生图配置保存失败")
    return ApiResponse(data=_to_data(runtime, settings))


@router.delete("/image/key", response_model=ApiResponse[SettingsData])
def clear_image_key(runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[SettingsData]:
    runtime.clear_image_key()
    return ApiResponse(data=_to_data(runtime, _load(runtime)))


@router.put("/voice", response_model=ApiResponse[SettingsData])
def set_voice(body: VoiceSettingInput, runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[SettingsData]:
    from yuni_agent.model.auth import VoiceAuth
    if body.enabled and (not body.base_url or not body.model_name.strip() or not body.voice.strip()):
        fail(400, 40000, "启用语音合成需要填写 Base URL、模型名和音色")
    settings = _load(runtime)
    auth = VoiceAuth(enabled=body.enabled, url=body.base_url, model=body.model_name,
                     voice=body.voice, trigger_mode=body.trigger_mode)
    api_key = body.api_key
    prev = runtime.voice_auth
    has_prev_key_for_url = bool(prev and prev.url == auth.url and prev.apikey.get_secret_value())
    if not (api_key or "").strip() and not has_prev_key_for_url and auth.url and runtime.model_api_key is not None:
        if auth.url == normalize_model_url(settings.model_base_url):
            api_key = runtime.model_api_key.get_secret_value()
    try:
        runtime.save_voice_auth(auth, api_key)
    except ValueError:
        fail(400, 40000, "更换语音合成地址时请填写新 Key")
    except AuthFileError:
        fail(500, 40000, "语音合成配置保存失败")
    return ApiResponse(data=_to_data(runtime, settings))


@router.delete("/voice/key", response_model=ApiResponse[SettingsData])
def clear_voice_key(runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[SettingsData]:
    runtime.clear_voice_key()
    return ApiResponse(data=_to_data(runtime, _load(runtime)))


@router.post("/voice/voices", response_model=ApiResponse[list[VoiceCatalogSummary]])
async def get_voices(
    body: VoiceListInput,
    response: Response,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[list[VoiceCatalogSummary]]:
    api_key = (body.api_key or "").strip()
    if (
        not api_key and runtime.voice_auth and runtime.voice_auth.url
        and normalize_model_url(runtime.voice_auth.url) == body.base_url
    ):
        api_key = runtime.voice_auth.apikey.get_secret_value()
    if (
        not api_key and runtime.model_api_key
        and normalize_model_url(_load(runtime).model_base_url) == body.base_url
    ):
        api_key = runtime.model_api_key.get_secret_value()
    if not api_key:
        fail(400, 40000, "请填写该地址的 API Key 后获取音色列表")
    try:
        entries = await list_voices(body.base_url, api_key)
    except VoiceCatalogError as exc:
        fail(502, 40303, str(exc))
    response.headers["Cache-Control"] = "private, no-store"
    return ApiResponse(data=[VoiceCatalogSummary(**entry) for entry in entries])


@router.post("/voice/preview")
async def preview_voice(body: VoicePreviewInput, runtime: Annotated[AgentRuntime, Depends(require_session)]):
    from yuni_agent.model.voice import VoiceGenerationError, sniff_voice_type, synthesize_speech
    api_key = body.api_key
    if not api_key:
        if runtime.voice_auth and runtime.voice_auth.url == normalize_model_url(body.base_url):
            api_key = runtime.voice_auth.apikey.get_secret_value()
    if not api_key and runtime.model_api_key:
        if normalize_model_url(body.base_url) == normalize_model_url(_load(runtime).model_base_url):
            api_key = runtime.model_api_key.get_secret_value()
    if not api_key:
        fail(400, 40000, "请填写 API Key")
    try:
        audio_bytes = await synthesize_speech(
            base_url=body.base_url, api_key=api_key, model=body.model_name, voice=body.voice, text=body.text
        )
        content_type = sniff_voice_type(audio_bytes)
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )
    except VoiceGenerationError as exc:
        fail(502, 40303, str(exc))


@router.post("/embedding/rebuild", response_model=ApiResponse[None])
def rebuild_embedding(runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[None]:
    from sqlalchemy import delete

    from yuni_agent.storage.models import ExpressionVector, MemoryEmbedding
    with runtime.embedding_lock:
        runtime.embedding_generation += 1
        with runtime.require_database().session() as session:
            session.execute(delete(ExpressionVector))
            session.execute(delete(MemoryEmbedding))
        runtime.clear_memory_index_files()
    return ApiResponse(data=None)
