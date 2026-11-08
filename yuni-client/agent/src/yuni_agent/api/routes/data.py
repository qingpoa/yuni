from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import Response
from pydantic import ValidationError
from sqlalchemy import DateTime, delete, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.chat.events import emit_event
from yuni_agent.chat.stickers import STICKER_TYPES, sticker_path
from yuni_agent.common.http import fail
from yuni_agent.model.image import (
    EXTENSIONS as IMAGE_EXTENSIONS,
)
from yuni_agent.model.image import (
    IMAGE_DIR,
    IMAGE_ID_RE,
    MAX_IMAGE_BYTES,
    image_path,
    sniff_image_type,
    store_image,
)
from yuni_agent.model.voice import MAX_VOICE_BYTES, VOICE_DIR, VOICE_ID_RE, store_voice, voice_path
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import (
    AllowedRoot,
    AvatarFile,
    ChatMessage,
    ChatTurn,
    ClientEvent,
    Companion,
    Conversation,
    Expression,
    ExpressionProgress,
    ExpressionSource,
    ExpressionVector,
    LocalProfile,
    Memory,
    MemoryCandidate,
    MemoryEmbedding,
    MemoryEntity,
    MemoryJob,
    MemorySource,
    Reminder,
    SafetyEvent,
    Settings,
    Sticker,
    ToolCall,
)

router = APIRouter(prefix="/api/data", tags=["data"])
BACKUP_VERSION = 1

BACKUP_MODELS = {
    "avatarFiles": AvatarFile,
    "localProfiles": LocalProfile,
    "companions": Companion,
    "conversations": Conversation,
    "stickers": Sticker,
    "chatMessages": ChatMessage,
    "chatTurns": ChatTurn,
    "memoryEntities": MemoryEntity,
    "memories": Memory,
    "memorySources": MemorySource,
    "reminders": Reminder,
    "expressions": Expression,
    "expressionSources": ExpressionSource,
    "expressionProgress": ExpressionProgress,
}


class ImportResult(CamelModel):
    has_companion: bool
    imported_messages: int
    imported_memories: int


def _serialize_row(item: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in item.__table__.columns:
        value = getattr(item, column.name)
        result[column.name] = iso_utc(value) if isinstance(value, datetime) else value
    return result


def _deserialize_row(model: type[Any], row: dict[str, Any]) -> Any:
    values: dict[str, Any] = {}
    columns = {column.name: column for column in model.__table__.columns}
    for name, value in row.items():
        column = columns.get(name)
        if column is None:
            continue
        if isinstance(column.type, DateTime) and isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        values[name] = value
    return model(**values)


def _settings_payload(item: Settings) -> dict[str, Any]:
    return {
        "memoryGenerationEnabled": item.memory_generation_enabled,
        "memoryRetrievalEnabled": item.memory_retrieval_enabled,
        "proactiveChatEnabled": item.proactive_chat_enabled,
        "proactiveIntervalMinutes": item.proactive_interval_minutes,
        "quietHoursStart": item.quiet_hours_start,
        "quietHoursEnd": item.quiet_hours_end,
        "modelProvider": item.model_provider,
        "modelBaseUrl": item.model_base_url,
        "chatModel": item.chat_model,
        "privacyVersion": item.privacy_version,
    }


@router.get("/export")
def export_data(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> Response:
    with runtime.require_database().session() as session:
        settings = session.get(Settings, 1)
        assert settings is not None
        tables = {
            name: [_serialize_row(item) for item in session.scalars(select(model))]
            for name, model in BACKUP_MODELS.items()
        }
        avatar_payloads: dict[str, str] = {}
        for avatar in session.scalars(select(AvatarFile)):
            path = runtime.config.data_dir / avatar.relative_path
            if path.is_file():
                avatar_payloads[str(avatar.id)] = base64.b64encode(path.read_bytes()).decode("ascii")
        sticker_payloads = {}
        for sticker in session.scalars(select(Sticker)):
            path = sticker_path(runtime.config.data_dir, sticker)
            if not path.is_file():
                fail(422, 40000, "表情素材缺失，无法完整导出")
            sticker_payloads[str(sticker.id)] = base64.b64encode(path.read_bytes()).decode("ascii")
        image_payloads: dict[str, str] = {}
        for row in tables["chatMessages"]:
            image = row.get("image")
            if not isinstance(image, dict) or image.get("id") in image_payloads:
                continue
            try:
                path = image_path(runtime.config.data_dir, str(image.get("id")), str(image.get("content_type")))
            except ValueError:
                fail(422, 40000, "生成图片记录无效，无法完整导出")
            if not path.is_file():
                fail(422, 40000, "生成图片文件缺失，无法完整导出")
            image_payloads[image["id"]] = base64.b64encode(path.read_bytes()).decode("ascii")
        voice_payloads: dict[str, str] = {}
        for row in tables["chatMessages"]:
            voice = row.get("voice")
            if not isinstance(voice, dict) or voice.get("id") in voice_payloads:
                continue
            try:
                path = voice_path(runtime.config.data_dir, str(voice.get("id")), str(voice.get("content_type")))
            except ValueError:
                fail(422, 40000, "语音记录无效，无法完整导出")
            if not path.is_file():
                fail(422, 40000, "语音文件缺失，无法完整导出")
            voice_payloads[voice["id"]] = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "format": "yuni-local-backup",
            "version": BACKUP_VERSION,
            "exportedAt": iso_utc(datetime.now().astimezone()),
            "settings": _settings_payload(settings),
            "tables": tables,
            "avatarPayloads": avatar_payloads,
            "stickerPayloads": sticker_payloads,
            "imagePayloads": image_payloads,
            "voicePayloads": voice_payloads,
        }
    return Response(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="yuni-backup.json"'},
    )


def _clear_database(session: Session) -> None:
    for model in (
        ClientEvent,
        MemoryEmbedding, ChatTurn, MemoryEntity, ExpressionVector, ExpressionSource, ExpressionProgress, Expression,
        SafetyEvent,
        Reminder,
        ToolCall,
        MemoryCandidate,
        MemoryJob,
        MemorySource,
        Memory,
        MemoryEntity,
        ChatMessage,
        Sticker,
        Conversation,
        Companion,
        LocalProfile,
        AvatarFile,
        AllowedRoot,
    ):
        session.execute(delete(model))


def _clear_avatar_files(data_dir: Path) -> None:
    for folder in ("avatars", "stickers", IMAGE_DIR, VOICE_DIR):
        target = data_dir / folder
        if target.resolve().parent != data_dir.resolve():
            raise ValueError("媒体目录无效")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


@router.delete("", response_model=ApiResponse[None])
async def clear_data(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[None]:
    await runtime.close_chat_queues()
    runtime.data_generation += 1
    runtime.embedding_generation += 1
    runtime.clear_memory_index_files()
    with runtime.require_database().session() as session:
        _clear_database(session)
        settings = session.get(Settings, 1)
        assert settings is not None
        settings.memory_generation_enabled = True
        settings.memory_retrieval_enabled = True
        settings.proactive_chat_enabled = True
        settings.proactive_interval_minutes = 120
        settings.quiet_hours_start = "23:30"
        settings.quiet_hours_end = "08:30"
        settings.model_provider = "openai_compatible"
        settings.model_base_url = "https://api.openai.com/v1"
        settings.chat_model = ""
        session.add(LocalProfile(id=1))
        runtime.ensure_workspace_root(session)
    _clear_avatar_files(runtime.config.data_dir)
    runtime.has_companion = False
    runtime.memory_generation_enabled = True
    runtime.memory_retrieval_enabled = True
    runtime.remove_model_auth()
    runtime.remove_embedding_auth()
    runtime.remove_image_auth()
    runtime.remove_voice_auth()
    emit_event(runtime, "state_reset", {"reason": "cleared"})
    return ApiResponse(data=None)


def _validate_backup(payload: Any) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not isinstance(payload, dict):
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "备份文件格式错误")
    if payload.get("format") != "yuni-local-backup" or payload.get("version") != BACKUP_VERSION:
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "备份版本不受支持")
    settings = payload.get("settings")
    tables = payload.get("tables")
    if not isinstance(settings, dict) or not isinstance(tables, dict):
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "备份内容不完整")
    for name in BACKUP_MODELS:
        if name in ("chatTurns", "expressions", "expressionSources", "expressionProgress", "memoryEntities", "stickers") and name not in tables:
            tables[name] = []
        if name == "localProfiles" and name not in tables:
            tables[name] = [{"id": 1, "nickname": "我", "avatar_type": "preset", "avatar_preset_id": "river"}]
        if not isinstance(tables.get(name), list):
            fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, f"备份缺少 {name}")
        if not all(isinstance(row, dict) for row in tables[name]):
            fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, f"备份中的 {name} 无效")
    for row in tables["companions"]:
        if row.get("gender") not in (None, "male", "female"):
            fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "备份中的伴侣性别无效")
        description = row.get("custom_description", "")
        if not isinstance(description, str) or len(description) > 2000:
            fail(status.HTTP_422_UNPROCESSABLE_CONTENT, 40000, "备份中的伴侣描述无效")
    expressions = {row.get("id"): row for row in tables["expressions"]}
    conversations = {row.get("id"): row for row in tables["conversations"]}
    messages = {row.get("id"): row for row in tables["chatMessages"]}
    for row in tables["chatTurns"]:
        message = messages.get(row.get("user_message_id"))
        anchor = messages.get(row.get("batch_user_message_id")) if row.get("batch_user_message_id") is not None else message
        if (not message or message.get("role") != "user" or not anchor or anchor.get("role") != "user"
            or message.get("conversation_id") != anchor.get("conversation_id")
            or row.get("status") not in ("queued", "running", "completed", "failed", "interrupted")):
            fail(422, 40000, "备份中的聊天批次无效")
        if row["status"] in ("queued", "running"):
            row["status"] = "interrupted"
    companions = {row.get("id") for row in tables["companions"]}
    from yuni_agent.memory.models import MemoryCandidate as StructuredCandidate
    entity_rows = {row.get("id"): row for row in tables["memoryEntities"]}
    memory_rows = {row.get("id"): row for row in tables["memories"]}
    try:
        for row in entity_rows.values():
            if type(row.get("id")) is not int or row.get("companion_id") not in companions or not isinstance(row.get("canonical_name"), str):
                raise ValueError("invalid entity")
            if row.get("entity_type", "person") not in {"person", "pet", "organization", "place", "other"} or row.get("status", "active") not in {"active", "merged", "deleted"}:
                raise ValueError("invalid entity type")
            if not isinstance(row.get("aliases", []), list) or not all(isinstance(alias,str) for alias in row.get("aliases", [])):
                raise ValueError("invalid aliases")
            target = entity_rows.get(row.get("merged_into_id"))
            if row.get("merged_into_id") and (target is None or target.get("companion_id") != row.get("companion_id")):
                raise ValueError("invalid entity merge")
        for row in memory_rows.values():
            row.setdefault("kind", "event" if row.get("type") == "important_event" else "fact")
            row.setdefault("domain", "real_world")
            StructuredCandidate.model_validate(row)
            if row.get("companion_id") not in companions or row.get("status", "active") not in {"active","historical","conflicted","retracted","deleted"}:
                raise ValueError("invalid memory")
            if not isinstance(row.get("entity_ids", []),list):
                raise ValueError("invalid entity ids")
            for entity_id in [*row.get("entity_ids", []), row.get("subject_entity_id")]:
                if entity_id is not None and (entity_id not in entity_rows or entity_rows[entity_id].get("companion_id") != row.get("companion_id")):
                    raise ValueError("cross-companion entity")
            for name in ("superseded_by_id", "conflict_with_id"):
                linked = memory_rows.get(row.get(name))
                if row.get(name) and (linked is None or linked.get("companion_id") != row.get("companion_id")):
                    raise ValueError("invalid memory link")
            if row.get("suppressed_until_message_id") is not None and (type(row["suppressed_until_message_id"]) is not int or row["suppressed_until_message_id"] < 0):
                raise ValueError("invalid deletion watermark")
        for row in tables["memorySources"]:
            memory, message = memory_rows.get(row.get("memory_id")), messages.get(row.get("message_id"))
            conversation = conversations.get(message.get("conversation_id")) if message else None
            if not memory or not conversation or conversation.get("companion_id") != memory.get("companion_id"):
                raise ValueError("invalid memory source")
            row.setdefault("message_sent_at", message.get("created_at"))
    except (ValueError, TypeError):
        fail(422, 40000, "备份中的结构化记忆、实体或来源关联无效")
    for row in tables["expressions"]:
        if type(row.get("id")) is not int or row.get("companion_id") not in companions or any(not isinstance(row.get(key), str) or not 1 <= len(row[key]) <= 20 for key in ("situation", "style")):
            fail(422, 40000, "备份中的表达无效")
    source_counts: dict[int, set[int]] = {}
    for row in tables["expressionSources"]:
        expression = expressions.get(row.get("expression_id"))
        message = messages.get(row.get("message_id"))
        conversation = conversations.get(message.get("conversation_id")) if message else None
        if not expression or not message or message.get("role") != "user" or message.get("status") != "succeeded" or not conversation or conversation.get("companion_id") != expression.get("companion_id"):
            fail(422, 40000, "备份中的表达来源无效")
        source_counts.setdefault(expression["id"], set()).add(message["id"])
    for row in tables["expressions"]:
        sourced = len(source_counts.get(row["id"], set()))
        raw_count = row.get("count")
        preserved = raw_count if type(raw_count) is int and raw_count >= 1 else 0
        row["count"] = max(sourced, preserved)
        if row["count"] < 1:
            fail(422, 40000, "备份中的表达缺少用户来源")
    for row in tables["expressionProgress"]:
        if row.get("conversation_id") not in conversations or type(row.get("until_message_id")) is not int:
            fail(422, 40000, "备份中的表达处理进度无效")
    from yuni_agent.api.routes.profile import ProfileInput
    profiles = tables["localProfiles"]
    if len(profiles) != 1 or profiles[0].get("id") != 1:
        fail(status.HTTP_422_UNPROCESSABLE_CONTENT, 40000, "备份中的本地用户资料无效")
    try:
        ProfileInput.model_validate(profiles[0])
    except ValidationError:
        fail(status.HTTP_422_UNPROCESSABLE_CONTENT, 40000, "备份中的昵称或头像无效")
    return settings, tables


@router.post("/import", response_model=ApiResponse[ImportResult])
async def import_data(
    file: UploadFile,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[ImportResult]:
    raw = await file.read(50 * 1024 * 1024 + 1)
    if len(raw) > 50 * 1024 * 1024:
        fail(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 40000, "备份文件不能超过 50 MB")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "备份文件不是有效 JSON")
    settings_payload, tables = _validate_backup(payload)
    avatar_payloads = payload.get("avatarPayloads", {})
    if not isinstance(avatar_payloads, dict):
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "头像备份格式错误")

    decoded_avatars: dict[int, bytes] = {}
    try:
        for row in tables["avatarFiles"]:
            avatar_id = int(row["id"])
            content = base64.b64decode(avatar_payloads[str(avatar_id)], validate=True)
            if len(content) != int(row["size_bytes"]):
                raise ValueError("avatar size mismatch")
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("avatar checksum mismatch")
            decoded_avatars[avatar_id] = content
    except (KeyError, ValueError, TypeError):
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, "头像备份已损坏")

    decoded_stickers: dict[int, bytes] = {}
    try:
        assets = payload.get("stickerPayloads", {})
        if not isinstance(assets, dict):
            raise ValueError("invalid sticker payloads")
        for row in tables["stickers"]:
            if (type(row.get("id")) is not int or row["id"] <= 0
                or row.get("content_type") not in STICKER_TYPES
                or not isinstance(row.get("description"), str) or len(row["description"]) > 500
                or type(row.get("enabled")) is not bool):
                raise ValueError("invalid sticker metadata")
            # 0014 fields are optional so pre-0014 backups still restore.
            row.setdefault("emotions", [])
            row.setdefault("usage", "")
            if (not isinstance(row["emotions"], list) or len(row["emotions"]) > 5
                    or not all(isinstance(item, str) and len(item) <= 20 for item in row["emotions"])
                    or not isinstance(row["usage"], str) or len(row["usage"]) > 300):
                raise ValueError("invalid sticker semantics")
            content = base64.b64decode(assets[str(row["id"])], validate=True)
            if not content or len(content) > 5 * 1024 * 1024 or len(content) != row["size_bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("sticker checksum mismatch")
            extension = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}[row["content_type"]]
            row["relative_path"] = f"stickers/{row['sha256']}.{extension}"
            decoded_stickers[row["id"]] = content
        for row in tables["chatMessages"]:
            sticker = row.get("sticker")
            if sticker is not None and (not isinstance(sticker, dict) or type(sticker.get("id")) is not int
                or sticker["id"] not in decoded_stickers or not isinstance(sticker.get("description"), str)
                or len(sticker["description"]) > 500
                or not isinstance(sticker.get("emotions", []), list)
                or not all(isinstance(item, str) for item in sticker.get("emotions", []))):
                raise ValueError("invalid message sticker")
    except (KeyError, ValueError, TypeError):
        fail(422, 40000, "表情备份已损坏")

    decoded_images: dict[str, bytes] = {}
    try:
        image_assets = payload.get("imagePayloads", {})
        if not isinstance(image_assets, dict):
            raise ValueError("invalid image payloads")
        for row in tables["chatMessages"]:
            image = row.get("image")
            if image is None:
                continue
            if (not isinstance(image, dict) or row.get("role") != "assistant"
                    or not isinstance(image.get("id"), str) or not IMAGE_ID_RE.fullmatch(image["id"])
                    or image.get("content_type") not in IMAGE_EXTENSIONS
                    or not isinstance(image.get("description", ""), str) or len(image.get("description", "")) > 500):
                raise ValueError("invalid message image")
            if image["id"] in decoded_images:
                continue
            content = base64.b64decode(image_assets[image["id"]], validate=True)
            if (not content or len(content) > MAX_IMAGE_BYTES or hashlib.sha256(content).hexdigest() != image["id"]
                    or sniff_image_type(content) != image["content_type"]):
                raise ValueError("image checksum mismatch")
            decoded_images[image["id"]] = content
    except (KeyError, ValueError, TypeError):
        fail(422, 40000, "生成图片备份已损坏")

    decoded_voices: dict[str, bytes] = {}
    try:
        voice_assets = payload.get("voicePayloads", {})
        if not isinstance(voice_assets, dict):
            raise ValueError("invalid voice payloads")
        for row in tables["chatMessages"]:
            voice = row.get("voice")
            if voice is None:
                continue
            if (not isinstance(voice, dict) or row.get("role") != "assistant"
                    or not isinstance(voice.get("id"), str) or not VOICE_ID_RE.fullmatch(voice["id"])
                    or voice.get("content_type") not in ("audio/mpeg", "audio/wav", "audio/ogg", "audio/aac")
                    or type(voice.get("duration")) not in (int, float)
                    or not math.isfinite(voice["duration"]) or not 0 < voice["duration"] <= 60
                    or row.get("image") is not None or row.get("sticker") is not None):
                raise ValueError("invalid message voice")
            if voice["id"] in decoded_voices:
                continue
            content = base64.b64decode(voice_assets[voice["id"]], validate=True)
            from yuni_agent.model.voice import sniff_voice_type
            if (not content or len(content) > MAX_VOICE_BYTES or hashlib.sha256(content).hexdigest() != voice["id"]
                    or sniff_voice_type(content) != voice["content_type"]):
                raise ValueError("voice checksum mismatch")
            decoded_voices[voice["id"]] = content
    except (KeyError, ValueError, TypeError):
        fail(422, 40000, "语音备份已损坏")

    await runtime.close_chat_queues()
    runtime.data_generation += 1
    runtime.embedding_generation += 1
    runtime.clear_memory_index_files()
    try:
        with runtime.require_database().session() as session:
            session.execute(text("PRAGMA defer_foreign_keys=ON"))
            _clear_database(session)
            db_settings = session.get(Settings, 1)
            assert db_settings is not None
            db_settings.memory_generation_enabled = bool(
                settings_payload.get("memoryGenerationEnabled", True)
            )
            db_settings.memory_retrieval_enabled = bool(
                settings_payload.get("memoryRetrievalEnabled", True)
            )
            db_settings.proactive_chat_enabled = bool(
                settings_payload.get("proactiveChatEnabled", True)
            )
            db_settings.proactive_interval_minutes = int(
                settings_payload.get("proactiveIntervalMinutes", 120)
            )
            db_settings.quiet_hours_start = str(
                settings_payload.get("quietHoursStart", "23:30")
            )
            db_settings.quiet_hours_end = str(
                settings_payload.get("quietHoursEnd", "08:30")
            )
            db_settings.model_provider = str(
                settings_payload.get("modelProvider", "openai_compatible")
            )
            db_settings.model_base_url = str(
                settings_payload.get("modelBaseUrl", "https://api.openai.com/v1")
            )
            db_settings.chat_model = str(settings_payload.get("chatModel", ""))
            db_settings.privacy_version = str(settings_payload.get("privacyVersion", "v1.0"))
            for name, model in BACKUP_MODELS.items():
                rows = tables[name]
                if rows and "id" in rows[0]:
                    rows = sorted(rows, key=lambda row: int(row["id"]))
                for row in rows:
                    session.add(_deserialize_row(model, row))
                session.flush()
            runtime.ensure_workspace_root(session)
    except (KeyError, TypeError, ValueError, ValidationError, SQLAlchemyError) as exc:
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40000, f"备份数据无法恢复：{exc}")

    _clear_avatar_files(runtime.config.data_dir)
    for row in tables["avatarFiles"]:
        path = runtime.config.data_dir / str(row["relative_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(decoded_avatars[int(row["id"])])
    for row in tables["stickers"]:
        path = runtime.config.data_dir / row["relative_path"]
        path.write_bytes(decoded_stickers[row["id"]])
    for content in decoded_images.values():
        store_image(runtime.config.data_dir, content, "")
    for content in decoded_voices.values():
        store_voice(runtime.config.data_dir, content, "")

    runtime.has_companion = bool(tables["companions"])
    runtime.memory_generation_enabled = bool(
        settings_payload.get("memoryGenerationEnabled", True)
    )
    runtime.memory_retrieval_enabled = bool(
        settings_payload.get("memoryRetrievalEnabled", True)
    )
    runtime.remove_model_auth()
    runtime.remove_embedding_auth()
    runtime.remove_image_auth()
    runtime.remove_voice_auth()
    emit_event(runtime, "state_reset", {"reason": "imported"})
    return ApiResponse(
        data=ImportResult(
            has_companion=runtime.has_companion,
            imported_messages=len(tables["chatMessages"]),
            imported_memories=len(tables["memories"]),
        )
    )
