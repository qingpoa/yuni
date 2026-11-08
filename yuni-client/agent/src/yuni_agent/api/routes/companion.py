from __future__ import annotations

import base64
import hashlib
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, UploadFile, status
from pydantic import Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.common.http import fail
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import AvatarFile, Companion, Conversation

router = APIRouter(prefix="/api/partner", tags=["companion"])

PERSONALITY_TAGS = {
    "gentle": "温柔",
    "reliable": "可靠",
    "patient": "耐心",
    "lively": "有趣",
    "rational": "理性",
    "delicate": "细腻",
}
INTERACTION_STYLES = {
    "daily_companion": "日常陪伴",
    "emotional_comfort": "情绪安慰",
    "light_humor": "轻松玩笑",
    "grow_together": "共同成长",
}
INTIMACY_LEVELS = {
    "reserved": "克制",
    "gentle_natural": "温柔自然",
    "warm_close": "亲近温暖",
}
PRESETS = [
    {"id": "mist", "name": "晨雾", "url": "mock-avatar://mist", "color": "#819587"},
    {"id": "river", "name": "青川", "url": "mock-avatar://river", "color": "#657b83"},
    {"id": "sunset", "name": "暖霞", "url": "mock-avatar://sunset", "color": "#a0837d"},
    {"id": "night", "name": "夜蓝", "url": "mock-avatar://night", "color": "#7b788d"},
]


class PartnerInput(CamelModel):
    name: str = Field(min_length=1, max_length=20)
    gender: Literal["male", "female"] | None = None
    avatar_type: Literal["preset", "uploaded"]
    avatar_preset_id: str | None = None
    avatar_file_id: int | None = None
    avatar_url: str = ""
    user_nickname: str = Field(min_length=1, max_length=20)
    personality_tags: list[str] = Field(default_factory=list, max_length=4)
    interaction_styles: list[str] = Field(default_factory=list, max_length=3)
    intimacy_level: str
    boundary_text: str = Field(default="", max_length=500)
    custom_description: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_options(self) -> PartnerInput:
        if not set(self.personality_tags) <= PERSONALITY_TAGS.keys():
            raise ValueError("包含不支持的性格标签")
        if not set(self.interaction_styles) <= INTERACTION_STYLES.keys():
            raise ValueError("包含不支持的相处风格")
        if self.intimacy_level not in INTIMACY_LEVELS:
            raise ValueError("亲密表达级别无效")
        if self.avatar_type == "preset" and self.avatar_preset_id not in {
            item["id"] for item in PRESETS
        }:
            raise ValueError("默认头像无效")
        if self.avatar_type == "uploaded" and self.avatar_file_id is None:
            raise ValueError("请先上传头像")
        return self


class PartnerData(CamelModel):
    id: int
    name: str
    gender: Literal["male", "female"] | None
    avatar_type: str
    avatar_preset_id: str | None
    avatar_file_id: int | None
    avatar_url: str
    user_nickname: str
    relationship_status: str
    personality_tags: list[str]
    interaction_styles: list[str]
    intimacy_level: str
    boundary_text: str
    custom_description: str
    created_at: str
    updated_at: str


class Option(CamelModel):
    value: str
    label: str


class PartnerOptions(CamelModel):
    personality_tags: list[Option]
    interaction_styles: list[Option]
    intimacy_levels: list[Option]


class AvatarPreset(CamelModel):
    id: str
    name: str
    url: str
    color: str


class UploadedAvatar(CamelModel):
    avatar_file_id: int
    avatar_url: str


def avatar_data_url(runtime: AgentRuntime, avatar: AvatarFile | None) -> str:
    if avatar is None:
        return ""
    path = runtime.config.data_dir / avatar.relative_path
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{avatar.content_type};base64,{encoded}"


def _to_data(runtime: AgentRuntime, item: Companion) -> PartnerData:
    preset = next((value for value in PRESETS if value["id"] == item.avatar_preset_id), None)
    avatar_url = (
        preset["url"] if item.avatar_type == "preset" and preset else avatar_data_url(runtime, item.avatar_file)
    )
    return PartnerData(
        id=item.id,
        name=item.name,
        gender=item.gender,
        avatar_type=item.avatar_type,
        avatar_preset_id=item.avatar_preset_id,
        avatar_file_id=item.avatar_file_id,
        avatar_url=avatar_url,
        user_nickname=item.user_nickname,
        relationship_status=item.relationship_status,
        personality_tags=item.personality_tags,
        interaction_styles=item.interaction_styles,
        intimacy_level=item.intimacy_level,
        boundary_text=item.boundary_text,
        custom_description=item.custom_description,
        created_at=iso_utc(item.created_at),
        updated_at=iso_utc(item.updated_at),
    )


def _load_current(runtime: AgentRuntime) -> Companion | None:
    database = runtime.require_database()
    with database.session() as session:
        return session.scalar(select(Companion).options(selectinload(Companion.avatar_file)).limit(1))


@router.get("/options", response_model=ApiResponse[PartnerOptions])
def get_options(_: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[PartnerOptions]:
    return ApiResponse(
        data=PartnerOptions(
            personality_tags=[Option(value=k, label=v) for k, v in PERSONALITY_TAGS.items()],
            interaction_styles=[Option(value=k, label=v) for k, v in INTERACTION_STYLES.items()],
            intimacy_levels=[Option(value=k, label=v) for k, v in INTIMACY_LEVELS.items()],
        )
    )


@router.get("/avatar-presets", response_model=ApiResponse[list[AvatarPreset]])
def get_presets(_: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[list[AvatarPreset]]:
    return ApiResponse(data=[AvatarPreset.model_validate(item) for item in PRESETS])


@router.get("", response_model=ApiResponse[PartnerData | None])
def get_current(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[PartnerData | None]:
    item = _load_current(runtime)
    return ApiResponse(data=_to_data(runtime, item) if item else None)


def validate_uploaded_avatar(runtime: AgentRuntime, avatar_file_id: int | None) -> None:
    if avatar_file_id is None:
        return
    with runtime.require_database().session() as session:
        if session.get(AvatarFile, avatar_file_id) is None:
            fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40201, "上传头像不存在")


@router.post("", response_model=ApiResponse[PartnerData], status_code=status.HTTP_201_CREATED)
def create_partner(
    body: PartnerInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[PartnerData]:
    if _load_current(runtime) is not None:
        fail(status.HTTP_409_CONFLICT, 40101, "已经创建过伴侣")
    validate_uploaded_avatar(runtime, body.avatar_file_id)
    with runtime.require_database().session() as session:
        item = Companion(
            singleton_key=1,
            name=body.name.strip(),
            gender=body.gender,
            avatar_type=body.avatar_type,
            avatar_preset_id=body.avatar_preset_id if body.avatar_type == "preset" else None,
            avatar_file_id=body.avatar_file_id if body.avatar_type == "uploaded" else None,
            user_nickname=body.user_nickname.strip(),
            personality_tags=list(dict.fromkeys(body.personality_tags)),
            interaction_styles=list(dict.fromkeys(body.interaction_styles)),
            intimacy_level=body.intimacy_level,
            boundary_text=body.boundary_text.strip(),
            custom_description=body.custom_description.strip(),
        )
        session.add(item)
        session.flush()
        session.add(Conversation(companion_id=item.id))
        session.flush()
        session.refresh(item, attribute_names=["avatar_file"])
        result = _to_data(runtime, item)
    runtime.has_companion = True
    return ApiResponse(data=result)


@router.put("", response_model=ApiResponse[PartnerData])
def update_partner(
    body: PartnerInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[PartnerData]:
    validate_uploaded_avatar(runtime, body.avatar_file_id)
    with runtime.require_database().session() as session:
        item = session.scalar(select(Companion).options(selectinload(Companion.avatar_file)).limit(1))
        if item is None:
            fail(status.HTTP_404_NOT_FOUND, 40102, "尚未创建伴侣")
        item.name = body.name.strip()
        if "gender" in body.model_fields_set:
            item.gender = body.gender
        item.avatar_type = body.avatar_type
        item.avatar_preset_id = body.avatar_preset_id if body.avatar_type == "preset" else None
        item.avatar_file_id = body.avatar_file_id if body.avatar_type == "uploaded" else None
        item.user_nickname = body.user_nickname.strip()
        if "personality_tags" in body.model_fields_set:
            item.personality_tags = list(dict.fromkeys(body.personality_tags))
        if "interaction_styles" in body.model_fields_set:
            item.interaction_styles = list(dict.fromkeys(body.interaction_styles))
        item.intimacy_level = body.intimacy_level
        item.boundary_text = body.boundary_text.strip()
        if "custom_description" in body.model_fields_set:
            item.custom_description = body.custom_description.strip()
        session.flush()
        session.refresh(item, attribute_names=["avatar_file"])
        result = _to_data(runtime, item)
    return ApiResponse(data=result)


@router.post("/avatar", response_model=ApiResponse[UploadedAvatar], status_code=status.HTTP_201_CREATED)
async def upload_avatar(
    file: UploadFile,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[UploadedAvatar]:
    content_types = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    extension = content_types.get(file.content_type or "")
    if extension is None:
        fail(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 40201, "仅支持 jpeg、png 或 webp")
    payload = await file.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        fail(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 40201, "头像不能超过 5 MB")
    if not payload:
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40201, "头像文件为空")

    avatar_dir = runtime.config.data_dir / "avatars"
    avatar_dir.mkdir(parents=True, exist_ok=True)
    relative_path = Path("avatars") / f"{uuid.uuid4().hex}{extension}"
    target = runtime.config.data_dir / relative_path
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)

    with runtime.require_database().session() as session:
        item = AvatarFile(
            relative_path=relative_path.as_posix(),
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        session.add(item)
        session.flush()
        avatar_file_id = item.id
    encoded = base64.b64encode(payload).decode("ascii")
    return ApiResponse(
        data=UploadedAvatar(
            avatar_file_id=avatar_file_id,
            avatar_url=f"data:{file.content_type};base64,{encoded}",
        )
    )
