from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, UploadFile
from pydantic import Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.routes.companion import (
    PRESETS,
    UploadedAvatar,
    avatar_data_url,
    upload_avatar,
    validate_uploaded_avatar,
)
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import LocalProfile

router = APIRouter(prefix="/api/profile", tags=["profile"])


class ProfileInput(CamelModel):
    nickname: str = Field(min_length=1, max_length=20)
    avatar_type: Literal["preset", "uploaded"]
    avatar_preset_id: str | None = None
    avatar_file_id: int | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> ProfileInput:
        self.nickname = self.nickname.strip()
        if not self.nickname:
            raise ValueError("请填写昵称")
        if self.avatar_type == "preset" and self.avatar_preset_id not in {item["id"] for item in PRESETS}:
            raise ValueError("默认头像无效")
        if self.avatar_type == "uploaded" and self.avatar_file_id is None:
            raise ValueError("请先上传头像")
        return self


class ProfileData(CamelModel):
    id: int
    nickname: str
    avatar_type: str
    avatar_preset_id: str | None
    avatar_file_id: int | None
    avatar_url: str
    created_at: str
    updated_at: str


def to_data(runtime: AgentRuntime, profile: LocalProfile) -> ProfileData:
    preset = next((item for item in PRESETS if item["id"] == profile.avatar_preset_id), None)
    return ProfileData(
        id=profile.id, nickname=profile.nickname, avatar_type=profile.avatar_type,
        avatar_preset_id=profile.avatar_preset_id, avatar_file_id=profile.avatar_file_id,
        avatar_url=preset["url"] if profile.avatar_type == "preset" and preset else avatar_data_url(runtime, profile.avatar_file),
        created_at=iso_utc(profile.created_at), updated_at=iso_utc(profile.updated_at),
    )


@router.get("", response_model=ApiResponse[ProfileData])
def get_profile(runtime: Annotated[AgentRuntime, Depends(require_session)]) -> ApiResponse[ProfileData]:
    with runtime.require_database().session() as session:
        profile = session.scalar(select(LocalProfile).options(selectinload(LocalProfile.avatar_file)).where(LocalProfile.id == 1))
        assert profile is not None
        return ApiResponse(data=to_data(runtime, profile))


@router.put("", response_model=ApiResponse[ProfileData])
def update_profile(
    body: ProfileInput, runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[ProfileData]:
    if body.avatar_type == "uploaded":
        validate_uploaded_avatar(runtime, body.avatar_file_id)
    with runtime.require_database().session() as session:
        profile = session.get(LocalProfile, 1)
        assert profile is not None
        profile.nickname = body.nickname
        profile.avatar_type = body.avatar_type
        profile.avatar_preset_id = body.avatar_preset_id if body.avatar_type == "preset" else None
        profile.avatar_file_id = body.avatar_file_id if body.avatar_type == "uploaded" else None
        session.flush()
        session.refresh(profile, attribute_names=["avatar_file"])
        return ApiResponse(data=to_data(runtime, profile))


@router.post("/avatar", response_model=ApiResponse[UploadedAvatar], status_code=201)
async def upload_profile_avatar(
    file: UploadFile, runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[UploadedAvatar]:
    return await upload_avatar(file, runtime)
