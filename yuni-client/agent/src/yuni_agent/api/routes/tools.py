from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, status
from pydantic import Field
from sqlalchemy import select

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.common.http import fail
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import AllowedRoot, ToolCall

router = APIRouter(tags=["tools"])


class RootInput(CamelModel):
    path: str = Field(min_length=1, max_length=1000)


class DirectoryBrowseInput(CamelModel):
    path: str | None = Field(default=None, min_length=1, max_length=1000)


class DirectoryEntry(CamelModel):
    name: str
    path: str


class DirectoryListing(CamelModel):
    path: str | None
    parent_path: str | None
    items: list[DirectoryEntry]


class RootData(CamelModel):
    id: int
    path: str
    display_name: str
    created_at: str


class ToolCallData(CamelModel):
    id: str
    tool_name: str
    risk: str
    status: str
    summary: str
    target_paths: list[str]
    diff_preview: str | None
    error_message: str | None
    created_at: str
    completed_at: str | None


class ToolDecisionInput(CamelModel):
    decision: Literal["approve", "reject"]


def _root_data(item: AllowedRoot) -> RootData:
    return RootData(
        id=item.id,
        path=item.canonical_path,
        display_name=item.display_name,
        created_at=iso_utc(item.created_at),
    )


def _tool_data(item: ToolCall) -> ToolCallData:
    summary = item.result_summary or item.tool_name
    return ToolCallData(
        id=item.id,
        tool_name=item.tool_name,
        risk=item.risk,
        status=item.status,
        summary=summary,
        target_paths=item.target_paths or [],
        diff_preview=item.diff_preview,
        error_message=item.error_message,
        created_at=iso_utc(item.created_at),
        completed_at=iso_utc(item.completed_at) if item.completed_at else None,
    )


@router.post("/api/directories/browse", response_model=ApiResponse[DirectoryListing])
def browse_directories(
    body: DirectoryBrowseInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[DirectoryListing]:
    if body.path is None:
        locations = [DirectoryEntry(name="用户文件夹", path=str(Path.home()))]
        if os.name == "nt":
            import ctypes
            import string

            drives = ctypes.windll.kernel32.GetLogicalDrives()
            locations.extend(
                DirectoryEntry(name=f"{letter}:\\", path=f"{letter}:\\")
                for index, letter in enumerate(string.ascii_uppercase)
                if drives & (1 << index)
            )
        else:
            locations.append(DirectoryEntry(name="文件系统", path="/"))
        return ApiResponse(data=DirectoryListing(path=None, parent_path=None, items=locations))

    requested = Path(body.path)
    if not requested.is_absolute():
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40601, "请选择一个有效目录")
    try:
        resolved = requested.resolve(strict=True)
        items = []
        with os.scandir(resolved) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        items.append(DirectoryEntry(name=entry.name, path=str(resolved / entry.name)))
                except OSError:
                    continue
    except (OSError, ValueError, RuntimeError):
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40601, "无法打开此目录，可能已移动或没有访问权限")
    items.sort(key=lambda item: (item.name.casefold(), item.name))
    return ApiResponse(data=DirectoryListing(
        path=str(resolved),
        parent_path=str(resolved.parent) if resolved.parent != resolved else None,
        items=items,
    ))


@router.get("/api/allowed-roots", response_model=ApiResponse[list[RootData]])
def list_roots(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[list[RootData]]:
    with runtime.require_database().session() as session:
        items = list(
            session.scalars(
                select(AllowedRoot)
                .where(AllowedRoot.status == "active")
                .order_by(AllowedRoot.created_at, AllowedRoot.id)
            )
        )
        result = [_root_data(item) for item in items]
    return ApiResponse(data=result)


@router.post("/api/allowed-roots", response_model=ApiResponse[RootData], status_code=201)
def add_root(
    body: RootInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[RootData]:
    requested = Path(body.path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError:
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40601, "授权目录不存在")
    if not resolved.is_dir():
        fail(status.HTTP_422_UNPROCESSABLE_ENTITY, 40601, "请选择一个目录")
    canonical = os.path.normcase(str(resolved))
    with runtime.require_database().session() as session:
        item = session.scalar(
            select(AllowedRoot).where(AllowedRoot.canonical_path == canonical)
        )
        if item is None:
            item = AllowedRoot(
                canonical_path=canonical,
                display_name=resolved.name or str(resolved),
                status="active",
            )
            session.add(item)
            session.flush()
        elif item.status == "active":
            fail(status.HTTP_409_CONFLICT, 40601, "该目录已经授权")
        else:
            item.status = "active"
            item.revoked_at = None
            session.flush()
        result = _root_data(item)
    return ApiResponse(data=result)


@router.delete("/api/allowed-roots/{root_id}", response_model=ApiResponse[None])
def remove_root(
    root_id: int,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[None]:
    with runtime.require_database().session() as session:
        item = session.get(AllowedRoot, root_id)
        if item is None or item.status != "active":
            fail(status.HTTP_404_NOT_FOUND, 40601, "授权目录不存在")
        from yuni_agent.storage.models import utc_now

        item.status = "revoked"
        item.revoked_at = utc_now()
    return ApiResponse(data=None)


@router.get("/api/tool-calls", response_model=ApiResponse[list[ToolCallData]])
def list_tool_calls(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[list[ToolCallData]]:
    with runtime.require_database().session() as session:
        items = list(
            session.scalars(select(ToolCall).order_by(ToolCall.created_at.desc()).limit(100))
        )
        result = [_tool_data(item) for item in items]
    return ApiResponse(data=result)


@router.post(
    "/api/tool-calls/{call_id}/decision", response_model=ApiResponse[ToolCallData]
)
async def decide_tool_call(
    call_id: str,
    body: ToolDecisionInput,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[ToolCallData]:
    with runtime.require_database().session() as session:
        item = session.get(ToolCall, call_id)
        if item is None:
            fail(status.HTTP_404_NOT_FOUND, 40603, "工具确认已失效")
        if item.status != "awaiting_confirmation":
            fail(status.HTTP_409_CONFLICT, 40603, "工具确认已失效")
        result = _tool_data(item)
    if not runtime.resolve_tool_decision(call_id, body.decision):
        fail(status.HTTP_409_CONFLICT, 40603, "工具确认已失效")
    return ApiResponse(data=result)
