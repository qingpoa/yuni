from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yuni_agent.memory.models import MemoryCandidate, MemoryType


class ToolRisk(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    content: str
    is_error: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class SaveMemoryArguments(MemoryCandidate):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_quote: str = Field(min_length=2, max_length=2000)
    source_quote: str = Field(min_length=2, max_length=2000)
    sensitivity: Literal["normal", "sensitive"]


class _ToolArguments(BaseModel):
    # Models occasionally attach stray keys; ignore them instead of failing the call.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)


_PATH_HINT = "相对路径（如 notes.txt）放在默认工作目录；也可用授权目录内的绝对路径"


class FindFilesArguments(_ToolArguments):
    path: str = Field(default="", max_length=4096, description="要搜索的目录；省略则搜索所有授权目录。" + _PATH_HINT)
    query: str = Field(max_length=200, description="文件名或文本关键词；空字符串表示列出目录")


class ReadTextFileArguments(_ToolArguments):
    path: str = Field(min_length=1, max_length=4096, description="文件路径。" + _PATH_HINT)


class CreateTextFileArguments(_ToolArguments):
    path: str = Field(min_length=1, max_length=4096, description="新文件路径。" + _PATH_HINT)
    content: str = Field(min_length=1)


class ReplaceTextArguments(_ToolArguments):
    path: str = Field(min_length=1, max_length=4096, description="文件路径。" + _PATH_HINT)
    old_text: str = Field(min_length=1)
    # Empty new_text deletes the matched passage.
    new_text: str


class CreateReminderArguments(_ToolArguments):
    title: str = Field(min_length=1, max_length=200)
    remind_at: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def non_blank_title(self):
        if not self.title.strip():
            raise ValueError("title 不能为空")
        return self


class GenerateImageArguments(_ToolArguments):
    mode: Literal["creative", "companion_photo"] = Field(
        default="creative",
        description="companion_photo=伴侣自拍/自己的照片/眼前的样子（尽量贴合头像本人）；creative=按要求画的其它图。")
    prompt: str = Field(min_length=1, max_length=1000,
                        description="完整画面描述：主体、动作、环境、光线、风格、构图；按用户要求写，不要加入用户未提的真实人物信息。")
    aspect_ratio: Literal["default", "1:1", "3:4", "4:3"] = Field(
        default="default", description="画幅：1:1 方图，3:4 竖图（如手机壁纸），4:3 横图；default 用设置里的默认尺寸。")

    @model_validator(mode="after")
    def non_blank_prompt(self):
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空")
        return self


class QueryMemoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=0, max_length=2000)
    type: MemoryType | None = None
    kind: Literal["all", "fact", "event", "ongoing_state"] = "all"
    include_history: bool = False
    time_from: str | None = None
    time_to: str | None = None
    entity_name: str = Field(default="", max_length=100)
    domain: Literal["real_world", "conversation", "fictional"] = "real_world"
    scene_key: str | None = None
    updated_after: datetime | None = None
    updated_before: datetime | None = None
    limit: int = Field(default=6, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def valid_range(self):
        from yuni_agent.memory.retriever import parse_time_bound
        first = parse_time_bound(self.time_from)
        last = parse_time_bound(self.time_to, end=True)
        if first and last and first >= last:
            raise ValueError("时间范围无效")
        if self.domain == "fictional" and not self.scene_key:
            raise ValueError("虚构范围须指定 scene_key")
        for value in (self.updated_after, self.updated_before):
            if value is not None and value.tzinfo is None:
                raise ValueError("时间筛选须带时区")
        if self.updated_after and self.updated_before and self.updated_after > self.updated_before:
            raise ValueError("时间范围无效")
        return self

