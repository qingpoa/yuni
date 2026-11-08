"""Declarative two-stage tool router (`request_tools`) with BM25 top-K filtering and disambiguation."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from yuni_agent.runtime import AgentRuntime
from yuni_agent.tools.errors import ToolExecutionError
from yuni_agent.tools.registry import RegisteredTool, ToolRegistry

MAX_MOUNTED_PER_REQUEST = 4

GroupId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
IntentStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


@dataclass(frozen=True, slots=True)
class ToolGroupSpec:
    id: str
    title: str
    description: str
    not_for: str = ""


DEFAULT_TOOL_GROUPS: dict[str, ToolGroupSpec] = {
    "memory": ToolGroupSpec(
        id="memory",
        title="长期记忆（memory）",
        description="伴侣脑海中的长期事实、偏好、近况与往事检索或保存",
        not_for="读写用户电脑里的本地文档或笔记请选 files；设置到点弹窗的定时提醒请选 reminder",
    ),
    "files": ToolGroupSpec(
        id="files",
        title="本地文件（files）",
        description="用户电脑已授权目录中的本地文本文件查找、读取、新建与局部替换",
        not_for="查询或保存伴侣脑海中的长期记忆与偏好请选 memory",
    ),
    "reminder": ToolGroupSpec(
        id="reminder",
        title="定时提醒（reminder）",
        description="创建到指定时间触发通知的本地定时提醒/闹钟",
        not_for="不需要到点提醒、仅需长期记住的用户事实或偏好请选 memory",
    ),
    "image": ToolGroupSpec(
        id="image",
        title="图像生成（image）",
        description="按用户要求生成伴侣自拍日常照或绘画创作图片",
        not_for="日常发送现成表情包由系统自动处理，无需调用此分组",
    ),
}


class RequestToolsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[GroupId] = Field(
        min_length=1,
        max_length=2,
        description="需要挂载的工具分组标识（最多同时选 2 个相近分组）。",
    )
    intent: IntentStr = Field(
        description="简要说明当前要执行的具体操作意图（1～300 字），用于在分组内精筛最匹配的工具。",
    )


def _tool_summary(tool: RegisteredTool) -> str:
    if tool.summary.strip():
        return tool.summary.strip()
    first = re.split(r"[。\n]", tool.description.strip(), maxsplit=1)[0].strip()
    return first or tool.name


def _group_spec(group_id: str, tools: list[RegisteredTool]) -> ToolGroupSpec:
    if group_id in DEFAULT_TOOL_GROUPS:
        return DEFAULT_TOOL_GROUPS[group_id]
    summaries = "；".join(_tool_summary(tool) for tool in tools[:3])
    return ToolGroupSpec(
        id=group_id,
        title=f"{group_id}",
        description=summaries or f"{group_id} 相关工具",
    )


def _tokens(value: str) -> list[str]:
    values: list[str] = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            values.extend(part[i : i + 2] for i in range(len(part) - 1))
            values.extend(list(part))
        else:
            values.append(part)
    return values


def _bm25(query: str, texts: dict[int, str]) -> dict[int, float]:
    documents = {key: Counter(_tokens(text)) for key, text in texts.items()}
    terms = set(_tokens(query))
    average = sum(sum(doc.values()) for doc in documents.values()) / max(1, len(documents)) or 1.0
    frequency = {term: sum(term in doc for doc in documents.values()) for term in terms}
    scores: dict[int, float] = {}
    for key, doc in documents.items():
        length = sum(doc.values())
        score = 0.0
        for term in terms:
            tf = doc.get(term, 0)
            if tf:
                idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
        if score:
            scores[key] = score
    return scores


def _available_tools_by_group(
    registry: ToolRegistry,
    runtime: AgentRuntime | None,
    exhausted_tools: Iterable[str],
) -> dict[str, list[RegisteredTool]]:
    exhausted = set(exhausted_tools)
    grouped: dict[str, list[RegisteredTool]] = {}
    for tool in registry.enabled_tools(runtime):
        if tool.name in exhausted:
            continue
        grouped.setdefault(tool.group, []).append(tool)
    return grouped


def build_request_tools_definition(
    registry: ToolRegistry,
    runtime: AgentRuntime | None,
    activated_tools: Iterable[str] = (),
    exhausted_tools: Iterable[str] = (),
) -> dict[str, Any] | None:
    activated = set(activated_tools)
    grouped = _available_tools_by_group(registry, runtime, exhausted_tools)
    unmounted_groups = [
        group_id
        for group_id, tools in grouped.items()
        if any(tool.name not in activated for tool in tools)
    ]
    if not unmounted_groups:
        return None

    group_lines: list[str] = []
    for group_id in unmounted_groups:
        spec = _group_spec(group_id, grouped[group_id])
        boundary = f"（注意区分：{spec.not_for}）" if spec.not_for else ""
        group_lines.append(f"- {spec.id}：{spec.description}{boundary}")

    description = (
        "按需挂载具体业务工具的完整定义（此操作仅在规划上下文中展开工具说明，不执行任何外部操作，也不消耗业务操作次数）。"
        "日常闲聊无需调用；当需要执行具体业务操作但当前工具列表中尚未挂载对应工具时，请先调用本工具选择 1～2 个相关分组并说明意图。\n"
        "当前可挂载分组：\n" + "\n".join(group_lines)
    )
    return {
        "type": "function",
        "function": {
            "name": "request_tools",
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["groups", "intent"],
                "properties": {
                    "groups": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {"type": "string", "enum": unmounted_groups},
                        "description": "需要挂载的工具分组 ID（最多同时选 2 个相近分组）。",
                    },
                    "intent": {
                        "type": "string",
                        "description": "简要说明当前要完成的具体操作意图（1～300 字），用于在分组内精筛最匹配的工具。",
                    },
                },
            },
        },
    }


def resolve_tool_request(
    registry: ToolRegistry,
    runtime: AgentRuntime | None,
    raw_arguments: dict[str, Any],
    activated_tools: Iterable[str] = (),
    exhausted_tools: Iterable[str] = (),
) -> tuple[list[str], str]:
    try:
        parsed = RequestToolsArguments.model_validate(raw_arguments)
    except ValidationError as exc:
        raise ToolExecutionError(
            "request_tools 参数无效：请提供 1～2 个有效分组（groups）与简要操作意图（intent）。"
        ) from exc

    grouped = _available_tools_by_group(registry, runtime, exhausted_tools)
    requested_groups = list(dict.fromkeys(parsed.groups))
    invalid_groups = [group_id for group_id in requested_groups if group_id not in grouped]
    if invalid_groups:
        available_str = "、".join(grouped.keys()) or "无"
        raise ToolExecutionError(
            f"请求的工具分组不可用或不存在：{'、'.join(invalid_groups)}。当前可用分组：{available_str}"
        )

    activated_set = set(activated_tools)
    candidates: list[RegisteredTool] = []
    for group_id in requested_groups:
        unmounted = [tool for tool in grouped[group_id] if tool.name not in activated_set]
        candidates.extend(unmounted or grouped[group_id])

    if len(candidates) <= MAX_MOUNTED_PER_REQUEST:
        selected = candidates
    else:
        corpus = {
            idx: f"{tool.name} {tool.summary} {tool.description} {tool.not_for}"
            for idx, tool in enumerate(candidates)
        }
        scores = _bm25(parsed.intent, corpus)
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda idx: (-scores.get(idx, 0.0), idx),
        )[:MAX_MOUNTED_PER_REQUEST]
        selected = [candidates[idx] for idx in ranked_indices]

    merged_activated = list(dict.fromkeys([*activated_tools, *(tool.name for tool in selected)]))
    lines = ["已挂载以下工具（请结合工具说明与区分提示选择最合适的一个调用；若均不适用可直接 reply）："]
    for tool in selected:
        summary = _tool_summary(tool)
        disambiguation = f" 【注意区分：{tool.not_for}】" if tool.not_for else ""
        lines.append(f"- {tool.name}：{summary}{disambiguation}")
    return merged_activated, "\n".join(lines)
