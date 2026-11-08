"""Local reminder tool pack (`create_reminder`)."""
from __future__ import annotations

from datetime import UTC, datetime

from yuni_agent.api.models import iso_utc
from yuni_agent.storage.models import Reminder
from yuni_agent.tools.errors import ToolExecutionError
from yuni_agent.tools.models import CreateReminderArguments, ToolResult, ToolRisk
from yuni_agent.tools.registry import RegisteredTool, ToolContext, ToolPreview


def parse_remind_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolExecutionError("提醒时间格式无效") from exc
    if parsed.tzinfo is None:
        raise ToolExecutionError("提醒时间必须包含时区")
    parsed = parsed.astimezone(UTC)
    if parsed <= datetime.now(UTC):
        raise ToolExecutionError("提醒时间必须晚于当前时间")
    return parsed


def _prepare_create_reminder(_context: ToolContext, arguments: CreateReminderArguments) -> ToolPreview:
    remind_at = parse_remind_at(arguments.remind_at)
    return ToolPreview(summary=f"创建提醒：{arguments.title.strip()}", diff_preview=f"提醒时间：{iso_utc(remind_at)}")


def _create_reminder(context: ToolContext, arguments: CreateReminderArguments) -> ToolResult:
    remind_at = parse_remind_at(arguments.remind_at)
    with context.runtime.require_database().session() as session:
        reminder = Reminder(title=arguments.title.strip()[:200], note=arguments.note[:2000],
                            remind_at=remind_at, status="pending")
        session.add(reminder)
        session.flush()
        reminder_id = reminder.id
    return ToolResult(content=f"提醒已创建，时间：{iso_utc(remind_at)}", details={"reminderId": reminder_id})


REMINDER_TOOLS: list[RegisteredTool] = [
    RegisteredTool(
        name="create_reminder",
        description="创建本地提醒。时间必须是带时区的 ISO 8601 时间，执行前获得用户确认。",
        arguments=CreateReminderArguments,
        risk=ToolRisk.WRITE,
        prepare=_prepare_create_reminder,
        run=_create_reminder,
        group="reminder",
        summary="创建到指定时间触发的本地定时提醒",
        not_for="不需要到点提醒、仅需长期记住的用户事实或偏好请用 save_memory",
    ),
]
