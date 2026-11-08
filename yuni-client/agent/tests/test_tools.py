from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from yuni_agent.config import StartupConfig
from yuni_agent.memory.models import MemoryCandidate
from yuni_agent.memory.retriever import SqliteMemoryRetriever
from yuni_agent.memory.writer import save_candidate
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import (
    AllowedRoot,
    ChatMessage,
    Companion,
    Conversation,
    Memory,
    MemorySource,
    Reminder,
    ToolCall,
)
from yuni_agent.tools.models import ToolRequest
from yuni_agent.tools.service import TOOL_DEFINITIONS, LocalToolExecutor, ToolExecutionError


def setup_runtime(tmp_path: Path) -> tuple[AgentRuntime, int, int, Path]:
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    root = tmp_path / "allowed"
    root.mkdir()
    with runtime.require_database().session() as session:
        session.add(
            AllowedRoot(
                canonical_path=str(root.resolve()), display_name="allowed", status="active"
            )
        )
        companion = Companion(
            name="Yuni",
            avatar_type="preset",
            avatar_preset_id="mist",
            user_nickname="你",
            personality_tags=["gentle"],
            interaction_styles=["daily_companion"],
            intimacy_level="gentle_natural",
            boundary_text="",
        )
        session.add(companion)
        session.flush()
        conversation = Conversation(companion_id=companion.id)
        session.add(conversation)
        session.flush()
        assistant = ChatMessage(
            conversation_id=conversation.id,
            role="assistant",
            content="",
            status="streaming",
        )
        session.add(assistant)
        session.flush()
        return runtime, conversation.id, assistant.id, root


async def execute_with_decision(
    executor: LocalToolExecutor,
    runtime: AgentRuntime,
    request: ToolRequest,
    conversation_id: int,
    assistant_id: int,
    decision: str,
    user_message_id: int | None = None,
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    task = asyncio.create_task(
        executor.execute(
            request,
            model_call_id="model-call",
            conversation_id=conversation_id,
            assistant_message_id=assistant_id,
            emit=lambda event, data: events.append((event, data)),
            user_message_id=user_message_id,
        )
    )
    for _ in range(100):
        if runtime.tool_decisions:
            break
        await asyncio.sleep(0.01)
    call_id = next(iter(runtime.tool_decisions))
    assert runtime.resolve_tool_decision(call_id, decision)
    return await task, events


@pytest.mark.asyncio
async def test_read_and_confirmed_write_tools(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    executor = LocalToolExecutor(runtime)
    source = root / "notes.txt"
    source.write_text("今天有太阳。", encoding="utf-8")
    try:
        read_events: list[tuple[str, dict[str, Any]]] = []
        read = await executor.execute(
            ToolRequest(name="read_text_file", arguments={"path": str(source)}),
            model_call_id="read-call",
            conversation_id=conversation_id,
            assistant_message_id=assistant_id,
            emit=lambda event, data: read_events.append((event, data)),
        )
        target = root / "journal.md"
        created, write_events = await execute_with_decision(
            executor,
            runtime,
            ToolRequest(
                name="create_text_file",
                arguments={"path": str(target), "content": "# 今天\n很好。"},
            ),
            conversation_id,
            assistant_id,
            "approve",
        )
    finally:
        runtime.close_storage()

    assert "今天有太阳" in read.content
    assert [event for event, _ in read_events] == ["tool_started", "tool_completed"]
    assert created.is_error is False
    assert target.read_text(encoding="utf-8") == "# 今天\n很好。"
    assert [event for event, _ in write_events] == [
        "tool_confirmation_required",
        "tool_started",
        "tool_completed",
    ]


@pytest.mark.asyncio
async def test_rejected_replace_and_confirmed_reminder(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    executor = LocalToolExecutor(runtime)
    source = root / "letter.txt"
    source.write_text("旧内容", encoding="utf-8")
    try:
        rejected, _ = await execute_with_decision(
            executor,
            runtime,
            ToolRequest(
                name="replace_text",
                arguments={"path": str(source), "old_text": "旧内容", "new_text": "新内容"},
            ),
            conversation_id,
            assistant_id,
            "reject",
        )
        remind_at = datetime.now(UTC) + timedelta(hours=1)
        reminder, _ = await execute_with_decision(
            executor,
            runtime,
            ToolRequest(
                name="create_reminder",
                arguments={
                    "title": "早点休息",
                    "note": "别再熬夜了",
                    "remind_at": remind_at.isoformat(),
                },
            ),
            conversation_id,
            assistant_id,
            "approve",
        )
        with runtime.require_database().session() as session:
            stored = session.scalar(select(Reminder))
    finally:
        runtime.close_storage()

    assert "拒绝" in rejected.content
    assert source.read_text(encoding="utf-8") == "旧内容"
    assert reminder.is_error is False
    assert stored is not None and stored.title == "早点休息"


def seed_memory_input(
    runtime: AgentRuntime, conversation_id: int, assistant_id: int,
    content: str = "请记住，我不吃香菜。",
) -> int:
    with runtime.require_database().session() as session:
        message = ChatMessage(
            conversation_id=conversation_id, role="user", content=content, status="succeeded",
        )
        session.add(message)
        session.flush()
        assistant = session.get(ChatMessage, assistant_id)
        assert assistant is not None
        assistant.parent_message_id = message.id
        return message.id


def memory_request(**overrides: Any) -> ToolRequest:
    return ToolRequest(name="save_memory", arguments={
        "type": "user_preference", "subject_key": "food.cilantro", "title": "不吃香菜",
        "content": "用户不吃香菜。", "request_quote": "请记住",
        "source_quote": "我不吃香菜", "sensitivity": "normal", **overrides,
    })


async def execute_memory(
    runtime: AgentRuntime, request: ToolRequest, conversation_id: int,
    assistant_id: int, user_message_id: int,
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    result = await LocalToolExecutor(runtime).execute(
        request, model_call_id="save-call", conversation_id=conversation_id,
        assistant_message_id=assistant_id, user_message_id=user_message_id,
        emit=lambda event, data: events.append((event, data)),
    )
    return result, events


def test_memory_tool_description_requires_explicit_user_intent() -> None:
    definition = next(item["function"] for item in TOOL_DEFINITIONS if item["function"]["name"] == "save_memory")
    assert "仅当当前用户明确表达保存意愿" in definition["description"]
    assert "不要求特定关键词" in definition["description"]
    assert "普通分享" in definition["description"]
    assert "询问记忆内容/保存状态" in definition["description"]
    assert "密码、API Key" in definition["description"]
    assert {"source_quote", "request_quote", "sensitivity"} <= set(definition["parameters"]["required"])


@pytest.mark.asyncio
async def test_save_memory_is_traceable_immediate_and_persistent(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    current_id = seed_memory_input(runtime, conversation_id, assistant_id)
    try:
        first, events = await execute_memory(runtime, memory_request(), conversation_id, assistant_id, current_id)
        second, _ = await execute_memory(runtime, memory_request(), conversation_id, assistant_id, current_id)
        with runtime.require_database().session() as session:
            memory = session.get(Memory, first.details["memoryId"])
            assert memory is not None and memory.content == "用户不吃香菜。"
            assert session.get(MemorySource, (memory.id, current_id)) is not None
            assert session.scalar(select(func.count(Memory.id))) == 1
            assert session.scalar(select(func.count(MemorySource.memory_id))) == 1
            calls = list(session.scalars(select(ToolCall)))
            assert all(call.status == "succeeded" and call.risk == "write" for call in calls)
            assert all(isinstance(call.arguments_json["source_quote"], dict) for call in calls)
        assert first.details["memoryStatus"] == "active" and first.details["outcome"] == "saved"
        assert second.details["memoryId"] == first.details["memoryId"]
        assert second.details["outcome"] == "unchanged"
        assert [event for event, _ in events] == ["tool_started", "tool_completed"]
        assert not runtime.tool_decisions
    finally:
        runtime.close_storage()
    restored = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    restored.initialize_storage()
    try:
        with restored.require_database().session() as session:
            conversation = session.get(Conversation, conversation_id)
            assert conversation is not None
            companion_id = conversation.companion_id
        retrieved = await SqliteMemoryRetriever(restored).retrieve(companion_id, "香菜")
        assert len(retrieved) == 1 and retrieved[0].id == first.details["memoryId"]
    finally:
        restored.close_storage()


@pytest.mark.asyncio
async def test_memory_tool_can_quote_previous_user_fact(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    previous_id = seed_memory_input(runtime, conversation_id, assistant_id, "我不吃香菜。")
    current_id = seed_memory_input(runtime, conversation_id, assistant_id, "把刚才这件事长期记着。")
    try:
        result, _ = await execute_memory(runtime, memory_request(request_quote="把刚才这件事长期记着"), conversation_id, assistant_id, current_id)
        with runtime.require_database().session() as session:
            assert session.get(MemorySource, (result.details["memoryId"], previous_id)) is not None
            assert session.get(MemorySource, (result.details["memoryId"], current_id)) is None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["request", "source", "assistant", "other_conversation", "future", "invalid", "disabled"])
async def test_memory_tool_rejects_invalid_sources_and_disabled_save(tmp_path: Path, case: str) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    current_id = seed_memory_input(runtime, conversation_id, assistant_id)
    request = memory_request()
    try:
        if case == "request":
            request.arguments["request_quote"] = "请永久保存我的生日"
        elif case in {"source", "assistant", "other_conversation", "future"}:
            request.arguments["source_quote"] = "我喜欢西兰花"
            with runtime.require_database().session() as session:
                if case == "assistant":
                    assistant = session.get(ChatMessage, assistant_id)
                    assert assistant is not None
                    assistant.content = "我喜欢西兰花"
                elif case == "other_conversation":
                    original = session.get(Conversation, conversation_id)
                    assert original is not None
                    other = Conversation(companion_id=original.companion_id)
                    session.add(other)
                    session.flush()
                    session.add(ChatMessage(conversation_id=other.id, role="user", content="我喜欢西兰花", status="succeeded"))
                elif case == "future":
                    session.add(ChatMessage(conversation_id=conversation_id, role="user", content="我喜欢西兰花", status="succeeded"))
        elif case == "invalid":
            request.arguments["type"] = "invented_type"
        else:
            runtime.memory_generation_enabled = False
            assert all(item["function"]["name"] != "save_memory" for item in LocalToolExecutor(runtime).definitions)
        with pytest.raises(ToolExecutionError):
            await execute_memory(runtime, request, conversation_id, assistant_id, current_id)
        with runtime.require_database().session() as session:
            assert session.scalar(select(func.count(Memory.id))) == 0
            assert session.scalar(select(func.count(ToolCall.id))) == 0
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,expected", [("approve", 1), ("reject", 0)])
async def test_sensitive_memory_requires_confirmation(tmp_path: Path, decision: str, expected: int) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    current_id = seed_memory_input(runtime, conversation_id, assistant_id)
    try:
        result, events = await execute_with_decision(
            LocalToolExecutor(runtime), runtime, memory_request(sensitivity="sensitive"),
            conversation_id, assistant_id, decision, current_id,
        )
        assert events[0][0] == "tool_confirmation_required"
        with runtime.require_database().session() as session:
            assert session.scalar(select(func.count(Memory.id))) == expected
            if expected:
                memory = session.get(Memory, result.details["memoryId"])
                assert memory is not None and memory.sensitivity == "sensitive"
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_sensitive_save_rechecks_setting_after_confirmation(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    current_id = seed_memory_input(runtime, conversation_id, assistant_id)
    try:
        def emit(event: str, _: Any) -> None:
            if event == "tool_started":
                runtime.memory_generation_enabled = False
        task = asyncio.create_task(LocalToolExecutor(runtime).execute(
            memory_request(sensitivity="sensitive"), model_call_id="sensitive-call",
            conversation_id=conversation_id, assistant_message_id=assistant_id,
            user_message_id=current_id, emit=emit,
        ))
        for _ in range(100):
            if runtime.tool_decisions:
                break
            await asyncio.sleep(0.01)
        assert runtime.resolve_tool_decision(next(iter(runtime.tool_decisions)), "approve")
        result = await task
        assert result.is_error and "关闭" in result.content
        with runtime.require_database().session() as session:
            assert session.scalar(select(func.count(Memory.id))) == 0
            call = session.scalar(select(ToolCall))
            assert call is not None and call.status == "failed"
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_revoked_root_after_preview_blocks_confirmed_write(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    target = root / "after-revoke.txt"
    try:
        def emit(event: str, _: Any) -> None:
            if event == "tool_confirmation_required":
                with runtime.require_database().session() as session:
                    allowed = session.scalar(
                        select(AllowedRoot).where(AllowedRoot.canonical_path == str(root.resolve()))
                    )
                    assert allowed is not None
                    allowed.status = "revoked"

        task = asyncio.create_task(LocalToolExecutor(runtime).execute(
            ToolRequest(name="create_text_file", arguments={"path": str(target), "content": "should not write"}),
            model_call_id="revoke-call",
            conversation_id=conversation_id,
            assistant_message_id=assistant_id,
            emit=emit,
        ))
        for _ in range(100):
            if runtime.tool_decisions:
                break
            await asyncio.sleep(0.01)
        assert runtime.resolve_tool_decision(next(iter(runtime.tool_decisions)), "approve")
        result = await task
        assert result.is_error and "授权目录" in result.content
        assert not target.exists()
        with runtime.require_database().session() as session:
            call = session.scalar(select(ToolCall))
            assert call is not None and call.status == "failed" and "授权目录" in (call.error_message or "")
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_memory_tool_and_background_writer_share_conflict_and_delete_rules(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    current_id = seed_memory_input(runtime, conversation_id, assistant_id)
    try:
        first, _ = await execute_memory(runtime, memory_request(), conversation_id, assistant_id, current_id)
        with runtime.require_database().session() as session:
            conversation = session.get(Conversation, conversation_id)
            assert conversation is not None
            companion_id = conversation.companion_id
            result = save_candidate(session, companion_id, MemoryCandidate(
                type="user_preference", subject_key="food.cilantro", title="不吃香菜",
                content="用户不吃香菜。", importance=3, confidence=0.95,
                source_message_ids=[current_id],
            ))
            assert result.memory_id == first.details["memoryId"] and result.outcome == "unchanged"
        new_id = seed_memory_input(runtime, conversation_id, assistant_id, "请记住，我现在喜欢吃香菜。")
        conflict, _ = await execute_memory(runtime, memory_request(
            content="用户现在喜欢吃香菜。", source_quote="我现在喜欢吃香菜",
        ), conversation_id, assistant_id, new_id)
        assert conflict.details["memoryStatus"] == "conflicted"
        assert "尚未成为可用记忆" in conflict.content
        assert await SqliteMemoryRetriever(runtime).retrieve(companion_id, "香菜") == []
        with runtime.require_database().session() as session:
            old = session.get(Memory, first.details["memoryId"])
            assert old is not None and old.content == "用户不吃香菜。"
            old.deleted_at = datetime.now(UTC)
        restored, _ = await execute_memory(runtime, memory_request(
            content="用户现在喜欢吃香菜。", source_quote="我现在喜欢吃香菜",
        ), conversation_id, assistant_id, new_id)
        assert restored.details["memoryId"] != first.details["memoryId"]
        # Deleting one branch does not constitute an explicit conflict resolution.
        assert restored.details["memoryStatus"] == "conflicted"
    finally:
        runtime.close_storage()


# --------------------------------------------------------------------------- registry / framework

def test_registry_specs_drive_definitions_and_risk() -> None:
    from yuni_agent.tools.models import ToolRisk
    from yuni_agent.tools.service import TOOL_REGISTRY
    names = [tool.name for tool in TOOL_REGISTRY.list()]
    assert names == ["query_memory", "save_memory", "find_files", "read_text_file",
                     "create_text_file", "replace_text", "create_reminder", "generate_image"]
    assert [item["function"]["name"] for item in TOOL_DEFINITIONS] == names
    risks = {tool.name: tool.risk for tool in TOOL_REGISTRY.list()}
    assert {name for name, risk in risks.items() if risk == ToolRisk.READ} == {"query_memory", "find_files", "read_text_file"}
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_DEFINITIONS}
    assert schemas["replace_text"]["required"] == ["path", "old_text", "new_text"]
    assert schemas["create_reminder"]["required"] == ["title", "remind_at"]
    assert "evidence_quote" not in schemas["save_memory"]["properties"]


def test_registry_rejects_duplicates() -> None:
    from yuni_agent.tools.registry import ToolRegistry
    from yuni_agent.tools.service import TOOL_REGISTRY
    registry = ToolRegistry(TOOL_REGISTRY.list())
    with pytest.raises(ValueError):
        registry.register(TOOL_REGISTRY.list()[0])


@pytest.mark.asyncio
async def test_unconfigured_image_tool_is_hidden_and_rejected(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    executor = LocalToolExecutor(runtime)
    try:
        assert "generate_image" not in {item["function"]["name"] for item in executor.definitions}
        with pytest.raises(ToolExecutionError, match="未配置"):
            await executor.execute(
                ToolRequest(name="generate_image", arguments={"prompt": "a mountain"}),
                model_call_id="unconfigured-image", conversation_id=conversation_id,
                assistant_message_id=assistant_id, emit=lambda *_: None,
            )
        with runtime.require_database().session() as session:
            assert session.scalar(select(func.count(ToolCall.id))) == 0
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments,message", [
    ("read_text_file", {}, "path"),
    ("read_text_file", {"path": 123}, "path"),
    ("create_text_file", {"path": "x"}, "content"),
    ("replace_text", {"path": "x", "old_text": "a"}, "new_text"),
    ("create_reminder", {"title": "  ", "remind_at": "2099-01-01T00:00:00+08:00"}, "参数无效"),
    ("create_reminder", {"title": "喝水", "remind_at": "2099-01-01T00:00:00"}, "时区"),
    ("unknown_tool", {}, "不支持"),
])
async def test_invalid_arguments_fail_before_any_tool_call(tmp_path: Path, name: str, arguments: dict, message: str) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    try:
        with pytest.raises(ToolExecutionError, match=message):
            await LocalToolExecutor(runtime).execute(
                ToolRequest(name=name, arguments=arguments), model_call_id="bad", conversation_id=conversation_id,
                assistant_message_id=assistant_id, emit=lambda *_: None,
            )
        with runtime.require_database().session() as session:
            assert session.scalar(select(func.count(ToolCall.id))) == 0
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_paths_outside_roots_and_existing_targets_are_rejected(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "exists.txt").write_text("x", encoding="utf-8")
    executor = LocalToolExecutor(runtime)
    try:
        for request, message in [
            (ToolRequest(name="read_text_file", arguments={"path": str(outside)}), "授权目录"),
            # Relative paths live in the workspace, so they cannot climb out of it.
            # workspace is <tmp>/data/workspace, so this is the existing <tmp>/secret.txt.
            (ToolRequest(name="read_text_file", arguments={"path": "../../secret.txt"}), "授权目录"),
            (ToolRequest(name="read_text_file", arguments={"path": "\\rooted.txt"}), "绝对路径"),
            (ToolRequest(name="create_text_file", arguments={"path": str(root / "exists.txt"), "content": "y"}), "已经存在"),
        ]:
            with pytest.raises(ToolExecutionError, match=message):
                await executor.execute(request, model_call_id="p", conversation_id=conversation_id,
                                       assistant_message_id=assistant_id, emit=lambda *_: None)
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_relative_paths_resolve_in_default_workspace(tmp_path: Path) -> None:
    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    workspace = runtime.config.workspace_path
    executor = LocalToolExecutor(runtime)
    # create_text_file requires the parent folder to exist already.
    (workspace / "notes").mkdir(parents=True)
    try:
        created, _ = await execute_with_decision(
            executor, runtime,
            ToolRequest(name="create_text_file", arguments={"path": "notes/今天.md", "content": "晴"}),
            conversation_id, assistant_id, "approve",
        )
        read = await executor.execute(
            ToolRequest(name="read_text_file", arguments={"path": "notes/今天.md"}), model_call_id="r",
            conversation_id=conversation_id, assistant_message_id=assistant_id, emit=lambda *_: None,
        )
        found = await executor.execute(
            ToolRequest(name="find_files", arguments={"query": "今天"}), model_call_id="f",
            conversation_id=conversation_id, assistant_message_id=assistant_id, emit=lambda *_: None,
        )
    finally:
        runtime.close_storage()
    assert created is not None and not created.is_error
    assert (workspace / "notes" / "今天.md").read_text(encoding="utf-8") == "晴"
    assert "晴" in read.content
    assert str((workspace / "notes" / "今天.md").resolve()) in found.content


def test_planner_sees_default_workspace() -> None:
    from yuni_agent.context.planner_builder import build_planner_context
    messages = build_planner_context([], "hi", allowed_roots=["D:/w"], default_workspace="D:/w")
    assert '"defaultWorkspace": "D:/w"' in messages[1]["content"]
    assert "defaultWorkspace" not in build_planner_context([], "hi", allowed_roots=[])[1]["content"]


@pytest.mark.asyncio
async def test_find_files_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from yuni_agent.tools import builtin
    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    for index in range(30):
        (root / f"note-{index:02}.txt").write_text("hello" if index % 2 else "bye", encoding="utf-8")
    deep = root
    for level in range(5):
        deep = deep / f"d{level}"
        deep.mkdir()
    (deep / "deep-hello.txt").write_text("hello", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "hello.txt").write_text("hello", encoding="utf-8")
    executor = LocalToolExecutor(runtime)

    async def find(query: str):
        return await executor.execute(
            ToolRequest(name="find_files", arguments={"query": query}), model_call_id="f",
            conversation_id=conversation_id, assistant_message_id=assistant_id, emit=lambda *_: None,
        )
    try:
        full = await find("hello")
        assert full.details["count"] == 16 and not full.details["truncated"]
        assert "node_modules" not in full.content
        monkeypatch.setattr(builtin, "FIND_MAX_DEPTH", 3)
        shallow = await find("hello")
        assert "deep-hello" not in shallow.content
        monkeypatch.setattr(builtin, "FIND_MAX_VISITED_FILES", 10)
        capped = await find("")
        assert capped.details["truncated"] and capped.details["visited"] == 10
        assert "已截断" in capped.content
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_blocking_tools_run_off_loop_and_time_out(tmp_path: Path) -> None:
    import threading
    import time
    from dataclasses import replace

    from yuni_agent.tools.models import ToolResult
    from yuni_agent.tools.registry import ToolRegistry
    from yuni_agent.tools.service import TOOL_REGISTRY
    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    (root / "a.txt").write_text("a", encoding="utf-8")
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def slow(_context, _arguments):
        seen.append(threading.get_ident())
        time.sleep(0.5)
        return ToolResult(content="late")
    registry = ToolRegistry([replace(TOOL_REGISTRY.require("read_text_file"), run=slow, timeout=0.1)])
    events: list[str] = []
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1
    task = asyncio.create_task(ticker())
    try:
        result = await LocalToolExecutor(runtime, registry).execute(
            ToolRequest(name="read_text_file", arguments={"path": str(root / "a.txt")}), model_call_id="t",
            conversation_id=conversation_id, assistant_message_id=assistant_id,
            emit=lambda event, _: events.append(event),
        )
        with runtime.require_database().session() as session:
            call = session.scalar(select(ToolCall))
            assert call is not None and call.status == "failed" and "超时" in (call.error_message or "")
    finally:
        task.cancel()
        runtime.close_storage()
    assert result.is_error and "超时" in result.content
    assert seen and seen[0] != loop_thread
    assert ticks >= 3
    assert events == ["tool_started", "tool_completed"]


@pytest.mark.asyncio
async def test_preview_and_locked_save_memory_time_out(tmp_path: Path) -> None:
    import time
    from dataclasses import replace

    from yuni_agent.tools.registry import ToolPreview, ToolRegistry
    from yuni_agent.tools.service import TOOL_REGISTRY

    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    (root / "a.txt").write_text("a", encoding="utf-8")
    current_id = seed_memory_input(runtime, conversation_id, assistant_id)

    def slow_prepare(_context, _arguments):
        time.sleep(0.5)
        return ToolPreview(summary="late preview")

    preview_registry = ToolRegistry([
        replace(TOOL_REGISTRY.require("read_text_file"), prepare=slow_prepare, timeout=0.1),
        replace(TOOL_REGISTRY.require("save_memory"), timeout=0.1),
    ])
    executor = LocalToolExecutor(runtime, preview_registry)
    try:
        preview_result = await executor.execute(
            ToolRequest(name="read_text_file", arguments={"path": str(root / "a.txt")}),
            model_call_id="preview-timeout",
            conversation_id=conversation_id,
            assistant_message_id=assistant_id,
            emit=lambda *_: None,
        )
        assert preview_result.is_error and "超时" in preview_result.content

        async with runtime.maintenance_lock:
            save_result = await executor.execute(
                memory_request(),
                model_call_id="save-timeout",
                conversation_id=conversation_id,
                assistant_message_id=assistant_id,
                user_message_id=current_id,
                emit=lambda *_: None,
            )
        assert save_result.is_error and "超时" in save_result.content
        with runtime.require_database().session() as session:
            assert session.scalar(select(func.count(Memory.id))) == 0
            calls = list(session.scalars(select(ToolCall)))
            assert len(calls) == 2 and all(call.status == "failed" and "超时" in (call.error_message or "") for call in calls)
    finally:
        runtime.close_storage()


# --------------------------------------------------------------------------- declarative tool router (TOOL-004)


def test_initial_planner_definitions_expose_only_request_tools_and_dynamic_groups(tmp_path: Path) -> None:
    from yuni_agent.model.auth import ImageAuth

    runtime, _, _, _ = setup_runtime(tmp_path)
    executor = LocalToolExecutor(runtime)
    try:
        initial = executor.planner_definitions([], set())
        assert [item["function"]["name"] for item in initial] == ["request_tools"]
        enum_groups = initial[0]["function"]["parameters"]["properties"]["groups"]["items"]["enum"]
        assert enum_groups == ["memory", "files", "reminder"]
        assert "image" not in enum_groups

        runtime.save_image_auth(ImageAuth(enabled=True, url="https://img.example.com/v1", model="flux"), "img-key")
        with_image = executor.planner_definitions([], set())
        assert [item["function"]["name"] for item in with_image] == ["request_tools"]
        enum_with_image = with_image[0]["function"]["parameters"]["properties"]["groups"]["items"]["enum"]
        assert enum_with_image == ["memory", "files", "reminder", "image"]

        all_names = [item["function"]["name"] for item in executor.definitions]
        fully_mounted = executor.planner_definitions(all_names, set())
        assert [item["function"]["name"] for item in fully_mounted] == all_names
        assert "request_tools" not in {item["function"]["name"] for item in fully_mounted}
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_planner_routes_via_request_tools_with_disambiguation_and_no_audit_row(tmp_path: Path) -> None:
    import json

    from planner_helpers import reply_event, stop_event

    from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
    from yuni_agent.context.builder import SqliteContextBuilder
    from yuni_agent.model.client import ModelStreamEvent, ModelToolCall

    runtime, conversation_id, assistant_id, root = setup_runtime(tmp_path)
    note_file = root / "diary.txt"
    note_file.write_text("春天的晚风很舒服。", encoding="utf-8")
    user_id = seed_memory_input(runtime, conversation_id, assistant_id, "帮我看看 diary.txt 里写了什么")
    seen_tool_rounds: list[list[str]] = []
    seen_receipts: list[dict[str, Any]] = []

    class RoutingModel:
        step = 0

        async def stream_chat_with_tools(self, messages, tools):
            seen_tool_rounds.append([item["function"]["name"] for item in tools])
            self.step += 1
            if self.step == 1:
                assert seen_tool_rounds[-1] == ["request_tools", "stop", "reply"]
                yield ModelStreamEvent(
                    type="tool_call",
                    tool_call=ModelToolCall(
                        id="route-files",
                        name="request_tools",
                        arguments={"groups": ["files"], "intent": "读取并查看本地笔记文件"},
                    ),
                )
            elif self.step == 2:
                assert {"find_files", "read_text_file", "create_text_file", "replace_text"} <= set(seen_tool_rounds[-1])
                tool_msg = next(m for m in reversed(messages) if m.get("role") == "tool")
                payload = json.loads(tool_msg["content"])
                seen_receipts.append(payload)
                yield ModelStreamEvent(
                    type="tool_call",
                    tool_call=ModelToolCall(
                        id="read-diary",
                        name="read_text_file",
                        arguments={"path": str(note_file)},
                    ),
                )
            elif self.step == 3:
                yield reply_event("告诉用户日记内容")
            else:
                yield stop_event()

        async def stream_chat(self, _messages):
            yield "看到啦，里面写着春天的晚风很舒服～"

    try:
        events = [
            event
            async for event in LangGraphChat().stream(
                {
                    "conversation_id": conversation_id,
                    "user_message_id": user_id,
                    "assistant_message_id": assistant_id,
                    "current_input": "帮我看看 diary.txt 里写了什么",
                },
                ChatGraphContext(
                    context_builder=SqliteContextBuilder(runtime),
                    model_client=RoutingModel(),
                    tool_executor=LocalToolExecutor(runtime),
                ),
            )
        ]
        with runtime.require_database().session() as session:
            audit_rows = list(session.scalars(select(ToolCall)))
    finally:
        runtime.close_storage()

    assert len(seen_receipts) == 1
    assert seen_receipts[0]["toolName"] == "request_tools"
    assert seen_receipts[0]["status"] == "mounted"
    assert "【注意区分：" in seen_receipts[0]["content"]
    assert "find_files" in seen_receipts[0]["content"] and "replace_text" in seen_receipts[0]["content"]
    assert len(audit_rows) == 1 and audit_rows[0].tool_name == "read_text_file"
    assert events[-1].data["content"] == "看到啦，里面写着春天的晚风很舒服～"


def test_resolve_tool_request_bm25_top_k_filtering_when_group_exceeds_four() -> None:
    from pydantic import BaseModel

    from yuni_agent.tools.models import ToolResult, ToolRisk
    from yuni_agent.tools.registry import RegisteredTool, ToolPreview, ToolRegistry
    from yuni_agent.tools.router import resolve_tool_request

    class DummyArgs(BaseModel):
        query: str = ""

    specs = [
        ("note_search", "搜索读书笔记与摘抄标签", "修改笔记请用 note_update"),
        ("note_create", "新建空白读书笔记卡片", "已有笔记追加请用 note_append"),
        ("note_update", "修改读书笔记的正文段落", "删除笔记请用 note_Archive"),
        ("note_export_pdf", "将读书笔记导出为排版精美的 PDF 文档并打印", "普通查看请用 note_search"),
        ("calendar_sync", "同步日历行程与会议安排表", "定时闹钟请用 create_reminder"),
        ("music_playlist", "检索并管理伴侣歌单与常听音乐曲目", "播放控制请用 music_player"),
        ("recipe_lookup", "查询家常菜谱食材配料与烹饪步骤", "点外卖记录请用 food_log"),
    ]
    registry = ToolRegistry([
        RegisteredTool(
            name=name,
            description=f"{summary}。详细说明：{name}",
            arguments=DummyArgs,
            risk=ToolRisk.READ,
            prepare=lambda _ctx, _args: ToolPreview(summary="preview"),
            run=lambda _ctx, _args: ToolResult(content="ok"),
            group="life_assistant",
            summary=summary,
            not_for=not_for,
        )
        for name, summary, not_for in specs
    ])

    mounted, receipt = resolve_tool_request(
        registry,
        None,
        {"groups": ["life_assistant"], "intent": "把我的读书笔记导出成 PDF 文档打印"},
    )
    assert len(mounted) == 4
    assert mounted[0] == "note_export_pdf"
    assert "note_export_pdf" in receipt and "【注意区分：普通查看请用 note_search】" in receipt


@pytest.mark.asyncio
async def test_declarative_custom_tool_auto_registers_group_prior_reply_and_redaction(tmp_path: Path) -> None:
    from planner_helpers import reply_event, stop_event
    from pydantic import BaseModel, Field

    from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
    from yuni_agent.context.builder import SqliteContextBuilder
    from yuni_agent.model.client import ModelStreamEvent, ModelToolCall
    from yuni_agent.tools.models import ToolResult, ToolRisk
    from yuni_agent.tools.registry import RegisteredTool, ToolPreview, ToolRegistry
    from yuni_agent.tools.service import TOOL_REGISTRY

    class CustomPackArgs(BaseModel):
        title: str = Field(min_length=1)
        secret_text: str = Field(min_length=1)

    custom_tool = RegisteredTool(
        name="seal_time_capsule",
        description="封存一段需要耗时加密的时间胶囊寄语。",
        arguments=CustomPackArgs,
        risk=ToolRisk.READ,
        prepare=lambda _ctx, args: ToolPreview(summary=f"封存胶囊：{args.title}", requires_confirmation=False),
        run=lambda _ctx, args: ToolResult(content=f"已封存：{args.title}"),
        group="custom_pack",
        summary="封存加密时间胶囊寄语",
        not_for="普通长期记忆请用 save_memory",
        requires_prior_reply=True,
        redacted_fields=("secret_text",),
    )

    runtime, conversation_id, assistant_id, _ = setup_runtime(tmp_path)
    user_id = seed_memory_input(runtime, conversation_id, assistant_id, "帮我封存一个时间胶囊")
    registry = ToolRegistry([*TOOL_REGISTRY.list(), custom_tool])
    executor = LocalToolExecutor(runtime, registry)

    initial = executor.planner_definitions([], set())
    enum_groups = initial[0]["function"]["parameters"]["properties"]["groups"]["items"]["enum"]
    assert "custom_pack" in enum_groups

    prior_reply_blocked = False

    class CustomFlowModel:
        step = 0

        async def stream_chat_with_tools(self, messages, tools):
            nonlocal prior_reply_blocked
            self.step += 1
            if self.step == 1:
                yield ModelStreamEvent(
                    type="tool_call",
                    tool_call=ModelToolCall(
                        id="req-custom",
                        name="request_tools",
                        arguments={"groups": ["custom_pack"], "intent": "封存时间胶囊"},
                    ),
                )
            elif self.step == 2:
                assert "seal_time_capsule" in {item["function"]["name"] for item in tools}
                # Try calling before any reply -> must be intercepted by requires_prior_reply!
                yield ModelStreamEvent(
                    type="tool_call",
                    tool_call=ModelToolCall(
                        id="call-early",
                        name="seal_time_capsule",
                        arguments={"title": "给未来的信", "secret_text": "这是不能明文落库的秘密寄语"},
                    ),
                )
            elif self.step == 3:
                if any("必须先 reply" in str(m.get("content") or "") for m in messages if m.get("role") == "system"):
                    prior_reply_blocked = True
                yield reply_event("先回应用户正在准备封存")
            elif self.step == 4:
                yield ModelStreamEvent(
                    type="tool_call",
                    tool_call=ModelToolCall(
                        id="call-after-reply",
                        name="seal_time_capsule",
                        arguments={"title": "给未来的信", "secret_text": "这是不能明文落库的秘密寄语"},
                    ),
                )
            elif self.step == 5:
                yield reply_event("告诉用户已经封存完成")
            else:
                yield stop_event()

        async def stream_chat(self, _messages):
            yield "好呀，帮你封存好啦～"

    try:
        events = [
            event
            async for event in LangGraphChat().stream(
                {
                    "conversation_id": conversation_id,
                    "user_message_id": user_id,
                    "assistant_message_id": assistant_id,
                    "current_input": "帮我封存一个时间胶囊",
                },
                ChatGraphContext(
                    context_builder=SqliteContextBuilder(runtime),
                    model_client=CustomFlowModel(),
                    tool_executor=executor,
                ),
            )
        ]
        with runtime.require_database().session() as session:
            calls = list(session.scalars(select(ToolCall).where(ToolCall.tool_name == "seal_time_capsule")))
    finally:
        runtime.close_storage()

    assert prior_reply_blocked is True
    assert len(calls) == 1
    assert calls[0].arguments_json["title"] == "给未来的信"
    assert isinstance(calls[0].arguments_json["secret_text"], dict)
    assert "sha256" in calls[0].arguments_json["secret_text"]
    assert "这是不能明文落库的秘密寄语" not in str(calls[0].arguments_json)
    assert events[-1].type.value == "completed"


@pytest.mark.asyncio
async def test_files_and_memory_capability_gating_in_router_and_prompts(tmp_path: Path) -> None:
    from yuni_agent.context.builder import SqliteContextBuilder
    from yuni_agent.context.models import ContextRequest

    runtime, conversation_id, _, _ = setup_runtime(tmp_path)
    try:
        executor = LocalToolExecutor(runtime)
        initial = executor.planner_definitions([], set())
        groups = initial[0]["function"]["parameters"]["properties"]["groups"]["items"]["enum"]
        assert "files" in groups and "memory" in groups

        # Remove allowed roots and disable memory -> neither files nor memory should appear in router or prompts.
        with runtime.require_database().session() as session:
            for row in session.scalars(select(AllowedRoot)):
                session.delete(row)
        runtime.memory_generation_enabled = False
        runtime.memory_retrieval_enabled = False

        gated = executor.planner_definitions([], set())
        gated_groups = gated[0]["function"]["parameters"]["properties"]["groups"]["items"]["enum"]
        assert "files" not in gated_groups
        assert "memory" not in gated_groups
        assert "reminder" in gated_groups

        built = await SqliteContextBuilder(runtime).build(
            ContextRequest(
                conversation_id=conversation_id,
                current_user_message_id=999,
                current_input="你好呀",
            )
        )
        planner_sys = "\n".join(item["content"] for item in built.tool_messages if item["role"] == "system")
        assert "save_memory" not in planner_sys
        assert "query_memory" not in planner_sys
        assert "allowedRoots" not in planner_sys
        assert "【当前语音能力】" not in planner_sys
    finally:
        runtime.close_storage()

