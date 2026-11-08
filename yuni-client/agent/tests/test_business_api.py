from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from yuni_agent.api.app import create_app
from yuni_agent.config import StartupConfig
from yuni_agent.context.builder import SqliteContextBuilder, _system_prompt
from yuni_agent.context.models import ContextRequest
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import ChatMessage, Companion, Conversation, Memory


def make_client(tmp_path: Path) -> tuple[TestClient, AgentRuntime, dict[str, str]]:
    config = StartupConfig.for_test(tmp_path)
    runtime = AgentRuntime(config=config)
    headers = {"X-Yuni-Session": config.session_token.get_secret_value()}
    return TestClient(create_app(runtime)), runtime, headers


def partner_payload() -> dict[str, object]:
    return {
        "name": "林夏",
        "avatarType": "preset",
        "avatarPresetId": "mist",
        "avatarFileId": None,
        "avatarUrl": "mock-avatar://mist",
        "userNickname": "小予",
        "personalityTags": ["gentle", "reliable"],
        "interactionStyles": ["daily_companion", "emotional_comfort"],
        "intimacyLevel": "gentle_natural",
        "boundaryText": "难过时先陪伴，再询问是否需要建议。",
    }


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


def test_companion_and_settings_persist(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        bootstrap = client.get("/local/bootstrap", headers=headers)
        created = client.post("/api/partner", headers=headers, json=partner_payload())
        settings = client.put(
            "/api/settings/memory", headers=headers, json={"memoryEnabled": False}
        )

    assert bootstrap.json()["data"]["initialized"] is True
    assert bootstrap.json()["data"]["hasCompanion"] is False
    assert created.status_code == 201
    assert created.json()["data"]["avatarUrl"] == "mock-avatar://mist"
    assert created.json()["data"]["gender"] is None
    assert settings.json()["data"]["memoryEnabled"] is False

    restarted_client, _, restarted_headers = make_client(tmp_path)
    with restarted_client:
        restarted = restarted_client.get("/local/bootstrap", headers=restarted_headers)
        companion = restarted_client.get("/api/partner", headers=restarted_headers)
        persisted_settings = restarted_client.get("/api/settings", headers=restarted_headers)

    assert restarted.json()["data"]["hasCompanion"] is True
    assert companion.json()["data"]["name"] == "林夏"
    assert persisted_settings.json()["data"]["memoryEnabled"] is False


@pytest.mark.parametrize("gender", ["male", "female"])
def test_companion_gender_create_edit_and_restart(tmp_path: Path, gender: str) -> None:
    client, _, headers = make_client(tmp_path)
    other_gender = "female" if gender == "male" else "male"
    with client:
        created = client.post(
            "/api/partner", headers=headers, json={**partner_payload(), "gender": gender}
        )
        assert created.status_code == 201
        assert created.json()["data"]["gender"] == gender
        invalid = client.put(
            "/api/partner", headers=headers, json={**partner_payload(), "gender": "invalid"}
        )
        assert invalid.status_code == 422
        updated = client.put(
            "/api/partner", headers=headers, json={**partner_payload(), "gender": other_gender}
        )
        assert updated.json()["data"]["gender"] == other_gender
        legacy_update = client.put("/api/partner", headers=headers, json=partner_payload())
        assert legacy_update.json()["data"]["gender"] == other_gender

    restarted, _, restarted_headers = make_client(tmp_path)
    with restarted:
        assert restarted.get("/api/partner", headers=restarted_headers).json()["data"]["gender"] == other_gender
        cleared = restarted.put(
            "/api/partner", headers=restarted_headers, json={**partner_payload(), "gender": None}
        )
        assert cleared.json()["data"]["gender"] is None


def test_invalid_gender_cannot_create_companion(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        response = client.post(
            "/api/partner", headers=headers, json={**partner_payload(), "gender": "invalid"}
        )
        assert response.status_code == 422
        assert client.get("/api/partner", headers=headers).json()["data"] is None


@pytest.mark.parametrize("gender", ["male", "female", None])
def test_gender_backup_compatibility(tmp_path: Path, gender: str | None) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json={**partner_payload(), "gender": gender})
        backup = client.get("/api/data/export", headers=headers).json()
        assert backup["tables"]["companions"][0]["gender"] == gender
        if gender is None:
            del backup["tables"]["companions"][0]["gender"]
        imported = client.post(
            "/api/data/import", headers=headers,
            files={"file": ("backup.json", json.dumps(backup).encode(), "application/json")},
        )
        assert imported.status_code == 200
        assert client.get("/api/partner", headers=headers).json()["data"]["gender"] == gender
        backup["tables"]["companions"][0]["gender"] = "invalid"
        denied = client.post(
            "/api/data/import", headers=headers,
            files={"file": ("invalid.json", json.dumps(backup).encode(), "application/json")},
        )
        assert denied.status_code == 422
        assert client.get("/api/partner", headers=headers).json()["data"]["gender"] == gender


def test_legacy_safety_chat_is_preserved_and_idempotently_replayed(tmp_path: Path) -> None:
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        payload = {"content": "我不想活了", "clientMessageId": "same-client-message-id"}
        # Existing persisted replies must remain readable and replayable after
        # removing input classification; no model or safety service is needed.
        with runtime.require_database().session() as session:
            companion_id = session.scalar(select(Companion.id))
            conversation = session.scalar(select(Conversation).where(Conversation.companion_id == companion_id))
            assert conversation is not None
            user = ChatMessage(
                conversation_id=conversation.id, role="user", content=payload["content"],
                client_message_id=payload["clientMessageId"], status="succeeded",
            )
            session.add(user)
            session.flush()
            session.add(ChatMessage(
                conversation_id=conversation.id, role="assistant", content="已有历史回复。",
                status="succeeded", message_type="safety", parent_message_id=user.id,
                help_resources=[{"title": "历史帮助资源", "description": "旧版本保存的内容"}],
            ))
        first = client.post("/api/chat/messages/stream", headers=headers, json=payload)
        replay = client.post("/api/chat/messages/stream", headers=headers, json=payload)
        history = client.get("/api/chat/messages?limit=30", headers=headers)

    first_events = parse_sse(first.text)
    replay_events = parse_sse(replay.text)
    assert first_events[0][0] == "message_created"
    assert first_events[-1][0] == "completed"
    assert first_events[-1][1]["messageType"] == "safety"
    assert replay_events == [
        ("message_created", first_events[0][1]),
        ("completed", first_events[-1][1]),
    ]

    items = history.json()["data"]["items"]
    assert [item["role"] for item in items] == ["user", "assistant"]
    assert items[1]["messageType"] == "safety"
    assert items[1]["replyParts"] == [items[1]["content"]]
    assert items[1]["helpResources"]
    assert items[1]["parentMessageId"] == items[0]["id"]


def test_memory_update_and_soft_delete(tmp_path: Path) -> None:
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        with runtime.require_database().session() as session:
            companion_id = session.query(Companion.id).scalar()
            memory = Memory(
                companion_id=companion_id,
                type="user_preference",
                subject_key="food.cilantro",
                title="不喜欢香菜",
                content="用户不喜欢香菜。",
                importance=4,
            )
            session.add(memory)
            session.flush()
            memory_id = memory.id

        listed = client.get("/api/memories?limit=50", headers=headers)
        updated = client.put(
            f"/api/memories/{memory_id}",
            headers=headers,
            json={
                "type": "user_preference",
                "title": "不吃香菜",
                "content": "用户明确表示不吃香菜。",
                "importance": 5,
            },
        )
        updated_id = updated.json()["data"]["id"]
        assert updated_id != memory_id  # Correction preserves the retracted version.
        removed = client.delete(f"/api/memories/{updated_id}", headers=headers)
        after_delete = client.get("/api/memories?limit=50", headers=headers)

    assert len(listed.json()["data"]["items"]) == 1
    assert updated.json()["data"]["importance"] == 5
    assert removed.status_code == 200
    assert after_delete.json()["data"]["items"] == []


def test_export_clear_and_import_round_trip(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        client.post(
            "/api/chat/messages/stream",
            headers=headers,
            json={"content": "我不想活了", "clientMessageId": "backup-message-id"},
        )
        exported = client.get("/api/data/export", headers=headers)
        cleared = client.delete("/api/data", headers=headers)
        empty_bootstrap = client.get("/local/bootstrap", headers=headers)
        imported = client.post(
            "/api/data/import",
            headers=headers,
            files={"file": ("backup.json", exported.content, "application/json")},
        )
        restored_history = client.get("/api/chat/messages", headers=headers)

    assert exported.status_code == 200
    assert cleared.status_code == 200
    assert empty_bootstrap.json()["data"]["hasCompanion"] is False
    assert imported.json()["data"]["hasCompanion"] is True
    assert imported.json()["data"]["importedMessages"] == 2
    assert len(restored_history.json()["data"]["items"]) == 2


def test_directory_browser_lists_only_folders_without_authorizing(tmp_path: Path) -> None:
    folder = tmp_path / "用户文档"
    folder.mkdir()
    (folder / "B folder").mkdir()
    (folder / "a folder").mkdir()
    (folder / "private.txt").write_text("not returned", encoding="utf-8")
    client, _, headers = make_client(tmp_path)
    with client:
        listing = client.post("/api/directories/browse", headers=headers, json={"path": str(folder)})
        empty = client.post("/api/directories/browse", headers=headers, json={"path": str(folder / "a folder")})
        roots = client.get("/api/allowed-roots", headers=headers)
    assert listing.status_code == 200
    data = listing.json()["data"]
    assert data["path"] == str(folder.resolve())
    assert data["parentPath"] == str(tmp_path.resolve())
    assert [item["name"] for item in data["items"]] == ["a folder", "B folder"]
    assert empty.json()["data"]["items"] == []
    # Browsing never authorizes; only the default workspace is registered.
    assert [item["displayName"] for item in roots.json()["data"]] == ["默认工作目录"]


def test_directory_browser_requires_session_and_rejects_invalid_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "note.txt"
    source.write_text("not a folder", encoding="utf-8")
    client, _, headers = make_client(tmp_path)
    with client:
        unauthorized = client.post("/api/directories/browse", json={"path": str(tmp_path)})
        assert unauthorized.status_code in (401, 403)
        for path in ("relative", str(tmp_path / "missing"), str(source), "bad\u0000path"):
            response = client.post("/api/directories/browse", headers=headers, json={"path": path})
            assert response.status_code == 422
            assert response.json()["code"] == 40601
        def denied(_path: object) -> None:
            raise PermissionError("denied")
        with monkeypatch.context() as context:
            context.setattr("yuni_agent.api.routes.tools.os.scandir", denied)
            response = client.post("/api/directories/browse", headers=headers, json={"path": str(tmp_path)})
            assert response.status_code == 422
            assert response.json()["code"] == 40601


def test_directory_browser_locations_and_filesystem_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    client, _, headers = make_client(tmp_path)
    with client:
        locations = client.post("/api/directories/browse", headers=headers, json={"path": None})
        root = client.post("/api/directories/browse", headers=headers, json={"path": tmp_path.anchor})
    assert locations.status_code == 200
    assert locations.json()["data"]["path"] is None
    assert locations.json()["data"]["items"][0]["path"] == str(tmp_path)
    assert root.status_code == 200
    assert root.json()["data"]["parentPath"] is None


def test_allowed_root_must_exist_and_can_be_revoked(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    with client:
        created = client.post(
            "/api/allowed-roots", headers=headers, json={"path": str(allowed)}
        )
        listed = client.get("/api/allowed-roots", headers=headers)
        removed = client.delete(
            f"/api/allowed-roots/{created.json()['data']['id']}", headers=headers
        )
        after_remove = client.get("/api/allowed-roots", headers=headers)

    assert created.status_code == 201
    assert len(listed.json()["data"]) == 2
    assert removed.status_code == 200
    assert [item["displayName"] for item in after_remove.json()["data"]] == ["默认工作目录"]


def test_default_workspace_is_registered_once_and_revocation_sticks(tmp_path: Path) -> None:
    import os
    client, runtime, headers = make_client(tmp_path)
    workspace = runtime.config.workspace_path
    with client:
        roots = client.get("/api/allowed-roots", headers=headers).json()["data"]
        assert workspace.is_dir()
        assert [item["path"] for item in roots] == [os.path.normcase(str(workspace.resolve()))]
        client.delete(f"/api/allowed-roots/{roots[0]['id']}", headers=headers)
    # Restart: a revoked workspace stays revoked.
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert client.get("/api/allowed-roots", headers=headers).json()["data"] == []
        # Clearing all data restores the default.
        assert client.delete("/api/data", headers=headers).status_code == 200
        assert len(client.get("/api/allowed-roots", headers=headers).json()["data"]) == 1


@pytest.mark.parametrize("revoke_previous", [False, True])
def test_workspace_switch_labels_current_root_without_losing_files_or_revocation(
    tmp_path: Path, revoke_previous: bool,
) -> None:
    import os

    from yuni_agent.storage.models import AllowedRoot
    from yuni_agent.tools.permissions import PathPolicy

    config = StartupConfig.for_test(tmp_path)
    headers = {"X-Yuni-Session": config.session_token.get_secret_value()}
    previous_workspace = config.workspace_path
    next_workspace = tmp_path / "project-workspace"
    documents = tmp_path / "documents"
    documents.mkdir()
    previous = AgentRuntime(config=config)
    with TestClient(create_app(previous)) as client:
        roots = client.get("/api/allowed-roots", headers=headers).json()["data"]
        previous_id = roots[0]["id"]
        (previous_workspace / "saved-note.txt").write_text("keep existing files", encoding="utf-8")
        assert client.post("/api/allowed-roots", headers=headers, json={"path": str(documents)}).status_code == 201
        if revoke_previous:
            assert client.delete(f"/api/allowed-roots/{previous_id}", headers=headers).status_code == 200

    switched = AgentRuntime(config=config.model_copy(update={"workspace_dir": next_workspace}))
    with TestClient(create_app(switched)) as client:
        roots = client.get("/api/allowed-roots", headers=headers).json()["data"]
        defaults = [root for root in roots if root["displayName"] == "默认工作目录"]
        assert len(defaults) == 1
        assert defaults[0]["path"] == os.path.normcase(str(next_workspace.resolve()))
        assert any(root["displayName"] == documents.name for root in roots)
        assert any(root["id"] == previous_id for root in roots) is not revoke_previous
        with switched.require_database().session() as session:
            old_root = session.get(AllowedRoot, previous_id)
            assert old_root.display_name == "历史工作目录"
            assert old_root.status == ("revoked" if revoke_previous else "active")
        assert PathPolicy(switched).resolve("new-note.txt", must_exist=False) == next_workspace / "new-note.txt"
        assert (previous_workspace / "saved-note.txt").read_text(encoding="utf-8") == "keep existing files"

    # Returning to the old startup mode relabels it without restoring revoked permission.
    for _ in range(2):
        restored = AgentRuntime(config=config)
        with TestClient(create_app(restored)) as client:
            roots = client.get("/api/allowed-roots", headers=headers).json()["data"]
            defaults = [root for root in roots if root["displayName"] == "默认工作目录"]
            assert len(defaults) == (0 if revoke_previous else 1)
            with restored.require_database().session() as session:
                rows = list(session.scalars(select(AllowedRoot)))
                assert len(rows) == 3
                assert session.get(AllowedRoot, previous_id).status == ("revoked" if revoke_previous else "active")
            assert any(root["displayName"] == "历史工作目录" for root in roots)


def test_partner_edit_persists_and_updates_next_context_without_snapshot(tmp_path: Path) -> None:
    client, runtime, headers = make_client(tmp_path)
    original = {**partner_payload(), "gender": "female", "customDescription": "习惯温柔安慰。"}
    updated = {**original, "name": "林辰", "gender": "male", "userNickname": "阿予", "customDescription": "说话直率，喜欢冷幽默。"}
    with client:
        created = client.post("/api/partner", headers=headers, json=original)
        assert created.status_code == 201
        with runtime.require_database().session() as session:
            conversation = session.scalar(select(Conversation))
            assert conversation is not None
            user = ChatMessage(conversation_id=conversation.id, role="user", content="介绍一下你自己", status="succeeded")
            session.add(user)
            session.flush()
            request = ContextRequest(conversation_id=conversation.id, current_user_message_id=user.id, current_input=user.content)
        builder = SqliteContextBuilder(runtime)
        before = asyncio.run(builder.build(request))
        assert "习惯温柔安慰" in before.messages[0]["content"]
        response = client.put("/api/partner", headers=headers, json=updated)
        assert response.status_code == 200
        with runtime.require_database().session() as session:
            saved = session.get(Companion, created.json()["data"]["id"])
            assert saved is not None
            assert (saved.name, saved.gender, saved.user_nickname, saved.custom_description) == ("林辰", "male", "阿予", updated["customDescription"])
        after = asyncio.run(builder.build(request))
        prompt = after.messages[0]["content"]
        for value in ("林辰", "关系身份：男朋友", "阿予", updated["customDescription"]):
            assert value in prompt
        assert "习惯温柔安慰" not in prompt
        assert "关系身份：女朋友" not in prompt
        assert all("本轮人物资料" not in item["content"] for item in after.messages)
        assert after.messages[-1] == {"role": "user", "content": request.current_input}
    restarted, restarted_runtime, _ = make_client(tmp_path)
    with restarted:
        restored = asyncio.run(SqliteContextBuilder(restarted_runtime).build(request))
        assert restored.messages[0]["content"] == prompt


def test_custom_description_edit_compatibility_and_restart(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        created = client.post("/api/partner", headers=headers, json=partner_payload())
        assert created.json()["data"]["customDescription"] == ""
        updated = client.put("/api/partner", headers=headers, json={**partner_payload(), "customDescription": "  嘴硬心软，喜欢冷幽默。  "})
        assert updated.json()["data"]["customDescription"] == "嘴硬心软，喜欢冷幽默。"
        legacy = client.put("/api/partner", headers=headers, json=partner_payload())
        assert legacy.json()["data"]["customDescription"] == "嘴硬心软，喜欢冷幽默。"
        invalid = client.put("/api/partner", headers=headers, json={**partner_payload(), "customDescription": "x" * 2001})
        assert invalid.status_code == 422
    restarted, _, restarted_headers = make_client(tmp_path)
    with restarted:
        assert restarted.get("/api/partner", headers=restarted_headers).json()["data"]["customDescription"] == "嘴硬心软，喜欢冷幽默。"
        cleared = restarted.put("/api/partner", headers=restarted_headers, json={**partner_payload(), "customDescription": ""})
        assert cleared.json()["data"]["customDescription"] == ""


def test_freeform_character_create_without_legacy_options(tmp_path: Path) -> None:
    payload = partner_payload()
    del payload["personalityTags"], payload["interactionStyles"]
    payload["customDescription"] = "  性格直率，喜欢冷幽默；谈到电影会很健谈。  "
    client, runtime, headers = make_client(tmp_path)
    with client:
        created = client.post("/api/partner", headers=headers, json=payload)
        assert created.status_code == 201
        data = created.json()["data"]
        assert data["personalityTags"] == [] and data["interactionStyles"] == []
        assert data["customDescription"] == str(payload["customDescription"]).strip()
        with runtime.require_database().session() as session:
            companion = session.get(Companion, data["id"])
            assert companion is not None
            prompt = _system_prompt(companion)
            assert data["customDescription"] in prompt
            assert "人物性格：温柔、可靠" not in prompt

    restarted, _, restarted_headers = make_client(tmp_path)
    with restarted:
        persisted = restarted.get("/api/partner", headers=restarted_headers).json()["data"]
        assert persisted["customDescription"] == data["customDescription"]
        assert persisted["personalityTags"] == [] and persisted["interactionStyles"] == []


def test_freeform_edit_preserves_omitted_legacy_options(tmp_path: Path) -> None:
    client, runtime, headers = make_client(tmp_path)
    with client:
        original = client.post("/api/partner", headers=headers, json=partner_payload()).json()["data"]
        payload = partner_payload()
        del payload["personalityTags"], payload["interactionStyles"]
        payload["customDescription"] = "不爱说教，喜欢直接表达自己的看法。"
        response = client.put("/api/partner", headers=headers, json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["personalityTags"] == original["personalityTags"]
        assert data["interactionStyles"] == original["interactionStyles"]
        assert data["intimacyLevel"] == original["intimacyLevel"]
        with runtime.require_database().session() as session:
            companion = session.get(Companion, data["id"])
            assert companion is not None and str(payload["customDescription"]) in _system_prompt(companion)
        invalid = client.put("/api/partner", headers=headers, json={**payload, "customDescription": "x" * 2001})
        assert invalid.status_code == 422
        assert client.get("/api/partner", headers=headers).json()["data"]["customDescription"] == payload["customDescription"]
        cleared = client.put("/api/partner", headers=headers, json={**payload, "customDescription": ""})
        assert cleared.status_code == 200 and cleared.json()["data"]["customDescription"] == ""


def profile_payload(**changes: object) -> dict[str, object]:
    return {"nickname": "小林", "avatarType": "preset", "avatarPresetId": "river", "avatarFileId": None, **changes}


def test_personal_profile_and_uploaded_avatar_persist_independently(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1sAAAAASUVORK5CYII=")
    with client:
        assert client.get("/api/profile").status_code == 401
        assert client.get("/api/profile", headers=headers).json()["data"]["nickname"] == "我"
        client.post("/api/partner", headers=headers, json=partner_payload())
        uploaded = client.post("/api/profile/avatar", headers=headers, files={"file": ("me.png", image, "image/png")})
        assert uploaded.status_code == 201
        avatar = uploaded.json()["data"]
        updated = client.put("/api/profile", headers=headers, json=profile_payload(nickname="  小林  ", avatarType="uploaded", avatarFileId=avatar["avatarFileId"]))
        assert updated.status_code == 200
        assert updated.json()["data"]["nickname"] == "小林"
        assert updated.json()["data"]["avatarUrl"] == avatar["avatarUrl"]
        companion = client.get("/api/partner", headers=headers).json()["data"]
        assert companion["userNickname"] == "小予" and companion["avatarType"] == "preset"
    restarted, _, restarted_headers = make_client(tmp_path)
    with restarted:
        profile = restarted.get("/api/profile", headers=restarted_headers).json()["data"]
        assert profile["nickname"] == "小林" and profile["avatarUrl"] == avatar["avatarUrl"]
        default = restarted.put("/api/profile", headers=restarted_headers, json=profile_payload(avatarPresetId="mist"))
        assert default.json()["data"]["avatarFileId"] is None
        assert default.json()["data"]["avatarUrl"] == "mock-avatar://mist"


@pytest.mark.parametrize("changes", [
    {"nickname": ""}, {"nickname": "   "}, {"nickname": "x" * 21},
    {"avatarPresetId": "unknown"}, {"avatarType": "uploaded", "avatarFileId": None},
    {"avatarType": "uploaded", "avatarFileId": 99999},
])
def test_invalid_personal_profile_does_not_mutate(tmp_path: Path, changes: dict[str, object]) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        response = client.put("/api/profile", headers=headers, json=profile_payload(**changes))
        assert response.status_code == 422
        assert client.get("/api/profile", headers=headers).json()["data"]["nickname"] == "我"


def test_profile_avatar_upload_errors(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        assert client.post("/api/profile/avatar", headers=headers, files={"file": ("bad.txt", b"text", "text/plain")}).status_code == 415
        assert client.post("/api/profile/avatar", headers=headers, files={"file": ("empty.png", b"", "image/png")}).status_code == 422
        assert client.post("/api/profile/avatar", headers=headers, files={"file": ("large.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")}).status_code == 413


def test_personalization_backup_and_legacy_defaults(tmp_path: Path) -> None:
    client, _, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json={**partner_payload(), "customDescription": "幽默但不刻意逗乐。"})
        avatar = client.post("/api/profile/avatar", headers=headers, files={"file": ("me.png", b"avatar fixture", "image/png")}).json()["data"]
        client.put("/api/profile", headers=headers, json=profile_payload(avatarType="uploaded", avatarFileId=avatar["avatarFileId"]))
        backup = client.get("/api/data/export", headers=headers).json()
        assert backup["tables"]["localProfiles"][0]["nickname"] == "小林"
        client.delete("/api/data", headers=headers)
        assert client.get("/api/profile", headers=headers).json()["data"]["nickname"] == "我"
        restored = client.post("/api/data/import", headers=headers, files={"file": ("backup.json", json.dumps(backup).encode(), "application/json")})
        assert restored.status_code == 200
        profile = client.get("/api/profile", headers=headers).json()["data"]
        assert profile["nickname"] == "小林" and profile["avatarUrl"] == avatar["avatarUrl"]
        assert client.get("/api/partner", headers=headers).json()["data"]["customDescription"] == "幽默但不刻意逗乐。"
        backup["tables"]["localProfiles"][0]["nickname"] = ""
        invalid = client.post("/api/data/import", headers=headers, files={"file": ("invalid.json", json.dumps(backup).encode(), "application/json")})
        assert invalid.status_code == 422
        assert client.get("/api/profile", headers=headers).json()["data"]["nickname"] == "小林"
        del backup["tables"]["localProfiles"]
        del backup["tables"]["companions"][0]["custom_description"]
        legacy = client.post("/api/data/import", headers=headers, files={"file": ("old.json", json.dumps(backup).encode(), "application/json")})
        assert legacy.status_code == 200
        assert client.get("/api/profile", headers=headers).json()["data"]["nickname"] == "我"
        assert client.get("/api/partner", headers=headers).json()["data"]["customDescription"] == ""
