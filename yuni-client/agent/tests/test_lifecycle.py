import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from yuni_agent.api.app import create_app
from yuni_agent.config import StartupConfig
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.migrations import upgrade_database


def make_client(tmp_path: Path) -> tuple[TestClient, AgentRuntime, str]:
    config = StartupConfig.for_test(tmp_path)
    runtime = AgentRuntime(config=config)
    return TestClient(create_app(runtime)), runtime, config.session_token.get_secret_value()


def test_health_is_public_and_uses_camel_case(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        response = client.get("/local/health")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "status": "ready",
        "agentVersion": "0.1.0",
        "protocolVersion": 1,
    }


def test_bootstrap_requires_session_token(tmp_path: Path) -> None:
    client, _, token = make_client(tmp_path)

    with client:
        denied = client.get("/local/bootstrap")
        accepted = client.get("/local/bootstrap", headers={"X-Yuni-Session": token})

    assert denied.status_code == 401
    assert denied.json()["code"] == 40100
    assert accepted.status_code == 200
    assert accepted.json()["data"]["memoryGenerationEnabled"] is True
    assert accepted.json()["data"]["memoryRetrievalEnabled"] is True


def test_storage_migration_is_stamped(tmp_path: Path) -> None:
    client, runtime, _ = make_client(tmp_path)

    with client:
        with runtime.require_database().engine.connect() as connection:
            revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(chat_messages)"))}
            assert "voice" in columns

    assert revision == "0016_voice_messages"


def test_gender_migration_preserves_existing_companion(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE companions (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
            "CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, content TEXT);"
            "INSERT INTO companions (id, name) VALUES (1, 'Legacy companion');"
            "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY);"
            "INSERT INTO alembic_version VALUES ('0002_reminders');"
        )

    upgrade_database(path)
    upgrade_database(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name, gender FROM companions").fetchone() == (
            "Legacy companion", None
        )
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0016_voice_messages",
        )


def test_personalization_migration_preserves_legacy_data(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE companions (id INTEGER PRIMARY KEY, name TEXT NOT NULL, gender TEXT);"
            "CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, content TEXT);"
            "INSERT INTO companions VALUES (1, 'Legacy', 'female');"
            "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY);"
            "INSERT INTO alembic_version VALUES ('0003_companion_gender');"
        )
    upgrade_database(path)
    upgrade_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name, gender, custom_description FROM companions").fetchone() == ("Legacy", "female", "")
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='local_profiles'").fetchone() is not None


def test_shutdown_requests_server_stop_after_response(tmp_path: Path) -> None:
    client, runtime, token = make_client(tmp_path)
    stopped = False

    def mark_stopped() -> None:
        nonlocal stopped
        stopped = True

    runtime.set_shutdown_callback(mark_stopped)
    with client:
        response = client.post("/local/shutdown", headers={"X-Yuni-Session": token})

    assert response.status_code == 200
    assert response.json()["data"]["accepted"] is True
    assert stopped is True
