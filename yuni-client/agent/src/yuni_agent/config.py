from __future__ import annotations

import json
import secrets
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROTOCOL_VERSION = 1
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_WORKSPACE_NAME = "默认工作目录"


def project_workspace_dir() -> Path:
    """<repo>/workspace for source-checkout runs (this file is at
    <repo>/yuni-client/agent/src/yuni_agent/config.py)."""
    return Path(__file__).resolve().parents[4] / "workspace"


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class StartupConfig(BaseModel):
    """Configuration supplied by Tauri through the sidecar stdin pipe."""

    model_config = ConfigDict(alias_generator=lambda name: _to_camel(name), populate_by_name=True, extra="forbid")

    data_dir: Path
    log_dir: Path
    # Default authorized folder for file tools; relative tool paths resolve here.
    # Omitted => <data_dir>/workspace (packaged app has no project folder).
    workspace_dir: Path | None = None
    session_token: SecretStr
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://tauri.localhost"])
    host: Literal["127.0.0.1"] = LOOPBACK_HOST
    port: int = Field(default=0, ge=0, le=65535)
    model_api_key: SecretStr | None = None

    @field_validator("session_token")
    @classmethod
    def validate_session_token(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("sessionToken must contain at least 32 bytes")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, value: list[str]) -> list[str]:
        if not value or "*" in value:
            raise ValueError("allowedOrigins must be a non-empty explicit allowlist")
        return list(dict.fromkeys(value))

    @property
    def workspace_path(self) -> Path:
        return self.workspace_dir or self.data_dir / "workspace"

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json_line(cls, line: str) -> StartupConfig:
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("startup configuration must be valid JSON") from exc
        return cls.model_validate(payload)

    @classmethod
    def development(cls) -> StartupConfig:
        root = Path(tempfile.gettempdir()) / "yuni-agent-dev"
        return cls(
            data_dir=root / "data",
            log_dir=root / "logs",
            workspace_dir=project_workspace_dir(),
            session_token=SecretStr("dev-session-token-not-for-production-0001"),
            allowed_origins=["http://127.0.0.1:5173", "http://tauri.localhost"],
            port=43127,
        )

    @classmethod
    def for_test(cls, root: Path) -> StartupConfig:
        return cls(
            data_dir=root / "data",
            log_dir=root / "logs",
            session_token=SecretStr(secrets.token_urlsafe(32)),
            allowed_origins=["http://127.0.0.1:5173"],
        )
