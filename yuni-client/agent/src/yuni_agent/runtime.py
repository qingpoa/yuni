from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import SecretStr
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from yuni_agent.chat.queue import ConversationQueue
from yuni_agent.config import StartupConfig
from yuni_agent.model.auth import (
    AuthFileError,
    ImageAuth,
    ModelAuth,
    VoiceAuth,
    load_auth,
    load_image_auth,
    load_voice_auth,
    remove_auth,
    save_auth,
    save_image_auth,
    save_voice_auth,
)
from yuni_agent.storage.database import Database
from yuni_agent.storage.migrations import upgrade_database
from yuni_agent.storage.models import (
    AllowedRoot,
    AppMeta,
    ChatMessage,
    ChatTurn,
    Companion,
    LocalProfile,
    MemoryJob,
    Settings,
    ToolCall,
)


class RuntimeStatus(StrEnum):
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"


@dataclass(slots=True)
class AgentRuntime:
    config: StartupConfig
    status: RuntimeStatus = RuntimeStatus.STARTING
    storage_ready: bool = False
    has_companion: bool = False
    memory_generation_enabled: bool = True
    memory_retrieval_enabled: bool = True
    database: Database | None = field(default=None, repr=False)
    model_api_key: SecretStr | None = field(default=None, repr=False)
    embedding_auth: ModelAuth | None = field(default=None, repr=False)
    embedding_generation: int = 0
    image_auth: ImageAuth | None = field(default=None, repr=False)
    voice_auth: VoiceAuth | None = field(default=None, repr=False)
    embedding_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    data_generation: int = 0
    proactive_generation: int = 0
    event_changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    chat_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    chat_queues: dict[int, ConversationQueue] = field(default_factory=dict, repr=False)
    chat_paused: bool = False
    maintenance_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    expression_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    memory_index_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    tool_decisions: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future[str]]] = field(
        default_factory=dict, repr=False
    )
    _shutdown_callback: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def database_path(self) -> Path:
        return self.config.data_dir / "yuni.db"

    @property
    def auth_path(self) -> Path:
        return self.config.data_dir / "auth.json"

    @property
    def embedding_auth_path(self) -> Path:
        return self.config.data_dir / "embedding-auth.json"

    @property
    def image_auth_path(self) -> Path:
        return self.config.data_dir / "image-auth.json"

    @property
    def voice_auth_path(self) -> Path:
        return self.config.data_dir / "voice-auth.json"

    @property
    def image_generation_ready(self) -> bool:
        return self.image_auth is not None and self.image_auth.ready

    @property
    def voice_generation_ready(self) -> bool:
        return self.voice_auth is not None and self.voice_auth.ready

    @property
    def embedding_ready(self) -> bool:
        return self.embedding_auth is not None and self.embedding_auth.ready

    def save_image_auth(self, auth: ImageAuth, key: str | None) -> None:
        """Blank key keeps the stored key, but only for the same URL."""
        previous = self.image_auth
        secret = (key or "").strip()
        if not secret and previous is not None:
            if previous.url and previous.url != auth.url and previous.apikey.get_secret_value():
                raise ValueError("更换生图地址时请填写新 Key")
            secret = previous.apikey.get_secret_value()
        updated = auth.model_copy(update={"apikey": SecretStr(secret)})
        save_image_auth(self.image_auth_path, updated)
        self.image_auth = updated

    def clear_image_key(self) -> None:
        if self.image_auth is not None:
            cleared = self.image_auth.model_copy(update={"apikey": SecretStr("")})
            save_image_auth(self.image_auth_path, cleared)
            self.image_auth = cleared

    def remove_image_auth(self) -> None:
        remove_auth(self.image_auth_path)
        self.image_auth = None

    def save_voice_auth(self, auth: VoiceAuth, key: str | None) -> None:
        """Blank key keeps the stored key, but only for the same URL."""
        previous = self.voice_auth
        secret = (key or "").strip()
        if not secret and previous is not None:
            if previous.url and previous.url != auth.url and previous.apikey.get_secret_value():
                raise ValueError("更换语音合成地址时请填写新 Key")
            secret = previous.apikey.get_secret_value()
        updated = auth.model_copy(update={"apikey": SecretStr(secret)})
        save_voice_auth(self.voice_auth_path, updated)
        self.voice_auth = updated

    def clear_voice_key(self) -> None:
        if self.voice_auth is not None:
            cleared = self.voice_auth.model_copy(update={"apikey": SecretStr("")})
            save_voice_auth(self.voice_auth_path, cleared)
            self.voice_auth = cleared

    def remove_voice_auth(self) -> None:
        remove_auth(self.voice_auth_path)
        self.voice_auth = None

    def save_embedding_auth(self, model: str, url: str, key: str | None) -> None:
        from sqlalchemy import delete

        from yuni_agent.storage.models import ExpressionVector, MemoryEmbedding
        with self.embedding_lock:
            previous = self.embedding_auth
            if previous and previous.url != url and not (key or '').strip():
                raise ValueError("更换 Embedding 地址时请填写新 Key")
            secret = (key or '').strip() or (previous.apikey.get_secret_value() if previous else '')
            auth = ModelAuth(model=model, url=url, apikey=SecretStr(secret))
            save_auth(self.embedding_auth_path, auth)
            self.embedding_auth = auth
            self.embedding_generation += 1
            if previous is None or (previous.url, previous.model) != (auth.url, auth.model):
                with self.require_database().session() as session:
                    session.execute(delete(ExpressionVector))
                    session.execute(delete(MemoryEmbedding))
                self.clear_memory_index_files()

    def remove_embedding_auth(self) -> None:
        with self.embedding_lock:
            remove_auth(self.embedding_auth_path)
            self.embedding_auth = None
            self.embedding_generation += 1
            self.clear_memory_index_files()

    def clear_memory_index_files(self) -> None:
        """Only our derived index directory; never remove original memory text."""
        with self.memory_index_lock:
            directory = self.config.data_dir / "memory-index"
            if directory.exists():
                for path in directory.iterdir():
                    if path.is_file() and path.suffix in {".faiss", ".json", ".tmp"}:
                        path.unlink(missing_ok=True)

    def save_model_auth(self, model: str, url: str, api_key: str | None, *, preserve_empty: bool = True) -> None:
        key = api_key.strip() if api_key and api_key.strip() else (
            self.model_api_key.get_secret_value() if self.model_api_key is not None else ""
        )
        if not preserve_empty:
            key = (api_key or "").strip()
        auth = ModelAuth(model=model, url=url, apikey=SecretStr(key))
        save_auth(self.auth_path, auth)
        self.model_api_key = auth.apikey if key else None

    def remove_model_auth(self) -> None:
        remove_auth(self.auth_path)
        self.model_api_key = None

    def initialize_storage(self) -> None:
        self.config.prepare_directories()
        upgrade_database(self.database_path)
        self.database = Database.open(self.database_path)
        self.model_api_key = self.config.model_api_key
        try:
            self.embedding_auth = load_auth(self.embedding_auth_path)
        except AuthFileError:
            self.embedding_auth = None
            logging.getLogger(__name__).warning("Embedding 配置无法读取，请重新保存")
        try:
            self.image_auth = load_image_auth(self.image_auth_path)
        except AuthFileError:
            self.image_auth = None
            logging.getLogger(__name__).warning("生图配置无法读取，请重新保存")
        try:
            self.voice_auth = load_voice_auth(self.voice_auth_path)
        except AuthFileError:
            self.voice_auth = None
            logging.getLogger(__name__).warning("语音合成配置无法读取，请重新保存")
        try:
            auth = load_auth(self.auth_path)
        except AuthFileError:
            logging.getLogger(__name__).warning("auth.json 无法读取或格式无效，请重新保存模型配置")
            auth = None
            self.model_api_key = None
        with self.database.session() as session:
            session.execute(update(ChatTurn).where(ChatTurn.status == "running").values(status="interrupted"))
            session.execute(
                update(ChatMessage)
                .where(ChatMessage.status.in_(["pending", "streaming"]))
                .values(
                    status="failed",
                    error_code=40304,
                    error_message="Agent 上次运行时回复中断",
                )
            )
            session.execute(
                update(MemoryJob)
                .where(MemoryJob.status == "running")
                .values(status="pending", error_message="Agent 重启后重新排队")
            )
            session.execute(
                update(ToolCall)
                .where(ToolCall.status.in_(["awaiting_confirmation", "running"]))
                .values(status="failed", error_message="Agent 重启，工具调用已失效")
            )
            settings = session.get(Settings, 1)
            if settings is None:
                settings = Settings(id=1)
                session.add(settings)
                session.flush()
            if session.get(LocalProfile, 1) is None:
                session.add(LocalProfile(id=1))
            if auth is not None:
                settings.model_base_url = auth.url
                settings.chat_model = auth.model
                self.model_api_key = auth.apikey if auth.apikey.get_secret_value() else None
            installation = session.get(AppMeta, "installation_id")
            if installation is None:
                session.add(AppMeta(key="installation_id", value=secrets.token_hex(16)))
            self.has_companion = session.scalar(select(Companion.id).limit(1)) is not None
            self.memory_generation_enabled = settings.memory_generation_enabled
            self.memory_retrieval_enabled = settings.memory_retrieval_enabled
            self.ensure_workspace_root(session)
        self.storage_ready = True

    def ensure_workspace_root(self, session: Session) -> None:
        """Register the current workspace once and label previous workspaces.

        Startup modes can share a database while using different workspaces.
        Preserve previous roots and their files, but reserve the default label
        for the current path. Revoked roots stay revoked, including on a switch
        back to a previous workspace. Clearing/importing data restores defaults.
        """
        from yuni_agent.config import DEFAULT_WORKSPACE_NAME
        workspace = self.config.workspace_path
        workspace.mkdir(parents=True, exist_ok=True)
        canonical = os.path.normcase(str(workspace.resolve()))
        session.execute(
            update(AllowedRoot)
            .where(
                AllowedRoot.display_name == DEFAULT_WORKSPACE_NAME,
                AllowedRoot.canonical_path != canonical,
            )
            .values(display_name="历史工作目录")
        )
        current = session.scalar(select(AllowedRoot).where(AllowedRoot.canonical_path == canonical))
        if current is None:
            session.add(AllowedRoot(canonical_path=canonical, display_name=DEFAULT_WORKSPACE_NAME, status="active"))
        else:
            current.display_name = DEFAULT_WORKSPACE_NAME
        session.flush()

    def close_storage(self) -> None:
        if self.database is not None:
            self.database.close()
            self.database = None
        self.storage_ready = False

    async def close_chat_queues(self) -> None:
        self.chat_paused = True
        try:
            queues, self.chat_queues = list(self.chat_queues.values()), {}
            await asyncio.gather(*(queue.close() for queue in queues))
        finally:
            self.chat_paused = False

    def require_database(self) -> Database:
        if self.database is None:
            raise RuntimeError("storage is not initialized")
        return self.database

    async def wait_for_tool_decision(self, call_id: str, timeout: float = 120) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self.tool_decisions[call_id] = (loop, future)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.tool_decisions.pop(call_id, None)

    def resolve_tool_decision(self, call_id: str, decision: str) -> bool:
        waiter = self.tool_decisions.get(call_id)
        if waiter is None:
            return False
        loop, future = waiter

        def resolve() -> None:
            if not future.done():
                future.set_result(decision)

        loop.call_soon_threadsafe(resolve)
        return True

    def mark_ready(self) -> None:
        self.status = RuntimeStatus.READY

    def mark_stopping(self) -> None:
        self.status = RuntimeStatus.STOPPING

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        self._shutdown_callback = callback

    def request_shutdown(self) -> None:
        self.mark_stopping()
        if self._shutdown_callback is not None:
            self._shutdown_callback()
