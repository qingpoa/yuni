from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AppMeta(Base):
    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class Settings(Base, TimestampMixin):
    __tablename__ = "settings"
    __table_args__ = (CheckConstraint("id = 1", name="ck_settings_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    memory_generation_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    memory_retrieval_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    proactive_chat_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    proactive_interval_minutes: Mapped[int] = mapped_column(Integer, default=120)
    quiet_hours_start: Mapped[str] = mapped_column(String(5), default="23:30")
    quiet_hours_end: Mapped[str] = mapped_column(String(5), default="08:30")
    model_provider: Mapped[str] = mapped_column(String(50), default="openai_compatible")
    model_base_url: Mapped[str] = mapped_column(String(500), default="https://api.openai.com/v1")
    chat_model: Mapped[str] = mapped_column(String(120), default="")
    privacy_version: Mapped[str] = mapped_column(String(30), default="v1.0")


class AvatarFile(Base, TimestampMixin):
    __tablename__ = "avatar_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    relative_path: Mapped[str] = mapped_column(String(500), unique=True)
    content_type: Mapped[str] = mapped_column(String(50))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)


class Companion(Base, TimestampMixin):
    __tablename__ = "companions"
    __table_args__ = (
        UniqueConstraint("singleton_key", name="uq_companions_singleton"),
        CheckConstraint("singleton_key = 1", name="ck_companions_singleton"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    singleton_key: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(20))
    gender: Mapped[str | None] = mapped_column(String(6), nullable=True)
    avatar_type: Mapped[str] = mapped_column(String(20), default="preset")
    avatar_preset_id: Mapped[str | None] = mapped_column(String(50))
    avatar_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("avatar_files.id", ondelete="SET NULL")
    )
    user_nickname: Mapped[str] = mapped_column(String(20))
    personality_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    interaction_styles: Mapped[list[str]] = mapped_column(JSON, default=list)
    intimacy_level: Mapped[str] = mapped_column(String(30), default="gentle_natural")
    boundary_text: Mapped[str] = mapped_column(Text, default="")
    custom_description: Mapped[str] = mapped_column(Text, default="")
    relationship_status: Mapped[str] = mapped_column(String(30), default="stable_partner")

    avatar_file: Mapped[AvatarFile | None] = relationship()
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="companion", cascade="all, delete-orphan"
    )


class LocalProfile(Base, TimestampMixin):
    __tablename__ = "local_profiles"
    __table_args__ = (CheckConstraint("id = 1", name="ck_local_profiles_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    nickname: Mapped[str] = mapped_column(String(20), default="我")
    avatar_type: Mapped[str] = mapped_column(String(20), default="preset")
    avatar_preset_id: Mapped[str | None] = mapped_column(String(50), default="river")
    avatar_file_id: Mapped[int | None] = mapped_column(ForeignKey("avatar_files.id", ondelete="SET NULL"))
    avatar_file: Mapped[AvatarFile | None] = relationship()


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    companion_id: Mapped[int] = mapped_column(ForeignKey("companions.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(100), default="日常聊天")
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_until_message_id: Mapped[int | None] = mapped_column(Integer)
    memory_processed_until_message_id: Mapped[int | None] = mapped_column(Integer)
    next_proactive_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_proactive_memory_id: Mapped[int | None] = mapped_column(Integer)

    companion: Mapped[Companion] = relationship(back_populates="conversations")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Sticker(Base, TimestampMixin):
    __tablename__ = "stickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    description: Mapped[str] = mapped_column(String(500), default="")
    # Emotions help the companion read a user's sticker; usage is reserved for choosing
    # stickers to send and is deliberately kept out of the chat context.
    emotions: Mapped[list[str]] = mapped_column(JSON, default=list)
    usage: Mapped[str] = mapped_column(String(300), default="")
    relative_path: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(50))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_timeline", "conversation_id", "created_at", "id"),
        Index("ix_chat_messages_parent_status", "parent_message_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    client_message_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(20))
    message_type: Mapped[str] = mapped_column(String(20), default="normal")
    content: Mapped[str] = mapped_column(Text, default="")
    reply_parts: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    sticker: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    # Generated image snapshot: {"id": sha256, "content_type": str, "description": str}.
    image: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    # Generated voice snapshot: {"id": sha256, "content_type": str, "duration": float}.
    voice: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    proactive_round_id: Mapped[str | None] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(20))
    parent_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="SET NULL")
    )
    help_resources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error_code: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(String(500))

    @property
    def context_content(self) -> str:
        if self.image and self.role == "assistant":
            description = str(self.image.get("description") or "").strip()
            return (
                f"{self.content}\n"
                f"[系统已发出生成的图片（仅作上下文记录，回复时严禁手写此标记）；画面：{description or '无描述'}]"
            ).strip()
        if self.voice and self.role == "assistant":
            from yuni_agent.model.voice import voice_context_content

            return voice_context_content(self.content, self.voice)
        if not self.sticker:
            if self.role == "assistant":
                from yuni_agent.model.voice import strip_voice_markers

                return strip_voice_markers(self.content)
            return self.content
        description = self.sticker.get("description", "").strip()
        emotions = [str(item) for item in self.sticker.get("emotions") or [] if str(item).strip()]
        emotion_part = f"表达情绪：{'、'.join(emotions)}；" if emotions else ""
        if self.role == "assistant":
            return (
                f"{self.content}\n"
                f"[系统已自动配发表情包（仅作上下文记录，回复时严禁手写此标记）；{emotion_part}素材描述：{description or '无描述，无法判断图片内容'}]"
            ).strip()
        note = (
            "顺手配图以润色语气、让表达不枯燥；请聚焦用户的文字与心情，勿把画面中的动作、场景或配字当成用户的真实处境"
            if self.content.strip()
            else "表情包是情绪与态度的象征，也常是顺手发来润色对话、让表达不枯燥或随手冒泡；画面中的具体动作、物品、场景和配字只是图案借喻，不代表用户此刻的真实状态或正在做的事，情绪标签也仅作氛围参考"
        )
        return (
            f"{self.content}\n"
            f"[发送了表情包；{emotion_part}素材描述：{description or '无描述，无法判断图片内容'}（注：{note}）]"
        ).strip()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Memory(Base, TimestampMixin):
    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_lookup", "companion_id", "status", "type"),
        Index("ix_memories_subject", "companion_id", "type", "subject_key"),
        Index("ix_memories_scope", "companion_id", "status", "kind", "domain"),
        Index("ix_memories_slot", "companion_id", "domain", "scene_key", "subject_entity_id", "subject_key", "qualifier", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    companion_id: Mapped[int] = mapped_column(ForeignKey("companions.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(50))
    kind: Mapped[str] = mapped_column(String(20), default="fact")
    domain: Mapped[str] = mapped_column(String(20), default="real_world")
    scene_key: Mapped[str | None] = mapped_column(String(200))
    subject_entity_id: Mapped[int | None] = mapped_column(ForeignKey("memory_entities.id", ondelete="SET NULL"))
    qualifier: Mapped[str] = mapped_column(String(200), default="")
    subject_key: Mapped[str] = mapped_column(String(200), default="")
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[int] = mapped_column(Integer, default=3)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    sensitivity: Mapped[str] = mapped_column(String(20), default="normal")
    status: Mapped[str] = mapped_column(String(20), default="active")
    conflict_content: Mapped[str | None] = mapped_column(Text)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    event_status: Mapped[str | None] = mapped_column(String(20))
    outcome: Mapped[str | None] = mapped_column(String(1000))
    time_text: Mapped[str | None] = mapped_column(String(200))
    time_precision: Mapped[str] = mapped_column(String(20), default="unknown")
    fresh_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    time_meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    entity_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    conflict_with_id: Mapped[int | None] = mapped_column(ForeignKey("memories.id", ondelete="SET NULL"))
    suppressed_until_message_id: Mapped[int | None] = mapped_column(Integer)
    superseded_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL")
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemorySource(Base):
    __tablename__ = "memory_sources"

    memory_id: Mapped[int] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True
    )
    evidence_text: Mapped[str] = mapped_column(Text)
    message_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryEntity(Base, TimestampMixin):
    __tablename__ = "memory_entities"
    __table_args__ = (Index("ix_memory_entities_companion", "companion_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    companion_id: Mapped[int] = mapped_column(ForeignKey("companions.id", ondelete="CASCADE"))
    entity_type: Mapped[str] = mapped_column(String(20), default="person")
    canonical_name: Mapped[str] = mapped_column(String(100))
    relation_to_user: Mapped[str | None] = mapped_column(String(100))
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="active")
    merged_into_id: Mapped[int | None] = mapped_column(ForeignKey("memory_entities.id", ondelete="SET NULL"))


class MemoryEmbedding(Base):
    """Rebuildable float32 cache; FAISS owns nearest-neighbour search."""
    __tablename__ = "memory_embeddings"
    memory_id: Mapped[int] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True)
    profile: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(120))
    dimension: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    vector: Mapped[bytes] = mapped_column(LargeBinary)


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    user_message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    batch_user_message_id: Mapped[int | None] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"))


class ClientEvent(Base):
    """Durable delivery log; IDs remain monotonic across data resets."""
    __tablename__ = "client_events"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    dedupe_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryJob(Base, TimestampMixin):
    __tablename__ = "memory_jobs"
    __table_args__ = (
        Index(
            "uq_memory_jobs_pending_conversation",
            "conversation_id",
            unique=True,
            sqlite_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    from_message_id: Mapped[int] = mapped_column(Integer)
    to_message_id: Mapped[int] = mapped_column(Integer)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String(500))


class MemoryCandidate(Base, TimestampMixin):
    __tablename__ = "memory_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("memory_jobs.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(50))
    subject_key: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    sensitivity: Mapped[str] = mapped_column(String(20))
    source_message_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    structured_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    reason: Mapped[str | None] = mapped_column(String(500))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AllowedRoot(Base):
    __tablename__ = "allowed_roots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_path: Mapped[str] = mapped_column(String(1000), unique=True)
    display_name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    assistant_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="SET NULL")
    )
    tool_name: Mapped[str] = mapped_column(String(100))
    risk: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30))
    target_paths: Mapped[list[str]] = mapped_column(JSON, default=list)
    arguments_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    diff_preview: Mapped[str | None] = mapped_column(Text)
    result_summary: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SafetyEvent(Base):
    __tablename__ = "safety_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    risk_type: Mapped[str] = mapped_column(String(50))
    risk_level: Mapped[str] = mapped_column(String(30))
    action: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Reminder(Base, TimestampMixin):
    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200))
    note: Mapped[str] = mapped_column(Text, default="")
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Expression(Base, TimestampMixin):
    __tablename__ = "expressions"
    __table_args__ = (UniqueConstraint("companion_id", "situation", "style", name="uq_expression_scope"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    companion_id: Mapped[int] = mapped_column(ForeignKey("companions.id", ondelete="CASCADE"), index=True)
    situation: Mapped[str] = mapped_column(String(100))
    style: Mapped[str] = mapped_column(String(100))
    count: Mapped[int] = mapped_column(Integer, default=1)


class ExpressionSource(Base):
    __tablename__ = "expression_sources"
    expression_id: Mapped[int] = mapped_column(ForeignKey("expressions.id", ondelete="CASCADE"), primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True)


class ExpressionProgress(Base):
    __tablename__ = "expression_progress"
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True)
    until_message_id: Mapped[int] = mapped_column(Integer, default=0)
    last_skipped_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ExpressionVector(Base):
    __tablename__ = "expression_vectors"
    expression_id: Mapped[int] = mapped_column(ForeignKey("expressions.id", ondelete="CASCADE"), primary_key=True)
    profile: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(120))
    dimension: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    vector: Mapped[list[float]] = mapped_column(JSON)
