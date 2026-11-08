from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MemoryType(StrEnum):
    USER_FACT = "user_fact"
    USER_PREFERENCE = "user_preference"
    IMPORTANT_EVENT = "important_event"
    RELATIONSHIP_PREFERENCE = "relationship_preference"
    EMOTION_PATTERN = "emotion_pattern"
    BOUNDARY = "boundary"


class EntityInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    entity_type: Literal["person", "pet", "organization", "place", "other"] = "person"
    relation_to_user: str | None = Field(default=None, max_length=100)
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def normalize_entity(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        etype = str(raw.get("entity_type") or "person").strip().lower()
        mapping = {
            "location": "place", "city": "place", "address": "place", "spot": "place",
            "company": "organization", "org": "organization", "school": "organization",
            "animal": "pet", "cat": "pet", "dog": "pet",
            "human": "person", "user": "person", "friend": "person", "colleague": "person", "family": "person",
        }
        etype = mapping.get(etype, etype)
        raw["entity_type"] = etype if etype in {"person", "pet", "organization", "place", "other"} else "other"
        if raw.get("aliases") is None:
            raw["aliases"] = []
        return raw


class TimeMeta(BaseModel):
    start_window_end: datetime | None = None
    end_window_start: datetime | None = None
    planned_start: datetime | None = None
    planned_end: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None


class MemoryCandidate(BaseModel):
    type: MemoryType
    subject_key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=2000)
    importance: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=1.0, ge=0, le=1)
    sensitivity: Literal["normal", "sensitive"] = "normal"
    source_message_ids: list[int] = Field(default_factory=list)
    kind: Literal["fact", "event", "ongoing_state"] = "fact"
    domain: Literal["real_world", "conversation", "fictional"] = "real_world"
    scene_key: str | None = Field(default=None, max_length=200)
    subject_person: str = Field(default="user", max_length=100)
    qualifier: str = Field(default="", max_length=200)
    entities: list[EntityInput] = Field(default_factory=list, max_length=20)
    event_status: Literal["planned", "in_progress", "completed", "cancelled"] | None = None
    outcome: str | None = Field(default=None, max_length=1000)
    time_text: str | None = Field(default=None, max_length=200)
    time_precision: Literal["exact", "day", "window", "recurring", "unknown"] = "unknown"
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    time_meta: TimeMeta = Field(default_factory=TimeMeta)
    fresh_days: Literal[5, 14] = 5
    operation_hint: Literal["auto", "create", "reinforce", "enrich", "transition", "correct", "coexist", "dispute"] = "auto"
    target_memory_id: int | None = None
    same_conclusion: bool = False
    evidence_quote: str = Field(default="", max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def normalize_candidate(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        for key, default in (
            ("qualifier", ""), ("subject_person", "user"), ("evidence_quote", ""),
            ("entities", []), ("time_meta", {}), ("time_precision", "unknown"),
            ("operation_hint", "auto"), ("same_conclusion", False),
        ):
            if raw.get(key) is None:
                raw[key] = default
        qual = str(raw.get("qualifier") or "").strip()
        if qual.lower() in {
            "like", "dislike", "love", "hate", "prefer", "avoid",
            "current_like", "current_dislike", "past_like", "past_dislike",
            "current", "past", "now", "before", "formerly", "previously",
            "喜欢", "不喜欢", "讨厌", "爱吃", "不爱吃", "不吃", "现在", "以前", "过去", "目前",
        }:
            raw["qualifier"] = ""
        if raw.get("time_precision") not in {"exact", "day", "window", "recurring", "unknown"}:
            raw["time_precision"] = "unknown"
        fresh = raw.get("fresh_days")
        raw["fresh_days"] = 14 if isinstance(fresh, (int, float)) and fresh >= 10 else 5
        kind = raw.get("kind") or ("event" if raw.get("type") == "important_event" else "fact")
        if kind != "event":
            raw["event_status"] = None
        elif raw.get("event_status") is not None:
            es = str(raw["event_status"]).strip().lower()
            es_map = {"ongoing": "in_progress", "done": "completed", "finished": "completed", "canceled": "cancelled"}
            es = es_map.get(es, es)
            raw["event_status"] = es if es in {"planned", "in_progress", "completed", "cancelled"} else None
        return raw

    @model_validator(mode="after")
    def validate_scope(self):
        if "kind" not in self.model_fields_set and self.type == MemoryType.IMPORTANT_EVENT:
            self.kind = "event"
        if self.domain == "fictional" and not self.scene_key:
            raise ValueError("虚构记忆必须指定 scene_key")
        if self.domain != "fictional":
            self.scene_key = None
        if self.domain == "conversation" and self.kind != "event":
            raise ValueError("聊天范围只能保存已经交付的互动事件")
        return self


class RetrievedMemory(BaseModel):
    id: int
    type: MemoryType
    content: str
    importance: int
    confidence: float
    kind: str = "fact"
    domain: str = "real_world"
    status: str = "active"
    subject_key: str = ""
    qualifier: str = ""
    event_status: str | None = None
    outcome: str | None = None
    time_text: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    fresh_until: datetime | None = None
    time_meta: dict = Field(default_factory=dict)
    time_certainty: str | None = None
    previous_facts: list[str] = Field(default_factory=list)

