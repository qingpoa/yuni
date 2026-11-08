from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 200
    message: str = "success"
    data: T


class HealthData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    agent_version: str = Field(serialization_alias="agentVersion")
    protocol_version: int = Field(serialization_alias="protocolVersion")


class BootstrapData(BaseModel):
    initialized: bool
    has_companion: bool = Field(serialization_alias="hasCompanion")
    memory_generation_enabled: bool = Field(serialization_alias="memoryGenerationEnabled")
    memory_retrieval_enabled: bool = Field(serialization_alias="memoryRetrievalEnabled")
    memory_enabled: bool = Field(serialization_alias="memoryEnabled")
    model_configured: bool = Field(serialization_alias="modelConfigured")
    agent_version: str = Field(serialization_alias="agentVersion")
    protocol_version: int = Field(serialization_alias="protocolVersion")


class ShutdownData(BaseModel):
    accepted: bool
