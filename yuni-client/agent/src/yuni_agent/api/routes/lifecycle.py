from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from yuni_agent import __version__
from yuni_agent.api.dependencies import get_runtime, require_session
from yuni_agent.api.schemas import ApiResponse, BootstrapData, HealthData, ShutdownData
from yuni_agent.config import PROTOCOL_VERSION
from yuni_agent.runtime import AgentRuntime

router = APIRouter(tags=["lifecycle"])


@router.get("/local/health", response_model=ApiResponse[HealthData])
def health(request: Request) -> ApiResponse[HealthData]:
    runtime = get_runtime(request)
    return ApiResponse(
        data=HealthData(
            status=runtime.status,
            agent_version=__version__,
            protocol_version=PROTOCOL_VERSION,
        )
    )


@router.get("/local/bootstrap", response_model=ApiResponse[BootstrapData])
def bootstrap(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[BootstrapData]:
    return ApiResponse(
        data=BootstrapData(
            initialized=runtime.storage_ready,
            has_companion=runtime.has_companion,
            memory_generation_enabled=runtime.memory_generation_enabled,
            memory_retrieval_enabled=runtime.memory_retrieval_enabled,
            memory_enabled=(
                runtime.memory_generation_enabled and runtime.memory_retrieval_enabled
            ),
            model_configured=runtime.model_api_key is not None,
            agent_version=__version__,
            protocol_version=PROTOCOL_VERSION,
        )
    )


@router.post("/local/shutdown", response_model=ApiResponse[ShutdownData])
def shutdown(
    background_tasks: BackgroundTasks,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[ShutdownData]:
    background_tasks.add_task(runtime.request_shutdown)
    return ApiResponse(data=ShutdownData(accepted=True))
