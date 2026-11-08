from __future__ import annotations

import os
import subprocess
import sys
from typing import Annotated

from fastapi import APIRouter, Depends

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.runtime import AgentRuntime

router = APIRouter(tags=["system"])


@router.post("/local/logs/open", response_model=ApiResponse[None])
def open_logs(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[None]:
    path = runtime.config.log_dir.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return ApiResponse(data=None)
