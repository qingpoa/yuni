from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from yuni_agent.runtime import AgentRuntime


def get_runtime(request: Request) -> AgentRuntime:
    return request.app.state.runtime


def require_session(
    request: Request,
    session_token: Annotated[str | None, Header(alias="X-Yuni-Session")] = None,
) -> AgentRuntime:
    runtime = get_runtime(request)
    expected = runtime.config.session_token.get_secret_value()
    if session_token is None or not secrets.compare_digest(session_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": 40100, "message": "invalid local session"},
        )
    return runtime

