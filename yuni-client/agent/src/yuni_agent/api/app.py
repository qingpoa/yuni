from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from yuni_agent import __version__
from yuni_agent.agent.proactive import ProactiveChatWorker
from yuni_agent.api.routes.chat import router as chat_router
from yuni_agent.api.routes.companion import router as companion_router
from yuni_agent.api.routes.data import router as data_router
from yuni_agent.api.routes.events import router as events_router
from yuni_agent.api.routes.lifecycle import router as lifecycle_router
from yuni_agent.api.routes.memory import router as memory_router
from yuni_agent.api.routes.profile import router as profile_router
from yuni_agent.api.routes.reminders import router as reminders_router
from yuni_agent.api.routes.settings import router as settings_router
from yuni_agent.api.routes.system import router as system_router
from yuni_agent.api.routes.tools import router as tools_router
from yuni_agent.chat.events import ReminderEventsWorker
from yuni_agent.context.compaction import ContextCompactionWorker
from yuni_agent.expressions.worker import ExpressionWorker
from yuni_agent.memory.index import MemoryIndexWorker
from yuni_agent.memory.worker import MemoryMaintenanceWorker
from yuni_agent.model.auth import AuthFileError
from yuni_agent.runtime import AgentRuntime


def create_app(runtime: AgentRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.initialize_storage()
        runtime.mark_ready()
        memory_task = asyncio.create_task(
            MemoryMaintenanceWorker(runtime).run_forever(), name="memory-maintenance"
        )
        compaction_task = asyncio.create_task(
            ContextCompactionWorker(runtime).run_forever(), name="context-compaction"
        )
        expression_task = asyncio.create_task(ExpressionWorker(runtime).run_forever(), name="expression-learning")
        memory_index_task = asyncio.create_task(MemoryIndexWorker(runtime).run_forever(), name="memory-vector-index")
        proactive_task = asyncio.create_task(ProactiveChatWorker(runtime).run_forever(), name="proactive-chat")
        reminder_task = asyncio.create_task(ReminderEventsWorker(runtime).run_forever(), name="reminder-events")
        try:
            yield
        finally:
            runtime.mark_stopping()
            await runtime.close_chat_queues()
            memory_task.cancel()
            compaction_task.cancel()
            expression_task.cancel()
            memory_index_task.cancel()
            proactive_task.cancel()
            reminder_task.cancel()
            with suppress(asyncio.CancelledError):
                await memory_task
            with suppress(asyncio.CancelledError):
                await compaction_task
            with suppress(asyncio.CancelledError):
                await expression_task
            with suppress(asyncio.CancelledError):
                await memory_index_task
            with suppress(asyncio.CancelledError):
                await proactive_task
            with suppress(asyncio.CancelledError):
                await reminder_task
            runtime.close_storage()

    app = FastAPI(
        title="Yuni Local Agent",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(AuthFileError)
    async def handle_auth_file_error(_: Request, __: AuthFileError) -> JSONResponse:
        return JSONResponse(status_code=500, content={
            "code": 50000,
            "message": "本地模型配置文件读写失败，请检查目录权限后重试",
            "data": None,
        })

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            code = int(exc.detail.get("code", exc.status_code))
            message = str(exc.detail.get("message", "request failed"))
        else:
            code = exc.status_code
            message = str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": code, "message": message, "data": None},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        message = str(errors[0].get("msg", "参数错误")) if errors else "参数错误"
        return JSONResponse(
            status_code=422,
            content={"code": 40000, "message": message, "data": None},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime.config.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Yuni-Session"],
    )
    app.include_router(lifecycle_router)
    app.include_router(companion_router)
    app.include_router(profile_router)
    app.include_router(settings_router)
    app.include_router(chat_router)
    app.include_router(memory_router)
    app.include_router(tools_router)
    app.include_router(system_router)
    app.include_router(data_router)
    app.include_router(reminders_router)
    app.include_router(events_router)
    return app
