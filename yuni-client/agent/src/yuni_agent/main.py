from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys

import uvicorn

from yuni_agent import __version__
from yuni_agent.api.app import create_app
from yuni_agent.common.logging import configure_logging
from yuni_agent.config import PROTOCOL_VERSION, StartupConfig
from yuni_agent.runtime import AgentRuntime

logger = logging.getLogger(__name__)


async def serve(config: StartupConfig) -> None:
    config.prepare_directories()
    configure_logging(config.log_dir)

    runtime = AgentRuntime(config=config)
    app = create_app(runtime)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((config.host, config.port))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.host,
            port=port,
            log_config=None,
            access_log=False,
        )
    )
    runtime.set_shutdown_callback(lambda: setattr(server, "should_exit", True))

    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started and not server_task.done():
        await asyncio.sleep(0.01)

    if server_task.done():
        await server_task
        raise RuntimeError("agent server stopped before becoming ready")

    ready = {
        "type": "ready",
        "port": port,
        "agentVersion": __version__,
        "protocolVersion": PROTOCOL_VERSION,
    }
    sys.stdout.write(json.dumps(ready, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    logger.info("Yuni Agent ready on %s:%s", config.host, port)

    await server_task


def run(config: StartupConfig) -> None:
    asyncio.run(serve(config))

