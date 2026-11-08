from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    message_id: int
    detached: bool = False
    subscribers: set[asyncio.Queue[str | None]] = field(default_factory=set)
    frames: list[str] = field(default_factory=list)


class ConversationQueue:
    """One producer per conversation; snapshot all pending inputs between turns.

    Subscribers never drive the producer's progress. Duplicate HTTP requests join
    the same input, and receive buffered events without executing it again.
    """

    def __init__(self, produce: Callable[[list[int]], AsyncIterator[str]], *,
                 quiet_seconds: float = 5.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.produce = produce
        self.quiet_seconds = quiet_seconds
        self._clock = clock
        self._pending_deadline = 0.0
        self._pending_changed = asyncio.Event()
        self.entries: dict[int, PendingMessage] = {}
        self.pending: dict[int, PendingMessage] = {}
        self.active: list[PendingMessage] = []
        self.task: asyncio.Task | None = None
        self.closed = False

    def _join(self, message_id: int, related_ids: list[int] | None = None) -> PendingMessage:
        if self.closed:
            raise RuntimeError("conversation queue is closed")
        entry = self.entries.get(message_id)
        if entry is None:
            added = False
            for related_id in related_ids or [message_id]:
                if related_id not in self.entries:
                    item = PendingMessage(related_id)
                    self.entries[related_id] = self.pending[related_id] = item
                    added = True
            if added:
                # Only new inputs reset the silence window. Reconnecting to an
                # existing entry must not postpone the batch.
                self._pending_deadline = self._clock() + self.quiet_seconds
                self._pending_changed.set()
            entry = self.entries[message_id]
        return entry

    def enqueue(self, message_id: int, related_ids: list[int] | None = None) -> None:
        self._join(message_id, related_ids).detached = True
        self._start()

    def stream(self, message_id: int, related_ids: list[int] | None = None, *, initial_frame: str = "") -> AsyncIterator[str]:
        entry = self._join(message_id, related_ids)
        subscriber: asyncio.Queue[str | None] = asyncio.Queue()
        if initial_frame:
            subscriber.put_nowait(initial_frame)
        for frame in entry.frames:
            subscriber.put_nowait(frame)
        entry.subscribers.add(subscriber)
        self._start()
        async def listen() -> AsyncIterator[str]:
            try:
                while (frame := await subscriber.get()) is not None:
                    yield frame
            finally:
                entry.subscribers.discard(subscriber)
                self._pending_changed.set()
                if self.active and not any(item.detached or item.subscribers for item in self.active):
                    # Retain the previous disconnect contract: orphaned tools must
                    # stop. Other, queued inputs still get their own next turn.
                    task = self.task
                    if task is not None and not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
        return listen()

    def _start(self) -> None:
        if (not self.closed and any(item.detached or item.subscribers for item in self.pending.values())
                and (self.task is None or self.task.done())):
            self.task = asyncio.create_task(self._run(), name="conversation-message-queue")

    @staticmethod
    def _publish(entries: list[PendingMessage], frame: str | None) -> None:
        for item in entries:
            if frame is not None:
                item.frames.append(frame)
            for subscriber in item.subscribers:
                subscriber.put_nowait(frame)

    async def _wait_for_quiet(self) -> bool:
        while self.pending and not self.closed and any(item.detached or item.subscribers for item in self.pending.values()):
            self._pending_changed.clear()
            remaining = self._pending_deadline - self._clock()
            if remaining <= 0:
                return True
            try:
                await asyncio.wait_for(self._pending_changed.wait(), timeout=remaining)
            except TimeoutError:
                # Recheck the deadline: a new input may have arrived at timeout.
                pass
        return False

    async def _run(self) -> None:
        try:
            while self.pending and not self.closed and any(item.detached or item.subscribers for item in self.pending.values()):
                if not await self._wait_for_quiet():
                    break
                # No await between snapshot and removal: later arrivals belong
                # to the following batch, even while context/model I/O awaits.
                self.active = [self.pending[key] for key in sorted(self.pending)]
                self.pending.clear()
                try:
                    async with aclosing(self.produce([item.message_id for item in self.active])) as frames:
                        async for frame in frames:
                            self._publish(self.active, frame)
                except Exception:
                    logger.exception("Conversation batch failed")
                    self._publish(self.active, 'event: error\ndata: {"code":40303,"message":"回复生成失败，请稍后重试"}\n\n')
                finally:
                    self._publish(self.active, None)
                    for item in self.active:
                        self.entries.pop(item.message_id, None)
                    self.active = []
        finally:
            self.task = None
            self._start()

    async def close(self) -> None:
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        self._publish(list(self.pending.values()), None)
        self.entries.clear()
        self.pending.clear()
