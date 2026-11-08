from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta

import faiss
import httpx
import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import select
from test_business_api import make_client, partner_payload
from test_chat_pipeline import initialized_runtime, seed_conversation

from yuni_agent.memory.index import MemoryIndex, MemoryIndexWorker, content_hash
from yuni_agent.memory.retriever import SqliteMemoryRetriever
from yuni_agent.model.embedding import EmbeddingClient
from yuni_agent.storage.models import Companion, Memory, MemoryEmbedding
from yuni_agent.tools.models import QueryMemoryArguments, ToolRequest
from yuni_agent.tools.service import LocalToolExecutor


def seed(runtime, companion_id):
    with runtime.require_database().session() as session:
        item = Memory(companion_id=companion_id, type="user_preference", subject_key="drink.coffee",
                      title="咖啡偏好", content="喝咖啡时不加糖。", status="active", importance=2, confidence=1)
        session.add(item)
        session.flush()
        return item.id


def install_embedding(runtime, monkeypatch):
    runtime.save_embedding_auth("embed", "https://isolated.example/v1", "fake")
    requests = []
    original = httpx.AsyncClient
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content)
        assert set(body) == {"model", "input"}
        requests.append(body)
        values = []
        for index, text in enumerate(body["input"]):
            vector = [1, 0, 0] if ("咖啡" in text or "提神" in text) else [0, 1, 0]
            values.append({"index": index, "embedding": vector})
        return httpx.Response(200, json={"model": "embed", "data": list(reversed(values))})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    return requests


@pytest.mark.asyncio
async def test_real_http_faiss_semantic_recall_persistence_and_keyword_fallback(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    memory_id = seed(runtime, companion_id)
    requests = install_embedding(runtime, monkeypatch)
    try:
        index = MemoryIndex(runtime)
        assert await index.backfill() == 2
        assert await index.backfill() == 0 and len(requests) == 1
        path = runtime.config.data_dir / "memory-index" / f"companion-{companion_id}.faiss"
        loaded = faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype="uint8").copy())
        assert loaded.ntotal == 2 and loaded.d == 3
        results = await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神饮品怎么选")
        assert [item.id for item in results] == [memory_id]
        assert all("香菜" not in item.content for item in results)  # unrelated importance=5 is insufficient
        original_bytes = path.read_bytes()
        runtime.close_storage()
        runtime = initialized_runtime(tmp_path)
        assert (await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神饮品怎么选"))[0].id == memory_id
        assert path.read_bytes() == original_bytes
        path.write_bytes(b"damaged-file")
        assert (await MemoryIndex(runtime).search(companion_id, "提神"))[memory_id] > .99
        assert path.read_bytes() != b"damaged-file"
        async def unavailable(*args):
            raise RuntimeError("isolated-provider-failure")
        monkeypatch.setattr(EmbeddingClient, "embed", unavailable)
        assert (await SqliteMemoryRetriever(runtime).retrieve(companion_id, "咖啡"))[0].id == memory_id
        assert await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神饮品怎么选") == []
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_unusable_rows_are_filtered_and_noncurrent_intervals_remain_indexable(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    memory_id = seed(runtime, companion_id)
    install_embedding(runtime, monkeypatch)
    try:
        index = MemoryIndex(runtime)
        await index.backfill()
        for change in ["edit", "conflicted", "deleted", "expired", "future"]:
            with runtime.require_database().session() as session:
                item = session.get(Memory, memory_id)
                item.content = "喝咖啡时不加糖。"
                item.status, item.deleted_at, item.valid_until, item.valid_from = "active", None, None, None
                if change == "edit": item.content = "喜欢果汁。"
                elif change == "conflicted": item.status = "conflicted"
                elif change == "deleted": item.deleted_at = datetime.now(UTC)
                elif change == "expired": item.valid_until = datetime.now(UTC) - timedelta(seconds=1)
                elif change == "future": item.valid_from = datetime.now(UTC) + timedelta(days=1)
            indexed = await index.search(companion_id, "提神")
            if change in {"expired", "future"}:
                # A time-filtered history query still needs these vectors.
                assert memory_id in indexed
                assert await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神") == []
            else:
                assert memory_id not in indexed
        assert await index.backfill() == 0
        with runtime.require_database().session() as session:
            assert session.get(MemoryEmbedding, memory_id) is not None
            session.get(Memory, memory_id).status = "deleted"
        assert await index.backfill() == 0
        with runtime.require_database().session() as session:
            assert session.get(MemoryEmbedding, memory_id) is None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["model", "data", "edit", "delete"])
async def test_inflight_embedding_cannot_restore_stale_data(tmp_path, change):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    memory_id = seed(runtime, companion_id)
    runtime.save_embedding_auth("embed", "https://isolated.example/v1", "fake")
    started, release = asyncio.Event(), asyncio.Event()
    class Slow:
        def __init__(self):
            self.auth = runtime.embedding_auth
            self.profile = EmbeddingClient(self.auth).profile
        async def embed(self, texts):
            started.set()
            await release.wait()
            return [[1, 0, 0] for _ in texts]
    try:
        task = asyncio.create_task(MemoryIndex(runtime).backfill(Slow()))
        await started.wait()
        if change == "model":
            runtime.save_embedding_auth("other", "https://isolated.example/v1", "fake")
        elif change == "data":
            runtime.data_generation += 1
        else:
            with runtime.require_database().session() as session:
                item = session.get(Memory, memory_id)
                if change == "edit": item.content = "更新后的原文"
                else: item.deleted_at = datetime.now(UTC)
        release.set()
        await task
        with runtime.require_database().session() as session:
            assert session.get(MemoryEmbedding, memory_id) is None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_query_tool_uses_current_companion_and_filters(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, user_id = seed_conversation(runtime)
    memory_id = seed(runtime, companion_id)
    install_embedding(runtime, monkeypatch)
    try:
        await MemoryIndex(runtime).backfill()
        executor = LocalToolExecutor(runtime)
        assert "query_memory" in str(executor.definitions)
        events = []
        result = await executor.execute(ToolRequest(name="query_memory", arguments={"query": "提神", "type": "user_preference"}),
            model_call_id="q", conversation_id=conversation_id, assistant_message_id=user_id,
            emit=lambda event, data: events.append(event))
        assert not result.is_error and json.loads(result.content)["memories"][0]["id"] == memory_id
        assert events == ["tool_started", "tool_completed"]
        assert await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神", memory_type="boundary") == []
        assert await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神", updated_after=datetime.now(UTC) + timedelta(days=1)) == []
        runtime.memory_retrieval_enabled = False
        assert "query_memory" not in str(executor.definitions)
        assert await SqliteMemoryRetriever(runtime).retrieve(companion_id, "咖啡") == []
    finally:
        runtime.close_storage()


def test_memory_api_rebuild_and_backup_keep_original_text(tmp_path, monkeypatch):
    async def unexpected_call(*args):
        raise AssertionError("This test must never call an embedding provider")
    monkeypatch.setattr(EmbeddingClient, "embed", unexpected_call)
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.post("/api/partner", headers=headers, json=partner_payload())
        with runtime.require_database().session() as session:
            companion_id = session.scalar(select(Companion.id))
        memory_id = seed(runtime, companion_id)
        runtime.save_embedding_auth("embed", "https://isolated.example/v1", "fake")
        # The HTTP embedding is never called in this test; only seed derived cache.
        with runtime.require_database().session() as session:
            item = session.get(Memory, memory_id)
            session.add(MemoryEmbedding(memory_id=memory_id, profile=EmbeddingClient(runtime.embedding_auth).profile,
                model="embed", dimension=2, content_hash=content_hash(item), vector=np.asarray([1, 0], dtype="<f4").tobytes()))
        listed = client.get("/api/memories", headers=headers).json()["data"]["items"]
        assert listed[0]["embeddingStatus"] == "indexed"
        updated = client.put(f"/api/memories/{memory_id}", headers=headers, json={"type": "user_preference", "title": "新偏好", "content": "改成不喝咖啡。", "importance": 3})
        assert updated.json()["data"]["embeddingStatus"] == "pending"
        assert client.post("/api/settings/embedding/rebuild", headers=headers).status_code == 200
        backup = client.get("/api/data/export", headers=headers).json()
        assert "memoryEmbeddings" not in backup["tables"] and "vector" not in str(backup["tables"]["memories"])
        active_rows = [row for row in backup["tables"]["memories"] if row["status"] == "active"]
        assert len(active_rows) == 1 and active_rows[0]["content"] == "改成不喝咖啡。"
        assert next(row for row in backup["tables"]["memories"] if row["id"] == memory_id)["status"] == "retracted"
        assert client.delete(f"/api/memories/{memory_id}", headers=headers).status_code == 200
        with runtime.require_database().session() as session:
            assert session.get(MemoryEmbedding, memory_id) is None


@pytest.mark.parametrize("arguments", [{"query": "x", "limit": 21}, {"query": "x", "limit": True}, {"query": "x", "updated_after": "2026-10-20T00:00:00"}])
def test_query_parameter_validation(arguments):
    with pytest.raises(ValidationError):
        QueryMemoryArguments.model_validate(arguments)


@pytest.mark.asyncio
async def test_retrieval_preserves_semantic_rank_and_utc_time_filters(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    memory_id = seed(runtime, companion_id)
    async def ranked(*args, **kwargs):
        return {memory_id: .95, memory_id - 1: .6}
    monkeypatch.setattr(MemoryIndex, "search", ranked)
    try:
        with runtime.require_database().session() as session:
            session.get(Memory, memory_id).updated_at = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
        rows = await SqliteMemoryRetriever(runtime).retrieve(companion_id, "提神饮品", limit=1,
            updated_after=datetime(2020, 1, 1, 11, 59, tzinfo=UTC),
            updated_before=datetime(2020, 1, 1, 12, 1, tzinfo=UTC))
        assert [item.id for item in rows] == [memory_id]
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_same_model_dimension_change_invalidates_then_backfills(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    memory_id = seed(runtime, companion_id)
    runtime.save_embedding_auth("embed", "https://isolated.example/v1", "fake")
    class Model:
        auth = runtime.embedding_auth
        profile = EmbeddingClient(auth).profile
        dimension = 2
        async def embed(self, texts):
            return [[1] + [0] * (self.dimension - 1) for _ in texts]
    model = Model()
    try:
        index = MemoryIndex(runtime)
        assert await index.backfill(model) == 2
        model.dimension = 4
        assert await index.search(companion_id, "query", client=model) == {}
        assert await index.backfill(model) == 2
        assert memory_id in await index.search(companion_id, "query", client=model)
        with runtime.require_database().session() as session:
            assert {item.dimension for item in session.scalars(select(MemoryEmbedding))} == {4}
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_worker_cancellation_joins_native_index_task_before_storage_close(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    runtime.save_embedding_auth("embed", "https://isolated.example/v1", "fake")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    async def embed(self, texts):
        return [[1, 0] for _ in texts]
    def sync(self, *args):
        started.set()
        release.wait(timeout=3)
        assert self.runtime.require_database() is not None
        finished.set()
    monkeypatch.setattr(EmbeddingClient, "embed", embed)
    monkeypatch.setattr(MemoryIndex, "_sync_all", sync)
    try:
        task = asyncio.create_task(MemoryIndexWorker(runtime).run_forever())
        for _ in range(300):
            if started.is_set(): break
            await asyncio.sleep(.01)
        assert started.is_set()
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        runtime.close_storage()
