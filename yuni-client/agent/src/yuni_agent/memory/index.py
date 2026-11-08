from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy import delete, or_, select

from yuni_agent.model.embedding import EmbeddingClient, normalized_vector
from yuni_agent.storage.models import Memory, MemoryEmbedding

logger = logging.getLogger(__name__)
MIN_SIMILARITY = 0.32


def memory_text(item: Memory) -> str:
    content = item.content.strip()
    prefixes = [value for value in (item.qualifier, item.subject_key) if value and value not in content]
    # Internal dotted/slashed keys are identifiers, not natural language.
    import re
    prefixes = [value for value in prefixes if not re.search(r"[._/]", value)]
    if prefixes:
        content = "，".join(dict.fromkeys(prefixes)) + "：" + content
    if item.kind == "event" and item.outcome and item.outcome not in content:
        content += f"（{item.outcome}）"
    return content


def content_hash(item: Memory) -> str:
    return hashlib.sha256(memory_text(item).encode()).hexdigest()


def active_memories(session, companion_id: int | None = None, *, include_history: bool = False) -> list[Memory]:
    now = datetime.now(UTC)
    current = (or_(Memory.valid_from.is_(None), Memory.valid_from <= now)
               & or_(Memory.valid_until.is_(None), Memory.valid_until > now))
    conditions = [Memory.deleted_at.is_(None)]
    if include_history:
        # The index covers every queryable version. Query-time filters decide
        # which interval applies, including past and future active records.
        conditions.append(Memory.status.in_(["active", "historical"]))
    else:
        conditions += [Memory.status == "active", current]
    if companion_id is not None:
        conditions.append(Memory.companion_id == companion_id)
    return list(session.scalars(select(Memory).where(*conditions).order_by(Memory.id)))


def _vector(stored: MemoryEmbedding) -> np.ndarray | None:
    if not 1 <= stored.dimension <= 16384 or len(stored.vector) != stored.dimension * 4:
        return None
    vector = np.frombuffer(stored.vector, dtype="<f4")
    if not np.isfinite(vector).all() or not np.linalg.norm(vector):
        return None
    return np.asarray(normalized_vector(vector.tolist()), dtype="float32")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MemoryIndex:
    """SQLite text is authoritative; float cache and FAISS files are disposable.

    Exact FlatIP over L2-normalized vectors is cosine similarity. Small local
    libraries need no trained quantizer or external database service.
    """

    def __init__(self, runtime):
        self.runtime = runtime

    def client(self):
        auth = self.runtime.embedding_auth
        return EmbeddingClient(auth) if auth and auth.ready else None

    def _snapshot(self, client=None):
        with self.runtime.embedding_lock:
            return (client or self.client(), self.runtime.embedding_generation, self.runtime.data_generation)

    def _current(self, generation, data_generation):
        return generation == self.runtime.embedding_generation and data_generation == self.runtime.data_generation

    async def _io(self, operation, *args):
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A native FAISS call cannot be cancelled. Join it before app shutdown
            # closes SQLite; do not leave an untracked writer behind.
            try:
                await task
            except Exception:
                pass
            raise

    async def backfill(self, client=None, batch_size=16) -> int:
        client, generation, data_generation = self._snapshot(client)
        if client is None or not self.runtime.memory_retrieval_enabled:
            return 0
        with self.runtime.embedding_lock, self.runtime.require_database().session() as session:
            if not self._current(generation, data_generation):
                return 0
            rows = active_memories(session, include_history=True)
            valid_ids = {item.id for item in rows}
            for stored in session.scalars(select(MemoryEmbedding)):
                if stored.memory_id not in valid_ids or stored.profile != client.profile:
                    session.delete(stored)
            pending = []
            for item in rows:
                stored = session.get(MemoryEmbedding, item.id)
                if stored is None or stored.profile != client.profile or stored.model != client.auth.model or stored.content_hash != content_hash(item) or _vector(stored) is None:
                    pending.append((item.id, content_hash(item), memory_text(item)))
                if len(pending) >= batch_size:
                    break
        saved = 0
        if pending:
            vectors = await client.embed([item[2] for item in pending])
            dimension = len(vectors[0])
            with self.runtime.embedding_lock, self.runtime.require_database().session() as session:
                if not self._current(generation, data_generation) or not self.runtime.memory_retrieval_enabled:
                    return 0
                session.execute(delete(MemoryEmbedding).where(
                    (MemoryEmbedding.profile != client.profile) | (MemoryEmbedding.dimension != dimension)))
                current = {item.id: item for item in active_memories(session, include_history=True)}
                for (item_id, digest, _), vector in zip(pending, vectors, strict=True):
                    item = current.get(item_id)
                    if item is None or content_hash(item) != digest:
                        continue
                    values = np.asarray(normalized_vector(vector), dtype="<f4")
                    session.merge(MemoryEmbedding(memory_id=item_id, profile=client.profile, model=client.auth.model,
                                                  dimension=len(values), content_hash=digest, vector=values.tobytes()))
                    saved += 1
        await self._io(self._sync_all, client, generation, data_generation)
        return saved

    def _index(self, companion_id, client, dimension, rows, stored_rows):
        directory = self.runtime.config.data_dir / "memory-index"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"companion-{companion_id}.faiss"
        manifest = directory / f"companion-{companion_id}.json"
        ids, vectors, hashes = [], [], []
        for item in rows:
            stored = stored_rows.get(item.id)
            if stored is None or stored.profile != client.profile or stored.model != client.auth.model or stored.dimension != dimension or stored.content_hash != content_hash(item):
                continue
            vector = _vector(stored)
            if vector is not None:
                ids.append(item.id)
                vectors.append(vector)
                hashes.append(stored.content_hash)
        signature = hashlib.sha256(json.dumps([client.profile, dimension, list(zip(ids, hashes))]).encode()).hexdigest()
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            payload = path.read_bytes()
            if meta.get("signature") == signature and meta.get("sha256") == hashlib.sha256(payload).hexdigest():
                index = faiss.deserialize_index(np.frombuffer(payload, dtype="uint8").copy())
                if isinstance(index, faiss.IndexIDMap2) and index.metric_type == faiss.METRIC_INNER_PRODUCT and index.d == dimension and index.ntotal == len(ids) and list(faiss.vector_to_array(index.id_map)) == ids:
                    return index
        except (OSError, ValueError, RuntimeError, AttributeError):
            pass
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        if ids:
            index.add_with_ids(np.asarray(vectors, dtype="float32"), np.asarray(ids, dtype="int64"))
        payload = faiss.serialize_index(index).tobytes()
        _atomic_write(path, payload)
        _atomic_write(manifest, json.dumps({"signature": signature, "sha256": hashlib.sha256(payload).hexdigest()}).encode())
        return index

    def _sync_all(self, client, generation, data_generation):
        with self.runtime.embedding_lock, self.runtime.memory_index_lock:
            if not self._current(generation, data_generation):
                return
            with self.runtime.require_database().session() as session:
                rows = active_memories(session, include_history=True)
                stored = {item.memory_id: item for item in session.scalars(select(MemoryEmbedding))}
            groups = {}
            for item in rows:
                groups.setdefault(item.companion_id, []).append(item)
            for companion_id, items in groups.items():
                dimension = next((stored[item.id].dimension for item in items if item.id in stored and stored[item.id].profile == client.profile and _vector(stored[item.id]) is not None), None)
                if dimension is not None:
                    self._index(companion_id, client, dimension, items, stored)
                else:
                    directory = self.runtime.config.data_dir / "memory-index"
                    for suffix in (".faiss", ".json"):
                        (directory / f"companion-{companion_id}{suffix}").unlink(missing_ok=True)
            directory = self.runtime.config.data_dir / "memory-index"
            if directory.exists():
                valid_names = {f"companion-{value}" for value in groups}
                for path in directory.iterdir():
                    if path.suffix in {".faiss", ".json"} and path.stem not in valid_names:
                        path.unlink(missing_ok=True)

    async def search(self, companion_id: int, query: str, limit: int = 50, client=None) -> dict[int, float]:
        client, generation, data_generation = self._snapshot(client)
        if client is None or not self.runtime.memory_retrieval_enabled:
            return {}
        # No model call for an empty/unindexed library. Backfill runs separately.
        with self.runtime.require_database().session() as session:
            available = session.scalar(select(MemoryEmbedding.memory_id).join(Memory).where(
                Memory.companion_id == companion_id, MemoryEmbedding.profile == client.profile).limit(1))
        if available is None:
            return {}
        query_vector = normalized_vector((await asyncio.wait_for(client.embed([query]), timeout=20))[0])
        return await self._io(self._search, companion_id, client, query_vector, limit, generation, data_generation)

    def _search(self, companion_id, client, query_vector, limit, generation, data_generation):
        with self.runtime.embedding_lock, self.runtime.memory_index_lock:
            if not self._current(generation, data_generation) or not self.runtime.memory_retrieval_enabled:
                return {}
            with self.runtime.require_database().session() as session:
                # Fresh rows after network I/O: edits/deletes/conflicts cannot leak
                # from an older snapshot, even before the next background pass.
                rows = active_memories(session, companion_id, include_history=True)
                session.execute(delete(MemoryEmbedding).where((MemoryEmbedding.profile != client.profile) | (MemoryEmbedding.dimension != len(query_vector))))
                stored = {item.memory_id: item for item in session.scalars(select(MemoryEmbedding))}
            index = self._index(companion_id, client, len(query_vector), rows, stored)
            if not index.ntotal:
                return {}
            scores, ids = index.search(np.asarray([query_vector], dtype="float32"), min(limit, index.ntotal))
            return {int(item_id): float(score) for item_id, score in zip(ids[0], scores[0], strict=True)
                    if item_id >= 0 and score >= MIN_SIMILARITY}


class MemoryIndexWorker:
    def __init__(self, runtime):
        self.runtime = runtime

    async def run_forever(self):
        index = MemoryIndex(self.runtime)
        while True:
            saved = 0
            try:
                saved = await asyncio.wait_for(index.backfill(), timeout=60)
            except Exception:
                # Provider exceptions may include private input; do not log them.
                logger.warning("长期记忆向量维护失败，将稍后重试")
            await asyncio.sleep(2 if saved else 30)
