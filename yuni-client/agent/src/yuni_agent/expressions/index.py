from __future__ import annotations

import hashlib

from sqlalchemy import delete, select

from yuni_agent.model.embedding import EmbeddingClient, normalized_vector
from yuni_agent.storage.models import Expression, ExpressionVector


def expression_text(item) -> str:
    return f"情境：{item.situation}\n表达方式：{item.style}"


def content_hash(item) -> str:
    return hashlib.sha256(expression_text(item).encode()).hexdigest()


from operator import mul


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("vector dimension mismatch")
    return sum(map(mul, left, right))


def rank_candidates(candidates, vectors, query, limit=50):
    """MaiBot item/cluster relevance (0.875/0.125) and incremental MMR (0.85)."""
    clusters = []
    assignments = []
    for vector in vectors:
        scores = [dot(vector, center) for center in clusters]
        if not scores or (max(scores) < 0.75 and len(clusters) < 16):
            assignments.append(len(clusters))
            clusters.append(vector)
        else:
            assignments.append(max(range(len(scores)), key=scores.__getitem__))
    centers = []
    for cluster in range(len(clusters)):
        members = [v for v, group in zip(vectors, assignments, strict=True) if group == cluster]
        summed = [sum(values) for values in zip(*members)]
        centers.append(normalized_vector(summed) if any(summed) else clusters[cluster])
    relevance = [0.875 * dot(vector, query) + 0.125 * dot(centers[group], query) for vector, group in zip(vectors, assignments, strict=True)]
    selected = []
    penalties = [float("-inf")] * len(candidates)
    remaining = sorted(range(len(candidates)))
    while remaining and len(selected) < min(50, limit):
        if selected:
            last_vec = vectors[selected[-1]]
            for index in remaining:
                sim = dot(vectors[index], last_vec)
                if sim > penalties[index]:
                    penalties[index] = sim
            best = max(remaining, key=lambda index: 0.85 * relevance[index] - 0.15 * penalties[index])
        else:
            best = max(remaining, key=relevance.__getitem__)
        selected.append(best)
        remaining.remove(best)
    return [candidates[index] for index in sorted(selected, key=lambda index: relevance[index], reverse=True)]


class ExpressionIndex:
    def __init__(self, runtime):
        self.runtime = runtime

    def client(self):
        auth = self.runtime.embedding_auth
        return EmbeddingClient(auth) if auth and auth.ready else None

    async def backfill(self, client=None, batch_size=16):
        with self.runtime.embedding_lock:
            client = client or self.client()
            generation = self.runtime.embedding_generation
            data_generation = self.runtime.data_generation
        if client is None:
            return 0
        with self.runtime.require_database().session() as session:
            items = list(session.scalars(select(Expression).order_by(Expression.id)))
            pending = []
            for item in items:
                stored = session.get(ExpressionVector, item.id)
                valid = False
                if stored:
                    try:
                        valid = len(normalized_vector(stored.vector)) == stored.dimension
                    except Exception:
                        pass
                if not valid or stored.profile != client.profile or stored.content_hash != content_hash(item):
                    pending.append((item.id, content_hash(item), expression_text(item)))
                if len(pending) >= batch_size:
                    break
        if not pending:
            return 0
        vectors = await client.embed([item[2] for item in pending])
        dimension = len(vectors[0])
        with self.runtime.embedding_lock, self.runtime.require_database().session() as session:
            if generation != self.runtime.embedding_generation or data_generation != self.runtime.data_generation:
                return 0
            session.execute(delete(ExpressionVector).where((ExpressionVector.profile != client.profile) | (ExpressionVector.dimension != dimension)))
            saved = 0
            for (item_id, digest, _), vector in zip(pending, vectors, strict=True):
                item = session.get(Expression, item_id)
                if item is None or content_hash(item) != digest:
                    continue
                session.merge(ExpressionVector(expression_id=item_id, profile=client.profile,
                    model=client.auth.model, dimension=dimension, content_hash=digest, vector=vector))
                saved += 1
        return saved

    async def recall(self, candidates, query, client=None):
        with self.runtime.embedding_lock:
            client = client or self.client()
            generation = self.runtime.embedding_generation
        if client is None:
            return []
        query_vector = (await client.embed([query]))[0]
        selected, vectors = [], []
        with self.runtime.embedding_lock, self.runtime.require_database().session() as session:
            if generation != self.runtime.embedding_generation:
                return []
            # A changed server-side dimension invalidates old projections, not learned facts.
            session.execute(delete(ExpressionVector).where((ExpressionVector.profile != client.profile) | (ExpressionVector.dimension != len(query_vector))))
            for candidate in candidates:
                stored = session.get(ExpressionVector, candidate['id'])
                item = session.get(Expression, candidate['id'])
                if not stored or not item or item.companion_id != candidate['companion_id'] or stored.content_hash != content_hash(item):
                    continue
                try:
                    vector = normalized_vector(stored.vector)
                except Exception:
                    continue
                if len(vector) != len(query_vector) or stored.model != client.auth.model:
                    continue
                selected.append(candidate)
                vectors.append(vector)
        return rank_candidates(selected, vectors, query_vector) if selected else []
