from __future__ import annotations

import hashlib
import json
import math

import httpx

from yuni_agent.model.auth import ModelAuth
from yuni_agent.model.client import ModelClientError


def profile_id(auth: ModelAuth) -> str:
    return hashlib.sha256(json.dumps([auth.url, auth.model]).encode()).hexdigest()


def normalized_vector(values: object) -> list[float]:
    if not isinstance(values, list) or not 1 <= len(values) <= 16384:
        raise ModelClientError("Embedding 响应无效")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ModelClientError("Embedding 响应无效")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ModelClientError("Embedding 响应无效")
    return [value / norm for value in values]


class EmbeddingClient:
    def __init__(self, auth: ModelAuth):
        self.auth = auth
        self.profile = profile_id(auth)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
                response = await client.post(self.auth.url.rstrip('/') + '/embeddings',
                    headers={"Authorization": "Bearer " + self.auth.apikey.get_secret_value()},
                    json={"model": self.auth.model, "input": texts})
                response.raise_for_status()
                if len(response.content) > 8 * 1024 * 1024:
                    raise ValueError("oversized embedding")
                payload = response.json()
                if payload.get("model") not in (None, self.auth.model):
                    raise ValueError("embedding model mismatch")
                rows = payload["data"]
                if len(rows) != len(texts) or sorted(row["index"] for row in rows) != list(range(len(texts))):
                    raise ValueError("embedding count mismatch")
                result = [normalized_vector(row["embedding"]) for row in sorted(rows, key=lambda row: row["index"])]
                if len({len(vector) for vector in result}) != 1:
                    raise ValueError("embedding dimension mismatch")
                return result
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError):
            raise ModelClientError("Embedding 请求失败或响应无效") from None
