from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field


class ModelToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelStreamEvent(BaseModel):
    type: str
    text: str = ""
    tool_call: ModelToolCall | None = None
    reasoning_content: str | None = Field(default=None, repr=False)


class ModelClient(Protocol):
    """Provider-neutral OpenAI-compatible model boundary."""

    async def stream_chat(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]: ...

    async def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]: ...

    async def stream_chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ModelStreamEvent]: ...


class ModelClientError(Exception):
    pass


class OpenAICompatibleClient:
    def __init__(self, *, base_url: str, api_key: str, model: str, thinking_enabled: bool | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.thinking_enabled = thinking_enabled

    def for_internal_json(self) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(base_url=self.base_url, api_key=self.api_key, model=self.model, thinking_enabled=False)

    def _provider_options(self) -> dict[str, Any]:
        if (urlsplit(self.base_url).hostname == "api.deepseek.com"
                or self.model.lower().split("/")[-1].startswith("deepseek")):
            return {"thinking": {"type": "disabled" if self.thinking_enabled is False else "enabled"}}
        return {}

    async def stream_chat(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        async for event in self.stream_chat_with_tools(messages, []):
            if event.type == "text":
                yield event.text

    async def stream_chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ModelStreamEvent]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": True,
            "temperature": 1.5,
            **self._provider_options(),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        calls: dict[int, dict[str, str]] = {}
        reasoning_chunks: list[str] = []
        reasoning_size = 0
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15)) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code in {401, 403}:
                        raise ModelClientError("模型凭据无效")
                    if response.status_code >= 400:
                        raise ModelClientError(f"模型服务返回 HTTP {response.status_code}")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            delta_payload = event["choices"][0]["delta"]
                            delta = delta_payload.get("content")
                            reasoning = delta_payload.get("reasoning_content")
                            for call in delta_payload.get("tool_calls", []):
                                index = int(call.get("index", 0))
                                current = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                                if call.get("id"):
                                    current["id"] = call["id"]
                                function = call.get("function", {})
                                current["name"] += function.get("name", "")
                                current["arguments"] += function.get("arguments", "")
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue
                        if tools and isinstance(reasoning, str) and reasoning:
                            reasoning_size += len(reasoning)
                            if reasoning_size > 64000:
                                raise ModelClientError("工具思考内容过长，请缩短请求")
                            reasoning_chunks.append(reasoning)
                        if isinstance(delta, str) and delta:
                            yield ModelStreamEvent(type="text", text=delta)
            for value in calls.values():
                try:
                    arguments = json.loads(value["arguments"] or "{}")
                except json.JSONDecodeError:
                    yield ModelStreamEvent(type="invalid_tool_call")
                    continue
                if value["id"] and value["name"] and isinstance(arguments, dict):
                    yield ModelStreamEvent(
                        type="tool_call",
                        tool_call=ModelToolCall(
                            id=value["id"], name=value["name"], arguments=arguments
                        ),
                        reasoning_content="".join(reasoning_chunks) or None,
                    )
                else:
                    yield ModelStreamEvent(type="invalid_tool_call")
        except httpx.TimeoutException as exc:
            raise ModelClientError("模型响应超时") from exc
        except httpx.HTTPError as exc:
            raise ModelClientError("无法连接模型服务") from exc

    async def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3 if self.thinking_enabled is False else 1.5,
            "response_format": {"type": "json_object"},
            **self._provider_options(),
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15)) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 400:
                    payload.pop("response_format", None)
                    response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                stripped = str(content).strip()
                if stripped.startswith("```"):
                    stripped = stripped.strip("`").removeprefix("json").strip()
                result, _ = json.JSONDecoder().raw_decode(stripped)
            if not isinstance(result, dict):
                raise ValueError("model JSON response must be an object")
            return result
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            raise ModelClientError("模型未返回有效 JSON") from exc
