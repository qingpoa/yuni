"""OpenAI-compatible image generation and local storage of generated images."""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from yuni_agent.model.catalog import v1_endpoint

IMAGE_DIR = "generated-images"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_RESPONSE_BYTES = 24 * 1024 * 1024  # base64 inflates ~4/3
# Upper bound for an avatar used as an image-to-image reference upload.
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
IMAGE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
SIZES = ("1024x1024", "768x1024", "1024x768")
ASPECT_SIZES = {"1:1": "1024x1024", "3:4": "768x1024", "4:3": "1024x768"}


class ImageGenerationError(Exception):
    """Message is safe to show to the model and the user (never includes secrets)."""


def sniff_image_type(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_path(data_dir: Path, image_id: str, content_type: str) -> Path:
    if not IMAGE_ID_RE.fullmatch(image_id) or content_type not in EXTENSIONS:
        raise ValueError("图片标识无效")
    root = (data_dir / IMAGE_DIR).resolve()
    target = (root / f"{image_id}.{EXTENSIONS[content_type]}").resolve()
    if not target.is_relative_to(root):
        raise ValueError("图片路径无效")
    return target


def image_data(image: dict[str, Any]) -> dict[str, str]:
    """Public DTO for messages and events."""
    return {"id": image["id"], "url": f"/api/chat/images/{image['id']}",
            "description": str(image.get("description") or "")}


def store_image(data_dir: Path, content: bytes, description: str) -> dict[str, Any]:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ImageGenerationError("生成的图片为空或超过 15 MB")
    content_type = sniff_image_type(content)
    if content_type is None:
        raise ImageGenerationError("生成结果不是 PNG/JPEG/WebP 图片")
    image_id = hashlib.sha256(content).hexdigest()
    path = image_path(data_dir, image_id, content_type)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".img-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"id": image_id, "content_type": content_type, "description": description[:500]}


async def _download(client: httpx.AsyncClient, url: str) -> bytes:
    if not url.lower().startswith(("http://", "https://")):
        raise ImageGenerationError("生图服务返回的图片地址无效")
    chunks: list[bytes] = []
    size = 0
    async with client.stream("GET", url) as response:
        if response.status_code >= 400:
            raise ImageGenerationError(f"下载生成图片失败（HTTP {response.status_code}）")
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise ImageGenerationError("生成的图片超过 15 MB")
            chunks.append(chunk)
    return b"".join(chunks)


async def _decode_response(client: httpx.AsyncClient, response: httpx.Response) -> bytes:
    if response.status_code in {401, 403}:
        raise ImageGenerationError("生图服务鉴权失败，请检查 API Key")
    if response.status_code >= 400:
        raise ImageGenerationError(f"生图请求失败（HTTP {response.status_code}）")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ImageGenerationError("生图响应过大")
    try:
        item = response.json()["data"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        raise ImageGenerationError("生图响应格式无效") from None
    if not isinstance(item, dict):
        raise ImageGenerationError("生图响应格式无效")
    if isinstance(item.get("b64_json"), str):
        try:
            return base64.b64decode(item["b64_json"], validate=True)
        except (binascii.Error, ValueError):
            raise ImageGenerationError("生图响应中的图片数据无效") from None
    if isinstance(item.get("url"), str):
        return await _download(client, item["url"])
    raise ImageGenerationError("生图响应中没有图片")


async def generate_image(*, base_url: str, api_key: str, model: str, prompt: str, size: str,
                         timeout: float = 55.0) -> bytes:
    """Text-to-image. Returns raw bytes; the image type is validated on store."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), follow_redirects=True) as client:
            response = await client.post(
                v1_endpoint(base_url, "images/generations"),
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "prompt": prompt, "n": 1, "size": size},
            )
            return await _decode_response(client, response)
    except httpx.TimeoutException:
        raise ImageGenerationError("生图超时，请稍后再试") from None
    except httpx.HTTPError:
        raise ImageGenerationError("生图服务连接失败") from None


async def edit_image(*, base_url: str, api_key: str, model: str, prompt: str,
                     image_bytes: bytes, image_type: str, size: str, timeout: float = 55.0) -> bytes:
    """Image-to-image via /v1/images/edits with a reference image (multipart)."""
    extension = EXTENSIONS.get(image_type, "png")
    files = {"image": (f"reference.{extension}", image_bytes, image_type)}
    data = {"model": model, "prompt": prompt, "n": "1", "size": size}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), follow_redirects=True) as client:
            response = await client.post(
                v1_endpoint(base_url, "images/edits"),
                headers={"Authorization": f"Bearer {api_key}"},
                data=data, files=files,
            )
            return await _decode_response(client, response)
    except httpx.TimeoutException:
        raise ImageGenerationError("生图超时，请稍后再试") from None
    except httpx.HTTPError:
        raise ImageGenerationError("生图服务连接失败") from None
