from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator


class AuthFileError(Exception):
    pass


def normalize_model_url(value: str) -> str:
    normalized = value.strip().rstrip('/')
    parsed = urlsplit(normalized)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError('Base URL 无效')
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('Base URL 仅支持 HTTP 或 HTTPS')
    return normalized


class ModelCredentials(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model: str = Field(min_length=1, max_length=120)
    url: str = Field(min_length=1, max_length=500)
    apikey: SecretStr = Field(repr=False)

    @field_validator('model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('模型名不能为空')
        return value

    @field_validator('url')
    @classmethod
    def validate_url(cls, value: str) -> str:
        return normalize_model_url(value)

    @field_validator('apikey')
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if len(key) > 1000:
            raise ValueError('API Key 过长')
        return SecretStr(key)

    @property
    def ready(self) -> bool:
        return bool(self.model and self.url and self.apikey.get_secret_value())


class ModelAuth(ModelCredentials):
    # Accept legacy Jev profiles without using or persisting their values.
    jev: object | None = Field(default=None, exclude=True, repr=False)


class ImageAuth(BaseModel):
    """Local image-generation settings (image-auth.json); the key never leaves this file."""
    model_config = ConfigDict(extra='ignore')

    enabled: bool = False
    model: str = Field(default='', max_length=120)
    url: str = Field(default='', max_length=500)
    apikey: SecretStr = Field(default=SecretStr(''), repr=False)
    default_size: Literal['1024x1024', '768x1024', '1024x768'] = '1024x1024'
    style_prefix: str = Field(default='', max_length=300)

    @field_validator('model', 'style_prefix')
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator('url')
    @classmethod
    def validate_url(cls, value: str) -> str:
        return normalize_model_url(value) if value.strip() else ''

    @field_validator('apikey')
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if len(key) > 1000:
            raise ValueError('API Key 过长')
        return SecretStr(key)

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.model and self.url and self.apikey.get_secret_value())


class VoiceAuth(BaseModel):
    """Local voice-generation settings (voice-auth.json); the key never leaves this file."""
    model_config = ConfigDict(extra='ignore')

    enabled: bool = False
    model: str = Field(default='', max_length=120)
    url: str = Field(default='', max_length=500)
    apikey: SecretStr = Field(default=SecretStr(''), repr=False)
    voice: str = Field(default='alloy', max_length=120)
    trigger_mode: Literal['explicit_only', 'occasional', 'always'] = 'occasional'

    @field_validator('model', 'voice')
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator('url')
    @classmethod
    def validate_url(cls, value: str) -> str:
        return normalize_model_url(value) if value.strip() else ''

    @field_validator('apikey')
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if len(key) > 1000:
            raise ValueError('API Key 过长')
        return SecretStr(key)

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.model and self.url and self.voice and self.apikey.get_secret_value())


def load_image_auth(path: Path) -> ImageAuth | None:
    try:
        if not path.exists():
            return None
        if path.stat().st_size > 16 * 1024:
            raise AuthFileError('image-auth.json 无效')
        return ImageAuth.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        raise AuthFileError('image-auth.json 无法读取或格式无效') from None


def save_image_auth(path: Path, auth: ImageAuth) -> None:
    payload = {**auth.model_dump(exclude={'apikey'}), 'apikey': auth.apikey.get_secret_value()}
    _atomic_json(path, payload)


def load_voice_auth(path: Path) -> VoiceAuth | None:
    try:
        if not path.exists():
            return None
        if path.stat().st_size > 16 * 1024:
            raise AuthFileError('voice-auth.json 无效')
        return VoiceAuth.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        raise AuthFileError('voice-auth.json 无法读取或格式无效') from None


def save_voice_auth(path: Path, auth: VoiceAuth) -> None:
    payload = {**auth.model_dump(exclude={'apikey'}), 'apikey': auth.apikey.get_secret_value()}
    _atomic_json(path, payload)


def load_auth(path: Path) -> ModelAuth | None:
    try:
        if not path.exists():
            return None
        if path.stat().st_size > 16 * 1024:
            raise AuthFileError('auth.json 无效')
        return ModelAuth.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        # Validation exceptions may contain the secret input; do not expose their text.
        raise AuthFileError('auth.json 无法读取或格式无效') from None


def save_auth(path: Path, auth: ModelAuth) -> None:
    _atomic_json(path, {'model': auth.model, 'url': auth.url, 'apikey': auth.apikey.get_secret_value()})


def _atomic_json(path: Path, payload: dict) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.auth-', suffix='.tmp', dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except OSError:
        raise AuthFileError('auth.json 无法保存') from None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def remove_auth(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        raise AuthFileError('auth.json 无法删除') from None
