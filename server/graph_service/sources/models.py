from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SourceKind = Literal['local', 'feishu', 'meego']
DEFAULT_GROUP_ID = 'neo4j'


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SourceDocument:
    external_id: str
    title: str
    content: str
    updated_at: datetime
    remote_version: str = ''
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        canonical = self.content.replace('\r\n', '\n').replace('\r', '\n').strip()
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def episode_body(self, source_kind: str) -> str:
        header = {
            'source': source_kind,
            'external_id': self.external_id,
            'title': self.title,
            'updated_at': self.updated_at.astimezone(timezone.utc).isoformat(),
            'url': self.url,
            **self.metadata,
        }
        return f'{json.dumps(header, ensure_ascii=False, sort_keys=True)}\n\n{self.content.strip()}'


class SourceCreateRequest(BaseModel):
    kind: SourceKind
    name: str = Field(min_length=1, max_length=120)
    wiki_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    group_id: str = Field(default=DEFAULT_GROUP_ID, pattern=r'^[a-zA-Z0-9_-]+$')
    connection_id: str | None = Field(default=None, min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator('name')
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('name 不能为空')
        return value


class SourceUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    group_id: str | None = Field(default=None, pattern=r'^[a-zA-Z0-9_-]+$')
    connection_id: str | None = Field(default=None, min_length=1, max_length=64)
    config: dict[str, Any] | None = None
    enabled: bool | None = None

    @field_validator('name')
    @classmethod
    def strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError('name 不能为空')
        return value


class FileUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1)
    modified_at: datetime | None = None

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = value.strip()
        if (
            not value
            or value in {'.', '..'}
            or '/' in value
            or '\\' in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError('filename 必须是不含路径的普通文件名')
        return value

    @field_validator('modified_at')
    @classmethod
    def validate_modified_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError('modified_at 必须包含时区')
        return value


class FileBatchUploadRequest(BaseModel):
    files: list[FileUploadRequest] = Field(min_length=1, max_length=50)
    sync: bool = True


class SyncRequest(BaseModel):
    full: bool = False
