from __future__ import annotations

import base64
import binascii
import json
import os
from datetime import timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import GroupsEdgesNotFoundError
from graphiti_core.nodes import EntityNode

from graph_service.config import Settings, get_settings
from graph_service.connections.manager import ConnectionManager, ConnectionManagerError
from graph_service.sources.connectors import SUPPORTED_SUFFIXES, ConnectorError
from graph_service.sources.models import (
    FileBatchUploadRequest,
    SourceCreateRequest,
    SourceUpdateRequest,
    SyncRequest,
)
from graph_service.sources.store import SourceStore
from graph_service.sources.sync import SyncManager
from graph_service.zep_graphiti import ZepGraphitiDep, is_llm_configured

router = APIRouter(prefix='/api', tags=['data-sources'])


def get_store(request: Request) -> SourceStore:
    return request.app.state.source_store


def get_sync_manager(request: Request) -> SyncManager:
    return request.app.state.sync_manager


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


StoreDep = Annotated[SourceStore, Depends(get_store)]
SyncManagerDep = Annotated[SyncManager, Depends(get_sync_manager)]
ConnectionManagerDep = Annotated[ConnectionManager, Depends(get_connection_manager)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

MAX_CONFIG_BYTES = 32 * 1024
MAX_CONFIG_LIST_ITEMS = 200
SENSITIVE_CONFIG_KEYS = {
    'api_key',
    'app_secret',
    'access_token',
    'password',
    'plugin_secret',
    'refresh_token',
    'token',
    'user_key',
}


def _not_found(kind: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f'{kind} {identifier} 不存在')


def _config_string(config: dict[str, Any], key: str, *, required: bool = False) -> str:
    value = config.get(key, '')
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f'config.{key} 必须是字符串')
    value = value.strip()
    if required and not value:
        raise HTTPException(status_code=422, detail=f'config.{key} 不能为空')
    if len(value) > 256 or any(ord(character) < 32 for character in value):
        raise HTTPException(status_code=422, detail=f'config.{key} 格式无效')
    return value


def _config_string_list(config: dict[str, Any], key: str) -> list[str]:
    values = config.get(key, [])
    if not isinstance(values, list):
        raise HTTPException(status_code=422, detail=f'config.{key} 必须是字符串列表')
    if len(values) > MAX_CONFIG_LIST_ITEMS:
        raise HTTPException(status_code=422, detail=f'config.{key} 条目过多')
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f'config.{key} 必须是字符串列表')
        value = value.strip()
        if value and value not in result:
            if len(value) > 256 or any(ord(character) < 32 for character in value):
                raise HTTPException(status_code=422, detail=f'config.{key} 含无效条目')
            result.append(value)
    return result


def _normalize_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        'local': set(),
        'feishu': {
            'folder_token',
            'document_tokens',
            'document_metadata',
            'recursive',
            'root_folder',
        },
        'meego': {'project_key', 'work_item_type_keys', 'page_size'},
    }[kind]
    rejected = set(config) - allowed
    if rejected:
        secret = rejected & SENSITIVE_CONFIG_KEYS
        if secret:
            names = ', '.join(sorted(secret))
            raise HTTPException(
                status_code=422,
                detail=f'凭证字段 {names} 不允许写入数据源；请使用账号授权连接',
            )
        raise HTTPException(
            status_code=422, detail=f'不支持的 config 字段：{", ".join(sorted(rejected))}'
        )

    normalized: dict[str, Any]
    if kind == 'local':
        normalized = {}
    elif kind == 'feishu':
        recursive = config.get('recursive', True)
        if not isinstance(recursive, bool):
            raise HTTPException(status_code=422, detail='config.recursive 必须是布尔值')
        root_folder = config.get('root_folder', False)
        if not isinstance(root_folder, bool):
            raise HTTPException(status_code=422, detail='config.root_folder 必须是布尔值')
        document_tokens = _config_string_list(config, 'document_tokens')
        raw_metadata = config.get('document_metadata', {})
        if not isinstance(raw_metadata, dict):
            raise HTTPException(status_code=422, detail='config.document_metadata 必须是对象')
        document_metadata: dict[str, dict[str, str]] = {}
        for token in document_tokens:
            item = raw_metadata.get(token, {})
            if not isinstance(item, dict):
                raise HTTPException(status_code=422, detail='config.document_metadata 条目无效')
            resource_type = str(item.get('type') or 'docx').strip()
            name = str(item.get('name') or token).strip()
            if resource_type not in {'docx', 'file'} or not name or len(name) > 255:
                raise HTTPException(status_code=422, detail='config.document_metadata 条目无效')
            document_metadata[token] = {'type': resource_type, 'name': name}
        normalized = {
            'folder_token': _config_string(config, 'folder_token'),
            'document_tokens': document_tokens,
            'document_metadata': document_metadata,
            'recursive': recursive,
            'root_folder': root_folder,
        }
        if not (
            normalized['root_folder'] or normalized['folder_token'] or normalized['document_tokens']
        ):
            raise HTTPException(status_code=422, detail='请从飞书空间选择根目录、文件夹或文档')
    else:
        page_size = config.get('page_size', 100)
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= 200
        ):
            raise HTTPException(status_code=422, detail='config.page_size 必须是 1 到 200 的整数')
        normalized = {
            'project_key': _config_string(config, 'project_key', required=True),
            'work_item_type_keys': _config_string_list(config, 'work_item_type_keys'),
            'page_size': page_size,
        }

    if len(json.dumps(normalized, ensure_ascii=False).encode('utf-8')) > MAX_CONFIG_BYTES:
        raise HTTPException(status_code=422, detail='数据源 config 过大')
    return normalized


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    result = dict(source)
    result['config'] = {
        key: value
        for key, value in source.get('config', {}).items()
        if key.casefold() not in SENSITIVE_CONFIG_KEYS
    }
    return result


def _suggested_group_id(settings: Settings) -> str:
    if settings.db_backend == 'falkordb':
        return settings.falkordb_database or 'default_db'
    return settings.neo4j_database or 'neo4j'


def _validate_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible name for tests and callers; returns normalized config."""
    return _normalize_config(kind, config)


def _ensure_source_idle(store: SourceStore, source_id: str) -> None:
    if store.active_job_for_source(source_id):
        raise HTTPException(status_code=409, detail='数据源正在同步，请等待任务结束后重试')


def _decode_uploads(
    request: FileBatchUploadRequest, settings: Settings
) -> list[tuple[str, bytes, float | None]]:
    decoded: list[tuple[str, bytes, float | None]] = []
    names: set[str] = set()
    total_size = 0
    for upload in request.files:
        filename = upload.filename
        canonical_name = filename.casefold()
        if canonical_name in names:
            raise HTTPException(status_code=422, detail=f'批次中包含重复文件名：{filename}')
        names.add(canonical_name)
        if Path(filename).suffix.casefold() not in SUPPORTED_SUFFIXES:
            supported = ', '.join(sorted(SUPPORTED_SUFFIXES))
            raise HTTPException(status_code=415, detail=f'{filename} 不受支持；支持：{supported}')
        try:
            data = base64.b64decode(upload.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f'{filename} 的 Base64 内容无效') from exc
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f'{filename} 超过 {settings.max_upload_bytes // (1024 * 1024)} MiB 限制',
            )
        total_size += len(data)
        if total_size > settings.max_upload_bytes * 4:
            raise HTTPException(status_code=413, detail='单次批量上传总大小超过限制')
        modified_at = (
            upload.modified_at.astimezone(timezone.utc).timestamp() if upload.modified_at else None
        )
        decoded.append((filename, data, modified_at))
    return decoded


@router.get('/status')
async def dashboard_status(request: Request, settings: SettingsDep, store: StoreDep):
    llm_ready = is_llm_configured(settings)
    connection_manager: ConnectionManager = request.app.state.connection_manager
    oauth_providers = connection_manager.provider_status()
    connections = connection_manager.store.list_connections()
    return {
        'service': 'Graphiti Studio',
        'database': {
            'provider': settings.db_backend,
            'configured': bool(
                settings.db_backend == 'falkordb'
                or (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password)
            ),
            'ready': bool(getattr(request.app.state, 'database_ready', False)),
            'error': getattr(request.app.state, 'database_error', None),
        },
        'llm': {
            'configured': llm_ready,
            'provider': 'ark/openai-compatible'
            if settings.openai_base_url and 'api.openai.com' not in settings.openai_base_url
            else 'openai',
            'model': settings.model_name,
            'structured_output_mode': settings.structured_output_mode,
        },
        'embedding': {
            'provider': settings.embedding_provider,
            'model': settings.embedding_model_name
            if settings.embedding_provider == 'openai'
            else 'local feature hash',
            'dimensions': settings.embedding_dim,
        },
        'connectors': {
            'feishu': oauth_providers['feishu'],
            'meego': oauth_providers['meego'],
            'local': True,
        },
        'oauth': {
            'providers': oauth_providers,
            'active_connections': sum(item['status'] == 'active' for item in connections),
        },
        'suggested_group_id': _suggested_group_id(settings),
        'mcp_url': settings.mcp_public_url,
        'stats': store.stats(),
    }


@router.get('/sources')
async def list_sources(store: StoreDep):
    return [_public_source(source) for source in store.list_sources()]


@router.post('/sources', status_code=status.HTTP_201_CREATED)
async def create_source(
    request: SourceCreateRequest,
    store: StoreDep,
    connections: ConnectionManagerDep,
):
    values = request.model_dump()
    values['config'] = _normalize_config(request.kind, request.config)
    if request.kind == 'local':
        if request.connection_id is not None:
            raise HTTPException(status_code=422, detail='本地数据源不能绑定远程账号')
    else:
        if not request.connection_id:
            raise HTTPException(status_code=422, detail='请先连接账号并选择可见资源')
        try:
            connections.validate_connection(request.connection_id, request.kind)
        except ConnectionManagerError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _public_source(store.create_source(**values))


@router.patch('/sources/{source_id}')
async def update_source(
    source_id: str,
    request: SourceUpdateRequest,
    store: StoreDep,
    connections: ConnectionManagerDep,
):
    values = request.model_dump(exclude_unset=True)
    try:
        current = store.get_source(source_id)
    except KeyError as exc:
        raise _not_found('数据源', source_id) from exc
    if 'config' in values:
        if values['config'] is None:
            raise HTTPException(status_code=422, detail='config 不能为空')
        values['config'] = _normalize_config(current['kind'], values['config'])
    if current['kind'] == 'local':
        if values.get('connection_id') is not None:
            raise HTTPException(status_code=422, detail='本地数据源不能绑定远程账号')
    else:
        connection_id = values.get('connection_id', current.get('connection_id'))
        if not connection_id:
            raise HTTPException(status_code=422, detail='远程数据源必须绑定授权账号')
        try:
            connections.validate_connection(connection_id, current['kind'])
        except ConnectionManagerError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    state_changed = any(
        key in values and values[key] != current.get(key)
        for key in ('config', 'group_id', 'connection_id')
    )
    if {'config', 'enabled', 'group_id', 'connection_id'} & values.keys():
        _ensure_source_idle(store, source_id)
    updated = store.update_source(source_id, values)
    if state_changed:
        store.reset_source_fingerprints(source_id)
        updated = store.get_source(source_id)
    return _public_source(updated)


@router.delete('/sources/{source_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(source_id: str, store: StoreDep):
    try:
        _ensure_source_idle(store, source_id)
        store.delete_source(source_id)
    except KeyError as exc:
        raise _not_found('数据源', source_id) from exc


@router.get('/sources/{source_id}/items')
async def list_source_items(
    source_id: str,
    store: StoreDep,
    limit: int = Query(default=100, ge=1, le=500),
):
    try:
        store.get_source(source_id)
    except KeyError as exc:
        raise _not_found('数据源', source_id) from exc
    return store.list_items(source_id, limit)


@router.post('/sources/{source_id}/files')
async def upload_files(
    source_id: str,
    request: FileBatchUploadRequest,
    store: StoreDep,
    manager: SyncManagerDep,
    settings: SettingsDep,
):
    try:
        source = store.get_source(source_id)
    except KeyError as exc:
        raise _not_found('数据源', source_id) from exc
    if source['kind'] != 'local':
        raise HTTPException(status_code=409, detail='只有本地数据源支持文件上传')
    if request.sync and not source['enabled']:
        raise HTTPException(status_code=409, detail='数据源已停用；请先启用或关闭自动同步')
    if request.sync:
        _ensure_source_idle(store, source_id)

    # Decode and validate the complete batch before touching the filesystem. A malformed later
    # entry must not leave earlier files silently committed.
    decoded = _decode_uploads(request, settings)

    source_root = (Path(settings.upload_root) / source_id).resolve()
    upload_root = Path(settings.upload_root).resolve()
    if upload_root not in source_root.parents:
        raise HTTPException(status_code=400, detail='无效的数据源路径')
    source_root.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, Any]] = []
    staged: list[tuple[Path, Path, float | None, int]] = []
    try:
        for filename, data, modified_at in decoded:
            target = source_root / filename
            temporary = source_root / f'.{filename}.{uuid4().hex}.uploading'
            with temporary.open('xb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, target, modified_at, len(data)))

        for temporary, target, modified_at, size in staged:
            temporary.replace(target)
            if modified_at is not None:
                os.utime(target, (modified_at, modified_at))
            saved.append({'filename': target.name, 'size': size})
    finally:
        for temporary, _, _, _ in staged:
            temporary.unlink(missing_ok=True)

    try:
        job = manager.enqueue(source_id) if request.sync else None
    except ConnectorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {'saved': saved, 'job': job}


@router.post('/sources/{source_id}/sync', status_code=status.HTTP_202_ACCEPTED)
async def sync_source(
    source_id: str,
    request: SyncRequest,
    store: StoreDep,
    manager: SyncManagerDep,
):
    try:
        store.get_source(source_id)
        return manager.enqueue(source_id, full_sync=request.full)
    except KeyError as exc:
        raise _not_found('数据源', source_id) from exc
    except ConnectorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get('/jobs')
async def list_jobs(store: StoreDep, limit: int = Query(default=50, ge=1, le=200)):
    return store.list_jobs(limit)


@router.get('/jobs/{job_id}')
async def get_job(job_id: str, store: StoreDep):
    try:
        return store.get_job(job_id)
    except KeyError as exc:
        raise _not_found('同步任务', job_id) from exc


@router.get('/graph')
async def graph_snapshot(
    graphiti: ZepGraphitiDep,
    group_id: str = Query(default='neo4j', pattern=r'^[a-zA-Z0-9_-]+$'),
    limit: int = Query(default=120, ge=1, le=500),
):
    nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id], limit=limit)
    try:
        edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id], limit=limit * 2)
    except GroupsEdgesNotFoundError:
        edges = []
    node_ids = {node.uuid for node in nodes}
    visible_edges = [
        edge
        for edge in edges
        if edge.source_node_uuid in node_ids and edge.target_node_uuid in node_ids
    ]
    return {
        'nodes': [
            {
                'id': node.uuid,
                'name': node.name,
                'summary': node.summary,
                'labels': node.labels,
                'created_at': node.created_at,
            }
            for node in nodes
        ],
        'edges': [
            {
                'id': edge.uuid,
                'source': edge.source_node_uuid,
                'target': edge.target_node_uuid,
                'name': edge.name,
                'fact': edge.fact,
                'valid_at': edge.valid_at,
                'invalid_at': edge.invalid_at,
                'expired_at': edge.expired_at,
            }
            for edge in visible_edges
        ],
    }
