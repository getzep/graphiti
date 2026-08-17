from __future__ import annotations

from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Request, status
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import GroupsEdgesNotFoundError
from graphiti_core.nodes import EntityNode

from graph_service.config import Settings, get_settings
from graph_service.sources.connectors import ConnectorError
from graph_service.sources.store import SourceStore
from graph_service.sources.sync import SyncManager
from graph_service.wikis.models import WikiCreateRequest, WikiSearchRequest
from graph_service.wikis.templates import project_wiki_plan
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter(prefix='/api/wikis', tags=['wikis'])


def get_store(request: Request) -> SourceStore:
    return request.app.state.source_store


def get_sync_manager(request: Request) -> SyncManager:
    return request.app.state.sync_manager


StoreDep = Annotated[SourceStore, Depends(get_store)]
SyncManagerDep = Annotated[SyncManager, Depends(get_sync_manager)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _not_found(wiki_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f'Wiki {wiki_id} 不存在')


def wiki_mcp_url(base_url: str, wiki_id: str) -> str:
    """Derive a per-Wiki URL while preserving an optional deployment path prefix."""
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip('/')
    if path.endswith('/mcp'):
        path = path[:-4]
    path = f'{path}/wiki/{wiki_id}/mcp'
    return urlunsplit((parsed.scheme, parsed.netloc, path, '', ''))


def _public_wiki(wiki: dict[str, Any], settings: Settings) -> dict[str, Any]:
    result = dict(wiki)
    result['mcp_url'] = wiki_mcp_url(settings.mcp_public_url, wiki['id'])
    return result


@router.post('', status_code=status.HTTP_201_CREATED)
async def create_wiki(request: WikiCreateRequest, store: StoreDep, settings: SettingsDep):
    try:
        values = request.model_dump()
        values['plan'] = project_wiki_plan(request.goal)
        wiki = store.create_wiki(**values)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _public_wiki(wiki, settings)


@router.get('')
async def list_wikis(store: StoreDep, settings: SettingsDep):
    return [_public_wiki(wiki, settings) for wiki in store.list_wikis()]


@router.get('/{wiki_id}')
async def get_wiki(wiki_id: str, store: StoreDep, settings: SettingsDep):
    try:
        return _public_wiki(store.get_wiki(wiki_id), settings)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc


@router.get('/{wiki_id}/plan')
async def get_wiki_plan(wiki_id: str, store: StoreDep):
    try:
        wiki = store.get_wiki(wiki_id)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    return {
        'wiki_id': wiki_id,
        'version': wiki['plan_version'],
        'goal': wiki['goal'],
        'plan': wiki['plan'],
        'created_at': wiki['created_at'],
    }


@router.get('/{wiki_id}/task')
async def get_wiki_task(wiki_id: str, store: StoreDep):
    try:
        wiki = store.refresh_wiki_build_status(wiki_id)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    jobs = store.list_wiki_jobs(wiki_id, created_since=wiki['candidate_started_at'])
    latest_started = max(
        (job['started_at'] or job['created_at'] for job in jobs),
        default=wiki['candidate_started_at'],
    )
    finished = [job['finished_at'] for job in jobs if job['finished_at']]
    return {
        'id': wiki['candidate_group_id'],
        'type': '首次构建' if not wiki['published_at'] else '增量更新',
        'status': wiki['candidate_status'],
        'started_at': latest_started,
        'finished_at': max(finished) if finished else None,
        'source_count': sum(source['enabled'] for source in store.list_sources(wiki_id)),
        'plan_version': wiki['plan_version'],
        'goal': wiki['goal'],
        'jobs': jobs,
    }


@router.get('/{wiki_id}/sources')
async def list_wiki_sources(wiki_id: str, store: StoreDep):
    try:
        store.get_wiki(wiki_id)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    return store.list_sources(wiki_id)


@router.post('/{wiki_id}/build', status_code=status.HTTP_202_ACCEPTED)
async def build_wiki(
    wiki_id: str,
    store: StoreDep,
    manager: SyncManagerDep,
    settings: SettingsDep,
):
    try:
        wiki, source_ids = store.prepare_wiki_build(wiki_id)
        jobs = [manager.enqueue(source_id, full_sync=True) for source_id in source_ids]
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    except (ValueError, ConnectorError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {'wiki': _public_wiki(wiki, settings), 'jobs': jobs}


@router.post('/{wiki_id}/publish')
async def publish_wiki(wiki_id: str, store: StoreDep, settings: SettingsDep):
    try:
        return _public_wiki(store.publish_wiki(wiki_id), settings)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post('/{wiki_id}/search')
async def search_published_wiki(
    wiki_id: str,
    request: WikiSearchRequest,
    store: StoreDep,
    graphiti: ZepGraphitiDep,
):
    try:
        group_id = store.get_published_group(wiki_id)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    edges = await graphiti.search(
        group_ids=[group_id], query=request.query, num_results=request.max_facts
    )
    return {
        'wiki_id': wiki_id,
        'published_group_id': group_id,
        'facts': [get_fact_result_from_edge(edge) for edge in edges],
    }


@router.get('/{wiki_id}/graph')
async def published_wiki_graph(
    wiki_id: str,
    store: StoreDep,
    graphiti: ZepGraphitiDep,
):
    try:
        group_id = store.get_published_group(wiki_id)
    except KeyError as exc:
        raise _not_found(wiki_id) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id], limit=120)
    try:
        edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id], limit=240)
    except GroupsEdgesNotFoundError:
        edges = []
    node_ids = {node.uuid for node in nodes}
    return {
        'wiki_id': wiki_id,
        'published_group_id': group_id,
        'nodes': [
            {
                'id': node.uuid,
                'name': node.name,
                'summary': node.summary,
                'labels': node.labels,
                'attributes': node.attributes,
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
            }
            for edge in edges
            if edge.source_node_uuid in node_ids and edge.target_node_uuid in node_ids
        ],
    }
