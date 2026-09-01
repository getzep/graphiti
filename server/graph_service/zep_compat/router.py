"""Zep Cloud v2 compatible routes backed by Graphiti.

Mounted under /api/v2 so the zep-cloud SDK, constructed with
base_url="http://host:port/api/v2", resolves every path it builds.

Route order matters: the concrete /graph/... routes are declared before the
/graph/{graph_id} catch-all.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from graphiti_core.edges import EntityEdge as GEntityEdge
from graphiti_core.errors import (
    EdgesNotFoundError,
    GroupsEdgesNotFoundError,
    GroupsNodesNotFoundError,
    NodeNotFoundError,
)
from graphiti_core.nodes import EntityNode as GEntityNode
from graphiti_core.nodes import EpisodicNode as GEpisodicNode
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    COMBINED_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from . import models as m
from .ontology import build_edge_types, build_entity_types
from .paging import page_edges, page_nodes, parse_cursor
from .runtime import Runtime, _parse_time, ingest_episode

logger = logging.getLogger(__name__)

router = APIRouter()

NEXT_CURSOR_HEADER = 'zep-next-cursor'


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, 'zep_runtime', None)
    if runtime is None:
        raise HTTPException(status_code=503, detail='Zep compatibility layer not ready')
    return runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


async def resolve_graph_for_uuid(runtime: Runtime, uuid: str, kind: str):
    """Find which graph a node/episode UUID lives in.

    Zep's GET graph/node/{uuid} and GET graph/episodes/{uuid} carry no graph_id,
    but graphiti 0.29.3 puts each graph in its own database, so we need one.
    The index is populated whenever we hand a UUID out; the scan is the fallback
    for UUIDs this process never served (e.g. after wiping the sqlite state).
    """
    graph_id = runtime.store.graph_id_for_uuid(uuid)
    if graph_id is not None:
        return graph_id, await runtime.pool.get(graph_id)
    return None, None


# ---------------------------------------------------------------------------
# converters: graphiti objects -> wire models
# ---------------------------------------------------------------------------


def node_out(node: GEntityNode) -> m.EntityNode:
    return m.EntityNode(
        uuid=node.uuid,
        name=node.name,
        summary=node.summary or '',
        created_at=m.iso(node.created_at) or m.now_iso(),
        labels=list(node.labels or []),
        attributes=dict(node.attributes or {}),
    )


def edge_out(edge: GEntityEdge) -> m.EntityEdge:
    return m.EntityEdge(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        created_at=m.iso(edge.created_at) or m.now_iso(),
        episodes=list(edge.episodes or []),
        valid_at=m.iso(edge.valid_at),
        invalid_at=m.iso(edge.invalid_at),
        expired_at=m.iso(edge.expired_at),
        attributes=dict(edge.attributes or {}),
    )


def episode_out(node: GEpisodicNode) -> m.Episode:
    source = node.source.value if hasattr(node.source, 'value') else str(node.source)
    return m.Episode(
        uuid=node.uuid,
        content=node.content or '',
        created_at=m.iso(node.created_at) or m.now_iso(),
        source=source if source in ('text', 'json', 'message', 'fact_triple') else 'text',
        source_description=node.source_description or '',
        # `processed` is what MiroFish polls in
        # zep_graph_memory_updater._wait_for_pending_episodes. Cloud ingests
        # asynchronously, so it can return an unprocessed episode; here
        # add_episode runs to completion before the EpisodicNode is readable,
        # so anything retrievable is by definition already processed. An
        # in-flight episode is a 404, not processed=False.
        processed=True,
    )


def graph_out(record: dict[str, Any]) -> m.Graph:
    return m.Graph(
        uuid=record['uuid'],
        graph_id=record['graph_id'],
        name=record.get('name'),
        description=record.get('description'),
        created_at=record.get('created_at'),
        time_zone=record.get('time_zone'),
    )


def _progress(counts: dict[str, int]) -> m.BatchProgress:
    total = counts.get('total', 0)
    succeeded = counts.get('succeeded', 0)
    failed = counts.get('failed', 0)
    skipped = counts.get('skipped', 0)
    settled = succeeded + failed + skipped + counts.get('canceled', 0)
    return m.BatchProgress(
        total_items=total,
        queued_items=counts.get('queued', 0) + counts.get('pending', 0),
        processing_items=counts.get('processing', 0),
        succeeded_items=succeeded,
        failed_items=failed,
        skipped_items=skipped,
        canceled_items=counts.get('canceled', 0),
        percent_complete=(100.0 * settled / total) if total else 0.0,
    )


def batch_out(record: dict[str, Any], counts: dict[str, int]) -> m.BatchSummary:
    return m.BatchSummary(
        batch_id=record['batch_id'],
        status=record['status'],
        item_count=counts.get('total', 0),
        metadata=record.get('metadata'),
        progress=_progress(counts),
        created_at=record.get('created_at'),
        updated_at=record.get('updated_at'),
        processed_at=record.get('processed_at'),
        completed_at=record.get('completed_at'),
        ignore_roles=record.get('ignore_roles'),
    )


def item_out(record: dict[str, Any]) -> m.BatchItemDetail:
    return m.BatchItemDetail(
        item_id=record['item_id'],
        batch_id=record['batch_id'],
        kind=record.get('kind') or 'graph_episode',
        status=record['status'],
        sequence_index=record['sequence_index'],
        graph_id=record.get('graph_id'),
        episode_uuid=record.get('episode_uuid'),
        error=record.get('error'),
        created_at=record.get('created_at'),
        updated_at=record.get('updated_at'),
    )


# ---------------------------------------------------------------------------
# ontology
# ---------------------------------------------------------------------------


@router.put('/entity-types', response_model=m.SuccessResponse)
async def set_entity_types(payload: m.SetEntityTypesRequest, runtime: RuntimeDep):
    """Zep applies an ontology to a set of graphs; we persist it per graph_id.

    Graphiti needs the types passed into every add_episode call, so the stored
    JSON is rebuilt into Pydantic models at ingest time.
    """
    graph_ids = payload.graph_ids or []
    if not graph_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='graph_ids is required; user-scoped ontologies are not supported',
        )
    entity_specs = [t.model_dump() for t in payload.entity_types]
    edge_specs = [t.model_dump() for t in payload.edge_types]

    # Fail loudly here rather than on every ingest if the ontology cannot be
    # expressed as Pydantic models.
    build_entity_types(entity_specs)
    build_edge_types(edge_specs)

    for graph_id in graph_ids:
        runtime.store.set_ontology(graph_id, entity_specs, edge_specs)
    return m.SuccessResponse(message=f'ontology set for {len(graph_ids)} graph(s)')


@router.get('/entity-types', response_model=m.EntityTypeResponse)
async def list_entity_types(runtime: RuntimeDep, graph_id: str | None = None):
    if not graph_id:
        return m.EntityTypeResponse(entity_types=[], edge_types=[])
    entity_specs, edge_specs = runtime.store.get_ontology(graph_id)
    return m.EntityTypeResponse(
        entity_types=[m.EntityType(**spec) for spec in entity_specs],
        edge_types=[m.EdgeType(**spec) for spec in edge_specs],
    )


# ---------------------------------------------------------------------------
# graph lifecycle + ingest
# ---------------------------------------------------------------------------


@router.post('/graph/create', response_model=m.Graph)
async def create_graph(payload: m.CreateGraphRequest, runtime: RuntimeDep):
    record = runtime.store.create_graph(
        payload.graph_id, payload.name, payload.description, payload.time_zone
    )
    return graph_out(record)


@router.post('/graph/search', response_model=m.GraphSearchResults)
async def search_graph(payload: m.SearchRequest, runtime: RuntimeDep):
    if not payload.graph_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='graph_id is required; user-scoped search is not supported',
        )
    # MiroFish sends reranker='cross_encoder' from the report tooling and
    # 'rrf' from the OASIS profile generator. Honour cross_encoder only when a
    # real local reranker is configured (GRAPHITI_RERANKER=bge); with the
    # passthrough encoder a cross_encoder recipe would just reorder by input
    # order, which is worse than RRF's own fusion. So default to RRF.
    scope = (payload.scope or 'edges').lower()
    want_cross_encoder = (payload.reranker or '').lower() == 'cross_encoder' and (
        os.environ.get('GRAPHITI_RERANKER', 'none').strip().lower() == 'bge'
    )
    if scope == 'nodes':
        config = (
            NODE_HYBRID_SEARCH_CROSS_ENCODER if want_cross_encoder else NODE_HYBRID_SEARCH_RRF
        ).model_copy(deep=True)
    elif scope in ('episodes', 'auto'):
        config = (
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER
            if want_cross_encoder
            else COMBINED_HYBRID_SEARCH_RRF
        ).model_copy(deep=True)
    else:
        config = (
            EDGE_HYBRID_SEARCH_CROSS_ENCODER if want_cross_encoder else EDGE_HYBRID_SEARCH_RRF
        ).model_copy(deep=True)
    # MiroFish already clamps to 50 via normalize_zep_search_limit; clamp again
    # so a hand-rolled client cannot ask for an unbounded scan.
    config.limit = max(1, min(int(payload.limit or 10), 50))

    graphiti = await runtime.pool.get(payload.graph_id)
    results = await graphiti.search_(
        query=payload.query,
        config=config,
        group_ids=[payload.graph_id],
        center_node_uuid=payload.center_node_uuid,
        bfs_origin_node_uuids=payload.bfs_origin_node_uuids,
    )
    runtime.store.remember_uuids(
        payload.graph_id, 'node', [n.uuid for n in results.nodes]
    )
    runtime.store.remember_uuids(
        payload.graph_id, 'episode', [e.uuid for e in results.episodes]
    )
    return m.GraphSearchResults(
        nodes=[node_out(n) for n in results.nodes],
        edges=[edge_out(e) for e in results.edges],
        episodes=[episode_out(e) for e in results.episodes],
    )


@router.post('/graph/node/graph/{graph_id}', response_model=list[m.EntityNode])
async def list_nodes_by_graph(
    graph_id: str,
    payload: m.ListByGraphRequest,
    response: Response,
    runtime: RuntimeDep,
):
    """MiroFish drives pagination off the zep-next-cursor RESPONSE header.

    An absent header means "last page"; a non-advancing cursor makes MiroFish
    raise. Graphiti's uuid_cursor is exactly the right primitive.
    """
    limit = max(1, min(int(payload.limit or 100), 100))
    offset = parse_cursor(payload.cursor or payload.uuid_cursor)
    graphiti = await runtime.pool.get(graph_id)
    try:
        nodes, next_offset = await page_nodes(graphiti.driver, graph_id, limit, offset)
    except GroupsNodesNotFoundError:
        nodes, next_offset = [], None
    # Remember where these live so a later GET graph/node/{uuid} can find them.
    runtime.store.remember_uuids(graph_id, 'node', [n.uuid for n in nodes])
    if next_offset is not None:
        response.headers[NEXT_CURSOR_HEADER] = str(next_offset)
    return [node_out(n) for n in nodes]


@router.post('/graph/edge/graph/{graph_id}', response_model=list[m.EntityEdge])
async def list_edges_by_graph(
    graph_id: str,
    payload: m.ListByGraphRequest,
    response: Response,
    runtime: RuntimeDep,
):
    limit = max(1, min(int(payload.limit or 100), 100))
    offset = parse_cursor(payload.cursor or payload.uuid_cursor)
    graphiti = await runtime.pool.get(graph_id)
    try:
        edges, next_offset = await page_edges(graphiti.driver, graph_id, limit, offset)
    except GroupsEdgesNotFoundError:
        # graphiti_core's own edge accessor raises on an empty result where the
        # node one returns []. Our query does not, but keep the guard: "no edges
        # yet" is normal for a fresh graph, and a 500 here gets retried three
        # times by MiroFish before failing the whole read.
        edges, next_offset = [], None
    if next_offset is not None:
        response.headers[NEXT_CURSOR_HEADER] = str(next_offset)
    return [edge_out(e) for e in edges]


@router.get('/graph/node/{node_uuid}/entity-edges', response_model=list[m.EntityEdge])
async def get_node_edges(node_uuid: str, runtime: RuntimeDep):
    _, graphiti = await resolve_graph_for_uuid(runtime, node_uuid, 'node')
    if graphiti is None:
        raise HTTPException(status_code=404, detail=f'node {node_uuid} not found')
    try:
        edges = await GEntityEdge.get_by_node_uuid(graphiti.driver, node_uuid)
    except (EdgesNotFoundError, GroupsEdgesNotFoundError):
        # An isolated node has no edges; that is data, not an error.
        edges = []
    return [edge_out(e) for e in edges]


@router.get('/graph/node/{node_uuid}', response_model=m.EntityNode)
async def get_node(node_uuid: str, runtime: RuntimeDep):
    _, graphiti = await resolve_graph_for_uuid(runtime, node_uuid, 'node')
    if graphiti is None:
        # MiroFish's zep_tools and zep_entity_reader both translate this 404
        # into "entity not found" rather than an error.
        raise HTTPException(status_code=404, detail=f'node {node_uuid} not found')
    try:
        node = await GEntityNode.get_by_uuid(graphiti.driver, node_uuid)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return node_out(node)


@router.get('/graph/episodes/{episode_uuid}', response_model=m.Episode)
async def get_episode(episode_uuid: str, runtime: RuntimeDep):
    """Report ingestion progress for one episode.

    Careful with 404 here. Unlike graph.get and graph.node.get, the zep-cloud
    SDK's `graph.episode.get` does NOT map 404 to NotFoundError — it only
    handles 400 and 500, so a 404 surfaces as a generic ApiError. MiroFish's
    is_retryable_zep_error treats a 404 ApiError as non-retryable, which would
    raise straight out of _wait_for_pending_episodes and abort the run.

    Episode UUIDs are handed to MiroFish at batch.add time, before the episode
    exists in the graph. So a UUID we know is queued must answer
    processed=false — the same thing Cloud does — and only a genuinely unknown
    UUID may 404.
    """
    # Check the batch item FIRST. The worker persists the EpisodicNode before
    # running extraction (add_episode(uuid=...) loads an existing node rather
    # than creating one), so the node is already readable while its item is
    # still processing. Trusting the graph alone would report processed=true
    # for work that has not finished.
    item = runtime.store.find_item_by_episode_uuid(episode_uuid)
    if item is not None and item['status'] not in ('succeeded', 'skipped'):
        if item['status'] == 'failed':
            # Terminal, and it will never become processed. Surfacing the
            # failure beats letting the caller poll until its deadline.
            raise HTTPException(
                status_code=500,
                detail=(item.get('error') or {}).get('message', 'ingestion failed'),
            )
        return m.Episode(
            uuid=episode_uuid,
            content=item.get('payload') or '',
            created_at=item.get('created_at') or m.now_iso(),
            source=item.get('data_type') or 'text',
            source_description=item.get('source_description') or '',
            processed=False,
        )

    graph_id = (item or {}).get('graph_id') or runtime.store.graph_id_for_uuid(episode_uuid)
    if graph_id is None:
        raise HTTPException(status_code=404, detail=f'episode {episode_uuid} not found')
    graphiti = await runtime.pool.get(graph_id)
    try:
        episode = await GEpisodicNode.get_by_uuid(graphiti.driver, episode_uuid)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return episode_out(episode)


@router.get('/graph/episodes/{episode_uuid}/mentions', response_model=m.EpisodeMentions)
async def get_episode_mentions(episode_uuid: str, runtime: RuntimeDep):
    graph_id, graphiti = await resolve_graph_for_uuid(runtime, episode_uuid, 'episode')
    if graphiti is None:
        raise HTTPException(status_code=404, detail=f'episode {episode_uuid} not found')
    results = await graphiti.get_nodes_and_edges_by_episode([episode_uuid])
    runtime.store.remember_uuids(graph_id, 'node', [n.uuid for n in results.nodes])
    return m.EpisodeMentions(
        nodes=[node_out(n) for n in results.nodes],
        edges=[edge_out(e) for e in results.edges],
    )


@router.post('/graph', response_model=m.Episode)
async def add_data(payload: m.AddDataRequest, runtime: RuntimeDep):
    """Synchronous single-episode ingest. Slow with a local LLM — MiroFish's
    bulk path goes through /batches instead."""
    if not payload.graph_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='graph_id is required; user graphs are not supported',
        )
    graphiti = await runtime.pool.get(payload.graph_id)
    # Let Graphiti mint the UUID here: unlike the batch path there is no UUID
    # promised to the client ahead of time.
    episode = await ingest_episode(
        graphiti,
        runtime.store,
        graph_id=payload.graph_id,
        episode_uuid=None,
        name='mirofish-episode',
        body=payload.data,
        data_type=payload.type,
        source_description=payload.source_description or '',
        reference_time=_parse_time(payload.created_at),
    )
    return episode_out(episode)


@router.get('/graph/{graph_id}', response_model=m.Graph)
async def get_graph(graph_id: str, runtime: RuntimeDep):
    record = runtime.store.get_graph(graph_id)
    if record is None:
        # MiroFish treats 404 here as "graph absent", not as a failure.
        raise HTTPException(status_code=404, detail=f'graph {graph_id} not found')
    return graph_out(record)


@router.delete('/graph/{graph_id}', response_model=m.SuccessResponse)
async def delete_graph(graph_id: str, runtime: RuntimeDep):
    # 404 for an unknown graph matches Cloud, and MiroFish's teardown in
    # app/api/graph.py explicitly catches NotFoundError to stay idempotent.
    if runtime.store.get_graph(graph_id) is None:
        raise HTTPException(status_code=404, detail=f'graph {graph_id} not found')
    # clear_data is the purpose-built helper and scopes by group_id across
    # Entity/Episodic/Community labels. Note EntityEdge has no
    # delete_by_group_id classmethod, so do not reach for one.
    graphiti = await runtime.pool.get(graph_id)
    await clear_data(graphiti.driver, group_ids=[graph_id])
    runtime.store.delete_graph(graph_id)
    # Drop the cached instance so a later graph with the same id starts clean.
    await runtime.pool.forget(graph_id)
    return m.SuccessResponse(message=f'graph {graph_id} deleted')


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------


@router.post('/batches', response_model=m.BatchSummary)
async def create_batch(payload: m.CreateBatchRequest, runtime: RuntimeDep):
    batch_id = runtime.store.create_batch(payload.metadata, payload.ignore_roles)
    record = runtime.store.get_batch(batch_id)
    assert record is not None
    return batch_out(record, runtime.store.item_counts(batch_id))


@router.get('/batches', response_model=m.BatchListResponse)
async def list_batches(
    runtime: RuntimeDep,
    limit: int = 100,
    cursor: int | None = None,
    status_filter: str | None = None,
):
    limit = max(1, min(int(limit or 100), 100))
    records, next_cursor = runtime.store.list_batches(limit, cursor, status_filter)
    return m.BatchListResponse(
        batches=[
            batch_out(record, runtime.store.item_counts(record['batch_id']))
            for record in records
        ],
        next_cursor=next_cursor,
    )


@router.post('/batches/{batch_id}/items', response_model=list[m.BatchItemDetail])
async def add_batch_items(
    batch_id: str, payload: m.AddBatchItemsRequest, runtime: RuntimeDep
):
    if runtime.store.get_batch(batch_id) is None:
        raise HTTPException(status_code=404, detail=f'batch {batch_id} not found')
    prepared = [
        {
            'graph_id': item.graph_id,
            'payload': item.payload(),
            'data_type': item.data_type or 'text',
            'name': item.name,
            'source_description': item.source_description,
            'reference_time': item.created_at,
            'metadata': item.metadata,
            'kind': item.type,
        }
        for item in payload.items
    ]
    created = runtime.store.add_items(batch_id, prepared)
    return [item_out(record) for record in created]


@router.get('/batches/{batch_id}/items', response_model=m.BatchItemListResponse)
async def list_batch_items(
    batch_id: str, runtime: RuntimeDep, limit: int = 100, cursor: int | None = None
):
    if runtime.store.get_batch(batch_id) is None:
        raise HTTPException(status_code=404, detail=f'batch {batch_id} not found')
    limit = max(1, min(int(limit or 100), 100))
    records, next_cursor = runtime.store.list_items(batch_id, limit, cursor)
    return m.BatchItemListResponse(
        items=[item_out(record) for record in records], next_cursor=next_cursor
    )


@router.post('/batches/{batch_id}/process', response_model=m.BatchSummary)
async def process_batch(batch_id: str, runtime: RuntimeDep):
    record = runtime.store.get_batch(batch_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f'batch {batch_id} not found')
    if record['status'] == 'draft':
        runtime.store.mark_items_queued(batch_id)
        runtime.store.set_batch_status(batch_id, 'queued', processed=True)
        runtime.worker.submit(batch_id)
    record = runtime.store.get_batch(batch_id)
    assert record is not None
    return batch_out(record, runtime.store.item_counts(batch_id))


@router.get('/batches/{batch_id}', response_model=m.BatchSummary)
async def get_batch(batch_id: str, runtime: RuntimeDep):
    record = runtime.store.get_batch(batch_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f'batch {batch_id} not found')
    return batch_out(record, runtime.store.item_counts(batch_id))
