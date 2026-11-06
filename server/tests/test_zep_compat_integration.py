"""Live integration: real FalkorDB + real Graphiti + the real zep-cloud SDK.

The only thing stubbed is the LLM/embedder pair, so entity extraction returns
nothing — that keeps the test deterministic and fast while still exercising every
database query, index, driver call and HTTP route for real. This is the layer
the pure-unit tests cannot cover, and it is where the raise-on-empty-edges bug
was found.

Opt in (a FalkorDB must be reachable):

    docker run -d --name falkordb-it -p 6399:6379 falkordb/falkordb:latest
    ZEP_COMPAT_IT=1 FALKORDB_IT_PORT=6399 \\
      uv run --extra dev pytest tests/test_zep_compat_integration.py -v
"""

from __future__ import annotations

import os
import random
import uuid as uuid_lib
from datetime import datetime, timezone

import httpx
import pytest

if not os.environ.get('ZEP_COMPAT_IT'):
    pytest.skip('set ZEP_COMPAT_IT=1 to run live FalkorDB tests', allow_module_level=True)

pytest.importorskip('zep_cloud')
pytest.importorskip('falkordb', reason='pip install graphiti-core[falkordb]')

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.llm_client.client import LLMClient  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from zep_cloud import BatchAddItem  # noqa: E402
from zep_cloud.client import AsyncZep  # noqa: E402

from graph_service.zep_compat.app import app  # noqa: E402
from graph_service.zep_compat.runtime import (  # noqa: E402
    BatchWorker,
    GraphitiPool,
    PassthroughCrossEncoder,
    Runtime,
)
from graph_service.zep_compat.store import Store  # noqa: E402

EMBED_DIM = 1024
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope='function')]


class DeterministicEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [random.random() for _ in range(EMBED_DIM)]

    async def create_batch(self, input_data_list):
        return [[random.random() for _ in range(EMBED_DIM)] for _ in input_data_list]


class EmptyExtractionLLM(LLMClient):
    """Answers every structured request with a schema-shaped empty value."""

    def __init__(self):
        super().__init__(LLMConfig(api_key='local', model='stub'), cache=False)

    async def _generate_response(
        self, messages, response_model=None, max_tokens=4096, model_size=None
    ):
        if response_model is None:
            return {}
        out: dict = {}
        for name, field in response_model.model_fields.items():
            annotation = str(field.annotation).lower()
            out[name] = [] if 'list' in annotation else None
        return out


@pytest.fixture
async def live(tmp_path):
    os.environ['EMBEDDING_DIM'] = str(EMBED_DIM)
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')

    # graphiti 0.29.3 maps group_id onto the database name, so the pool builds
    # one instance per graph with its driver pointed at a database named exactly
    # graph_id. Do NOT prefix it: add_episode compares group_id to
    # driver._database and clones away if they differ, which sends the ingest to
    # a different database than the pre-saved episode. Tests get isolation from
    # a unique graph_id instead.
    def factory(graph_id: str) -> Graphiti:
        instance = Graphiti(
            graph_driver=FalkorDriver(
                host=os.environ.get('FALKORDB_IT_HOST', '127.0.0.1'),
                port=int(os.environ.get('FALKORDB_IT_PORT', '6399')),
                database=graph_id,
            ),
            llm_client=EmptyExtractionLLM(),
            embedder=DeterministicEmbedder(),
            cross_encoder=PassthroughCrossEncoder(),
            max_coroutines=2,
        )
        return instance

    pool = GraphitiPool(factory=factory, build_indices=True)
    store = Store(tmp_path / 'it.sqlite3')
    worker = BatchWorker(pool=pool, store=store, concurrency=2)
    app.state.zep_runtime = Runtime(pool=pool, store=store, worker=worker)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://shim') as http:
        client = AsyncZep(api_key='local', base_url='http://shim/api/v2', httpx_client=http)
        try:
            yield client, pool, store, f'it{uuid_lib.uuid4().hex[:8]}'
        finally:
            app.state.zep_runtime = None
            store.close()
            await pool.close()


async def _await_terminal(client, batch_id, tries=600):
    import asyncio

    for _ in range(tries):
        summary = await client.batch.get(batch_id=batch_id)
        if summary.status in {'succeeded', 'partial', 'failed', 'invalid', 'canceled'}:
            return summary
        await asyncio.sleep(0.05)
    raise AssertionError('batch never settled')


async def test_fresh_graph_edge_listing_is_empty_not_an_error(live):
    """The regression that motivated this file: a brand-new graph has zero
    edges, and graphiti_core raises rather than returning []."""
    client, _, _, gid = live
    await client.graph.create(graph_id=gid, name='it', description='d')
    response = await client.graph.edge.with_raw_response.get_by_graph_id(gid, limit=100)
    assert response.data == []


async def test_fresh_graph_node_listing_is_empty(live):
    client, _, _, gid = live
    await client.graph.create(graph_id=gid, name='it', description='d')
    response = await client.graph.node.with_raw_response.get_by_graph_id(gid, limit=100)
    assert response.data == []


async def test_batch_ingest_persists_episodes_with_the_promised_uuids(live):
    """The episode UUIDs handed out at batch.add time must be the UUIDs that
    end up in the graph — MiroFish records them and polls them later."""
    client, _, _, gid = live
    await client.graph.create(graph_id=gid, name='it', description='d')

    batch = await client.batch.create(metadata={'mirofish_operation_id': 'op-it'})
    details = await client.batch.add(
        batch_id=batch.batch_id,
        items=[
            BatchAddItem(
                type='graph_episode',
                graph_id=gid,
                data=f'Document chunk number {i}.',
                data_type='text',
                source_description='MiroFish source document chunk',
            )
            for i in range(3)
        ],
    )
    promised = [d.episode_uuid for d in details]
    await client.batch.process(batch_id=batch.batch_id)

    summary = await _await_terminal(client, batch.batch_id)
    assert summary.status == 'succeeded', summary.progress

    for episode_uuid in promised:
        episode = await client.graph.episode.get(uuid_=episode_uuid)
        assert episode.uuid_ == episode_uuid
        assert episode.processed is True
        assert episode.content.startswith('Document chunk number')


async def test_queued_episode_is_not_processed_yet(live):
    """Before /process runs, a promised UUID must answer processed=false rather
    than 404 — a 404 here is a non-retryable ApiError inside MiroFish."""
    client, _, _, gid = live
    await client.graph.create(graph_id=gid, name='it', description='d')
    batch = await client.batch.create(metadata=None)
    details = await client.batch.add(
        batch_id=batch.batch_id,
        items=[BatchAddItem(type='graph_episode', graph_id=gid, data='x', data_type='text')],
    )
    episode = await client.graph.episode.get(uuid_=details[0].episode_uuid)
    assert episode.processed is False


async def test_search_runs_against_the_real_index(live):
    """Extraction is stubbed so there is nothing to find; what matters is that
    the hybrid search executes against real indexes without erroring."""
    client, _, _, gid = live
    await client.graph.create(graph_id=gid, name='it', description='d')
    results = await client.graph.search(
        graph_id=gid, query='anything', limit=10, scope='edges', reranker='rrf'
    )
    assert results.edges in (None, [])
    assert results.nodes in (None, [])


async def test_graph_delete_clears_the_group_and_then_404s(live):
    client, pool, _, gid = live
    await client.graph.create(graph_id=gid, name='it', description='d')
    await client.graph.add(
        graph_id=gid,
        type='text',
        data='Something happened.',
        created_at=datetime.now(timezone.utc).isoformat(),
        source_description='MiroFish simulation activity batch',
    )
    await client.graph.delete(graph_id=gid)

    from zep_cloud import NotFoundError

    with pytest.raises(NotFoundError):
        await client.graph.get(gid)


async def test_pagination_over_a_real_multi_page_node_set(live):
    """Drive the zep-next-cursor loop exactly as MiroFish's zep_paging does,
    against real rows and the driver's own uuid cursor."""
    client, pool, _, gid = live
    from graphiti_core.nodes import EntityNode

    await client.graph.create(graph_id=gid, name='it', description='d')
    graphiti = await pool.get(gid)
    for i in range(120):
        node = EntityNode(
            uuid=str(uuid_lib.uuid4()),
            name=f'Entity{i}',
            group_id=gid,
            labels=['Entity'],
            created_at=datetime.now(timezone.utc),
            summary=f'entity {i}',
        )
        await node.generate_name_embedding(graphiti.embedder)
        await node.save(graphiti.driver)

    seen, cursor, cursors = [], None, set()
    while True:
        kwargs = {'limit': 50}
        if cursor is not None:
            kwargs['cursor'] = cursor
        response = await client.graph.node.with_raw_response.get_by_graph_id(gid, **kwargs)
        seen.extend(response.data or [])
        next_cursor = next(
            (v for k, v in response.headers.items() if k.lower() == 'zep-next-cursor'), None
        )
        if next_cursor is None:
            break
        assert next_cursor != cursor and next_cursor not in cursors
        cursors.add(next_cursor)
        cursor = next_cursor

    assert len(seen) == 120
    assert len({n.uuid_ for n in seen}) == 120
