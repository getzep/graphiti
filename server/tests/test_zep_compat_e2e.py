"""End-to-end: drive the shim with the REAL zep-cloud SDK over an ASGI transport.

This is the test that actually proves the shim is a drop-in. It uses
zep_cloud's own generated client — the same code MiroFish runs — so paths, HTTP
methods, request bodies, response parsing, error mapping and the
`zep-next-cursor` header are all exercised for real. No network, no graph
database, no LLM: Graphiti's retrieval calls are stubbed with in-memory objects.

MiroFish uses the sync `Zep` client; httpx can only drive an ASGI app
asynchronously, so we use `AsyncZep`. Both are generated from one spec and share
identical paths, payload shapes and response parsers.

Run:  uv run --extra dev pytest tests/test_zep_compat_e2e.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

pytest.importorskip('zep_cloud', reason='zep-cloud is a dev-only dependency')
pytest.importorskip('graphiti_core')

from graphiti_core.edges import EntityEdge as GEntityEdge  # noqa: E402
from graphiti_core.nodes import EntityNode as GEntityNode  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.nodes import EpisodicNode as GEpisodicNode  # noqa: E402
from zep_cloud import BatchAddItem, EntityEdgeSourceTarget, NotFoundError  # noqa: E402
from zep_cloud.client import AsyncZep  # noqa: E402
from zep_cloud.external_clients.ontology import EdgeModel, EntityModel, EntityText  # noqa: E402
from pydantic import Field  # noqa: E402

from graph_service.zep_compat import router as router_mod  # noqa: E402
from graph_service.zep_compat.app import app  # noqa: E402
from graph_service.zep_compat.runtime import (  # noqa: E402
    BatchWorker,
    GraphitiPool,
    Runtime,
)
from graph_service.zep_compat.store import Store  # noqa: E402

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


def make_node(uuid: str, name: str = 'N', group_id: str = 'g-1') -> GEntityNode:
    return GEntityNode(
        uuid=uuid,
        name=name,
        group_id=group_id,
        labels=['Entity'],
        created_at=NOW,
        summary=f'summary of {name}',
        attributes={'role': 'tester'},
    )


def make_edge(uuid: str, src: str = 'n-1', dst: str = 'n-2', group_id: str = 'g-1'):
    return GEntityEdge(
        uuid=uuid,
        group_id=group_id,
        source_node_uuid=src,
        target_node_uuid=dst,
        created_at=NOW,
        name='WORKS_FOR',
        fact=f'fact for {uuid}',
        episodes=['ep-1'],
        valid_at=NOW,
        attributes={'since': '2020'},
    )


def make_episode(uuid: str, group_id: str = 'g-1') -> GEpisodicNode:
    return GEpisodicNode(
        uuid=uuid,
        name='ep',
        group_id=group_id,
        labels=[],
        created_at=NOW,
        source=EpisodeType.text,
        source_description='chunk',
        content='hello world',
        valid_at=NOW,
        entity_edges=[],
    )


class StubGraphiti:
    """Just enough Graphiti surface for the router, with recorded calls."""

    def __init__(self):
        self.driver = object()
        self.added: list[dict] = []
        self.cleared: list[list[str]] = []
        self.saved_episodes: list = []
        self.fail_payloads: set[str] = set()

    async def add_episode(self, **kwargs):
        if kwargs.get('episode_body') in self.fail_payloads:
            raise RuntimeError('simulated extraction failure')
        self.added.append(kwargs)
        from graphiti_core.graphiti import AddEpisodeResults

        return AddEpisodeResults(
            episode=make_episode(kwargs.get('uuid') or 'ep-new', kwargs['group_id']),
            episodic_edges=[],
            nodes=[],
            edges=[],
            communities=[],
            community_edges=[],
        )

    async def search_(self, **kwargs):
        from graphiti_core.search.search_config import SearchResults

        self.last_search = kwargs
        return SearchResults(
            nodes=[make_node('n-1', 'Alice')],
            edges=[make_edge('e-1')],
            episodes=[make_episode('ep-1')],
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids):
        from graphiti_core.search.search_config import SearchResults

        return SearchResults(nodes=[make_node('n-1')], edges=[make_edge('e-1')])

    async def close(self):
        pass


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Mount the real app with a real Store and a stubbed Graphiti."""
    store = Store(tmp_path / 'compat.sqlite3')
    graphiti = StubGraphiti()
    # One Graphiti per graph in production (group_id is the database name); the
    # stub stands in for all of them here.
    pool = GraphitiPool(factory=lambda graph_id: graphiti, build_indices=False)  # type: ignore[arg-type]
    worker = BatchWorker(pool=pool, store=store, concurrency=2)

    nodes = {'n-1': make_node('n-1', 'Alice'), 'n-2': make_node('n-2', 'Acme')}
    episodes = {'ep-1': make_episode('ep-1')}

    async def fake_node_get_by_uuid(driver, uuid):
        from graphiti_core.errors import NodeNotFoundError

        if uuid not in nodes:
            raise NodeNotFoundError(uuid)
        return nodes[uuid]

    async def fake_episode_get_by_uuid(driver, uuid):
        from graphiti_core.errors import NodeNotFoundError

        if uuid not in episodes:
            raise NodeNotFoundError(uuid)
        return episodes[uuid]

    # 250 nodes/edges so pagination is genuinely multi-page.
    all_nodes = [make_node(f'n-{i:04d}', f'Node{i}') for i in range(250)]
    all_edges = [make_edge(f'e-{i:04d}') for i in range(250)]

    # The router pages with its own SKIP/LIMIT query (see paging.py) because
    # graphiti_core's uuid_cursor is silently ignored on FalkorDB.
    def offset_page(items, limit, offset):
        window = items[offset : offset + limit]
        has_more = len(items) > offset + limit
        return window, (offset + limit if has_more else None)

    async def fake_page_nodes(driver, group_id, limit, offset):
        return offset_page(all_nodes, limit, offset)

    async def fake_page_edges(driver, group_id, limit, offset):
        return offset_page(all_edges, limit, offset)

    async def fake_edges_by_node(driver, node_uuid):
        return [make_edge('e-1', src=node_uuid), make_edge('e-2', dst=node_uuid)]

    async def fake_clear_data(driver, group_ids=None):
        graphiti.cleared.append(list(group_ids or []))

    # The worker persists the EpisodicNode before calling add_episode, because
    # graphiti's add_episode(uuid=...) loads an existing node rather than
    # creating one. Record those saves instead of touching a database.
    async def fake_episode_save(self, driver):
        graphiti.saved_episodes.append(self)
        episodes[self.uuid] = self

    monkeypatch.setattr(router_mod.GEntityNode, 'get_by_uuid', fake_node_get_by_uuid)
    monkeypatch.setattr(router_mod, 'page_nodes', fake_page_nodes)
    monkeypatch.setattr(router_mod.GEpisodicNode, 'get_by_uuid', fake_episode_get_by_uuid)
    monkeypatch.setattr(router_mod, 'page_edges', fake_page_edges)
    monkeypatch.setattr(router_mod.GEntityEdge, 'get_by_node_uuid', fake_edges_by_node)
    monkeypatch.setattr(router_mod, 'clear_data', fake_clear_data)
    monkeypatch.setattr(GEpisodicNode, 'save', fake_episode_save)

    # Node/episode UUIDs resolve to a graph through this index, which production
    # populates whenever a UUID is handed out by a graph-scoped read.
    store.create_graph('g-1', 'n', 'd', None)
    store.remember_uuids('g-1', 'node', list(nodes))
    store.remember_uuids('g-1', 'episode', list(episodes))

    app.state.zep_runtime = Runtime(pool=pool, store=store, worker=worker)
    try:
        yield app.state.zep_runtime, graphiti, store
    finally:
        store.close()
        app.state.zep_runtime = None


@pytest.fixture
async def client(wired):
    """A real AsyncZep pointed at the in-process app, exactly as MiroFish
    configures it: base_url ending in /api/v2."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://shim') as http:
        yield AsyncZep(api_key='local', base_url='http://shim/api/v2', httpx_client=http)


pytestmark = pytest.mark.asyncio(loop_scope='function')


# ---------------------------------------------------------------------------
# graph lifecycle
# ---------------------------------------------------------------------------


async def test_create_then_get_graph(client):
    created = await client.graph.create(
        graph_id='g-1', name='N', description='MiroFish Social Simulation Graph'
    )
    assert created.graph_id == 'g-1'
    fetched = await client.graph.get('g-1')
    assert fetched.graph_id == 'g-1'
    assert fetched.uuid_


async def test_get_unknown_graph_raises_not_found(client):
    """MiroFish reconciles a lost create by calling graph.get and treating
    NotFoundError as 'the create really did fail'."""
    with pytest.raises(NotFoundError):
        await client.graph.get('missing')


async def test_delete_graph_clears_the_group(client, wired):
    _, graphiti, _ = wired
    await client.graph.create(graph_id='g-1', name='N', description='d')
    await client.graph.delete(graph_id='g-1')
    assert graphiti.cleared == [['g-1']]
    with pytest.raises(NotFoundError):
        await client.graph.get('g-1')


async def test_delete_unknown_graph_raises_not_found(client):
    """app/api/graph.py catches NotFoundError to keep teardown idempotent."""
    with pytest.raises(NotFoundError):
        await client.graph.delete(graph_id='missing')


# ---------------------------------------------------------------------------
# ontology
# ---------------------------------------------------------------------------


async def test_set_ontology_through_the_sdk_helper(client, wired):
    """Exercises zep_cloud's EntityModel -> wire-schema conversion, the same
    path graph_builder.set_ontology uses."""
    _, _, store = wired

    class Journalist(EntityModel):
        """A reporter."""

        outlet: EntityText = Field(description='Employer', default=None)

    class ReportsOn(EdgeModel):
        """Covers a topic."""

        beat: EntityText = Field(description='Topic area', default=None)

    await client.graph.set_ontology(
        graph_ids=['g-1'],
        entities={'Journalist': Journalist},
        edges={
            'REPORTS_ON': (
                ReportsOn,
                [EntityEdgeSourceTarget(source='Journalist', target='Entity')],
            )
        },
    )

    entity_specs, edge_specs = store.get_ontology('g-1')
    assert [e['name'] for e in entity_specs] == ['Journalist']
    assert entity_specs[0]['description'] == 'A reporter.'
    assert [p['name'] for p in entity_specs[0]['properties']] == ['outlet']
    assert edge_specs[0]['source_targets'] == [
        {'source': 'Journalist', 'target': 'Entity'}
    ]


async def test_edge_only_ontology_is_accepted(client, wired):
    """graph_builder passes entities={} for edge-only ontologies."""
    _, _, store = wired

    class Knows(EdgeModel):
        """Knows someone."""

    await client.graph.set_ontology(graph_ids=['g-1'], entities={}, edges={'KNOWS': Knows})
    entity_specs, edge_specs = store.get_ontology('g-1')
    assert entity_specs == []
    assert [e['name'] for e in edge_specs] == ['KNOWS']


# ---------------------------------------------------------------------------
# single-episode ingest
# ---------------------------------------------------------------------------


async def test_graph_add_returns_an_episode_uuid(client, wired):
    """zep_graph_memory_updater raises if graph.add returns no episode UUID."""
    _, graphiti, _ = wired
    await client.graph.create(graph_id='g-1', name='N', description='d')
    episode = await client.graph.add(
        graph_id='g-1',
        type='text',
        data='agent 7 posted something',
        created_at='2024-01-01T00:00:00+00:00',
        source_description='MiroFish simulation activity batch',
        metadata={'source': 'mirofish_simulation', 'activity_count': 1},
    )
    assert episode.uuid_
    assert graphiti.added[0]['group_id'] == 'g-1'
    assert graphiti.added[0]['episode_body'] == 'agent 7 posted something'


async def test_episode_get_reports_processed(client):
    """The teardown drain polls episode.get until processed is True."""
    episode = await client.graph.episode.get(uuid_='ep-1')
    assert episode.processed is True
    assert episode.content == 'hello world'


async def test_episode_get_unknown_raises_a_404_api_error(client):
    """Note the asymmetry: zep-cloud's graph.episode.get handles only 400 and
    500, so a 404 arrives as a generic ApiError, NOT NotFoundError. MiroFish
    treats a 404 ApiError as non-retryable and lets it propagate — which is why
    a queued episode must report processed=false instead of 404."""
    from zep_cloud.core.api_error import ApiError

    with pytest.raises(ApiError) as excinfo:
        await client.graph.episode.get(uuid_='nope')
    assert excinfo.value.status_code == 404
    assert not isinstance(excinfo.value, NotFoundError)


async def test_queued_episode_reports_not_processed_instead_of_404(client, wired):
    """batch.add hands out episode UUIDs before the episodes exist. Polling one
    must not 404, or _wait_for_pending_episodes aborts the run."""
    _, _, store = wired
    batch_id = store.create_batch({'mirofish_operation_id': 'op-x'}, None)
    created = store.add_items(
        batch_id, [{'graph_id': 'g-1', 'payload': 'queued chunk', 'data_type': 'text'}]
    )
    episode_uuid = created[0]['episode_uuid']

    episode = await client.graph.episode.get(uuid_=episode_uuid)
    assert episode.uuid_ == episode_uuid
    assert episode.processed is False
    assert episode.content == 'queued chunk'


async def test_failed_episode_surfaces_an_error_rather_than_polling_forever(
    client, wired
):
    _, _, store = wired
    batch_id = store.create_batch(None, None)
    created = store.add_items(
        batch_id, [{'graph_id': 'g-1', 'payload': 'doomed', 'data_type': 'text'}]
    )
    store.set_item_status(created[0]['item_id'], 'failed', {'message': 'llm exploded'})

    from zep_cloud.core.api_error import ApiError

    with pytest.raises(ApiError) as excinfo:
        await client.graph.episode.get(uuid_=created[0]['episode_uuid'])
    assert excinfo.value.status_code == 500


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


async def test_node_get_returns_the_fields_mirofish_reads(client):
    node = await client.graph.node.get(uuid_='n-1')
    assert node.uuid_ == 'n-1'
    assert node.name == 'Alice'
    assert node.labels == ['Entity']
    assert node.summary == 'summary of Alice'
    assert node.attributes == {'role': 'tester'}


async def test_node_get_unknown_raises_not_found(client):
    """This 404 is the only way 'entity not found' is expressed."""
    with pytest.raises(NotFoundError):
        await client.graph.node.get(uuid_='missing')


async def test_node_get_edges_returns_a_bare_array(client):
    edges = await client.graph.node.get_edges(node_uuid='n-1')
    assert isinstance(edges, list)
    assert {e.uuid_ for e in edges} == {'e-1', 'e-2'}
    assert edges[0].fact
    assert edges[0].source_node_uuid


async def test_search_returns_nodes_and_edges(client):
    results = await client.graph.search(
        graph_id='g-1', query='what happened', limit=10, scope='edges',
        reranker='cross_encoder',
    )
    assert results.edges and results.edges[0].fact == 'fact for e-1'
    assert results.nodes and results.nodes[0].name == 'Alice'


async def test_search_scope_nodes_and_rrf_reranker(client, wired):
    _, graphiti, _ = wired
    await client.graph.search(
        graph_id='g-1', query='q', limit=20, scope='nodes', reranker='rrf'
    )
    assert graphiti.last_search['group_ids'] == ['g-1']
    assert graphiti.last_search['config'].limit == 20


async def test_search_limit_is_clamped_to_50(client, wired):
    _, graphiti, _ = wired
    await client.graph.search(graph_id='g-1', query='q', limit=5000, scope='edges')
    assert graphiti.last_search['config'].limit == 50


# ---------------------------------------------------------------------------
# pagination via the zep-next-cursor response header
# ---------------------------------------------------------------------------


async def _drain_pages(api_call, graph_id):
    """Mirror MiroFish's zep_paging._fetch_all loop, assertions included."""
    items, cursor, seen = [], None, set()
    while True:
        kwargs = {'limit': 100}
        if cursor is not None:
            kwargs['cursor'] = cursor
        response = await api_call(graph_id, **kwargs)
        items.extend(response.data or [])
        next_cursor = next(
            (v for k, v in response.headers.items() if k.lower() == 'zep-next-cursor'),
            None,
        )
        if next_cursor is None:
            break
        assert next_cursor != cursor, 'cursor must advance or MiroFish raises'
        assert next_cursor not in seen
        seen.add(next_cursor)
        cursor = next_cursor
    return items


async def test_fetch_all_nodes_pagination(client):
    nodes = await _drain_pages(
        client.graph.node.with_raw_response.get_by_graph_id, 'g-1'
    )
    assert len(nodes) == 250
    assert len({n.uuid_ for n in nodes}) == 250


async def test_fetch_all_edges_pagination(client):
    edges = await _drain_pages(
        client.graph.edge.with_raw_response.get_by_graph_id, 'g-1'
    )
    assert len(edges) == 250
    assert len({e.uuid_ for e in edges}) == 250


# ---------------------------------------------------------------------------
# the batch pipeline
# ---------------------------------------------------------------------------


TERMINAL = {'succeeded', 'partial', 'failed', 'invalid', 'canceled'}


async def _await_terminal(client, batch_id, tries=200):
    """Mirror MiroFish's _wait_for_batch poll, minus the 3s sleep."""
    import asyncio

    for _ in range(tries):
        summary = await client.batch.get(batch_id=batch_id)
        if summary.status in TERMINAL:
            return summary
        await asyncio.sleep(0.01)
    raise AssertionError(f'batch {batch_id} never reached a terminal status')


async def _run_batch(client, chunks, graph_id='g-1', per_call=None):
    batch = await client.batch.create(
        metadata={'mirofish_operation_id': 'op-1', 'graph_id': graph_id}
    )
    assert batch.status == 'draft'
    per_call = per_call or len(chunks)
    episode_uuids = []
    for start in range(0, len(chunks), per_call):
        details = await client.batch.add(
            batch_id=batch.batch_id,
            items=[
                BatchAddItem(
                    type='graph_episode',
                    graph_id=graph_id,
                    data=chunk,
                    data_type='text',
                    source_description='MiroFish source document chunk',
                    metadata={'chunk_index': start + i},
                )
                for i, chunk in enumerate(chunks[start : start + per_call])
            ],
        )
        episode_uuids.extend(d.episode_uuid for d in details)
    await client.batch.process(batch_id=batch.batch_id)
    return batch.batch_id, episode_uuids


async def test_full_batch_lifecycle_reaches_succeeded(client, wired):
    _, graphiti, _ = wired
    batch_id, episode_uuids = await _run_batch(client, [f'chunk {i}' for i in range(5)])

    assert all(episode_uuids) and len(set(episode_uuids)) == 5

    summary = await _await_terminal(client, batch_id)
    assert summary.status == 'succeeded'
    assert summary.item_count == 5
    assert summary.progress.succeeded_items == 5
    assert summary.progress.percent_complete == 100.0

    # The pre-assigned episode UUIDs are the ones actually ingested.
    assert {kw['uuid'] for kw in graphiti.added} == set(episode_uuids)


async def test_sequence_index_is_global_across_add_calls(client):
    """add_text_batches asserts indexes == set(range(expected_count))."""
    batch_id, _ = await _run_batch(
        client, [f'chunk {i}' for i in range(7)], per_call=3
    )
    await _await_terminal(client, batch_id)
    page = await client.batch.list_items(batch_id=batch_id, limit=100)
    assert {i.sequence_index for i in page.items} == set(range(7))


async def test_batch_item_pagination_cursor_advances(client):
    batch_id, _ = await _run_batch(client, [f'chunk {i}' for i in range(150)])
    items, cursor, seen = [], None, set()
    while True:
        page = await client.batch.list_items(
            batch_id=batch_id, limit=100, cursor=cursor
        )
        items.extend(page.items or [])
        if page.next_cursor is None:
            break
        assert page.next_cursor != cursor
        assert page.next_cursor not in seen
        seen.add(page.next_cursor)
        cursor = page.next_cursor
    assert len(items) == 150


async def test_batch_can_be_found_by_operation_id(client):
    """Mirrors _find_batch_by_operation_id after an ambiguous create."""
    await _run_batch(client, ['a'])
    matches, cursor = [], None
    while True:
        page = await client.batch.list(limit=100, cursor=cursor)
        matches.extend(
            b
            for b in (page.batches or [])
            if (b.metadata or {}).get('mirofish_operation_id') == 'op-1'
            and (b.metadata or {}).get('graph_id') == 'g-1'
        )
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert len(matches) == 1, 'MiroFish raises on more than one match'


async def test_partial_batch_surfaces_the_failed_item_error(client, wired):
    _, graphiti, _ = wired
    graphiti.fail_payloads = {'chunk 1'}
    batch_id, _ = await _run_batch(client, ['chunk 0', 'chunk 1', 'chunk 2'])

    summary = await _await_terminal(client, batch_id)
    assert summary.status == 'partial'

    page = await client.batch.list_items(batch_id=batch_id, limit=100)
    failed = [i for i in page.items if i.status not in {'succeeded', 'skipped'}]
    assert len(failed) == 1
    assert 'simulated extraction failure' in failed[0].error['message']


async def test_all_items_failing_yields_failed_not_partial(client, wired):
    _, graphiti, _ = wired
    graphiti.fail_payloads = {'only'}
    batch_id, _ = await _run_batch(client, ['only'])
    summary = await _await_terminal(client, batch_id)
    assert summary.status == 'failed'


async def test_process_is_idempotent(client):
    """A lost /process response makes MiroFish reconcile with a GET, but a
    duplicate POST must not re-run the batch either."""
    batch_id, _ = await _run_batch(client, ['a', 'b'])
    again = await client.batch.process(batch_id=batch_id)
    assert again.status != 'draft'


async def test_batch_get_unknown_raises_not_found(client):
    with pytest.raises(NotFoundError):
        await client.batch.get(batch_id='missing')


async def test_ontology_is_applied_to_batch_ingest(client, wired):
    """The stored ontology must reach add_episode as Pydantic types."""
    _, graphiti, store = wired
    store.set_ontology(
        'g-1',
        [
            {
                'name': 'Journalist',
                'description': 'A reporter',
                'properties': [
                    {'name': 'outlet', 'description': 'Employer', 'type': 'Text'}
                ],
            }
        ],
        [
            {
                'name': 'REPORTS_ON',
                'description': 'Covers',
                'properties': [],
                'source_targets': [{'source': 'Journalist', 'target': 'Entity'}],
            }
        ],
    )
    batch_id, _ = await _run_batch(client, ['chunk'])
    await _await_terminal(client, batch_id)
    call = graphiti.added[0]
    assert set(call['entity_types']) == {'Journalist'}
    assert set(call['edge_types']) == {'REPORTS_ON'}
    assert call['edge_type_map'] == {('Journalist', 'Entity'): ['REPORTS_ON']}


# ---------------------------------------------------------------------------
# empty-graph regression (found by smoke-testing against a real FalkorDB)
# ---------------------------------------------------------------------------


async def test_edge_listing_on_a_graph_with_no_edges_returns_empty(client, wired, monkeypatch):
    """graphiti_core 0.29.3: EntityNode.get_by_group_ids returns [] when empty,
    but EntityEdge.get_by_group_ids RAISES GroupsEdgesNotFoundError. A fresh
    graph has no edges, so letting that escape 500s a completely normal read —
    and MiroFish retries a 500 three times before failing the whole drain."""
    from graphiti_core.errors import GroupsEdgesNotFoundError

    async def raise_empty(driver, group_id, limit, offset):
        raise GroupsEdgesNotFoundError([group_id])

    monkeypatch.setattr(router_mod, 'page_edges', raise_empty)

    response = await client.graph.edge.with_raw_response.get_by_graph_id('g-1', limit=100)
    assert response.data == []
    assert 'zep-next-cursor' not in {k.lower() for k in response.headers}


async def test_node_listing_survives_an_empty_graph(client, wired, monkeypatch):
    from graphiti_core.errors import GroupsNodesNotFoundError

    async def raise_empty(driver, group_id, limit, offset):
        raise GroupsNodesNotFoundError([group_id])

    monkeypatch.setattr(router_mod, 'page_nodes', raise_empty)
    response = await client.graph.node.with_raw_response.get_by_graph_id('g-1', limit=100)
    assert response.data == []


async def test_isolated_node_has_no_edges_without_erroring(client, wired, monkeypatch):
    from graphiti_core.errors import EdgesNotFoundError

    async def raise_empty(driver, node_uuid):
        raise EdgesNotFoundError([node_uuid])

    monkeypatch.setattr(router_mod.GEntityEdge, 'get_by_node_uuid', raise_empty)
    assert await client.graph.node.get_edges(node_uuid='n-1') == []
