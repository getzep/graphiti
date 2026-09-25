"""Regression tests for FalkorDB MCP graph routing."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.search.search_config import SearchResults

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import graphiti_mcp_server as server  # noqa: E402
from services.queue_service import QueueService  # noqa: E402
from utils.falkor_routing import (  # noqa: E402
    ENTITY_EDGE_UUID_QUERY,
    EPISODE_UUIDS_QUERY,
    AmbiguousUuidError,
    driver_for_uuid,
    drivers_for_uuids,
)


class FakeFalkorClient:
    def __init__(self, graph_uuids: dict[str, set[str]]):
        self.graph_uuids = graph_uuids
        self.queries: list[tuple[str, tuple[str, ...]]] = []
        self.clone_calls = 0

    async def list_graphs(self) -> list[str]:
        return list(self.graph_uuids)


class FakeFalkorDriver:
    def __init__(
        self,
        graph_uuids: dict[str, set[str]],
        database: str = 'default_db',
        client: FakeFalkorClient | None = None,
    ):
        self._database = database
        self.client = client or FakeFalkorClient(graph_uuids)

    def clone(self, database: str) -> 'FakeFalkorDriver':
        self.client.clone_calls += 1
        return FakeFalkorDriver(self.client.graph_uuids, database, self.client)

    async def execute_query(
        self,
        _query: str,
        *,
        uuid: str | None = None,
        uuids: list[str] | None = None,
        **_kwargs,
    ):
        requested = (uuid,) if uuid is not None else tuple(uuids or [])
        self.client.queries.append((self._database, requested))
        matches = self.client.graph_uuids[self._database].intersection(requested)
        return [{'uuid': value} for value in matches], None, None


class FakeGraphiti:
    def __init__(self, driver: FakeFalkorDriver):
        self.driver = driver
        self.max_coroutines = 10
        self.deleted_episodes: list[tuple[str, str]] = []
        self.provenance_calls: list[tuple[str, tuple[str, ...]]] = []

    async def remove_episode(self, uuid: str) -> None:
        self.deleted_episodes.append((self.driver._database, uuid))

    async def get_nodes_and_edges_by_episode(self, uuids: list[str]) -> SearchResults:
        self.provenance_calls.append((self.driver._database, tuple(uuids)))
        return SearchResults()


class FakeGraphitiService:
    def __init__(self, client: FakeGraphiti):
        self.client = client

    async def get_client(self) -> FakeGraphiti:
        return self.client


@pytest.mark.asyncio
async def test_driver_for_uuid_finds_owning_graph_without_rebinding_shared_driver():
    driver = FakeFalkorDriver({'default_db': set(), 'group-a': {'edge-1'}})

    scoped, group_id = await driver_for_uuid(driver, ENTITY_EDGE_UUID_QUERY, 'edge-1')

    assert group_id == 'group-a'
    assert scoped._database == 'group-a'
    assert driver._database == 'default_db'
    assert driver.client.clone_calls == 0


@pytest.mark.asyncio
async def test_driver_for_uuid_rejects_cross_graph_collision():
    driver = FakeFalkorDriver({'group-a': {'duplicate'}, 'group-b': {'duplicate'}})

    with pytest.raises(AmbiguousUuidError, match='group-a, group-b'):
        await driver_for_uuid(driver, ENTITY_EDGE_UUID_QUERY, 'duplicate')

    assert driver.client.clone_calls == 0


@pytest.mark.asyncio
async def test_delete_episode_routes_cascade_to_owning_graph(monkeypatch):
    client = FakeGraphiti(FakeFalkorDriver({'default_db': set(), 'group-a': {'episode-1'}}))
    monkeypatch.setattr(server, 'graphiti_service', FakeGraphitiService(client))

    response = await server.delete_episode('episode-1')

    assert response['message'] == 'Episode with UUID episode-1 deleted successfully'
    assert client.deleted_episodes == [('group-a', 'episode-1')]
    assert client.driver._database == 'default_db'


@pytest.mark.asyncio
async def test_get_and_delete_entity_edge_use_owning_graph(monkeypatch):
    client = FakeGraphiti(FakeFalkorDriver({'default_db': set(), 'group-b': {'edge-1'}}))
    monkeypatch.setattr(server, 'graphiti_service', FakeGraphitiService(client))

    edge = SimpleNamespace(delete=AsyncMock())
    get_by_uuid = AsyncMock(return_value=edge)
    monkeypatch.setattr(server.EntityEdge, 'get_by_uuid', get_by_uuid)
    monkeypatch.setattr(server, 'format_fact_result', lambda _edge: {'uuid': 'edge-1'})

    assert await server.get_entity_edge('edge-1') == {'uuid': 'edge-1'}
    response = await server.delete_entity_edge('edge-1')

    assert response['message'] == 'Entity edge with UUID edge-1 deleted successfully'
    assert [call.args[0]._database for call in get_by_uuid.await_args_list] == [
        'group-b',
        'group-b',
    ]
    assert [call.args[0]._database for call in edge.delete.await_args_list] == ['group-b']
    assert client.driver._database == 'default_db'


@pytest.mark.asyncio
async def test_get_episode_entities_batches_uuids_by_owning_graph(monkeypatch):
    client = FakeGraphiti(
        FakeFalkorDriver(
            {
                'default_db': set(),
                'group-a': {'episode-a'},
                'group-b': {'episode-b'},
            }
        )
    )
    monkeypatch.setattr(server, 'graphiti_service', FakeGraphitiService(client))

    response = await server.get_episode_entities(['episode-b', 'episode-a'])

    assert response['nodes'] == []
    assert response['edges'] == []
    assert set(client.provenance_calls) == {
        ('group-b', ('episode-b',)),
        ('group-a', ('episode-a',)),
    }
    assert client.driver._database == 'default_db'
    assert client.driver.client.queries == [
        ('default_db', ('episode-b', 'episode-a')),
        ('group-a', ('episode-b', 'episode-a')),
        ('group-b', ('episode-b', 'episode-a')),
    ]
    assert client.driver.client.clone_calls == 0


@pytest.mark.asyncio
async def test_drivers_for_uuids_rejects_cross_graph_collision():
    driver = FakeFalkorDriver({'group-a': {'duplicate'}, 'group-b': {'duplicate'}})

    with pytest.raises(AmbiguousUuidError, match='group-a, group-b'):
        await drivers_for_uuids(driver, EPISODE_UUIDS_QUERY, ['duplicate'])


@pytest.mark.asyncio
async def test_queue_serializes_driver_rebinding_across_groups():
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingGraphiti:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.groups: list[str] = []

        async def add_episode(self, **kwargs) -> None:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.groups.append(kwargs['group_id'])
            if len(self.groups) == 1:
                first_entered.set()
                await release_first.wait()
            self.active -= 1

    client = BlockingGraphiti()
    service = QueueService()
    await service.initialize(client)
    common = {
        'name': 'episode',
        'content': 'body',
        'source_description': 'test',
        'episode_type': 'text',
        'entity_types': None,
        'uuid': None,
    }

    await service.add_episode(group_id='group-a', **common)
    await service.add_episode(group_id='group-b', **common)
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    await asyncio.sleep(0)

    assert client.max_active == 1
    release_first.set()
    await asyncio.wait_for(service._episode_queues['group-a'].join(), timeout=1)
    await asyncio.wait_for(service._episode_queues['group-b'].join(), timeout=1)

    assert client.groups == ['group-a', 'group-b']
    assert client.max_active == 1
