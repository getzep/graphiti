"""Exercise MCP episode reads through the real handler and node query/parser."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.driver.driver import GraphDriver, GraphProvider

# Support running from either the repository root or mcp_server/.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from mcp_server.src import graphiti_mcp_server as server  # noqa: E402


def record(uuid, group_id):
    return dict(
        uuid=uuid,
        group_id=group_id,
        name=uuid,
        content='episode body',
        source='text',
        source_description='routing regression',
        entity_edges=[],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class PartitionedDriver:
    """In-memory query transport; not evidence of live FalkorDB behavior."""

    provider = GraphProvider.FALKORDB
    graph_operations_interface = None
    default_group_id = '_'
    with_database = GraphDriver.with_database

    def __init__(self, database='base'):
        self._database = database
        self.graphs = {
            'base': [],
            'a': [record(uuid, 'a') for uuid in ('09', '05', '01')],
            'b': [record(uuid, 'b') for uuid in ('08', '06', '02')],
            'outside': [record('99', 'outside')],
            'default_db': [record('07', '_')],
        }
        self.calls = []

    def clone(self, database):
        if self.provider != GraphProvider.FALKORDB or database == self._database:
            return self
        raise AssertionError('Reads must not start clone index-building tasks')

    async def execute_query(self, query, **params):
        await asyncio.sleep(0)
        self.calls.append((self._database, params['group_ids'], params['limit']))
        assert 'ORDER BY uuid DESC' in query
        limit = params['limit']
        if limit < 0:
            raise ValueError('LIMIT must be non-negative')
        rows = [
            row for row in self.graphs[self._database] if row['group_id'] in params['group_ids']
        ]
        rows.sort(key=lambda row: row['uuid'], reverse=True)
        return rows[:limit], None, None


@pytest.fixture
def install_client(monkeypatch, config):
    monkeypatch.setattr(server, 'config', config, raising=False)
    monkeypatch.setattr(server, '_group_drivers', {})

    def install(driver):
        client = SimpleNamespace(driver=driver, clients=SimpleNamespace(driver=driver))
        service = SimpleNamespace(get_client=AsyncMock(return_value=client))
        monkeypatch.setattr(server, 'graphiti_service', service)
        return client

    return install


@pytest.mark.parametrize('database', ['base', 'a'])
@pytest.mark.parametrize('groups', [['a', 'b'], ['b', 'a'], ['a', 'b', 'a']])
async def test_global_order_and_limit(install_client, database, groups):
    driver = PartitionedDriver(database)
    client = install_client(driver)
    result = await server.get_episodes(group_ids=groups, max_episodes=2)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '08']
    assert {episode['group_id'] for episode in result['episodes']} == {'a', 'b'}
    assert len(driver.calls) == 2
    assert all(len(groups) == 1 and limit == 2 for _, groups, limit in driver.calls)
    assert client.driver is driver
    assert client.clients.driver is driver
    assert driver._database == database


async def test_duplicate_groups_do_not_duplicate_episodes(install_client):
    driver = PartitionedDriver()
    install_client(driver)
    result = await server.get_episodes(group_ids=['a', 'a'], max_episodes=10)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '05', '01']
    assert driver.calls == [('a', ['a'], 10)]


@pytest.mark.parametrize('database', ['base', '_'])
async def test_default_group_uses_default_physical_graph(install_client, database):
    driver = PartitionedDriver(database)
    driver.graphs['_'] = driver.graphs['default_db']
    install_client(driver)
    result = await server.get_episodes(group_ids=['_', 'b'], max_episodes=2)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['08', '07']
    expected_database = '_' if database == '_' else 'default_db'
    assert driver.calls == [(expected_database, ['_'], 2), ('b', ['b'], 2)]


@pytest.mark.parametrize('limit', [0, -1])
async def test_zero_and_negative_limits(install_client, limit):
    install_client(PartitionedDriver())
    result = await server.get_episodes(group_ids=['a', 'b'], max_episodes=limit)
    if limit == 0:
        assert 'episodes' in result, result
        assert result['episodes'] == []
    else:
        assert 'error' in result, result
        assert 'LIMIT must be non-negative' in result['error']


@pytest.mark.parametrize(
    'provider', [GraphProvider.NEO4J, GraphProvider.NEPTUNE, GraphProvider.KUZU]
)
async def test_other_providers_keep_upstream_group_queries(install_client, provider):
    driver = PartitionedDriver()
    driver.provider = provider
    driver.graphs['base'] = driver.graphs['a'] + driver.graphs['b']
    install_client(driver)
    result = await server.get_episodes(group_ids=['a', 'b'], max_episodes=2)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '05']
    assert driver.calls == [('base', ['a'], 2), ('base', ['b'], 2)]


async def test_single_configured_group_is_unchanged(install_client):
    driver = PartitionedDriver('a')
    install_client(driver)
    result = await server.get_episodes(group_ids='a', max_episodes=2)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '05']
    assert driver.calls == [('a', ['a'], 2)]


async def test_empty_groups_do_not_query(install_client):
    driver = PartitionedDriver()
    install_client(driver)
    result = await server.get_episodes(group_ids=[], max_episodes=2)
    assert 'episodes' in result, result
    assert result['episodes'] == []
    assert driver.calls == []


async def test_concurrent_handlers_preserve_driver(install_client):
    driver = PartitionedDriver()
    client = install_client(driver)
    results = await asyncio.gather(
        server.get_episodes(group_ids=['a', 'b'], max_episodes=2),
        server.get_episodes(group_ids=['b', 'a'], max_episodes=1),
    )
    episode_ids = []
    for result in results:
        assert 'episodes' in result, result
        episode_ids.append([episode['uuid'] for episode in result['episodes']])
    assert episode_ids == [['09', '08'], ['09']]
    assert client.driver is client.clients.driver is driver
    assert driver._database == 'base'


@pytest.mark.parametrize('limit, expected', [(2, ['09', '08']), (4, ['09', '08', '06', '05'])])
async def test_global_limit_uses_the_same_order_as_each_graph(
    install_client, monkeypatch, limit, expected
):
    driver = PartitionedDriver()
    # UUIDs are not chronological. A timestamp merge after a UUID-limited
    # per-graph query cannot produce chronological top K, so use UUID throughout.
    for group in ('a', 'b'):
        for row in driver.graphs[group]:
            row['created_at'] = datetime(2026, 1, 10 - int(row['uuid']), tzinfo=timezone.utc)
    # Allow upstream's clone path in the baseline reproduction so this test
    # isolates the merge comparator from the separate clone-side-effect tests.
    monkeypatch.setattr(driver, 'clone', lambda database: driver.with_database(database))
    install_client(driver)
    result = await server.get_episodes(group_ids=['b', 'a'], max_episodes=limit)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == expected


@pytest.mark.parametrize('groups', [None, 'a', ['a']])
async def test_single_group_routes_from_the_base_graph(install_client, monkeypatch, config, groups):
    driver = PartitionedDriver()
    # Single-group behavior comes from upstream's cached driver helper.
    calls = []

    def clone(database):
        calls.append(database)
        return driver.with_database(database)

    monkeypatch.setattr(driver, 'clone', clone)
    install_client(driver)
    config.graphiti.group_id = 'a'
    result = await server.get_episodes(group_ids=groups, max_episodes=2)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '05']
    assert calls == ['a']
    assert driver._database == 'base'


async def test_single_group_preserves_upstream_timestamp_sort(install_client):
    driver = PartitionedDriver('a')
    for row in driver.graphs['a']:
        row['created_at'] = datetime(2026, 1, 10 - int(row['uuid']), tzinfo=timezone.utc)
    install_client(driver)
    result = await server.get_episodes(group_ids='a', max_episodes=2)
    assert 'episodes' in result, result
    assert [episode['uuid'] for episode in result['episodes']] == ['05', '09']
    assert driver.calls == [('a', ['a'], 2)]
