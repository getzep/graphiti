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
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '05', '01']
    assert driver.calls == [('a', ['a'], 10)]


async def test_default_group_uses_default_physical_graph(install_client):
    driver = PartitionedDriver()
    install_client(driver)
    result = await server.get_episodes(group_ids=['_', 'b'], max_episodes=2)
    assert [episode['uuid'] for episode in result['episodes']] == ['08', '07']
    assert driver.calls == [('default_db', ['_'], 2), ('b', ['b'], 2)]


@pytest.mark.parametrize('limit', [0, -1])
async def test_zero_and_negative_limits(install_client, limit):
    install_client(PartitionedDriver())
    result = await server.get_episodes(group_ids=['a', 'b'], max_episodes=limit)
    if limit == 0:
        assert result['episodes'] == []
    else:
        assert 'LIMIT must be non-negative' in result['error']


@pytest.mark.parametrize(
    'provider', [GraphProvider.NEO4J, GraphProvider.NEPTUNE, GraphProvider.KUZU]
)
async def test_other_providers_keep_combined_query(install_client, provider):
    driver = PartitionedDriver()
    driver.provider = provider
    driver.graphs['base'] = driver.graphs['a'] + driver.graphs['b']
    install_client(driver)
    result = await server.get_episodes(group_ids=['a', 'b'], max_episodes=2)
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '08']
    assert driver.calls == [('base', ['a', 'b'], 2)]


async def test_single_configured_group_is_unchanged(install_client):
    driver = PartitionedDriver('a')
    install_client(driver)
    result = await server.get_episodes(group_ids='a', max_episodes=2)
    assert [episode['uuid'] for episode in result['episodes']] == ['09', '05']
    assert driver.calls == [('a', ['a'], 2)]


async def test_empty_groups_do_not_query(install_client):
    driver = PartitionedDriver()
    install_client(driver)
    result = await server.get_episodes(group_ids=[], max_episodes=2)
    assert result['episodes'] == []
    assert driver.calls == []


async def test_concurrent_handlers_preserve_driver(install_client):
    driver = PartitionedDriver()
    client = install_client(driver)
    results = await asyncio.gather(
        server.get_episodes(group_ids=['a', 'b'], max_episodes=2),
        server.get_episodes(group_ids=['b', 'a'], max_episodes=1),
    )
    assert [[episode['uuid'] for episode in result['episodes']] for result in results] == [
        ['09', '08'],
        ['09'],
    ]
    assert client.driver is client.clients.driver is driver
    assert driver._database == 'base'
