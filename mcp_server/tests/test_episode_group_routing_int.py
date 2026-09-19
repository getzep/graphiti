"""Real FalkorDB/MCP-handler regression, without an LLM or MCP transport.

Opt in with GRAPHITI_TEST_FALKORDB_URL=redis://localhost:6379 and run:
    pytest -c mcp_server/pytest.ini --confcutdir=mcp_server \
        mcp_server/tests/test_episode_group_routing_int.py

Only randomly named graphs owned by this test are created/deleted.
"""

import os
import sys
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

# Support running from either the repository root or mcp_server/.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from mcp_server.src import graphiti_mcp_server as server  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
def live_driver():
    uri = os.environ.get('GRAPHITI_TEST_FALKORDB_URL')
    if not uri:
        pytest.skip('Set GRAPHITI_TEST_FALKORDB_URL to run against a real FalkorDB')
    from falkordb.asyncio import FalkorDB
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    # A synchronous fixture constructs outside the event loop, so no index task
    # starts. All subsequent scopes share this connection via with_database.
    driver = FalkorDriver(
        falkor_db=FalkorDB.from_url(uri, socket_timeout=5, socket_connect_timeout=5),
        database=f'mcprouting{uuid4().hex}',
    )
    assert driver._init_task is None
    return driver


@pytest.mark.parametrize('base_is_group', [False, True])
async def test_live_multigroup_read(monkeypatch, config, live_driver, base_is_group):
    driver = live_driver
    base = driver._database
    group_a, group_b = f'{base}a', f'{base}b'
    owned_graphs = [base, group_a, group_b]
    client = SimpleNamespace(driver=driver, clients=SimpleNamespace(driver=driver))
    monkeypatch.setattr(server, 'config', config, raising=False)
    monkeypatch.setattr(
        server, 'graphiti_service', SimpleNamespace(get_client=AsyncMock(return_value=client))
    )
    try:
        # A physical base graph with no episodes reproduces a restarted client.
        await driver.execute_query('CREATE (:RoutingTest)')
        for group, uuids in ((group_a, ['09', '05', '01']), (group_b, ['08', '06', '02'])):
            scoped = driver.with_database(group)
            for uuid in uuids:
                await scoped.execute_query(
                    'CREATE (e:Episodic {uuid: $uuid, group_id: $group_id, name: $uuid, '
                    'content: "body", source: "text", source_description: "routing test", '
                    'entity_edges: [], created_at: "2026-01-01T00:00:00Z", '
                    'valid_at: "2026-01-01T00:00:00Z"})',
                    uuid=uuid,
                    group_id=group,
                )
        if base_is_group:
            driver = driver.with_database(group_a)
            client.driver = driver
            client.clients.driver = driver
        expected_database = driver._database

        for groups in ([group_a, group_b], [group_b, group_a], [group_a, group_b, group_a]):
            result = await server.get_episodes(group_ids=groups, max_episodes=2)
            assert 'error' not in result, result
            assert [episode['uuid'] for episode in result['episodes']] == ['09', '08']
            assert client.driver is client.clients.driver is driver
            assert driver._database == expected_database
            assert driver._init_task is None

        empty = await server.get_episodes(group_ids=[group_a, group_b], max_episodes=0)
        assert empty['episodes'] == []
        invalid = await server.get_episodes(group_ids=[group_a, group_b], max_episodes=-1)
        assert 'error' in invalid
    finally:
        # No broad clear/delete: every graph name belongs to this invocation.
        from redis.exceptions import ResponseError

        for graph in owned_graphs:
            with suppress(ResponseError):  # A failure during setup may leave a graph absent.
                await live_driver.client.select_graph(graph).delete()
        await live_driver.close()
