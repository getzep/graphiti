"""Unit tests for per-group driver routing in the MCP tools.

graphiti-core 0.30.2 stores each non-default group_id in its own backend graph
(FalkorDB clones the driver per database). These tests verify that the tools
route their reads and writes through a driver bound to the requested group.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import graphiti_mcp_server as server


class FakeDriver:
    """Records clone() calls and returns a distinct driver per database."""

    def __init__(self, database: str = 'default_db', share: list | None = None):
        self.database = database
        self.databases = share if share is not None else []

    def clone(self, database: str):
        self.databases.append(database)
        return FakeDriver(database=database, share=self.databases)


class SameDriver:
    """A driver whose clone() is a no-op (Neo4j/Neptune behaviour)."""

    def clone(self, database: str):
        return self


class FakeClients(SimpleNamespace):
    """Stands in for the pydantic GraphitiClients holder."""

    def model_copy(self, update: dict):
        return FakeClients(**{**vars(self), **update})


class FakeClient:
    """Minimal Graphiti stand-in: driver, a clients holder, and a call log.

    copy.copy() shares the seen list between the original and the scoped copy,
    so remove_episode records which object the call actually ran on.
    """

    def __init__(self, driver):
        self.driver = driver
        self.clients = FakeClients(driver=driver)
        self.seen: list = []

    async def remove_episode(self, uuid):
        self.seen.append((self, uuid))


def make_client(driver) -> FakeClient:
    return FakeClient(driver)


def install_service(monkeypatch, client, default_group_id='main'):
    service = SimpleNamespace(get_client=lambda: _async_return(client))
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(
        server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id=default_group_id)),
        raising=False,
    )
    return service


async def _async_return(value):
    return value


def test_driver_for_group_clones_per_group():
    client = make_client(FakeDriver())

    driver = server._driver_for_group(client, 'group-a')

    assert driver is not client.driver
    assert driver.database == 'group-a'


def test_driver_for_group_returns_same_driver_when_clone_is_noop():
    client = make_client(SameDriver())

    assert server._driver_for_group(client, 'group-a') is client.driver


def test_client_for_group_returns_same_client_when_clone_is_noop():
    client = make_client(SameDriver())

    assert server._client_for_group(client, 'group-a') is client


def test_client_for_group_scopes_driver_without_touching_original():
    client = make_client(FakeDriver())

    scoped = server._client_for_group(client, 'group-b')

    assert scoped is not client
    assert scoped.driver.database == 'group-b'
    assert scoped.clients.driver is scoped.driver
    assert client.driver.database == 'default_db'
    assert client.clients.driver.database == 'default_db'


async def test_get_episodes_queries_each_group_with_its_own_driver(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client, default_group_id='')
    calls = []

    def episode(uuid, created_at):
        return SimpleNamespace(
            uuid=uuid,
            name='ep',
            content='body',
            created_at=created_at,
            source='text',
            source_description='d',
            group_id='g',
        )

    old = datetime(2024, 1, 1, tzinfo=timezone.utc)
    mid = datetime(2024, 6, 1, tzinfo=timezone.utc)
    new = datetime(2025, 1, 1, tzinfo=timezone.utc)
    replies = {
        'g1': [episode('old', old)],
        'g2': [episode('new', new), episode('mid', mid)],
    }

    async def fake_get_by_group_ids(driver, group_ids, limit=None, uuid_cursor=None):
        calls.append((driver, group_ids, limit))
        return replies[group_ids[0]]

    monkeypatch.setattr(
        server.EpisodicNode, 'get_by_group_ids', staticmethod(fake_get_by_group_ids)
    )

    response = await server.get_episodes(group_ids=['g1', 'g2'], max_episodes=2)

    assert {call[0].database for call in calls} == {'g1', 'g2'}
    assert all(call[0] is not client.driver for call in calls)
    # Newest first, truncated to max_episodes.
    assert [e['uuid'] for e in response['episodes']] == ['new', 'mid']


async def test_clear_graph_clears_each_group_with_its_own_driver(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client, default_group_id='')
    calls = []

    async def fake_clear_data(driver, group_ids=None):
        calls.append((driver, group_ids))

    monkeypatch.setattr(server, 'clear_data', fake_clear_data)

    response = await server.clear_graph(group_ids=['g1', 'g2'])

    assert [call[0].database for call in calls] == ['g1', 'g2']
    assert [call[1] for call in calls] == [['g1'], ['g2']]
    assert 'g1' in response['message'] and 'g2' in response['message']


async def test_get_entity_edge_uses_cloned_driver_for_explicit_group(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client)
    calls = []

    async def fake_get_by_uuid(driver, uuid):
        calls.append(driver)
        return SimpleNamespace(
            uuid=uuid,
            fact='f',
            name='e',
            group_id='g',
            created_at=None,
            valid_at=None,
            invalid_at=None,
            expired_at=None,
            source_node_uuid='a',
            target_node_uuid='b',
            attributes={},
            episodes=[],
        )

    monkeypatch.setattr(server.EntityEdge, 'get_by_uuid', staticmethod(fake_get_by_uuid))
    monkeypatch.setattr(server, 'format_fact_result', lambda edge: {'uuid': edge.uuid})

    await server.get_entity_edge('uuid-1', group_id='group-x')

    assert calls[0].database == 'group-x'
    assert calls[0] is not client.driver


async def test_get_entity_edge_uses_base_driver_without_any_group(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client, default_group_id='')
    calls = []

    async def fake_get_by_uuid(driver, uuid):
        calls.append(driver)
        return SimpleNamespace(uuid=uuid)

    monkeypatch.setattr(server.EntityEdge, 'get_by_uuid', staticmethod(fake_get_by_uuid))
    monkeypatch.setattr(server, 'format_fact_result', lambda edge: {'uuid': edge.uuid})

    await server.get_entity_edge('uuid-1')

    assert calls == [client.driver]


async def test_delete_entity_edge_uses_cloned_driver_for_explicit_group(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client)
    calls = []

    class FakeEdge:
        async def delete(self, driver):
            calls.append(('delete', driver))

    async def fake_get_by_uuid(driver, uuid):
        calls.append(('get', driver))
        return FakeEdge()

    monkeypatch.setattr(server.EntityEdge, 'get_by_uuid', staticmethod(fake_get_by_uuid))

    response = await server.delete_entity_edge('uuid-1', group_id='group-y')

    assert calls == [('get', calls[0][1]), ('delete', calls[0][1])]
    assert calls[0][1].database == 'group-y'
    assert 'deleted successfully' in response['message']


async def test_delete_episode_calls_remove_episode_on_scoped_client(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client)

    response = await server.delete_episode('uuid-9', group_id='group-z')

    assert client.seen[0][1] == 'uuid-9'
    # The call ran on a scoped copy whose driver is the group clone.
    assert client.seen[0][0] is not client
    assert client.seen[0][0].driver.database == 'group-z'
    assert client.driver.database == 'default_db'
    assert 'deleted successfully' in response['message']


async def test_delete_episode_uses_base_client_without_any_group(monkeypatch):
    client = make_client(FakeDriver())
    install_service(monkeypatch, client, default_group_id='')

    await server.delete_episode('uuid-9')

    assert client.seen[0][0] is client
