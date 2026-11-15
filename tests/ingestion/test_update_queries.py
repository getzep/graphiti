"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Tests for the `MATCH`-only update queries and for the write time limit.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.edges import EntityEdge
from graphiti_core.ingestion.phases import admit_bulk, admit_episode
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_update_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_entity_node_update_bulk_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.utils import bulk_utils
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    execute_write_with_timeout,
    update_entity_edges_tx,
    update_entity_nodes_tx,
)
from tests.ingestion.ingestion_fixtures import GROUP, T0, Harness

NODES = [{'uuid': 'n1', 'name': 'Ana', 'labels': ['Entity', 'Person']}]


def query_text(queries) -> str:
    if isinstance(queries, str):
        return queries
    return '\n'.join(query[0] if isinstance(query, tuple) else query for query in queries)


@pytest.mark.parametrize('provider', list(GraphProvider))
def test_edge_update_query_has_no_merge(provider):
    save = get_entity_edge_save_bulk_query(provider)
    update = get_entity_edge_update_bulk_query(provider)
    assert 'MERGE' in save
    assert 'MERGE' not in update
    # The update query is the save query with one word changed.
    assert update == save.replace('MERGE (source)-[', 'MATCH (source)-[')


@pytest.mark.parametrize('provider', list(GraphProvider))
def test_node_update_query_has_no_merge(provider):
    save = get_entity_node_save_bulk_query(provider, NODES)
    update = get_entity_node_update_bulk_query(provider, NODES)
    assert type(update) is type(save)
    assert 'MERGE' in query_text(save)
    assert 'MERGE' not in query_text(update)
    assert query_text(update) == query_text(save).replace(
        'MERGE (n:Entity {uuid: ', 'MATCH (n:Entity {uuid: '
    )


class FakeSession:
    def __init__(self):
        self.execute_write = AsyncMock(return_value='result')
        self.close = AsyncMock()


async def test_bulk_write_with_no_timeout_uses_the_session_call_of_today(monkeypatch):
    session = FakeSession()
    driver = SimpleNamespace(provider=GraphProvider.NEO4J, session=lambda: session)
    with_timeout = AsyncMock()
    monkeypatch.setattr(bulk_utils, 'execute_write_with_timeout', with_timeout)

    await add_nodes_and_edges_bulk(driver, [], [], [], [], AsyncMock())

    with_timeout.assert_not_awaited()
    args, kwargs = session.execute_write.await_args
    assert args[0] is bulk_utils.add_nodes_and_edges_bulk_tx
    assert kwargs == {'driver': driver}
    session.close.assert_awaited_once()


async def test_neo4j_write_gets_the_limit_as_transaction_timeout():
    session = FakeSession()
    driver = SimpleNamespace(provider=GraphProvider.NEO4J, session=lambda: session)

    async def work(tx, value):
        return value

    assert await execute_write_with_timeout(driver, 7.5, work, 'x') == 'result'
    wrapped = session.execute_write.await_args.args[0]
    assert wrapped.timeout == 7.5
    assert session.execute_write.await_args.args[1:] == ('x',)
    session.close.assert_awaited_once()


async def test_falkordb_write_gets_the_limit_as_query_timeout():
    graph = SimpleNamespace(query=AsyncMock())
    session = FalkorDriverSession(graph)  # type: ignore[arg-type]
    driver = SimpleNamespace(provider=GraphProvider.FALKORDB, session=lambda: session)

    async def work(tx):
        await tx.run('RETURN 1', value=1)

    await execute_write_with_timeout(driver, 2.5, work)
    assert graph.query.await_args.kwargs == {'timeout': 2500}

    # A session with no limit sends no timeout argument.
    plain_graph = SimpleNamespace(query=AsyncMock())
    await FalkorDriverSession(plain_graph).run('RETURN 1', value=1)  # type: ignore[arg-type]
    assert plain_graph.query.await_args.kwargs == {}


async def test_write_is_cancelled_at_the_limit_on_a_driver_with_no_server_limit():
    import asyncio

    class SlowSession(FakeSession):
        def __init__(self):
            super().__init__()

            async def slow(func, *args, **kwargs):
                await asyncio.sleep(5)

            self.execute_write = slow

    session = SlowSession()
    driver = SimpleNamespace(provider=GraphProvider.NEPTUNE, session=lambda: session)
    with pytest.raises(asyncio.TimeoutError):
        await execute_write_with_timeout(driver, 0.05, AsyncMock())
    session.close.assert_awaited_once()


# --- a driver with `graph_operations_interface` -------------------------------------


class SaveOnlyInterface(GraphOperationsInterface):
    """An interface that has the bulk saves of today and no update operation."""

    async def episodic_node_save_bulk(self, _cls, driver, transaction, nodes, batch_size=100):
        return None

    async def node_save_bulk(self, _cls, driver, transaction, nodes, batch_size=100):
        return None

    async def episodic_edge_save_bulk(
        self, _cls, driver, transaction, episodic_edges, batch_size=100
    ):
        return None

    async def edge_save_bulk(self, _cls, driver, transaction, edges, batch_size=100):
        return None


class FullInterface(SaveOnlyInterface):
    calls: list[tuple[str, list[str]]] = []

    async def node_update_bulk(self, _cls, driver, transaction, nodes, batch_size=100):
        self.calls.append(('node_update_bulk', [node['uuid'] for node in nodes]))

    async def edge_update_bulk(self, _cls, driver, transaction, edges, batch_size=100):
        self.calls.append(('edge_update_bulk', [edge['uuid'] for edge in edges]))


async def admit_with_interface(monkeypatch, interface, bulk: bool):
    h = Harness(monkeypatch)
    h.graphiti.retrieve_episodes = AsyncMock(return_value=[])
    h.clients.driver.graph_operations_interface = interface
    if bulk:
        return h, admit_bulk(
            h.graphiti,
            h.store,
            h.config,
            unit_id='u1',
            group_id=GROUP,
            bulk_episodes=[
                RawEpisode(
                    name='episode',
                    content='content',
                    source_description='test',
                    source=EpisodeType.text,
                    reference_time=T0,
                )
            ],
        )
    return h, admit_episode(
        h.graphiti,
        h.store,
        h.config,
        unit_id='u1',
        group_id=GROUP,
        name='episode',
        episode_body='content',
        source_description='test',
        reference_time=T0,
    )


@pytest.mark.parametrize('bulk', [False, True], ids=['episode', 'bulk'])
async def test_admission_rejects_an_interface_with_no_update_operation(monkeypatch, bulk):
    h, admission = await admit_with_interface(monkeypatch, SaveOnlyInterface(), bulk)

    with pytest.raises(ValueError, match='edge_update_bulk, node_update_bulk'):
        await admission

    # The error is before the admission: no unit and no graph write.
    assert h.store._groups == {}
    assert h.graph.log == []
    assert h.graph.episodes == {}


async def test_admission_names_each_operation_that_an_interface_does_not_have(monkeypatch):
    h, admission = await admit_with_interface(monkeypatch, GraphOperationsInterface(), False)

    with pytest.raises(ValueError) as error:
        await admission

    for name in (
        'edge_save_bulk',
        'edge_update_bulk',
        'episodic_edge_save_bulk',
        'episodic_node_save_bulk',
        'node_save_bulk',
        'node_update_bulk',
    ):
        assert name in str(error.value)


async def test_admission_accepts_an_interface_with_each_operation(monkeypatch):
    h, admission = await admit_with_interface(monkeypatch, FullInterface(), False)

    unit = await admission

    assert unit.seq == 1


async def test_update_functions_call_the_interface_of_the_driver():
    interface = FullInterface()
    interface.calls.clear()
    driver = MagicMock()
    driver.graph_operations_interface = interface
    embedder = MagicMock()
    node = EntityNode(uuid='n1', name='Acme', group_id=GROUP, name_embedding=[0.1, 0.2])
    edge = EntityEdge(
        uuid='e1',
        group_id=GROUP,
        source_node_uuid='n1',
        target_node_uuid='n2',
        name='RELATES',
        fact='fact',
        created_at=T0,
        fact_embedding=[0.1, 0.2],
    )
    tx = MagicMock()
    tx.run = AsyncMock()

    await update_entity_nodes_tx(tx, [node], embedder, driver)
    await update_entity_edges_tx(tx, [edge], embedder, driver)

    assert interface.calls == [('node_update_bulk', ['n1']), ('edge_update_bulk', ['e1'])]
    tx.run.assert_not_awaited()
