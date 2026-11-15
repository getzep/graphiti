"""Tests for the `MATCH`-only update queries and for the write time limit."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_update_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_entity_node_update_bulk_query,
)
from graphiti_core.utils import bulk_utils
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk, execute_write_with_timeout

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
