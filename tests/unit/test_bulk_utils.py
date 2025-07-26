from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx


class DummySession(GraphDriverSession):
    def __init__(self):
        self.calls = []

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def run(self, query: str, **kwargs):
        self.calls.append((query, kwargs))

    async def close(self):
        pass

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)


class DummyDriver(GraphDriver):
    provider = 'falkordb'

    def session(self, database: str | None = None) -> GraphDriverSession:
        return DummySession()

    async def execute_query(self, cypher_query_: str, **kwargs):
        pass

    async def close(self):
        pass

    async def delete_all_indexes(self):
        pass


@pytest.mark.asyncio
async def test_bulk_tx_serializes_episodic_nodes():
    tx = DummySession()
    driver = DummyDriver()
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    episode = EpisodicNode(
        name='ep',
        group_id='g',
        labels=[],
        source=EpisodeType.text,
        source_description='desc',
        content='c',
        created_at=now,
        valid_at=now,
    )
    await add_nodes_and_edges_bulk_tx(tx, [episode], [], [], [], embedder=None, driver=driver)
    assert tx.calls, 'tx.run should have been called'
    first_kwargs = tx.calls[0][1]
    episodes = first_kwargs.get('episodes')
    assert isinstance(episodes[0]['created_at'], str)
    assert isinstance(episodes[0]['valid_at'], str)
    assert episodes[0]['source'] == 'text'
