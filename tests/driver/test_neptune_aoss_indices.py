from typing import Any, cast

import pytest

from graphiti_core.driver.driver import COMMUNITY_INDEX_NAME, GraphProvider
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.neptune_driver import aoss_indices
from graphiti_core.nodes import CommunityNode


class FakeNeptuneDriver:
    provider = GraphProvider.NEPTUNE
    graph_operations_interface = None

    def __init__(self):
        self.saved_to_aoss: list[tuple[str, list[dict[str, Any]]]] = []
        self.execute_query_kwargs: dict[str, Any] | None = None

    def save_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int:
        self.saved_to_aoss.append((name, data))
        return len(data)

    async def execute_query(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.execute_query_kwargs = kwargs
        return []


class FakeAOSSSearchDriver:
    def __init__(self):
        self.queries: list[tuple[str, str, int]] = []

    def run_aoss_query(self, name: str, query: str, limit: int = 10) -> dict[str, Any]:
        self.queries.append((name, query, limit))
        return {'hits': {'total': {'value': 0}, 'hits': []}}


def test_default_community_index_matches_registered_aoss_index():
    registered_indices = {index['index_name'] for index in aoss_indices}

    assert COMMUNITY_INDEX_NAME == 'community_name'
    assert COMMUNITY_INDEX_NAME in registered_indices


@pytest.mark.asyncio
async def test_community_node_fallback_saves_to_registered_aoss_index():
    driver = FakeNeptuneDriver()
    node = CommunityNode(uuid='community-1', name='Community 1', group_id='group-1')

    await node.save(driver)  # type: ignore[arg-type]

    assert driver.saved_to_aoss == [
        (
            COMMUNITY_INDEX_NAME,
            [{'name': 'Community 1', 'uuid': 'community-1', 'group_id': 'group-1'}],
        )
    ]
    assert driver.execute_query_kwargs is not None
    assert driver.execute_query_kwargs['uuid'] == 'community-1'


@pytest.mark.asyncio
async def test_neptune_community_search_uses_registered_aoss_index():
    driver = FakeAOSSSearchDriver()
    operations = NeptuneSearchOperations(cast(Any, driver))

    results = await operations.community_fulltext_search(
        cast(Any, None), 'community query', limit=3
    )

    assert results == []
    assert driver.queries == [(COMMUNITY_INDEX_NAME, 'community query', 3)]
