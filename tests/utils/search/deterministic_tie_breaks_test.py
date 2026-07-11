from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)


class _CapturingExecutor:
    provider = GraphProvider.FALKORDB
    search_interface = None

    def __init__(self) -> None:
        self.queries: list[str] = []

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 128,
    ) -> str:
        return query

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> tuple[list, list, None]:
        self.queries.append(cypher_query_)
        return [], [], None


LegacySearchCall = Callable[[_CapturingExecutor, SearchFilters], Awaitable[list]]
OperationsSearchCall = Callable[
    [FalkorSearchOperations, _CapturingExecutor, SearchFilters], Awaitable[list]
]


async def _legacy_edge_fulltext(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await edge_fulltext_search(driver, "query", filters)


async def _legacy_edge_similarity(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await edge_similarity_search(driver, [1.0], None, None, filters)


async def _legacy_node_fulltext(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await node_fulltext_search(driver, "query", filters)


async def _legacy_node_similarity(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await node_similarity_search(driver, [1.0], filters)


async def _legacy_episode_fulltext(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await episode_fulltext_search(driver, "query", filters)


async def _legacy_community_fulltext(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await community_fulltext_search(driver, "query")


async def _legacy_community_similarity(driver: _CapturingExecutor, filters: SearchFilters) -> list:
    return await community_similarity_search(driver, [1.0])


async def _ops_node_fulltext(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.node_fulltext_search(executor, "query", filters)


async def _ops_node_similarity(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.node_similarity_search(executor, [1.0], filters)


async def _ops_edge_fulltext(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.edge_fulltext_search(executor, "query", filters)


async def _ops_edge_similarity(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.edge_similarity_search(executor, [1.0], None, None, filters)


async def _ops_episode_fulltext(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.episode_fulltext_search(executor, "query", filters)


async def _ops_community_fulltext(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.community_fulltext_search(executor, "query")


async def _ops_community_similarity(
    operations: FalkorSearchOperations, executor: _CapturingExecutor, filters: SearchFilters
) -> list:
    return await operations.community_similarity_search(executor, [1.0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_call", "expected_order"),
    [
        (_legacy_edge_fulltext, "ORDER BY score DESC, uuid ASC"),
        (_legacy_edge_similarity, "ORDER BY score DESC, uuid ASC"),
        (_legacy_node_fulltext, "ORDER BY score DESC, n.uuid ASC"),
        (_legacy_node_similarity, "ORDER BY score DESC, uuid ASC"),
        (_legacy_episode_fulltext, "ORDER BY score DESC, uuid ASC"),
        (_legacy_community_fulltext, "ORDER BY score DESC, uuid ASC"),
        (_legacy_community_similarity, "ORDER BY score DESC, uuid ASC"),
    ],
)
async def test_legacy_falkor_search_orders_equal_scores_by_uuid(
    search_call: LegacySearchCall, expected_order: str
) -> None:
    driver = _CapturingExecutor()

    await search_call(driver, SearchFilters())

    assert len(driver.queries) == 1
    assert expected_order in driver.queries[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_call", "expected_order"),
    [
        (_ops_node_fulltext, "ORDER BY score DESC, n.uuid ASC"),
        (_ops_node_similarity, "ORDER BY score DESC, uuid ASC"),
        (_ops_edge_fulltext, "ORDER BY score DESC, uuid ASC"),
        (_ops_edge_similarity, "ORDER BY score DESC, uuid ASC"),
        (_ops_episode_fulltext, "ORDER BY score DESC, uuid ASC"),
        (_ops_community_fulltext, "ORDER BY score DESC, uuid ASC"),
        (_ops_community_similarity, "ORDER BY score DESC, uuid ASC"),
    ],
)
async def test_falkor_search_operations_order_equal_scores_by_uuid(
    search_call: OperationsSearchCall, expected_order: str
) -> None:
    operations = FalkorSearchOperations()
    executor = _CapturingExecutor()

    await search_call(operations, executor, SearchFilters())

    assert len(executor.queries) == 1
    assert expected_order in executor.queries[0]


def test_rrf_orders_equal_scores_by_uuid() -> None:
    ranked_uuids, scores = rrf([["uuid-b"], ["uuid-a"]])

    assert ranked_uuids == ["uuid-a", "uuid-b"]
    assert scores == [1.0, 1.0]
