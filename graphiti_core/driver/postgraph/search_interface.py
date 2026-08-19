"""SearchInterface adapter for the PostGraph driver.

graphiti_core/search/search_utils.py checks `if driver.search_interface:` and
otherwise falls back to raw Cypher, which PostgreSQL rejects outright
("syntax error at or near MATCH"). The driver already implements every search
in operations/search_ops.py; without this adapter none of them is ever reached,
so persistence works and the first search fails.

The interface takes `driver`, the operations take a QueryExecutor, and the
driver is one — the adapter is a signature translation, not new search logic.
"""
from typing import Any

from graphiti_core.driver.postgraph.operations.search_ops import PGSearchOperations
from graphiti_core.driver.search_interface.search_interface import SearchInterface


class PostGraphSearchInterface(SearchInterface):
    """Routes Graphiti's searches to the driver's SQL implementations."""

    class Config:
        arbitrary_types_allowed = True

    async def edge_fulltext_search(self, driver: Any, query: str, search_filter: Any,
                                   group_ids: list[str] | None = None,
                                   limit: int = 100) -> list[Any]:
        return await PGSearchOperations().edge_fulltext_search(driver, query, search_filter,
                                                      group_ids, limit)

    async def edge_similarity_search(self, driver: Any, search_vector: list[float],
                                     source_node_uuid: str | None = None,
                                     target_node_uuid: str | None = None,
                                     search_filter: Any = None,
                                     group_ids: list[str] | None = None,
                                     limit: int = 100,
                                     min_score: float = 0.7) -> list[Any]:
        return await PGSearchOperations().edge_similarity_search(
            driver, search_vector, source_node_uuid, target_node_uuid,
            search_filter, group_ids, limit, min_score)

    async def node_fulltext_search(self, driver: Any, query: str, search_filter: Any,
                                   group_ids: list[str] | None = None,
                                   limit: int = 100) -> list[Any]:
        return await PGSearchOperations().node_fulltext_search(driver, query, search_filter,
                                                      group_ids, limit)

    async def node_similarity_search(self, driver: Any, search_vector: list[float],
                                     search_filter: Any = None,
                                     group_ids: list[str] | None = None,
                                     limit: int = 100,
                                     min_score: float = 0.7) -> list[Any]:
        return await PGSearchOperations().node_similarity_search(
            driver, search_vector, search_filter, group_ids, limit, min_score)

    async def episode_fulltext_search(self, driver: Any, query: str, search_filter: Any,
                                      group_ids: list[str] | None = None,
                                      limit: int = 100) -> list[Any]:
        return await PGSearchOperations().episode_fulltext_search(
            driver, query, search_filter, group_ids, limit)

    async def edge_bfs_search(self, driver: Any, bfs_origin_node_uuids: list[str] | None,
                              bfs_max_depth: int, search_filter: Any,
                              limit: int = 100) -> list[Any]:
        return await PGSearchOperations().edge_bfs_search(
            driver, bfs_origin_node_uuids or [], bfs_max_depth, search_filter,
            None, limit)

    async def node_bfs_search(self, driver: Any, bfs_origin_node_uuids: list[str] | None,
                              search_filter: Any, bfs_max_depth: int,
                              limit: int = 100) -> list[Any]:
        return await PGSearchOperations().node_bfs_search(
            driver, bfs_origin_node_uuids or [], search_filter, bfs_max_depth,
            None, limit)

    async def community_fulltext_search(self, driver: Any, query: str,
                                        group_ids: list[str] | None = None,
                                        limit: int = 100) -> list[Any]:
        return await PGSearchOperations().community_fulltext_search(driver, query, group_ids, limit)

    async def community_similarity_search(self, driver: Any, search_vector: list[float],
                                          group_ids: list[str] | None = None,
                                          limit: int = 100,
                                          min_score: float = 0.7) -> list[Any]:
        return await PGSearchOperations().community_similarity_search(
            driver, search_vector, group_ids, limit, min_score)
