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

from typing import Any

from pydantic import BaseModel


class SearchInterface(BaseModel):
    """
    Interface for implementing custom search logic.

    All methods use `Any` type hints to avoid circular imports. See docstrings
    for expected concrete types.

    Type reference:
        - driver: GraphDriver
        - search_filter: SearchFilters
        - EntityNode, EpisodicNode, CommunityNode from graphiti_core.nodes
        - EntityEdge from graphiti_core.edges
    """

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform fulltext search over edge facts and names.

        Args:
            driver: GraphDriver instance
            query: Search query string
            search_filter: SearchFilters instance for filtering results
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            list[EntityEdge]: List of matching EntityEdge objects
        """
        raise NotImplementedError

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        """
        Perform vector similarity search over edge fact embeddings.

        Args:
            driver: GraphDriver instance
            search_vector: Query embedding vector
            source_node_uuid: Optional source node UUID to filter by
            target_node_uuid: Optional target node UUID to filter by
            search_filter: SearchFilters instance for filtering results
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)

        Returns:
            list[EntityEdge]: List of matching EntityEdge objects
        """
        raise NotImplementedError

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform fulltext search over node names and summaries.

        Args:
            driver: GraphDriver instance
            query: Search query string
            search_filter: SearchFilters instance for filtering results
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            list[EntityNode]: List of matching EntityNode objects
        """
        raise NotImplementedError

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        """
        Perform vector similarity search over node name embeddings.

        Args:
            driver: GraphDriver instance
            search_vector: Query embedding vector
            search_filter: SearchFilters instance for filtering results
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)

        Returns:
            list[EntityNode]: List of matching EntityNode objects
        """
        raise NotImplementedError

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform fulltext search over episode content.

        Args:
            driver: GraphDriver instance
            query: Search query string
            search_filter: SearchFilters instance (kept for interface parity)
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            list[EpisodicNode]: List of matching EpisodicNode objects
        """
        raise NotImplementedError

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform breadth-first search for edges starting from origin nodes.

        Args:
            driver: GraphDriver instance
            bfs_origin_node_uuids: List of starting node UUIDs (Entity or Episodic).
                Returns empty list if None or empty.
            bfs_max_depth: Maximum traversal depth (must be >= 1)
            search_filter: SearchFilters instance for filtering results
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            list[EntityEdge]: List of EntityEdge objects found within the search depth
        """
        raise NotImplementedError

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform breadth-first search for nodes starting from origin nodes.

        Args:
            driver: GraphDriver instance
            bfs_origin_node_uuids: List of starting node UUIDs (Entity or Episodic).
                Returns empty list if None or empty.
            search_filter: SearchFilters instance for filtering results
            bfs_max_depth: Maximum traversal depth (must be >= 1, returns empty if < 1)
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            list[EntityNode]: List of EntityNode objects found within the search depth
        """
        raise NotImplementedError

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Perform fulltext search over community names.

        Args:
            driver: GraphDriver instance
            query: Search query string
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return

        Returns:
            list[CommunityNode]: List of matching CommunityNode objects
        """
        raise NotImplementedError

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        """
        Perform vector similarity search over community name embeddings.

        Args:
            driver: GraphDriver instance
            search_vector: Query embedding vector
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)

        Returns:
            list[CommunityNode]: List of matching CommunityNode objects
        """
        raise NotImplementedError

    async def get_embeddings_for_communities(
        self,
        driver: Any,
        communities: list[Any],
    ) -> dict[str, list[float]]:
        """
        Load name embeddings for a list of community nodes.

        Args:
            driver: GraphDriver instance
            communities: List of CommunityNode objects to load embeddings for

        Returns:
            dict[str, list[float]]: Mapping of community UUID to name embedding vector
        """
        raise NotImplementedError

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """
        Rerank nodes by their graph distance to a center node.

        Nodes directly connected to the center node get score 1.0, the center node
        itself gets score 0.1 (if in the input list), and unconnected nodes get
        score approaching 0 (1/infinity).

        Args:
            driver: GraphDriver instance
            node_uuids: List of node UUIDs to rerank. The center_node_uuid will be
                filtered out during processing but included in results if present.
            center_node_uuid: UUID of the center node to measure distances from
            min_score: Minimum score threshold. Nodes with 1/distance < min_score
                are excluded from results.

        Returns:
            tuple[list[str], list[float]]: Tuple of (sorted_uuids, scores) where
                scores are 1/distance values, sorted by distance ascending
        """
        raise NotImplementedError

    async def episode_mentions_reranker(
        self,
        driver: Any,
        node_uuids: list[list[str]],
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """
        Rerank nodes by their episode mention count.

        Uses RRF (Reciprocal Rank Fusion) as a preliminary ranker, then reranks
        by the number of episodes that mention each node.

        Args:
            driver: GraphDriver instance
            node_uuids: List of ranked UUID lists (e.g., from multiple search results)
                to be merged and reranked
            min_score: Minimum mention count threshold. Nodes with fewer mentions
                are excluded from results.

        Returns:
            tuple[list[str], list[float]]: Tuple of (sorted_uuids, mention_counts)
                sorted by mention count descending
        """
        raise NotImplementedError

    # ---------- SEARCH FILTERS (sync) ----------
    def build_node_search_filters(self, search_filters: Any) -> Any:
        """
        Build provider-specific node search filters.

        Args:
            search_filters: SearchFilters instance

        Returns:
            Provider-specific filter representation
        """
        raise NotImplementedError

    def build_edge_search_filters(self, search_filters: Any) -> Any:
        """
        Build provider-specific edge search filters.

        Args:
            search_filters: SearchFilters instance

        Returns:
            Provider-specific filter representation
        """
        raise NotImplementedError

    class Config:
        arbitrary_types_allowed = True
