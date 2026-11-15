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

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError, EdgesNotFoundError, NodeNotFoundError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search import search
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.edge_operations import _pair_search_filter, _same_pair_edges
from graphiti_core.utils.maintenance.node_operations import _collect_candidate_nodes


class GraphReader:
    """The graph reads of the commit and of the enrichment.

    The commit reads the graph only through this class, so a test can replace the
    class with an in-memory object. Each search uses the stored embedding of the
    record and does not call the embedder.
    """

    def __init__(self, clients: GraphitiClients):
        self.clients = clients

    async def node_candidates(
        self,
        nodes: list[EntityNode],
        identity_properties: dict[str, list[str]] | None = None,
    ) -> list[list[EntityNode]]:
        """Return the dedupe candidates of each node, with the searches of serial ingestion."""
        if not nodes:
            return []
        return await _collect_candidate_nodes(
            self.clients,
            nodes,
            identity_properties=identity_properties,
            reuse_name_embeddings=True,
        )

    async def edge_duplicate_candidates(self, edges: list[EntityEdge]) -> list[list[EntityEdge]]:
        """Return the stored edges between the endpoints of each edge, in either direction."""
        results = await semaphore_gather(
            *[
                search(
                    self.clients,
                    edge.fact,
                    group_ids=[edge.group_id],
                    config=EDGE_HYBRID_SEARCH_RRF,
                    search_filter=_pair_search_filter(edge),
                    query_vector=edge.fact_embedding,
                )
                for edge in edges
            ],
            max_coroutines=getattr(self.clients, 'max_coroutines', None),
        )
        return [
            _same_pair_edges(edge, result.edges)
            for edge, result in zip(edges, results, strict=True)
        ]

    async def edge_invalidation_candidates(self, edges: list[EntityEdge]) -> list[list[EntityEdge]]:
        """Return the graph-wide contradiction candidates of each edge, with no endpoint filter."""
        results = await semaphore_gather(
            *[
                search(
                    self.clients,
                    edge.fact,
                    group_ids=[edge.group_id],
                    config=EDGE_HYBRID_SEARCH_RRF,
                    search_filter=SearchFilters(),
                    query_vector=edge.fact_embedding,
                )
                for edge in edges
            ],
            max_coroutines=getattr(self.clients, 'max_coroutines', None),
        )
        return [list(result.edges) for result in results]

    async def nodes_by_uuids(self, uuids: list[str], group_id: str) -> list[EntityNode]:
        """Return the current state of the nodes that exist."""
        if not uuids:
            return []
        try:
            return await EntityNode.get_by_uuids(self.clients.driver, uuids, group_id=group_id)
        except NodeNotFoundError:
            return []

    async def episodes_by_uuids(self, uuids: list[str]) -> list[EpisodicNode]:
        """Return the current state of the episode nodes that exist."""
        if not uuids:
            return []
        return await EpisodicNode.get_by_uuids(self.clients.driver, uuids)

    async def edges_by_uuids(self, uuids: list[str]) -> list[EntityEdge]:
        """Return the current state of the edges that exist."""
        if not uuids:
            return []
        try:
            return await EntityEdge.get_by_uuids(self.clients.driver, uuids)
        except (EdgeNotFoundError, EdgesNotFoundError):
            return []
