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

import logging
from collections import defaultdict

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

logger = logging.getLogger(__name__)


def build_episodic_edges(
    entity_nodes: list[EntityNode], episode: EpisodicNode
) -> list[EpisodicEdge]:
    edges: list[EpisodicEdge] = []

    for node in entity_nodes:
        edges.append(
            EpisodicEdge(
                source_node_uuid=episode.uuid,
                target_node_uuid=node.uuid,
                created_at=episode.created_at,
            )
        )

    return edges


def chunk_edges_by_nodes(edges: list[EntityEdge]) -> list[list[EntityEdge]]:
    # We only want to dedupe edges that are between the same pair of nodes
    # We build a map of the edges based on their source and target nodes.
    edge_chunk_map: dict[str, list[EntityEdge]] = defaultdict(list)
    for edge in edges:
        # We drop loop edges
        if edge.source_node_uuid == edge.target_node_uuid:
            continue

        # Keep the order of the two nodes consistent, we want to be direction agnostic during edge resolution
        pointers = [edge.source_node_uuid, edge.target_node_uuid]
        pointers.sort()

        edge_chunk_map[pointers[0] + pointers[1]].append(edge)

    edge_chunks = [chunk for chunk in edge_chunk_map.values()]

    return edge_chunks
