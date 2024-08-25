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

from graphiti_core.edges import EpisodicEdge
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
