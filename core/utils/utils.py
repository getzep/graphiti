import logging

from core.edges import EpisodicEdge
from core.nodes import EntityNode, EpisodicNode

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
