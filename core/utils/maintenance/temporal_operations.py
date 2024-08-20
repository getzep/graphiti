from datetime import datetime
from typing import List
from core.llm_client import LLMClient
from core.edges import EntityEdge
from core.nodes import EntityNode
from core.prompts import prompt_library
import logging

logger = logging.getLogger(__name__)


class EdgeWithNodes:
    def __init__(
        self, edge: EntityEdge, source_node: EntityNode, target_node: EntityNode
    ):
        self.edge = edge
        self.source_node = source_node
        self.target_node = target_node


def prepare_edges_for_invalidation(
    existing_edges: list[EntityEdge],
    new_edges: list[EntityEdge],
    nodes: list[EntityNode],
) -> tuple[list[EdgeWithNodes], list[EdgeWithNodes]]:
    existing_edges_pending_invalidation = []
    new_edges_with_nodes = []

    for edge_list, result_list in [
        (existing_edges, existing_edges_pending_invalidation),
        (new_edges, new_edges_with_nodes),
    ]:
        for edge in edge_list:
            source_node = next(
                (node for node in nodes if node.uuid == edge.source_node_uuid), None
            )
            target_node = next(
                (node for node in nodes if node.uuid == edge.target_node_uuid), None
            )

            if source_node and target_node:
                result_list.append(
                    EdgeWithNodes(
                        edge=edge,
                        source_node=source_node,
                        target_node=target_node,
                    )
                )

    return existing_edges_pending_invalidation, new_edges_with_nodes


async def invalidate_edges(
    llm_client: LLMClient,
    existing_edges_pending_invalidation: List[EdgeWithNodes],
    new_edges: List[EdgeWithNodes],
) -> List[EntityEdge]:
    invalidated_edges = []

    context = prepare_invalidation_context(
        existing_edges_pending_invalidation, new_edges
    )
    llm_response = await llm_client.generate_response(
        prompt_library.invalidate_edges.v1(context)
    )

    edges_to_invalidate = llm_response.get("invalidated_edges", [])
    invalidated_edges = process_edge_invalidation_llm_response(
        edges_to_invalidate, existing_edges_pending_invalidation
    )

    return invalidated_edges


def prepare_invalidation_context(
    existing_edges: List[EdgeWithNodes], new_edges: List[EdgeWithNodes]
) -> dict:
    return {
        "existing_edges": [
            f"{edge_data.edge.uuid} | {edge_data.source_node.name} - {edge_data.edge.name} - {edge_data.target_node.name} ({edge_data.edge.created_at.isoformat()})"
            for edge_data in sorted(
                existing_edges, key=lambda x: x.edge.created_at, reverse=True
            )
        ],
        "new_edges": [
            f"{edge_data.edge.uuid} | {edge_data.source_node.name} - {edge_data.edge.name} - {edge_data.target_node.name} ({edge_data.edge.created_at.isoformat()})"
            for edge_data in sorted(
                new_edges, key=lambda x: x.edge.created_at, reverse=True
            )
        ],
    }


def process_edge_invalidation_llm_response(
    edges_to_invalidate: List[dict], existing_edges: List[EdgeWithNodes]
) -> List[EntityEdge]:
    invalidated_edges = []
    for edge_to_invalidate in edges_to_invalidate:
        edge_uuid = edge_to_invalidate["edge_uuid"]
        edge_to_update = next(
            (
                edge_data.edge
                for edge_data in existing_edges
                if edge_data.edge.uuid == edge_uuid
            ),
            None,
        )
        if edge_to_update:
            edge_to_update.expired_at = datetime.now()
            invalidated_edges.append(edge_to_update)
            logger.info(
                f"Invalidated edge: {edge_to_update.name} (UUID: {edge_to_update.uuid}). Reason: {edge_to_invalidate['reason']}"
            )
    return invalidated_edges
