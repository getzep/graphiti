import logging
from datetime import datetime
from typing import List

from core.edges import EntityEdge
from core.llm_client import LLMClient
from core.nodes import EntityNode
from core.prompts import prompt_library

logger = logging.getLogger(__name__)

NodeEdgeNodeTriplet = tuple[EntityNode, EntityEdge, EntityNode]


def prepare_edges_for_invalidation(
    existing_edges: list[EntityEdge],
    new_edges: list[EntityEdge],
    nodes: list[EntityNode],
) -> tuple[list[NodeEdgeNodeTriplet], list[NodeEdgeNodeTriplet]]:
    existing_edges_pending_invalidation = []  # TODO: this is not yet used?
    new_edges_with_nodes = []  # TODO: this is not yet used?

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
                result_list.append((source_node, edge, target_node))

    return existing_edges_pending_invalidation, new_edges_with_nodes


async def invalidate_edges(
    llm_client: LLMClient,
    existing_edges_pending_invalidation: List[NodeEdgeNodeTriplet],
    new_edges: List[NodeEdgeNodeTriplet],
) -> List[EntityEdge]:
    invalidated_edges = []  # TODO: this is not yet used?

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
    existing_edges: List[NodeEdgeNodeTriplet], new_edges: List[NodeEdgeNodeTriplet]
) -> dict:
    return {
        "existing_edges": [
            f"{edge.uuid} | {source_node.name} - {edge.name} - {target_node.name} ({edge.created_at.isoformat()})"
            for source_node, edge, target_node in sorted(
                existing_edges, key=lambda x: x[1].created_at, reverse=True
            )
        ],
        "new_edges": [
            f"{edge.uuid} | {source_node.name} - {edge.name} - {target_node.name} ({edge.created_at.isoformat()})"
            for source_node, edge, target_node in sorted(
                new_edges, key=lambda x: x[1].created_at, reverse=True
            )
        ],
    }


def process_edge_invalidation_llm_response(
    edges_to_invalidate: List[dict], existing_edges: List[NodeEdgeNodeTriplet]
) -> List[EntityEdge]:
    invalidated_edges = []
    for edge_to_invalidate in edges_to_invalidate:
        edge_uuid = edge_to_invalidate["edge_uuid"]
        edge_to_update = next(
            (edge for _, edge, _ in existing_edges if edge.uuid == edge_uuid),
            None,
        )
        if edge_to_update:
            edge_to_update.expired_at = datetime.now()
            invalidated_edges.append(edge_to_update)
            logger.info(
                f"Invalidated edge: {edge_to_update.name} (UUID: {edge_to_update.uuid}). Reason: {edge_to_invalidate['reason']}"
            )
    return invalidated_edges
