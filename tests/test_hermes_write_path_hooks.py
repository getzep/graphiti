from types import SimpleNamespace
from typing import Any, cast

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search import search_utils
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance import edge_operations, node_operations
from graphiti_core.write_path_hooks import (
    clear_hooks,
    register_hook,
    reset_write_path_context,
    set_write_path_context,
)


@pytest.fixture(autouse=True)
def clear_write_path_hooks():
    clear_hooks()
    yield
    clear_hooks()


class CandidateHook:
    async def collect_candidate_nodes(self, original, clients, extracted_nodes, existing_nodes_override, *, context):
        context.stats['candidate_hook_called'] = True
        return [[] for _ in extracted_nodes]


@pytest.mark.asyncio
async def test_candidate_node_hook_wraps_collect_candidate_nodes() -> None:
    token = set_write_path_context(stats={})
    try:
        register_hook('candidate_nodes', CandidateHook())
        node = EntityNode(name='Alice', group_id='g', labels=['Entity'])

        result = await node_operations._collect_candidate_nodes(cast(Any, SimpleNamespace()), [node], None)

        assert result == [[]]
    finally:
        reset_write_path_context(token)


class NodeResolutionHook:
    async def resolve_extracted_nodes(self, original, clients, extracted_nodes, *args, context, **kwargs):
        context.stats['node_resolution_hook_called'] = True
        return extracted_nodes, {node.uuid: node.uuid for node in extracted_nodes}, []


@pytest.mark.asyncio
async def test_node_resolution_hook_wraps_resolve_extracted_nodes() -> None:
    stats = {}
    token = set_write_path_context(stats=stats)
    try:
        register_hook('node_resolution', NodeResolutionHook())
        node = EntityNode(name='Alice', group_id='g', labels=['Entity'])

        nodes, uuid_map, duplicate_pairs = await node_operations.resolve_extracted_nodes(
            cast(Any, SimpleNamespace()), [node]
        )

        assert nodes == [node]
        assert uuid_map == {node.uuid: node.uuid}
        assert duplicate_pairs == []
        assert stats['node_resolution_hook_called'] is True
    finally:
        reset_write_path_context(token)


class EdgeResolutionHook:
    async def resolve_extracted_edge(self, original, llm_client, extracted_edge, *args, context, **kwargs):
        context.stats['edge_resolution_hook_called'] = True
        return extracted_edge, [], []


@pytest.mark.asyncio
async def test_edge_resolution_hook_wraps_resolve_extracted_edge() -> None:
    stats = {}
    token = set_write_path_context(stats=stats)
    try:
        register_hook('edge_resolution', EdgeResolutionHook())
        edge = EntityEdge(
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATES_TO',
            group_id='g',
            fact='Alice knows Bob',
            created_at=utc_now(),
            episodes=[],
        )

        resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
            cast(Any, SimpleNamespace()), edge, [], [], cast(Any, SimpleNamespace(valid_at=None))
        )

        assert resolved is edge
        assert invalidated == []
        assert duplicates == []
        assert stats['edge_resolution_hook_called'] is True
    finally:
        reset_write_path_context(token)


class EdgeSimilaritySearchHook:
    async def filter_edges(self, edges, *, context, record_scores=None, **kwargs):
        context.stats['edge_similarity_search_hook_called'] = True
        context.stats['edge_similarity_record_score'] = (record_scores or {}).get(edges[0].uuid)
        return edges[:1]


@pytest.mark.asyncio
async def test_edge_similarity_search_hook_filters_results() -> None:
    stats = {}
    token = set_write_path_context(stats=stats)
    try:
        register_hook('edge_similarity_search', EdgeSimilaritySearchHook())
        edge_a = EntityEdge(
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATES_TO',
            group_id='g',
            fact='Alice knows Bob',
            created_at=utc_now(),
            episodes=[],
        )
        edge_b = EntityEdge(
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATES_TO',
            group_id='g',
            fact='Alice dislikes Bob',
            created_at=utc_now(),
            episodes=[],
        )

        result = await search_utils._apply_edge_similarity_search_hook(
            [edge_a, edge_b],
            driver=cast(Any, SimpleNamespace()),
            search_vector=[1.0],
            group_ids=['g'],
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=cast(Any, SimpleNamespace()),
            record_scores={edge_a.uuid: 0.91, edge_b.uuid: 0.2},
        )

        assert result == [edge_a]
        assert stats['edge_similarity_search_hook_called'] is True
        assert stats['edge_similarity_record_score'] == 0.91
    finally:
        reset_write_path_context(token)
