from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'limit,min_score,expected_uuids,expected_scores',
    [
        (4, 0, ['center-1', 'center-2', 'neighbor', 'unconnected'], [10.0, 10.0, 1.0, 0.0]),
        (2, 0, ['center-1', 'center-2'], [10.0, 10.0]),
        (4, 0.001, ['center-1', 'center-2', 'neighbor'], [10.0, 10.0, 1.0]),
    ],
)
async def test_node_distance_scores_follow_each_edge(
    monkeypatch, limit, min_score, expected_uuids, expected_scores
):
    """Facts sharing a source inherit its score, including after filtering and limiting."""
    candidates = [
        EntityEdge(
            uuid=uuid,
            source_node_uuid=source,
            target_node_uuid=f'target-{uuid}',
            name='RELATES_TO',
            group_id='group',
            fact=f'{source} relates to {uuid}',
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for uuid, source in [
            ('neighbor', 'neighbor-source'),
            ('center-1', 'center-source'),
            ('center-2', 'center-source'),
            ('unconnected', 'unconnected-source'),
        ]
    ]
    fulltext_search = AsyncMock(return_value=candidates)
    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', fulltext_search)
    driver = AsyncMock(spec=GraphDriver)
    driver.provider = GraphProvider.NEO4J
    driver.search_interface = None
    # Exercise the real distance reranker with one adjacent source and one with no path.
    driver.execute_query = AsyncMock(
        return_value=([{'uuid': 'neighbor-source', 'score': 1}], None, None)
    )

    edges, scores = await edge_search(
        driver=driver,
        cross_encoder=AsyncMock(spec=CrossEncoderClient),
        query='related facts',
        query_vector=[],
        group_ids=['group'],
        config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25],
            reranker=EdgeReranker.node_distance,
        ),
        search_filter=SearchFilters(),
        center_node_uuid='center-source',
        limit=limit,
        reranker_min_score=min_score,
    )

    assert [edge.uuid for edge in edges] == expected_uuids
    assert scores == expected_scores
