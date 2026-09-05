from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.utils.maintenance.community_operations import get_community_clusters

pytestmark = pytest.mark.asyncio


def _entity_record(uuid: str, group_id: str) -> dict:
    return {
        'uuid': uuid,
        'name': uuid,
        'group_id': group_id,
        'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
        'summary': f'Summary for {uuid}',
        'labels': ['Entity'],
        'attributes': {},
    }


def _mock_driver(node_records: list[dict], projection_records: list[dict]) -> GraphDriver:
    async def execute_query(query: str, **kwargs):
        if 'WHERE n.group_id IN $group_ids' in query:
            return node_records, None, None
        if 'source_uuid' in query:
            return projection_records, None, None
        raise AssertionError(f'Unexpected per-node or cluster query: {query}')

    driver = MagicMock(spec=GraphDriver)
    driver.provider = GraphProvider.NEO4J
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(side_effect=execute_query)
    return driver


async def test_get_community_clusters_fetches_large_projection_in_two_queries():
    group_id = 'large-group'
    node_records = [_entity_record(f'node-{index}', group_id) for index in range(1_000)]
    driver = _mock_driver(node_records, [])

    clusters = await get_community_clusters(driver, [group_id])

    assert len(clusters) == 1_000
    assert all(len(cluster) == 1 for cluster in clusters)
    assert driver.execute_query.await_count == 2


async def test_get_community_clusters_builds_clusters_from_batched_projection():
    group_id = 'connected-group'
    node_records = [_entity_record(node_uuid, group_id) for node_uuid in ('a', 'b', 'c', 'd')]
    projection_records = [
        {'source_uuid': 'a', 'target_uuid': 'b', 'edge_count': 1},
        {'source_uuid': 'b', 'target_uuid': 'a', 'edge_count': 1},
        {'source_uuid': 'c', 'target_uuid': 'd', 'edge_count': 1},
        {'source_uuid': 'd', 'target_uuid': 'c', 'edge_count': 1},
    ]
    driver = _mock_driver(node_records, projection_records)

    clusters = await get_community_clusters(driver, [group_id])

    assert {frozenset(node.uuid for node in cluster) for cluster in clusters} == {
        frozenset({'a', 'b'}),
        frozenset({'c', 'd'}),
    }
    assert driver.execute_query.await_count == 2
