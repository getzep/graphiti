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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.community_operations import (
    build_community,
    get_community_clusters,
)


def _mock_driver():
    driver = MagicMock()
    driver.graph_operations_interface = None
    driver.graph_ops = None
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


def _entity(name: str) -> EntityNode:
    return EntityNode(name=name, group_id='g1', labels=['Entity'], summary=f'{name} summary')


@pytest.mark.asyncio
async def test_get_community_clusters_skips_empty_clusters():
    """An unresolvable cluster must not reach build_community.

    ``EntityNode.get_by_uuids`` returns only the nodes it finds and does not
    error on the rest, so a node deleted or filtered out between label
    propagation and this fetch yields an empty cluster. The driver-specific
    implementations already skip those (see
    ``driver/neo4j/operations/graph_ops.py``); this fallback path did not, and
    ``build_community`` raises ``IndexError`` on ``summaries[0]`` for one.
    """
    driver = _mock_driver()

    with (
        patch.object(
            EntityNode, 'get_by_group_ids', new=AsyncMock(return_value=[_entity('alice')])
        ),
        patch.object(
            EntityNode,
            'get_by_uuids',
            new=AsyncMock(side_effect=[[], [_entity('bob')]]),
        ),
        patch(
            'graphiti_core.utils.maintenance.community_operations.label_propagation',
            return_value=[['missing-uuid'], ['bob-uuid']],
        ),
    ):
        clusters = await get_community_clusters(driver, ['g1'])

    assert all(cluster for cluster in clusters), (
        f'an empty cluster survived into the result: {clusters}'
    )
    assert len(clusters) == 1
    assert clusters[0][0].name == 'bob'


@pytest.mark.asyncio
async def test_get_community_clusters_skips_empty_uuid_lists():
    """An empty uuid list is not even worth a fetch."""
    driver = _mock_driver()
    get_by_uuids = AsyncMock(return_value=[_entity('bob')])

    with (
        patch.object(
            EntityNode, 'get_by_group_ids', new=AsyncMock(return_value=[_entity('alice')])
        ),
        patch.object(EntityNode, 'get_by_uuids', new=get_by_uuids),
        patch(
            'graphiti_core.utils.maintenance.community_operations.label_propagation',
            return_value=[[], ['bob-uuid']],
        ),
    ):
        clusters = await get_community_clusters(driver, ['g1'])

    assert get_by_uuids.await_count == 1, 'the empty uuid list should not be fetched'
    assert len(clusters) == 1


@pytest.mark.asyncio
async def test_get_community_clusters_keeps_healthy_clusters_intact():
    """The healthy path is unchanged: every resolvable cluster is returned."""
    driver = _mock_driver()

    with (
        patch.object(
            EntityNode, 'get_by_group_ids', new=AsyncMock(return_value=[_entity('alice')])
        ),
        patch.object(
            EntityNode,
            'get_by_uuids',
            new=AsyncMock(side_effect=[[_entity('a'), _entity('b')], [_entity('c')]]),
        ),
        patch(
            'graphiti_core.utils.maintenance.community_operations.label_propagation',
            return_value=[['a', 'b'], ['c']],
        ),
    ):
        clusters = await get_community_clusters(driver, ['g1'])

    assert [len(c) for c in clusters] == [2, 1]


@pytest.mark.asyncio
async def test_build_community_rejects_an_empty_cluster():
    """Called directly, build_community reports the contract it needs.

    ``semaphore_gather`` wraps ``asyncio.gather`` without
    ``return_exceptions``, so a bare IndexError here fails the entire
    ``build_communities`` batch and discards the communities already built
    alongside it. A named error beats an index crash for whoever reads the log.
    """
    with pytest.raises(ValueError, match='non-empty community_cluster'):
        await build_community(MagicMock(), [])
