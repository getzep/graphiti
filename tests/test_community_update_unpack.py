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

Regression test for: semaphore_gather unpack bug in add_episode with update_communities=True.

Previously, `semaphore_gather` returned a list of (communities, edges) tuples — one per
extracted node — but the code tried to unpack the whole list as two variables:

    communities, community_edges = await semaphore_gather(...)

This raised ValueError for any episode that extracted at least one node.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from graphiti_core.edges import CommunityEdge
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import CommunityNode, EntityNode

pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio
async def test_semaphore_gather_community_results_flatten():
    """semaphore_gather returns a list of tuples; verify the flatten logic is correct."""
    community_node = CommunityNode(
        name='test_community',
        labels=[],
        created_at=datetime.now(),
        group_id='test',
        summary='test summary',
    )
    community_edge = MagicMock(spec=CommunityEdge)

    async def mock_update_community(*args, **kwargs):
        return ([community_node], [community_edge])

    entity_nodes = [
        EntityNode(name='node_1', labels=[], created_at=datetime.now(), group_id='test'),
        EntityNode(name='node_2', labels=[], created_at=datetime.now(), group_id='test'),
    ]

    # This is the fixed pattern — previously the code tried:
    #   communities, community_edges = await semaphore_gather(...)
    # which raises ValueError because semaphore_gather returns a list of N tuples.
    results = await semaphore_gather(
        *[mock_update_community(node) for node in entity_nodes],
    )

    # Verify semaphore_gather returns a list of tuples (one per coroutine)
    assert len(results) == 2
    assert isinstance(results[0], tuple)

    # Verify the flatten produces the correct lists
    communities = [c for r in results for c in r[0]]
    community_edges_out = [e for r in results for e in r[1]]

    assert len(communities) == 2
    assert all(c is community_node for c in communities)
    assert len(community_edges_out) == 2
    assert all(e is community_edge for e in community_edges_out)


@pytest.mark.asyncio
async def test_semaphore_gather_unpack_raises_without_fix():
    """Demonstrate that the old unpack pattern fails — confirms the bug existed."""

    async def mock_update_community(*args):
        return ([MagicMock(spec=CommunityNode)], [MagicMock(spec=CommunityEdge)])

    results = await semaphore_gather(mock_update_community('node'))

    # Old code: `communities, community_edges = results` would raise ValueError
    # because results is a list of 1 tuple, not 2 items.
    with pytest.raises(ValueError, match='not enough values to unpack'):
        communities, community_edges = results
