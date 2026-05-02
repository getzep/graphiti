from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti import Graphiti
from graphiti_core.utils.maintenance import community_operations as community_ops


@pytest.mark.asyncio
async def test_build_community_passes_max_coroutines_to_summary_gather(monkeypatch):
    gather_limits: list[int | None] = []

    async def immediate_gather(*aws, max_coroutines=None):
        gather_limits.append(max_coroutines)
        return [await aw for aw in aws]

    monkeypatch.setattr(community_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(community_ops, 'summarize_pair', AsyncMock(side_effect=['ab', 'cd', 'abcd']))
    monkeypatch.setattr(
        community_ops, 'generate_summary_description', AsyncMock(return_value='community')
    )
    monkeypatch.setattr(community_ops, 'build_community_edges', MagicMock(return_value=[]))

    cluster = [
        SimpleNamespace(summary='a', group_id='group-1'),
        SimpleNamespace(summary='b', group_id='group-1'),
        SimpleNamespace(summary='c', group_id='group-1'),
        SimpleNamespace(summary='d', group_id='group-1'),
    ]

    await community_ops.build_community(
        MagicMock(),
        cluster,
        max_coroutines=3,
    )

    assert gather_limits == [3, 3]


@pytest.mark.asyncio
async def test_build_communities_passes_max_coroutines_to_nested_calls(monkeypatch):
    received_limits: list[int | None] = []

    async def fake_get_community_clusters(driver, group_ids):
        return [[SimpleNamespace(group_id='group-1')], [SimpleNamespace(group_id='group-1')]]

    async def fake_build_community(llm_client, cluster, max_coroutines=None):
        received_limits.append(max_coroutines)
        return (f'community-{len(received_limits)}', [f'edge-{len(received_limits)}'])

    async def immediate_gather(*aws, max_coroutines=None):
        received_limits.append(max_coroutines)
        return [await aw for aw in aws]

    monkeypatch.setattr(community_ops, 'get_community_clusters', fake_get_community_clusters)
    monkeypatch.setattr(community_ops, 'build_community', fake_build_community)
    monkeypatch.setattr(community_ops, 'semaphore_gather', immediate_gather)

    community_nodes, community_edges = await community_ops.build_communities(
        MagicMock(),
        MagicMock(),
        None,
        max_coroutines=3,
    )

    assert received_limits[0] == 3
    assert received_limits[1:] == [3, 3]
    assert community_nodes == ['community-2', 'community-3']
    assert community_edges == ['edge-2', 'edge-3']


@pytest.mark.asyncio
async def test_graphiti_build_communities_passes_instance_max_coroutines(monkeypatch):
    captured: dict[str, int | None] = {}

    async def fake_remove_communities(driver):
        return None

    async def fake_build_communities(driver, llm_client, group_ids, max_coroutines=None):
        captured['max_coroutines'] = max_coroutines
        return ([], [])

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr('graphiti_core.graphiti.remove_communities', fake_remove_communities)
    monkeypatch.setattr('graphiti_core.graphiti.build_communities', fake_build_communities)
    monkeypatch.setattr('graphiti_core.graphiti.semaphore_gather', immediate_gather)
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients', lambda **kwargs: SimpleNamespace(**kwargs)
    )

    graphiti = Graphiti(
        graph_driver=MagicMock(),
        llm_client=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        max_coroutines=2,
    )

    await graphiti.build_communities()

    assert captured['max_coroutines'] == 2
