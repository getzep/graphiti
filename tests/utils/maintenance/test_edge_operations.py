from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def mock_extracted_edge():
    return EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='Test fact',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )


@pytest.fixture
def mock_related_edges():
    return [
        EntityEdge(
            source_node_uuid='source_uuid_2',
            target_node_uuid='target_uuid_2',
            name='related_edge',
            group_id='group_1',
            fact='Related fact',
            episodes=['episode_2'],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            valid_at=datetime.now(timezone.utc) - timedelta(days=1),
            invalid_at=None,
        )
    ]


@pytest.fixture
def mock_existing_edges():
    return [
        EntityEdge(
            source_node_uuid='source_uuid_3',
            target_node_uuid='target_uuid_3',
            name='existing_edge',
            group_id='group_1',
            fact='Existing fact',
            episodes=['episode_3'],
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
            valid_at=datetime.now(timezone.utc) - timedelta(days=2),
            invalid_at=None,
        )
    ]


@pytest.fixture
def mock_current_episode():
    return EpisodicNode(
        uuid='episode_1',
        content='Current episode content',
        valid_at=datetime.now(timezone.utc),
        name='Current Episode',
        group_id='group_1',
        source='message',
        source_description='Test source description',
    )


@pytest.fixture
def mock_previous_episodes():
    return [
        EpisodicNode(
            uuid='episode_2',
            content='Previous episode content',
            valid_at=datetime.now(timezone.utc) - timedelta(days=1),
            name='Previous Episode',
            group_id='group_1',
            source='message',
            source_description='Test source description',
        )
    ]


@pytest.mark.asyncio
async def test_resolve_extracted_edge_no_changes(
    mock_llm_client,
    mock_extracted_edge,
    mock_related_edges,
    mock_existing_edges,
    mock_current_episode,
    mock_previous_episodes,
    monkeypatch: MonkeyPatch,
):
    # Mock the function calls
    dedupe_mock = AsyncMock(return_value=mock_extracted_edge)
    get_contradictions_mock = AsyncMock(return_value=[])

    # Patch the function calls
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.edge_operations.dedupe_extracted_edge', dedupe_mock
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.edge_operations.get_edge_contradictions',
        get_contradictions_mock,
    )

    resolved_edge, invalidated_edges = await resolve_extracted_edge(
        mock_llm_client,
        mock_extracted_edge,
        mock_related_edges,
        mock_existing_edges,
    )

    assert resolved_edge.uuid == mock_extracted_edge.uuid
    assert invalidated_edges == []
    dedupe_mock.assert_called_once()
    get_contradictions_mock.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_extracted_edge_with_invalidation(
    mock_llm_client,
    mock_extracted_edge,
    mock_related_edges,
    mock_existing_edges,
    mock_current_episode,
    mock_previous_episodes,
    monkeypatch: MonkeyPatch,
):
    valid_at = datetime.now(timezone.utc) - timedelta(days=1)
    mock_extracted_edge.valid_at = valid_at

    invalidation_candidate = EntityEdge(
        source_node_uuid='source_uuid_4',
        target_node_uuid='target_uuid_4',
        name='invalidation_candidate',
        group_id='group_1',
        fact='Invalidation candidate fact',
        episodes=['episode_4'],
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc) - timedelta(days=2),
        invalid_at=None,
    )

    # Mock the function calls
    dedupe_mock = AsyncMock(return_value=mock_extracted_edge)
    get_contradictions_mock = AsyncMock(return_value=[invalidation_candidate])

    # Patch the function calls
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.edge_operations.dedupe_extracted_edge', dedupe_mock
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.edge_operations.get_edge_contradictions',
        get_contradictions_mock,
    )

    resolved_edge, invalidated_edges = await resolve_extracted_edge(
        mock_llm_client,
        mock_extracted_edge,
        mock_related_edges,
        mock_existing_edges,
    )

    assert resolved_edge.uuid == mock_extracted_edge.uuid
    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == invalidation_candidate.uuid
    assert invalidated_edges[0].invalid_at == valid_at
    assert invalidated_edges[0].expired_at is not None


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])
