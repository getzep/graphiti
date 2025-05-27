from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode


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


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])
