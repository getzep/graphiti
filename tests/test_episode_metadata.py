from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.record_parsers import episodic_node_from_record
from graphiti_core.nodes import EpisodeType, EpisodicNode


def _record(metadata):
    return {
        'content': 'content',
        'created_at': datetime(2025, 1, 1, tzinfo=timezone.utc),
        'valid_at': datetime(2025, 1, 1, tzinfo=timezone.utc),
        'uuid': 'episode-1',
        'group_id': 'group',
        'source': 'text',
        'name': 'Episode',
        'source_description': 'test',
        'entity_edges': [],
        'episode_metadata': metadata,
    }


def test_episodic_node_record_restores_metadata():
    metadata = {'customer_id': 'customer-1', 'priority': 3}

    episode = episodic_node_from_record(_record(metadata))

    assert episode.episode_metadata == metadata


@pytest.mark.asyncio
async def test_episodic_node_save_passes_metadata():
    driver = MagicMock(provider=GraphProvider.NEO4J, graph_operations_interface=None)
    driver.execute_query = AsyncMock(return_value=([], [], None))
    episode = EpisodicNode(
        name='Episode',
        group_id='group',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        episode_metadata={'customer_id': 'customer-1'},
    )

    await episode.save(driver)

    kwargs = driver.execute_query.await_args.kwargs
    assert kwargs['episode_data']['customer_id'] == 'customer-1'

