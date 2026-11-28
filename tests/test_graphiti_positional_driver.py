"""Public retrieval calls must preserve positional arguments when routing graphs."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType


@pytest.mark.asyncio
@pytest.mark.parametrize('group_ids', [['customer'], ['customer', 'other'], ['default_db']])
@pytest.mark.parametrize('positional', [True, False], ids=['positional', 'keyword'])
@pytest.mark.parametrize('explicit_driver', [False, True], ids=['none-driver', 'driver'])
async def test_retrieve_episodes_preserves_driver_and_following_arguments(
    group_ids, positional, explicit_driver
):
    reference_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    databases = {}

    def make_driver(database):
        driver = Mock(spec=GraphDriver)
        driver.provider = GraphProvider.FALKORDB
        driver._database = database
        driver.graph_operations_interface = None
        driver.clone.side_effect = make_driver
        # Only the database response is supplied. Graphiti's public method,
        # argument routing, query construction and episode parsing run normally.
        driver.execute_query = AsyncMock(
            return_value=(
                [
                    {
                        'uuid': f'episode-{database}',
                        'name': 'fixture',
                        'group_id': database,
                        'source': 'text',
                        'source_description': 'synthetic fixture',
                        'content': 'Alice lives in Paris.',
                        'created_at': reference_time,
                        'valid_at': reference_time,
                        'entity_edges': [],
                    }
                ],
                None,
                None,
            )
        )
        databases[database] = driver
        return driver

    driver = make_driver('default_db')
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
    )
    supplied_driver = driver if explicit_driver else None
    if positional:
        episodes = await graphiti.retrieve_episodes(
            reference_time, 3, group_ids, EpisodeType.text, supplied_driver, 'conversation'
        )
    else:
        episodes = await graphiti.retrieve_episodes(
            reference_time,
            3,
            group_ids,
            EpisodeType.text,
            driver=supplied_driver,
            saga='conversation',
        )

    assert {episode.uuid for episode in episodes} == {f'episode-{gid}' for gid in group_ids}
    for gid in group_ids:
        database = databases[gid]
        database.execute_query.assert_awaited_once()
        parameters = database.execute_query.call_args.kwargs
        assert parameters['reference_time'] == reference_time
        assert parameters['num_episodes'] == 3
        assert parameters['group_id'] == gid
        assert parameters['source'] == 'text'
        assert parameters['saga_name'] == 'conversation'
    assert graphiti.driver is driver
    assert graphiti.clients.driver is driver
    assert driver._database == 'default_db'
