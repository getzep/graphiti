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

# Regression tests for add_episode with a caller-supplied uuid (issue #2019).
#
# Previously, passing a uuid caused add_episode to look the episode up with
# EpisodicNode.get_by_uuid, which raises NodeNotFoundError when the episode does
# not exist yet. That made it impossible to add a new episode with a chosen
# uuid. The fix treats a missing episode as a create, using the supplied uuid.

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.tracer import Tracer

pytest_plugins = ('pytest_asyncio',)


class FakeDriver(GraphDriver):
    provider = GraphProvider.NEO4J

    def __init__(self, database: str = 'test_group'):
        self._database = database

    def clone(self, database: str) -> 'FakeDriver':
        return FakeDriver(database=database)

    async def execute_query(self, cypher_query_: str, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def session(self, database: str | None = None):  # pragma: no cover
        raise NotImplementedError

    def close(self):  # pragma: no cover
        raise NotImplementedError

    def delete_all_indexes(self):  # pragma: no cover
        raise NotImplementedError

    async def build_indices_and_constraints(
        self, delete_existing: bool = False
    ):  # pragma: no cover
        raise NotImplementedError


def _make_graphiti() -> Graphiti:
    driver = FakeDriver()
    clients = GraphitiClients(  # type: ignore[call-arg]
        driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
        tracer=Mock(spec=Tracer),
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = clients
    graphiti.store_raw_episode_content = True
    graphiti.max_coroutines = 1
    graphiti.tracer = Mock(spec=Tracer)
    graphiti.tracer.start_span.return_value.__enter__ = Mock(return_value=Mock())
    graphiti.tracer.start_span.return_value.__exit__ = Mock(return_value=False)
    return graphiti


async def _run_add_episode(graphiti: Graphiti, uuid: str | None):
    """Run add_episode with the extraction pipeline stubbed out."""
    captured: dict[str, EpisodicNode] = {}

    async def fake_process_episode_data(episode, *args, **kwargs):
        captured['episode'] = episode
        return [], episode

    with (
        patch(
            'graphiti_core.graphiti.retrieve_episodes',
            new=AsyncMock(return_value=[]),
        ),
        patch(
            'graphiti_core.graphiti.extract_nodes',
            new=AsyncMock(return_value=([], {})),
        ),
        patch(
            'graphiti_core.graphiti.resolve_extracted_nodes',
            new=AsyncMock(return_value=([], {}, [])),
        ),
        patch.object(
            graphiti,
            '_extract_and_resolve_edges',
            new=AsyncMock(return_value=([], [], [])),
        ),
        patch(
            'graphiti_core.graphiti.extract_attributes_from_nodes',
            new=AsyncMock(return_value=[]),
        ),
        patch.object(
            graphiti,
            '_process_episode_data',
            new=fake_process_episode_data,
        ),
    ):
        result = await graphiti.add_episode(
            name='test episode',
            episode_body='test body',
            source_description='test source',
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.text,
            group_id='test_group',
            uuid=uuid,
        )

    return result, captured['episode']


@pytest.mark.asyncio
async def test_add_episode_creates_episode_with_custom_uuid_when_missing():
    graphiti = _make_graphiti()
    custom_uuid = '0d39ada6-ba77-5f1d-be0d-45c02b248357'

    with patch.object(
        EpisodicNode,
        'get_by_uuid',
        new=AsyncMock(side_effect=NodeNotFoundError(custom_uuid)),
    ):
        result, episode = await _run_add_episode(graphiti, custom_uuid)

    assert episode.uuid == custom_uuid
    assert result.episode.uuid == custom_uuid


@pytest.mark.asyncio
async def test_add_episode_reuses_existing_episode_for_custom_uuid():
    graphiti = _make_graphiti()
    custom_uuid = '0d39ada6-ba77-5f1d-be0d-45c02b248357'
    existing = EpisodicNode(
        uuid=custom_uuid,
        name='existing',
        group_id='test_group',
        labels=[],
        source=EpisodeType.text,
        content='existing body',
        source_description='existing source',
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )

    with patch.object(
        EpisodicNode,
        'get_by_uuid',
        new=AsyncMock(return_value=existing),
    ):
        result, episode = await _run_add_episode(graphiti, custom_uuid)

    assert episode is existing
    assert result.episode.uuid == custom_uuid


@pytest.mark.asyncio
async def test_add_episode_generates_uuid_when_none_supplied():
    graphiti = _make_graphiti()

    result, episode = await _run_add_episode(graphiti, None)

    assert episode.uuid
    assert result.episode.uuid == episode.uuid
