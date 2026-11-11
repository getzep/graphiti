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

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from tests.helpers_test import group_id

pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def empty_extraction_llm_client():
    """LLM mock that extracts nothing, so add_episode's real pipeline runs LLM-free."""
    llm = Mock(spec=LLMClient)
    llm.generate_response = AsyncMock(
        return_value={'extracted_entities': [], 'edges': [], 'entity_resolutions': []}
    )
    return llm


@pytest.fixture
def null_embedder():
    embedder = Mock(spec=EmbedderClient)
    embedder.create = AsyncMock(return_value=[0.0] * 1024)
    embedder.create_batch = AsyncMock(return_value=[])
    return embedder


@pytest.fixture
def null_cross_encoder():
    return Mock(spec=CrossEncoderClient)


def make_graphiti(graph_driver, llm, embedder, cross_encoder) -> Graphiti:
    return Graphiti(
        graph_driver=graph_driver,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


@pytest.mark.asyncio
async def test_add_episode_with_fresh_uuid_creates_episode(
    graph_driver, empty_extraction_llm_client, null_embedder, null_cross_encoder
):
    """A client-supplied uuid that does not exist yet must CREATE the episode
    with exactly that uuid instead of raising NodeNotFoundError."""
    graphiti = make_graphiti(
        graph_driver, empty_extraction_llm_client, null_embedder, null_cross_encoder
    )
    supplied_uuid = str(uuid4())

    result = await graphiti.add_episode(
        name='fresh-uuid episode',
        episode_body='episode created with a client-supplied uuid',
        source_description='test',
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id=group_id,
        uuid=supplied_uuid,
    )

    assert result.episode.uuid == supplied_uuid
    stored = await EpisodicNode.get_by_uuid(graphiti.driver, supplied_uuid)
    assert stored.uuid == supplied_uuid
    assert stored.name == 'fresh-uuid episode'
    assert stored.content == 'episode created with a client-supplied uuid'
    assert stored.group_id == group_id


@pytest.mark.asyncio
async def test_add_episode_with_existing_uuid_reprocesses_in_place(
    graph_driver, empty_extraction_llm_client, null_embedder, null_cross_encoder
):
    """Upstream semantics preserved: an existing uuid loads the stored episode
    (its stored content wins) and does not create a duplicate node."""
    graphiti = make_graphiti(
        graph_driver, empty_extraction_llm_client, null_embedder, null_cross_encoder
    )
    now = datetime.now(timezone.utc)
    stored = await graphiti.add_episode(
        name='stored episode',
        episode_body='stored content',
        source_description='test',
        reference_time=now,
        source=EpisodeType.text,
        group_id=group_id,
    )
    existing_uuid = stored.episode.uuid

    result = await graphiti.add_episode(
        name='ignored name',
        episode_body='ignored body',
        source_description='test',
        reference_time=now,
        source=EpisodeType.text,
        group_id=group_id,
        uuid=existing_uuid,
    )

    assert result.episode.uuid == existing_uuid
    assert result.episode.content == 'stored content'
    records, _, _ = await graphiti.driver.execute_query(
        'MATCH (e:Episodic {uuid: $uuid}) RETURN count(e) AS n',
        uuid=existing_uuid,
    )
    assert records[0]['n'] == 1
