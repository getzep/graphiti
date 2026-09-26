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

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from tests.helpers_test import get_edge_count, get_node_count, group_id

pytest_plugins = ('pytest_asyncio',)

EMBEDDING = [0.1] * 384


@pytest.fixture
def embedder():
    client = Mock(spec=EmbedderClient)
    client.create = AsyncMock(return_value=EMBEDDING)
    client.create_batch = AsyncMock(side_effect=lambda texts: [EMBEDDING for _ in texts])
    return client


@pytest.fixture
def llm_client():
    """Summarizes every entity in the prompt from the episode content it was given."""
    client = Mock(spec=LLMClient)

    async def generate_response(messages, response_model=None, **kwargs):
        prompt = '\n'.join(message.content for message in messages)
        summaries = []
        for name in ('Alice', 'Bob'):
            if f'"name": "{name}"' in prompt:
                sources = [text for text in ('first source', 'second source') if text in prompt]
                summaries.append({'name': name, 'summary': f'{name} from ' + ' and '.join(sources)})
        return {'summaries': summaries}

    client.generate_response = AsyncMock(side_effect=generate_response)
    return client


def _episode(name: str, content: str, valid_at: datetime) -> EpisodicNode:
    return EpisodicNode(
        name=name,
        group_id=group_id,
        labels=[],
        created_at=valid_at,
        source=EpisodeType.message,
        source_description='conversation message',
        content=content,
        valid_at=valid_at,
    )


def _entity(name: str, now: datetime, summary: str) -> EntityNode:
    return EntityNode(
        name=name,
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary=summary,
        name_embedding=EMBEDDING,
    )


def _mention(episode: EpisodicNode, node: EntityNode) -> EpisodicEdge:
    return EpisodicEdge(
        source_node_uuid=episode.uuid,
        target_node_uuid=node.uuid,
        created_at=episode.created_at,
        group_id=group_id,
    )


async def _shared_fact_graph(graph_driver, embedder):
    """Two episodes state the same fact; the entity summaries carry both episodes."""
    now = datetime.now()
    first = _episode('first', 'Alice likes Bob (first source)', now - timedelta(days=1))
    second = _episode('second', 'Alice likes Bob (second source)', now)
    alice = _entity('Alice', now, 'Alice from first source and second source')
    bob = _entity('Bob', now, 'Bob from first source and second source')
    fact = EntityEdge(
        source_node_uuid=alice.uuid,
        target_node_uuid=bob.uuid,
        created_at=now,
        name='LIKES',
        fact='Alice likes Bob',
        episodes=[first.uuid, second.uuid],
        valid_at=now - timedelta(days=1),
        group_id=group_id,
        fact_embedding=EMBEDDING,
    )
    first.entity_edges = [fact.uuid]
    second.entity_edges = [fact.uuid]
    await add_nodes_and_edges_bulk(
        graph_driver,
        [first, second],
        [
            _mention(first, alice),
            _mention(first, bob),
            _mention(second, alice),
            _mention(second, bob),
        ],
        [alice, bob],
        [fact],
        embedder,
    )
    return first, second, alice, bob, fact


def _graphiti(graph_driver, llm_client, embedder) -> Graphiti:
    return Graphiti(
        graph_driver=graph_driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=Mock(spec=CrossEncoderClient),
    )


@pytest.mark.asyncio
async def test_remove_first_episode_keeps_fact_supported_by_another_episode(
    graph_driver, llm_client, embedder
):
    graphiti = _graphiti(graph_driver, llm_client, embedder)
    await graphiti.build_indices_and_constraints()
    first, second, alice, bob, fact = await _shared_fact_graph(graph_driver, embedder)

    await graphiti.remove_episode(first.uuid)

    assert await get_node_count(graph_driver, [first.uuid]) == 0
    assert await get_node_count(graph_driver, [alice.uuid, bob.uuid]) == 2
    surviving = await EntityEdge.get_by_uuid(graph_driver, fact.uuid)
    assert surviving.episodes == [second.uuid]
    assert surviving.fact == 'Alice likes Bob'


@pytest.mark.asyncio
async def test_remove_later_episode_removes_it_from_the_surviving_fact(
    graph_driver, llm_client, embedder
):
    graphiti = _graphiti(graph_driver, llm_client, embedder)
    await graphiti.build_indices_and_constraints()
    first, second, _, _, fact = await _shared_fact_graph(graph_driver, embedder)

    await graphiti.remove_episode(second.uuid)

    surviving = await EntityEdge.get_by_uuid(graph_driver, fact.uuid)
    assert surviving.episodes == [first.uuid]


@pytest.mark.asyncio
async def test_remove_episode_rebuilds_summaries_from_remaining_episodes(
    graph_driver, llm_client, embedder
):
    graphiti = _graphiti(graph_driver, llm_client, embedder)
    await graphiti.build_indices_and_constraints()
    first, _, alice, bob, _ = await _shared_fact_graph(graph_driver, embedder)

    await graphiti.remove_episode(first.uuid)

    assert (await EntityNode.get_by_uuid(graph_driver, alice.uuid)).summary == (
        'Alice from second source'
    )
    assert (await EntityNode.get_by_uuid(graph_driver, bob.uuid)).summary == (
        'Bob from second source'
    )
    prompts = [
        '\n'.join(message.content for message in call.args[0])
        for call in llm_client.generate_response.call_args_list
    ]
    assert prompts
    assert all('first source' not in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_remove_last_supporting_episode_deletes_fact_and_entities(
    graph_driver, llm_client, embedder
):
    graphiti = _graphiti(graph_driver, llm_client, embedder)
    await graphiti.build_indices_and_constraints()
    first, second, alice, bob, fact = await _shared_fact_graph(graph_driver, embedder)

    await graphiti.remove_episode(first.uuid)
    await graphiti.remove_episode(second.uuid)

    assert await get_edge_count(graph_driver, [fact.uuid]) == 0
    assert await get_node_count(graph_driver, [first.uuid, second.uuid, alice.uuid, bob.uuid]) == 0
