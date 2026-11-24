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

# add_episode_bulk resolves the edges of each episode against the stored graph. Two
# episodes can thus change two copies of one stored edge. These database-free tests check
# that add_episode_bulk saves one merged edge, not the last copy (issue #2210).

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import pytest

from graphiti_core import graphiti as graphiti_module
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.tracer import NoOpTracer
from graphiti_core.utils.bulk_utils import RawEpisode

DEC = datetime(2025, 12, 1, tzinfo=timezone.utc)
JAN = datetime(2026, 1, 10, tzinfo=timezone.utc)
MAR = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _make_graphiti() -> Graphiti:
    driver = Mock()
    driver.provider = GraphProvider.NEO4J
    driver._database = 'default_db'
    clients = GraphitiClients.model_construct(
        driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
        tracer=NoOpTracer(),
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = clients
    graphiti.tracer = NoOpTracer()
    graphiti.max_coroutines = None
    return graphiti


def _initech_copy(episodes: list[str], invalid_at: datetime | None = None) -> EntityEdge:
    """Return one in-memory copy of the stored fact that Alice works at Initech."""
    return EntityEdge(
        uuid='initech',
        name='WORKS_AT',
        fact='Alice works at Initech',
        group_id='group',
        source_node_uuid='alice',
        target_node_uuid='initech',
        valid_at=DEC,
        created_at=DEC,
        episodes=episodes,
        invalid_at=invalid_at,
        expired_at=invalid_at,
    )


@pytest.mark.asyncio
async def test_add_episode_bulk_saves_one_merged_copy_of_a_stored_edge(monkeypatch):
    graphiti = _make_graphiti()
    saved: dict[str, Any] = {}
    stored_episode = 'stored-episode'

    async def fake_add_nodes_and_edges_bulk(driver, *args, **kwargs):
        edges = kwargs['entity_edges'] if 'entity_edges' in kwargs else args[3]
        if edges:
            saved['edges'] = list(edges)

    async def fake_previous_episodes(driver, episodes, max_coroutines=None):
        return [(episode, []) for episode in episodes]

    async def fake_dedupe_edges_bulk(clients, extracted_edges, episode_context, *args):
        return {episode.uuid: [] for episode, _ in episode_context}

    async def fake_extract(self, episode_context, *args, **kwargs):
        return {episode.uuid: [] for episode, _ in episode_context}, {}, [[]]

    async def fake_resolve(self, nodes_by_episode, edges_by_episode, episode_context, *a, **kw):
        restating = episode_context[0][0]
        # The first episode restates the stored fact. The other two contradict it.
        resolved = [_initech_copy([stored_episode, restating.uuid])]
        invalidated = [
            _initech_copy([stored_episode], invalid_at=MAR),
            _initech_copy([stored_episode], invalid_at=JAN),
        ]
        return [], resolved, invalidated, {}

    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', fake_add_nodes_and_edges_bulk)
    monkeypatch.setattr(graphiti_module, 'retrieve_previous_episodes_bulk', fake_previous_episodes)
    monkeypatch.setattr(graphiti_module, 'dedupe_edges_bulk', fake_dedupe_edges_bulk)
    monkeypatch.setattr(Graphiti, '_extract_and_dedupe_nodes_bulk', fake_extract)
    monkeypatch.setattr(Graphiti, '_resolve_nodes_and_edges_bulk', fake_resolve)

    results = await graphiti.add_episode_bulk(
        [
            RawEpisode(
                name=name,
                content=name,
                source=EpisodeType.text,
                source_description='HR log',
                reference_time=when,
            )
            for name, when in (('payslip', JAN), ('acme', JAN), ('globex', MAR))
        ],
        group_id='group',
    )

    restating = results.episodes[0]
    assert [edge.uuid for edge in saved['edges']] == ['initech']
    initech = saved['edges'][0]
    assert initech.episodes == [stored_episode, restating.uuid]
    assert initech.invalid_at == JAN
    assert initech.expired_at == JAN
    assert results.edges == saved['edges']
    assert restating.entity_edges == ['initech']
