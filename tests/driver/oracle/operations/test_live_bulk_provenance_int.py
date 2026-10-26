# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import graphiti_core.graphiti as graphiti_module
import graphiti_core.utils.bulk_utils as bulk_utils
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from tests.driver.oracle.conformance.backends import DeterministicEmbedder
from tests.driver.oracle.operations.test_live_episode_removal_int import graph_for
from tests.driver.oracle.operations.test_live_surface_int import NOW, VECTOR_A, stable_uuid

pytestmark = [pytest.mark.oracle, pytest.mark.asyncio]


async def test_live_bulk_shared_provenance_and_reingestion_have_no_duplicate_rows(
    live_driver, monkeypatch
) -> None:
    """Real bulk orchestration/dedup/persistence, with deterministic semantic decisions."""

    driver, fixture = live_driver
    scoped = driver.clone(fixture.group_id)
    graph = graph_for(scoped)
    graph.embedder = DeterministicEmbedder()
    graph.tracer = MagicMock()
    graph.clients = GraphitiClients.model_construct(
        driver=scoped,
        llm_client=graph.llm_client,
        embedder=graph.embedder,
        cross_encoder=object(),
        tracer=graph.tracer,
        max_coroutines=2,
    )
    token = uuid4().hex
    nodes = [
        EntityNode(
            uuid=stable_uuid(fixture.group_id, f'bulk-node-{index}'),
            name=f'Bulk entity {index} {token}',
            group_id=fixture.group_id,
            summary='Shared source context',
            name_embedding=VECTOR_A,
            created_at=NOW,
        )
        for index in range(2)
    ]
    raws = [
        RawEpisode(
            uuid=stable_uuid(fixture.group_id, f'bulk-episode-{index}'),
            name=f'Native bulk source {index}',
            content=('Evidência ação observação. ' * 1400) + f' BulkSource{index}{token}',
            source_description='synthetic library certification',
            source=EpisodeType.text,
            reference_time=NOW,
        )
        for index in range(2)
    ]

    async def deterministic_extraction(context, *_args, **_kwargs):
        by_episode = {episode.uuid: nodes for episode, _ in context}
        edges = [
            [
                EntityEdge(
                    uuid=stable_uuid(fixture.group_id, f'bulk-fact-{index}'),
                    source_node_uuid=nodes[0].uuid,
                    target_node_uuid=nodes[1].uuid,
                    group_id=fixture.group_id,
                    name='SHARED_BULK_FACT',
                    fact=('Fato técnico ação conexão. ' * 1500) + token,
                    fact_embedding=VECTOR_A,
                    episodes=[episode.uuid],
                    attributes={'section': 'Memória técnica de observação. ' * 600},
                    created_at=NOW,
                )
            ]
            for index, (episode, _) in enumerate(context)
        ]
        return by_episode, {}, edges

    async def deterministic_resolution(_llm, edge, candidates, *_args, **_kwargs):
        duplicates = [candidate for candidate in candidates if candidate.fact == edge.fact]
        return edge, [], duplicates

    async def existing_graph_resolution(_by_episode, edges_by_episode, *_args, **_kwargs):
        unique_edges = {edge.uuid: edge for edges in edges_by_episode.values() for edge in edges}
        return nodes, list(unique_edges.values()), [], {}

    async def deterministic_summary(_llm, entities, sources, *_args, **_kwargs):
        for entity in entities:
            entity.summary += '\n'.join(source.content for source in sources)

    monkeypatch.setattr(graph, '_extract_and_dedupe_nodes_bulk', deterministic_extraction)
    monkeypatch.setattr(graph, '_resolve_nodes_and_edges_bulk', existing_graph_resolution)
    monkeypatch.setattr(bulk_utils, 'resolve_extracted_edge', deterministic_resolution)
    monkeypatch.setattr(graphiti_module, '_extract_entity_summaries_batch', deterministic_summary)

    first = await graph.add_episode_bulk(raws, group_id=fixture.group_id)
    fact_uuid = first.edges[0].uuid
    expected_episodes = {raw.uuid for raw in raws}
    for episode in first.episodes:
        persisted = await driver.episode_node_ops.get_by_uuid(scoped, episode.uuid)
        assert persisted.entity_edges == [fact_uuid]
        assert persisted.content == next(raw.content for raw in raws if raw.uuid == episode.uuid)
    fact = await driver.entity_edge_ops.get_by_uuid(scoped, fact_uuid)
    assert set(fact.episodes) == expected_episodes

    await graph.remove_episode(first.episodes[0].uuid)
    remaining_fact = await driver.entity_edge_ops.get_by_uuid(scoped, fact_uuid)
    assert remaining_fact.episodes == [first.episodes[1].uuid]

    await graph.add_episode_bulk(raws, group_id=fixture.group_id)
    await graph.add_episode_bulk(raws, group_id=fixture.group_id)

    episodes = await driver.episode_node_ops.get_by_group_ids(scoped, [fixture.group_id])
    facts = await driver.entity_edge_ops.get_by_group_ids(scoped, [fixture.group_id])
    assert {episode.uuid for episode in episodes} == expected_episodes
    assert len(episodes) == 2
    assert len(facts) == 1
    assert set(facts[0].episodes) == expected_episodes
    assert len(facts[0].episodes) == 2
    assert facts[0].fact == first.edges[0].fact
    assert facts[0].attributes == first.edges[0].attributes
    counts, _, _ = await scoped.execute_query(
        'SELECT COUNT(*) AS edge_count FROM graphiti_edges '
        "WHERE kind = 'MENTIONS' AND group_id = :group_id",
        group_id=fixture.group_id,
    )
    assert counts[0]['edge_count'] == 4


pytestmark = [pytest.mark.integration, pytest.mark.oracle]
