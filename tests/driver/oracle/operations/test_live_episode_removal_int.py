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

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.driver.oracle.errors import EpisodeRemovalConflict, EpisodeSummaryRebuildError
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from tests.driver.oracle.operations.test_live_surface_int import (
    NOW,
    VECTOR_A,
    stable_uuid,
)

pytestmark = [pytest.mark.oracle, pytest.mark.asyncio]


def graph_for(driver) -> Graphiti:
    graph = object.__new__(Graphiti)
    graph.driver = driver
    graph.llm_client = object()
    graph.clients = SimpleNamespace(driver=driver)
    graph.max_coroutines = 2
    return graph


async def seed_shared_episode_graph(driver, group_id: str, episode_count: int = 2):
    scoped = driver.clone(group_id)
    token = uuid4().hex
    nodes = [
        EntityNode(
            uuid=stable_uuid(group_id, f'removal-node-{index}'),
            name=f'Removal entity {index}',
            group_id=group_id,
            summary=f'Old summary {token}',
            attributes={'original': True, 'index': index},
            name_embedding=VECTOR_A,
            created_at=NOW,
        )
        for index in range(2)
    ]
    shared_uuid = stable_uuid(group_id, 'shared-fact')
    episodes = [
        EpisodicNode(
            uuid=stable_uuid(group_id, f'removal-episode-{index}'),
            name=f'Removal source {index}',
            group_id=group_id,
            source=EpisodeType.text,
            source_description='synthetic library certification',
            content=('Evidência UTF-8 ação observação. ' * 1000)
            + f' SurvivingSource{index}{token}',
            entity_edges=[shared_uuid],
            created_at=NOW,
            valid_at=NOW,
        )
        for index in range(episode_count)
    ]
    shared = EntityEdge(
        uuid=shared_uuid,
        source_node_uuid=nodes[0].uuid,
        target_node_uuid=nodes[1].uuid,
        group_id=group_id,
        name='SHARED_EVIDENCE',
        fact=f'SharedFact{token}: ' + 'Evidência técnica ação. ' * 1600,
        fact_embedding=VECTOR_A,
        episodes=[episode.uuid for episode in episodes],
        attributes={'section': 'Memória técnica de observação. ' * 600},
        created_at=NOW,
        valid_at=NOW,
    )
    async with scoped.transaction() as tx:
        await driver.entity_node_ops.save_bulk(scoped, nodes, tx=tx)
        await driver.episode_node_ops.save_bulk(scoped, episodes, tx=tx)
        await driver.entity_edge_ops.save(scoped, shared, tx=tx)
        for episode in episodes:
            for node in nodes:
                await driver.episodic_edge_ops.save(
                    scoped,
                    EpisodicEdge(
                        uuid=stable_uuid(group_id, f'mention-{episode.uuid}-{node.uuid}'),
                        source_node_uuid=episode.uuid,
                        target_node_uuid=node.uuid,
                        group_id=group_id,
                        created_at=NOW,
                    ),
                    tx=tx,
                )
    return scoped, nodes, episodes, shared


@pytest.mark.parametrize('removed_index', [0, 1])
async def test_live_remove_episode_preserves_shared_fact_and_exact_survivor_summary(
    live_driver, monkeypatch, removed_index: int
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)
    removed = episodes[removed_index]
    remaining = episodes[1 - removed_index]
    summary_inputs: list[list[str]] = []

    async def deterministic_summary(_llm, entities, sources, *_args, **_kwargs):
        summary_inputs.append([source.uuid for source in sources])
        for entity in entities:
            entity.summary += '\n'.join(source.content for source in sources)

    monkeypatch.setattr(graphiti_module, '_extract_entity_summaries_batch', deterministic_summary)

    await graph_for(scoped).remove_episode(removed.uuid)

    with pytest.raises(NodeNotFoundError):
        await driver.episode_node_ops.get_by_uuid(scoped, removed.uuid)
    persisted = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert persisted.episodes == [remaining.uuid]
    assert persisted.fact == shared.fact
    assert persisted.fact_embedding == VECTOR_A
    assert persisted.attributes == shared.attributes
    for original in nodes:
        node = await driver.entity_node_ops.get_by_uuid(scoped, original.uuid)
        assert node.summary == remaining.content
        assert node.attributes == original.attributes
        assert node.name_embedding == VECTOR_A
    assert summary_inputs == [[remaining.uuid], [remaining.uuid]]

    await graph_for(scoped).remove_episode(remaining.uuid)

    with pytest.raises(EdgeNotFoundError):
        await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    for node in nodes:
        with pytest.raises(NodeNotFoundError):
            await driver.entity_node_ops.get_by_uuid(scoped, node.uuid)


async def test_live_remove_episode_summary_failure_leaves_graph_unchanged(
    live_driver, monkeypatch
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)

    async def unavailable_summary(*_args, **_kwargs):
        raise RuntimeError('synthetic summary service failure')

    monkeypatch.setattr(graphiti_module, '_extract_entity_summaries_batch', unavailable_summary)

    with pytest.raises(RuntimeError, match='summary'):
        await graph_for(scoped).remove_episode(episodes[0].uuid)

    for episode in episodes:
        persisted_episode = await driver.episode_node_ops.get_by_uuid(scoped, episode.uuid)
        assert persisted_episode.content == episode.content
    persisted_edge = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert persisted_edge.episodes == shared.episodes
    for original in nodes:
        persisted = await driver.entity_node_ops.get_by_uuid(scoped, original.uuid)
        assert persisted.summary == original.summary
        assert persisted.name_embedding == original.name_embedding


async def test_live_remove_legacy_bulk_episode_uses_reverse_fact_provenance(
    live_driver, monkeypatch
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)
    episodes[0].entity_edges = []
    await driver.episode_node_ops.save(scoped, episodes[0])

    async def deterministic_summary(_llm, entities, sources, *_args, **_kwargs):
        for entity in entities:
            entity.summary += '\n'.join(source.content for source in sources)

    monkeypatch.setattr(graphiti_module, '_extract_entity_summaries_batch', deterministic_summary)

    await graph_for(scoped).remove_episode(episodes[0].uuid)

    persisted = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert persisted.episodes == [episodes[1].uuid]
    assert persisted.fact == shared.fact
    persisted_node = await driver.entity_node_ops.get_by_uuid(scoped, nodes[0].uuid)
    assert persisted_node.summary == episodes[1].content


async def test_live_remove_episode_rolls_back_every_write_on_persistence_error(
    live_driver, monkeypatch
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)

    async def deterministic_summary(_llm, entities, sources, *_args, **_kwargs):
        for entity in entities:
            entity.summary += '\n'.join(source.content for source in sources)

    original_delete = driver.episode_node_ops.delete

    async def fail_after_delete(executor, episode, tx=None):
        await original_delete(executor, episode, tx=tx)
        raise RuntimeError('synthetic removal persistence failure')

    monkeypatch.setattr(graphiti_module, '_extract_entity_summaries_batch', deterministic_summary)
    monkeypatch.setattr(driver.episode_node_ops, 'delete', fail_after_delete)

    with pytest.raises(RuntimeError, match='persistence failure'):
        await graph_for(scoped).remove_episode(episodes[0].uuid)

    persisted = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert persisted.episodes == shared.episodes
    assert persisted.fact == shared.fact
    for episode in episodes:
        persisted_episode = await driver.episode_node_ops.get_by_uuid(scoped, episode.uuid)
        assert persisted_episode.content == episode.content
    for original in nodes:
        node = await driver.entity_node_ops.get_by_uuid(scoped, original.uuid)
        assert node.summary == original.summary
    mentioned = await driver.graph_ops.get_mentioned_nodes(scoped, episodes)
    assert {node.uuid for node in mentioned} == {node.uuid for node in nodes}


async def test_live_remove_episode_rejects_changed_snapshot_without_partial_removal(
    live_driver, monkeypatch
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)
    changed = False

    async def update_source_during_summary(_llm, entities, sources, *_args, **_kwargs):
        nonlocal changed
        if not changed:
            episodes[1].content += ' Fresh source revision'
            await driver.episode_node_ops.save(scoped, episodes[1])
            changed = True
        for entity in entities:
            entity.summary += '\n'.join(source.content for source in sources)

    monkeypatch.setattr(
        graphiti_module, '_extract_entity_summaries_batch', update_source_during_summary
    )

    with pytest.raises(EpisodeRemovalConflict, match='changed'):
        await graph_for(scoped).remove_episode(episodes[0].uuid)

    persisted_fact = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert persisted_fact.episodes == shared.episodes
    persisted_episode = await driver.episode_node_ops.get_by_uuid(scoped, episodes[0].uuid)
    assert persisted_episode.uuid == episodes[0].uuid
    persisted_node = await driver.entity_node_ops.get_by_uuid(scoped, nodes[0].uuid)
    assert persisted_node.summary == nodes[0].summary
    persisted_source = await driver.episode_node_ops.get_by_uuid(scoped, episodes[1].uuid)
    assert persisted_source.content.endswith('Fresh source revision')


async def test_live_remove_episode_cannot_escape_request_scope(live_driver) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)
    foreign = driver.clone(fixture.group_id + '-foreign')

    with pytest.raises(NodeNotFoundError):
        await graph_for(foreign).remove_episode(episodes[0].uuid)

    persisted_fact = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert persisted_fact.episodes == shared.episodes
    for original in nodes:
        persisted = await driver.entity_node_ops.get_by_uuid(scoped, original.uuid)
        assert persisted.summary == original.summary


async def test_live_remove_episode_summaries_cover_all_survivors_in_bounded_batches(
    live_driver, monkeypatch
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, _ = await seed_shared_episode_graph(
        driver, fixture.group_id, episode_count=13
    )
    batches: list[list[str]] = []

    async def deterministic_summary(_llm, entities, sources, *_args, **_kwargs):
        batches.append([source.uuid for source in sources])
        for entity in entities:
            prefix = '\n' if entity.summary else ''
            entity.summary += prefix + '\n'.join(source.content for source in sources)

    monkeypatch.setattr(graphiti_module, '_extract_entity_summaries_batch', deterministic_summary)

    await graph_for(scoped).remove_episode(episodes[0].uuid)

    expected_sources = sorted(episodes[1:], key=lambda source: source.uuid)
    assert [len(batch) for batch in batches] == [10, 2, 10, 2]
    assert batches[0] + batches[1] == [source.uuid for source in expected_sources]
    expected_summary = '\n'.join(source.content for source in expected_sources)
    for node in nodes:
        persisted = await driver.entity_node_ops.get_by_uuid(scoped, node.uuid)
        assert persisted.summary == expected_summary
        assert persisted.name_embedding == VECTOR_A


async def test_live_remove_episode_rejects_incomplete_model_output_before_any_write(
    live_driver,
) -> None:
    driver, fixture = live_driver
    scoped, nodes, episodes, shared = await seed_shared_episode_graph(driver, fixture.group_id)
    graph = graph_for(scoped)
    graph.llm_client = SimpleNamespace(generate_response=AsyncMock(return_value={'summaries': []}))

    with pytest.raises(EpisodeSummaryRebuildError, match='summary preparation'):
        await graph.remove_episode(episodes[0].uuid)

    fact = await driver.entity_edge_ops.get_by_uuid(scoped, shared.uuid)
    assert fact.episodes == shared.episodes
    assert (await driver.episode_node_ops.get_by_uuid(scoped, episodes[0].uuid)).uuid
    for original in nodes:
        persisted = await driver.entity_node_ops.get_by_uuid(scoped, original.uuid)
        assert persisted.summary == original.summary
        assert persisted.attributes == original.attributes


pytestmark = [pytest.mark.integration, pytest.mark.oracle]
