"""Tests that Neptune bulk operations actually respect batch_size instead of
sending every item in a single unbounded request (or, for entity nodes,
mixing nodes with different label sets into the wrong query)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.neptune.operations.entity_edge_ops import NeptuneEntityEdgeOperations
from graphiti_core.driver.neptune.operations.entity_node_ops import NeptuneEntityNodeOperations
from graphiti_core.driver.neptune.operations.episode_node_ops import NeptuneEpisodeNodeOperations
from graphiti_core.driver.neptune.operations.episodic_edge_ops import (
    NeptuneEpisodicEdgeOperations,
)
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeExecutor:
    """Records every execute_query call instead of hitting a real driver."""

    def __init__(self, responses: list[tuple[list[dict[str, Any]], None, None]] | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = list(responses) if responses is not None else None

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.calls.append((cypher_query_, kwargs))
        if self._responses is not None:
            return self._responses.pop(0)
        return [], None, None


def make_entity_node(name: str, labels: list[str]) -> EntityNode:
    return EntityNode(name=name, group_id='g', labels=labels, created_at=NOW)


def make_entity_edge(name: str) -> EntityEdge:
    return EntityEdge(
        name=name,
        fact=f'{name} fact',
        group_id='g',
        source_node_uuid='src',
        target_node_uuid='tgt',
        created_at=NOW,
    )


def make_episodic_node(name: str) -> EpisodicNode:
    return EpisodicNode(
        name=name,
        group_id='g',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=NOW,
        created_at=NOW,
    )


def make_episodic_edge() -> EpisodicEdge:
    return EpisodicEdge(
        group_id='g',
        source_node_uuid='episode',
        target_node_uuid='entity',
        created_at=NOW,
    )


class TestEntityNodeBatching:
    @pytest.mark.asyncio
    async def test_save_bulk_chunks_by_batch_size(self):
        executor = FakeExecutor()
        ops = NeptuneEntityNodeOperations()
        nodes = [make_entity_node(f'n{i}', ['Person']) for i in range(5)]

        await ops.save_bulk(executor, nodes, batch_size=2)

        assert len(executor.calls) == 3
        for _, kwargs in executor.calls:
            assert len(kwargs['nodes']) <= 2
        seen_uuids = {n['uuid'] for _, kwargs in executor.calls for n in kwargs['nodes']}
        assert seen_uuids == {n.uuid for n in nodes}

    @pytest.mark.asyncio
    async def test_save_bulk_does_not_cross_contaminate_label_groups(self):
        executor = FakeExecutor()
        ops = NeptuneEntityNodeOperations()
        people = [make_entity_node(f'person{i}', ['Person']) for i in range(3)]
        orgs = [make_entity_node(f'org{i}', ['Organization']) for i in range(2)]

        await ops.save_bulk(executor, people + orgs, batch_size=100)

        # Each request's query text is specific to one label combination, so
        # a Person's data must never be UNWOUND through the Organization
        # query (or vice versa) -- doing so would tag it with the wrong label.
        assert len(executor.calls) == 2
        for query, kwargs in executor.calls:
            uuids_in_call = {n['uuid'] for n in kwargs['nodes']}
            if 'Person' in query:
                assert uuids_in_call == {n.uuid for n in people}
            else:
                assert 'Organization' in query
                assert uuids_in_call == {n.uuid for n in orgs}

    @pytest.mark.asyncio
    async def test_save_bulk_noop_on_empty_list(self):
        executor = FakeExecutor()
        ops = NeptuneEntityNodeOperations()

        await ops.save_bulk(executor, [])

        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_load_embeddings_bulk_chunks_by_batch_size(self):
        nodes = [make_entity_node(f'n{i}', ['Person']) for i in range(5)]
        responses = [
            ([{'uuid': n.uuid, 'name_embedding': [0.1]} for n in chunk], None, None)
            for chunk in (nodes[0:2], nodes[2:4], nodes[4:5])
        ]
        executor = FakeExecutor(responses)
        ops = NeptuneEntityNodeOperations()

        await ops.load_embeddings_bulk(executor, nodes, batch_size=2)

        assert len(executor.calls) == 3
        for _, kwargs in executor.calls:
            assert len(kwargs['uuids']) <= 2
        assert all(n.name_embedding == [0.1] for n in nodes)


class TestEntityEdgeBatching:
    @pytest.mark.asyncio
    async def test_save_bulk_chunks_by_batch_size(self):
        executor = FakeExecutor()
        ops = NeptuneEntityEdgeOperations()
        edges = [make_entity_edge(f'e{i}') for i in range(5)]

        await ops.save_bulk(executor, edges, batch_size=2)

        assert len(executor.calls) == 3
        for _, kwargs in executor.calls:
            assert len(kwargs['entity_edges']) <= 2
        seen_uuids = {e['uuid'] for _, kwargs in executor.calls for e in kwargs['entity_edges']}
        assert seen_uuids == {e.uuid for e in edges}

    @pytest.mark.asyncio
    async def test_save_bulk_noop_on_empty_list(self):
        executor = FakeExecutor()
        ops = NeptuneEntityEdgeOperations()

        await ops.save_bulk(executor, [])

        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_load_embeddings_bulk_chunks_by_batch_size(self):
        edges = [make_entity_edge(f'e{i}') for i in range(5)]
        responses = [
            ([{'uuid': e.uuid, 'fact_embedding': [0.2]} for e in chunk], None, None)
            for chunk in (edges[0:2], edges[2:4], edges[4:5])
        ]
        executor = FakeExecutor(responses)
        ops = NeptuneEntityEdgeOperations()

        await ops.load_embeddings_bulk(executor, edges, batch_size=2)

        assert len(executor.calls) == 3
        for _, kwargs in executor.calls:
            assert len(kwargs['edge_uuids']) <= 2
        assert all(e.fact_embedding == [0.2] for e in edges)


class TestEpisodeNodeBatching:
    @pytest.mark.asyncio
    async def test_save_bulk_chunks_by_batch_size(self):
        executor = FakeExecutor()
        ops = NeptuneEpisodeNodeOperations()
        nodes = [make_episodic_node(f'ep{i}') for i in range(5)]

        await ops.save_bulk(executor, nodes, batch_size=2)

        assert len(executor.calls) == 3
        for _, kwargs in executor.calls:
            assert len(kwargs['episodes']) <= 2

    @pytest.mark.asyncio
    async def test_save_bulk_noop_on_empty_list(self):
        executor = FakeExecutor()
        ops = NeptuneEpisodeNodeOperations()

        await ops.save_bulk(executor, [])

        assert executor.calls == []


class TestEpisodicEdgeBatching:
    @pytest.mark.asyncio
    async def test_save_bulk_chunks_by_batch_size(self):
        executor = FakeExecutor()
        ops = NeptuneEpisodicEdgeOperations()
        edges = [make_episodic_edge() for _ in range(5)]

        await ops.save_bulk(executor, edges, batch_size=2)

        assert len(executor.calls) == 3
        for _, kwargs in executor.calls:
            assert len(kwargs['episodic_edges']) <= 2

    @pytest.mark.asyncio
    async def test_save_bulk_noop_on_empty_list(self):
        executor = FakeExecutor()
        ops = NeptuneEpisodicEdgeOperations()

        await ops.save_bulk(executor, [])

        assert executor.calls == []
