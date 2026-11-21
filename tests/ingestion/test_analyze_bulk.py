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

# Tests for the analysis of a bulk unit.

from unittest.mock import AsyncMock

from graphiti_core import graphiti as graphiti_module
from graphiti_core.graphiti import Graphiti
from graphiti_core.ingestion import phases
from graphiti_core.ingestion.change_set import PatchNode
from graphiti_core.ingestion.phases import AdmittedUnit, analyze_bulk
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance import node_operations as node_ops
from tests.ingestion.ingestion_fixtures import GROUP, T0, Harness, X, make_episode, make_node


async def admit_bulk_unit(h: Harness, episode_uuids: list[str]) -> AdmittedUnit:
    admission = await h.store.begin_admit(GROUP, 'bulk', h.config.lease_seconds)
    assert admission is not None
    await h.store.end_admit(GROUP, admission.seq)
    return AdmittedUnit(
        group_id=GROUP,
        unit_id='bulk',
        seq=admission.seq,
        created_at=T0,
        bulk=True,
        episodes=[make_episode(uuid) for uuid in episode_uuids],
        previous_episodes=[[] for _ in episode_uuids],
    )


async def test_bulk_analysis_does_not_create_a_node_that_is_in_the_graph(monkeypatch):
    """The first pass of the batch pipeline resolves an extracted node to a stored node.

    The stored node must not be in `create_nodes`, because the commit writes each
    record of that list in full.
    """
    h = Harness(monkeypatch)
    h.graphiti._extract_and_dedupe_nodes_bulk = Graphiti._extract_and_dedupe_nodes_bulk.__get__(
        h.graphiti
    )
    h.graphiti._resolve_nodes_and_edges_bulk = Graphiti._resolve_nodes_and_edges_bulk.__get__(
        h.graphiti
    )
    stored = make_node('stored', 'Ana Lopez', summary='the summary in the graph')
    extracted = [
        [
            EntityNode(
                uuid='x1',
                name='Ana Lopez',
                group_id=GROUP,
                labels=['Entity', 'Person'],
                attributes={'email': 'ana@example.com'},
            ),
            EntityNode(uuid='x2', name='Acme', group_id=GROUP, labels=['Entity']),
        ],
        [EntityNode(uuid='x3', name='Ana Lopez', group_id=GROUP, labels=['Entity'])],
    ]
    monkeypatch.setattr(
        graphiti_module,
        'extract_nodes_and_edges_bulk',
        AsyncMock(return_value=(extracted, [[], []])),
    )

    async def collect_candidates(clients, nodes, *args, **kwargs):
        return [[stored] if node.name == 'Ana Lopez' else [] for node in nodes]

    monkeypatch.setattr(node_ops, '_collect_candidate_nodes', collect_candidates)
    monkeypatch.setattr(phases, 'dedupe_edges_bulk', AsyncMock(return_value={'ep1': [], 'ep2': []}))
    monkeypatch.setattr(
        graphiti_module, 'resolve_extracted_edges', AsyncMock(return_value=([], [], []))
    )
    h.clients.embedder.create = AsyncMock(return_value=X)
    unit = await admit_bulk_unit(h, ['ep1', 'ep2'])

    change_set = await analyze_bulk(h.graphiti, h.store, h.config, unit)

    assert [node.uuid for node in change_set.create_nodes] == ['x2']
    assert change_set.node_uuids == ['stored', 'x2']
    assert change_set.seen_node_candidates['x1'] == ['stored']
    assert change_set.seen_node_candidates['x3'] == ['stored']
    assert (
        PatchNode(
            node_uuid='stored',
            labels=['Entity', 'Person'],
            attributes={'email': 'ana@example.com'},
        )
        in change_set.operations
    )
    assert sorted(
        (edge.source_node_uuid, edge.target_node_uuid) for edge in change_set.episodic_edges
    ) == [
        ('ep1', 'stored'),
        ('ep1', 'x2'),
        ('ep2', 'stored'),
    ]
    h.llm.assert_not_awaited()
