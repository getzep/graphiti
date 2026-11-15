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

# Replay tests for the commit of concurrent ingestion (steps C1 to C10).
#
# Each test makes change sets with no model call, commits them in sequence order on an
# in-memory graph, and checks the repair. The model client raises an error on a call
# that a test does not expect.

from unittest.mock import AsyncMock

import pytest

from graphiti_core.edges import HasEpisodeEdge, NextEpisodeEdge
from graphiti_core.errors import ChangeSetRejected, NodeNotFoundError, UnitExpiredError
from graphiti_core.ingestion.change_set import (
    AddEpisodesToEdge,
    ChangeSet,
    EndEdge,
    PatchNode,
    SetEdgeAttributes,
)
from graphiti_core.ingestion.phases import UnitLease
from graphiti_core.ingestion.state_store import STATUS_APPLYING, STATUS_FAILED
from graphiti_core.nodes import SagaNode
from tests.ingestion.ingestion_fixtures import (
    GROUP,
    NEAR_X,
    T0,
    T1,
    T2,
    FakeClock,
    Harness,
    Y,
    Z,
    make_edge,
    make_node,
)

TYPED = pytest.mark.parametrize('typed', [False, True], ids=['untyped', 'typed'])


def labels_for(typed: bool, label: str) -> list[str]:
    return ['Entity', label] if typed else ['Entity']


async def commit_two(h: Harness, fields_1: dict, fields_2: dict, **commit_kwargs):
    """Analyze two units on the same snapshot, then commit them in sequence order."""
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    change_set_1 = await h.analyze(unit_1, **fields_1)
    change_set_2 = await h.analyze(unit_2, **fields_2)
    assert change_set_1.snapshot_seq == change_set_2.snapshot_seq == 0
    result_1 = await h.commit(unit_1, change_set_1, **commit_kwargs)
    result_2 = await h.commit(unit_2, change_set_2, **commit_kwargs)
    return (unit_1, result_1), (unit_2, result_2)


def use_llm(h: Harness, response: dict) -> None:
    h.llm.side_effect = None
    h.llm.return_value = response


# --- C1 --------------------------------------------------------------------------


async def test_c1_retry_applies_the_stored_final_change_set(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit = await h.admit('u1')
    change_set = await h.analyze(unit, create_nodes=[make_node('a1', 'Acme')])
    # An earlier attempt stored the final change set and stopped before the write.
    turn = await h.store.begin_commit(GROUP, unit.seq, h.config.lease_seconds)
    final = change_set.model_copy(deep=True)
    final.create_nodes = [make_node('final', 'Final Node')]
    final.node_uuids = ['final']
    await h.store.save_final(GROUP, unit.seq, turn.fence, final.to_bytes())
    # The retry gets the turn only after the deadline of the earlier attempt.
    clock.advance(h.config.lease_seconds + 1)

    result = await h.commit(unit, change_set)

    assert result.report['retry'] == 1
    assert set(h.graph.nodes) == {'final'}
    assert 'read:node_candidates' not in h.graph.log
    h.llm.assert_not_awaited()


# --- C3 --------------------------------------------------------------------------


@TYPED
async def test_c3_merges_a_node_that_an_earlier_unit_of_the_window_created(monkeypatch, typed):
    h = Harness(monkeypatch)
    attributes = {'domain': 'acme.com'} if typed else {}
    entity_types = {'Company': object} if typed else None
    (unit_1, result_1), (unit_2, result_2) = await commit_two(
        h,
        {'create_nodes': [make_node('a1', 'Acme')]},
        {
            'create_nodes': [
                make_node('a2', 'acme', labels=labels_for(typed, 'Company'), attributes=attributes),
                make_node('b2', 'Bo', Y),
            ],
            'create_edges': [make_edge('e2', 'a2', 'b2', 'Acme hired Bo', Z)],
        },
        entity_types=entity_types,
    )

    final = result_2.change_set
    assert [node.uuid for node in final.create_nodes] == ['b2']
    assert final.uuid_map == {'a2': 'a1'}
    assert final.node_uuids == ['a1', 'b2']
    assert final.node_first_episode == {'a1': 'ep-u2', 'b2': 'ep-u2'}
    assert (
        PatchNode(node_uuid='a1', labels=labels_for(typed, 'Company'), attributes=attributes)
        in final.operations
    )
    # C4: the created edge and the episodic edge point at the surviving node.
    assert set(h.graph.nodes) == {'a1', 'b2'}
    assert h.graph.edges['e2'].source_node_uuid == 'a1'
    assert ('ep-u2', 'a1') in h.graph.episodic_edges
    assert ('ep-u2', 'a2') not in h.graph.episodic_edges
    assert result_2.report['node.checked'] == 2
    assert result_2.report['node.collided'] == 1
    assert result_2.report['repair.c3.node_merged'] == result_2.report['node.merged'] == 1
    assert result_2.report['delta.unit_count'] == 1
    h.llm.assert_not_awaited()

    # The Enrich phase applies the patch to the current node, in sequence order.
    await h.enrich(unit_1, result_1)
    await h.enrich(unit_2, result_2)
    merged = h.graph.nodes['a1']
    assert merged.summary == '+ep-u1+ep-u2'
    assert merged.labels == labels_for(typed, 'Company')
    assert merged.attributes == attributes


async def test_c3_asks_the_model_when_the_rules_do_not_resolve_a_collision(monkeypatch):
    h = Harness(monkeypatch)
    use_llm(
        h,
        {'entity_resolutions': [{'id': 0, 'name': 'Zenith GmbH', 'duplicate_candidate_id': 0}]},
    )
    _, (_, result_2) = await commit_two(
        h,
        {'create_nodes': [make_node('a1', 'Zenith Corporation')]},
        {'create_nodes': [make_node('a2', 'Zenith GmbH', NEAR_X)]},
    )

    assert result_2.change_set.uuid_map == {'a2': 'a1'}
    assert result_2.report['repair.llm_calls'] == 1
    assert h.llm.await_count == 1


async def test_c3_keeps_a_node_when_the_model_finds_no_duplicate(monkeypatch):
    h = Harness(monkeypatch)
    use_llm(
        h,
        {'entity_resolutions': [{'id': 0, 'name': 'Zenith GmbH', 'duplicate_candidate_id': -1}]},
    )
    _, (_, result_2) = await commit_two(
        h,
        {'create_nodes': [make_node('a1', 'Zenith Corporation')]},
        {'create_nodes': [make_node('a2', 'Zenith GmbH', NEAR_X)]},
    )

    assert result_2.change_set.uuid_map == {}
    assert set(h.graph.nodes) == {'a1', 'a2'}
    assert result_2.report['node.collided'] == 1
    assert result_2.report['node.merged'] == 0


async def test_c3_finds_a_delta_node_that_the_search_does_not_return(monkeypatch):
    h = Harness(monkeypatch)
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    change_set_1 = await h.analyze(unit_1, create_nodes=[make_node('a1', 'Acme')])
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('a2', 'Acme')])
    await h.commit(unit_1, change_set_1)
    # The search index does not have the node of unit 1 at this time.
    h.graph.hidden.add('a1')

    result_2 = await h.commit(unit_2, change_set_2)

    assert result_2.change_set.uuid_map == {'a2': 'a1'}
    assert set(h.graph.nodes) == {'a1'}


async def test_c3_index_lag_covers_a_unit_that_committed_before_the_snapshot(monkeypatch):
    h = Harness(monkeypatch)
    await h.run('u1', create_nodes=[make_node('a1', 'Acme')])
    h.graph.hidden.add('a1')
    # Unit 2 starts after the commit of unit 1, so its snapshot contains unit 1.
    unit_2 = await h.admit('u2')
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('a2', 'Acme')])
    assert change_set_2.snapshot_seq == 1

    result_2 = await h.commit(unit_2, change_set_2)

    assert result_2.report['delta.unit_count'] == 1
    assert result_2.change_set.uuid_map == {'a2': 'a1'}


async def test_c3_with_no_index_lag_the_delta_has_no_unit_before_the_snapshot(monkeypatch):
    h = Harness(monkeypatch, index_lag_seconds=0.0, clock=FakeClock())
    await h.run('u1', create_nodes=[make_node('a1', 'Acme')])
    h.graph.hidden.add('a1')
    h.clock.advance(1)
    unit_2, result_2 = await h.run('u2', create_nodes=[make_node('a2', 'Acme')])

    assert result_2.report['delta.unit_count'] == 0
    assert result_2.change_set.uuid_map == {}


async def test_c3_does_not_decide_again_a_candidate_that_the_analysis_saw(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Zenith Corporation'))
    _, result = await h.run(
        'u1',
        create_nodes=[make_node('a1', 'Zenith GmbH', NEAR_X)],
        seen_node_candidates={'a1': ['s1']},
    )

    assert result.report['node.collided'] == 0
    assert set(h.graph.nodes) == {'s1', 'a1'}
    h.llm.assert_not_awaited()


async def test_c3_decides_a_stored_candidate_that_the_analysis_did_not_see(monkeypatch):
    # A writer outside the sequence created the node after the analysis.
    h = Harness(monkeypatch)
    unit = await h.admit('u1')
    change_set = await h.analyze(unit, create_nodes=[make_node('a1', 'Acme')])
    h.graph.add(make_node('s1', 'Acme'))

    result = await h.commit(unit, change_set)

    assert result.change_set.uuid_map == {'a1': 's1'}
    assert result.report['delta.unit_count'] == 0


async def test_c3_identity_key_of_a_delta_patch_merges_a_seen_candidate(monkeypatch):
    h = Harness(monkeypatch)
    person = ['Entity', 'Person']
    h.graph.add(make_node('s1', 'Ana Lopez', labels=person))
    identity_properties = {'Person': ['email']}
    _, (_, result_2) = await commit_two(
        h,
        {
            'operations': [
                PatchNode(node_uuid='s1', labels=person, attributes={'email': 'ana@example.com'})
            ],
            'node_uuids': ['s1'],
        },
        {
            # The analysis saw `s1` with no email and found no duplicate.
            'create_nodes': [
                make_node(
                    'p2', 'A. Lopez', Y, labels=person, attributes={'email': 'ana@example.com'}
                )
            ],
            'seen_node_candidates': {'p2': ['s1']},
        },
        identity_properties=identity_properties,
        entity_types={'Person': object},
    )

    # The Enrich phase of unit 1 did not run, so the graph node has no email.
    assert h.graph.nodes['s1'].attributes == {}
    assert result_2.change_set.uuid_map == {'p2': 's1'}
    assert result_2.report['repair.llm_calls'] == 0
    h.llm.assert_not_awaited()


async def test_c3_a_delta_patch_with_no_new_data_does_not_make_a_collision(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Zenith Corporation', labels=['Entity', 'Company']))
    _, (_, result_2) = await commit_two(
        h,
        {
            'operations': [PatchNode(node_uuid='s1', labels=['Entity', 'Company'])],
            'node_uuids': ['s1'],
        },
        {
            'create_nodes': [make_node('a2', 'Zenith GmbH', NEAR_X)],
            'seen_node_candidates': {'a2': ['s1']},
        },
    )

    assert result_2.report['node.collided'] == 0
    h.llm.assert_not_awaited()


async def test_c3_a_delta_patch_that_promotes_a_label_keeps_a_seen_candidate(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Zenith Corporation'))
    use_llm(
        h,
        {'entity_resolutions': [{'id': 0, 'name': 'Zenith GmbH', 'duplicate_candidate_id': 0}]},
    )
    _, (_, result_2) = await commit_two(
        h,
        {
            'operations': [PatchNode(node_uuid='s1', labels=['Entity', 'Company'])],
            'node_uuids': ['s1'],
        },
        {
            'create_nodes': [make_node('a2', 'Zenith GmbH', NEAR_X, labels=['Entity', 'Company'])],
            'seen_node_candidates': {'a2': ['s1']},
        },
        entity_types={'Company': object},
    )

    assert result_2.report['node.collided'] == 1
    assert result_2.change_set.uuid_map == {'a2': 's1'}


async def test_c3_a_label_promotion_that_the_enrichment_wrote_keeps_a_seen_candidate(
    monkeypatch,
):
    """The enrichment of unit 1 wrote the promoted label before the repair of unit 2.

    The analysis of unit 2 saw the candidate with no type label. The candidate must
    stay a candidate, because its labels changed after the analysis saw it.
    """
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Zenith Corporation'))
    use_llm(
        h,
        {'entity_resolutions': [{'id': 0, 'name': 'Zenith GmbH', 'duplicate_candidate_id': 0}]},
    )
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    change_set_1 = await h.analyze(
        unit_1,
        operations=[PatchNode(node_uuid='s1', labels=['Entity', 'Company'])],
        node_uuids=['s1'],
    )
    change_set_2 = await h.analyze(
        unit_2,
        create_nodes=[make_node('a2', 'Zenith GmbH', NEAR_X, labels=['Entity', 'Company'])],
        seen_node_candidates={'a2': ['s1']},
        seen_node_labels={'s1': ['Entity']},
    )
    result_1 = await h.commit(unit_1, change_set_1, entity_types={'Company': object})
    await h.enrich(unit_1, result_1, entity_types={'Company': object})
    assert h.graph.nodes['s1'].labels == ['Entity', 'Company']

    result_2 = await h.commit(unit_2, change_set_2, entity_types={'Company': object})

    assert result_2.report['node.collided'] == 1
    assert result_2.change_set.uuid_map == {'a2': 's1'}


async def test_c3_a_label_that_the_analysis_saw_does_not_make_a_collision(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Zenith Corporation'))
    _, (_, result_2) = await commit_two(
        h,
        {
            'operations': [PatchNode(node_uuid='s1', labels=['Entity', 'Company'])],
            'node_uuids': ['s1'],
        },
        {
            'create_nodes': [make_node('a2', 'Zenith GmbH', NEAR_X)],
            'seen_node_candidates': {'a2': ['s1']},
            # The analysis of unit 2 read the node after the label was in the graph.
            'seen_node_labels': {'s1': ['Entity', 'Company']},
        },
    )

    assert result_2.report['node.collided'] == 0
    h.llm.assert_not_awaited()


# --- C2a -------------------------------------------------------------------------


@pytest.mark.parametrize('partial', [False, True], ids=['not-written', 'partly-written'])
async def test_c2a_apply_failed_unit_gives_only_the_records_that_exist(monkeypatch, partial):
    h = Harness(monkeypatch, max_apply_attempts=2)
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    change_set_1 = await h.analyze(unit_1, create_nodes=[make_node('a1', 'Acme')])
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('a2', 'Acme')])
    h.graph.fail_bulk_writes = 2
    h.graph.partial_bulk_write = partial
    with pytest.raises(RuntimeError):
        await h.commit(unit_1, change_set_1)
    assert ('a1' in h.graph.nodes) is partial

    result_2 = await h.commit(unit_2, change_set_2)

    assert result_2.report['delta.unit_count'] == 1
    if partial:
        assert result_2.change_set.uuid_map == {'a2': 'a1'}
        assert set(h.graph.nodes) == {'a1'}
    else:
        # A new node is never merged into a record that is not in the graph.
        assert result_2.change_set.uuid_map == {}
        assert set(h.graph.nodes) == {'a2'}


# --- C4 and C4a ------------------------------------------------------------------


async def test_c4_drops_an_edge_that_a_node_merge_makes_a_self_edge(monkeypatch):
    h = Harness(monkeypatch)
    _, (_, result_2) = await commit_two(
        h,
        {'create_nodes': [make_node('a1', 'Acme')]},
        {
            'create_nodes': [make_node('a2', 'Acme'), make_node('a3', 'ACME')],
            'create_edges': [make_edge('e2', 'a2', 'a3', 'Acme is also named ACME', Z)],
        },
    )

    final = result_2.change_set
    assert final.uuid_map == {'a2': 'a1', 'a3': 'a1'}
    assert final.create_edges == []
    assert final.episodes[0].entity_edges == []
    assert [(edge.source_node_uuid, edge.target_node_uuid) for edge in final.episodic_edges] == [
        ('ep-u2', 'a1')
    ]
    assert h.graph.edges == {}


async def test_c4a_removes_a_duplicate_between_created_edges_with_no_model_call(monkeypatch):
    h = Harness(monkeypatch)
    _, (_, result_2) = await commit_two(
        h,
        {'create_nodes': [make_node('a1', 'Acme')]},
        {
            'create_nodes': [
                make_node('a2', 'Acme'),
                make_node('a3', 'ACME'),
                make_node('b2', 'Bo', Y),
            ],
            'create_edges': [
                make_edge('e2', 'a2', 'b2', 'Acme hired Bo', Z),
                make_edge('e3', 'a3', 'b2', 'acme  hired bo', Z, episodes=['ep-other']),
                make_edge('e4', 'a3', 'b2', 'Acme pays Bo', Z),
            ],
        },
    )

    final = result_2.change_set
    assert [edge.uuid for edge in final.create_edges] == ['e2', 'e4']
    assert final.create_edges[0].episodes == ['ep-u2', 'ep-other']
    assert final.episodes[0].entity_edges == ['e2', 'e4']
    assert result_2.report['repair.c4a.edge_removed'] == 1
    assert result_2.report['edge.deduplicated'] == 1
    assert set(h.graph.edges) == {'e2', 'e4'}
    h.llm.assert_not_awaited()


# --- C5 to C7 --------------------------------------------------------------------


def stored_people(h: Harness) -> None:
    h.graph.add(make_node('ana', 'Ana'), make_node('acme', 'Acme', Y), make_node('bolt', 'Bolt', Z))


async def test_c5_exact_fact_of_an_earlier_unit_is_a_duplicate_with_no_model_call(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    _, (_, result_2) = await commit_two(
        h,
        {'create_edges': [make_edge('e1', 'ana', 'acme', 'Ana works at Acme')]},
        {'create_edges': [make_edge('e2', 'ana', 'acme', 'ana works at  ACME')]},
    )

    final = result_2.change_set
    assert final.create_edges == []
    assert final.episodes[0].entity_edges == ['e1']
    assert AddEpisodesToEdge(edge_uuid='e1', episode_uuids=['ep-u2']) in final.operations
    assert set(h.graph.edges) == {'e1'}
    assert h.graph.edges['e1'].episodes == ['ep-u1', 'ep-u2']
    assert [edge.uuid for edge in result_2.edges] == ['e1']
    assert result_2.report['edge.collided'] == 1
    assert result_2.report['repair.c5.edge_merged'] == 1
    assert result_2.report['operation.applied'] == 1
    h.llm.assert_not_awaited()


async def test_c5_model_decides_a_duplicate_with_a_different_fact_text(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    use_llm(h, {'duplicate_facts': [0], 'contradicted_facts': []})
    _, (_, result_2) = await commit_two(
        h,
        {'create_edges': [make_edge('e1', 'ana', 'acme', 'Ana works at Acme')]},
        # The reverse direction is also a duplicate candidate.
        {'create_edges': [make_edge('e2', 'acme', 'ana', 'Acme employs Ana')]},
    )

    assert set(h.graph.edges) == {'e1'}
    assert h.graph.edges['e1'].episodes == ['ep-u1', 'ep-u2']
    assert result_2.report['repair.llm_calls'] == 1
    assert h.llm.await_count == 1


async def test_c6_new_fact_ends_a_contradicted_fact_of_an_earlier_unit(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    use_llm(h, {'duplicate_facts': [], 'contradicted_facts': [0]})
    _, (_, result_2) = await commit_two(
        h,
        {'create_edges': [make_edge('e1', 'ana', 'acme', 'Ana works at Acme', valid_at=T0)]},
        # The edge has different endpoints. Only the graph-wide pass finds the candidate.
        {
            'create_edges': [
                make_edge('e2', 'ana', 'bolt', 'Ana works at Bolt', NEAR_X, valid_at=T1)
            ]
        },
    )

    ended = h.graph.edges['e1']
    assert ended.invalid_at == T1
    assert ended.expired_at is not None
    assert h.graph.edges['e2'].expired_at is None
    final = result_2.change_set
    assert [type(operation) for operation in final.operations] == [EndEdge]
    assert final.operations[0].only_if_open is False
    assert final.episodes[0].entity_edges == ['e2', 'e1']
    assert {edge.uuid for edge in result_2.edges} == {'e1', 'e2'}
    assert result_2.report['repair.c6.edge_ended'] == result_2.report['edge.ended'] == 1


async def test_c7_a_candidate_that_starts_later_ends_the_new_fact(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    use_llm(h, {'duplicate_facts': [], 'contradicted_facts': [0]})
    _, (_, result_2) = await commit_two(
        h,
        {'create_edges': [make_edge('e1', 'ana', 'acme', 'Ana works at Acme', valid_at=T1)]},
        {
            'create_edges': [
                make_edge('e2', 'ana', 'bolt', 'Ana works at Bolt', NEAR_X, valid_at=T0)
            ]
        },
    )

    assert h.graph.edges['e1'].invalid_at is None
    assert h.graph.edges['e1'].expired_at is None
    assert h.graph.edges['e2'].invalid_at == T1
    assert h.graph.edges['e2'].expired_at is not None
    assert result_2.change_set.operations == []
    assert result_2.report['edge.ended'] == 1


@pytest.mark.parametrize(
    ('ended_by_candidate', 'expected_invalid_at'),
    [(True, T1), (False, T2)],
    ids=['end-date-from-a-candidate', 'end-date-from-the-episode'],
)
async def test_c7_an_earlier_contradiction_of_the_delta_moves_the_end_date_of_the_new_fact(
    monkeypatch, ended_by_candidate, expected_invalid_at
):
    """The analysis ended the new fact at T2, the start of a stored fact that it saw.

    A fact of an earlier unit of the window contradicts the new fact and starts at T1.
    Serial ingestion ends the new fact at the earliest contradiction that starts later.
    An end date that the episode gave stays.
    """
    h = Harness(monkeypatch)
    stored_people(h)
    use_llm(h, {'duplicate_facts': [], 'contradicted_facts': [0]})
    new_fact = make_edge(
        'e2', 'ana', 'bolt', 'Ana works at Bolt', NEAR_X, valid_at=T0, invalid_at=T2, expired_at=T2
    )
    _, (_, result_2) = await commit_two(
        h,
        {'create_edges': [make_edge('e1', 'ana', 'acme', 'Ana works at Acme', valid_at=T1)]},
        {
            'create_edges': [new_fact],
            'edges_ended_by_candidate': ['e2'] if ended_by_candidate else [],
        },
    )

    assert result_2.report['edge.collided'] == 1
    assert h.graph.edges['e2'].invalid_at == expected_invalid_at
    assert h.graph.edges['e2'].expired_at == T2
    # The fact of the earlier unit starts later. It stays open.
    assert h.graph.edges['e1'].invalid_at is None


async def test_c6_finds_a_delta_edge_that_the_search_does_not_return(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    use_llm(h, {'duplicate_facts': [], 'contradicted_facts': [0]})
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    change_set_1 = await h.analyze(
        unit_1, create_edges=[make_edge('e1', 'ana', 'acme', 'Ana works at Acme', valid_at=T0)]
    )
    change_set_2 = await h.analyze(
        unit_2,
        create_edges=[make_edge('e2', 'ana', 'bolt', 'Ana works at Bolt', NEAR_X, valid_at=T1)],
    )
    await h.commit(unit_1, change_set_1)
    h.graph.hidden.add('e1')

    await h.commit(unit_2, change_set_2)

    assert h.graph.edges['e1'].invalid_at == T1


async def test_c7_an_edge_with_no_new_candidate_makes_no_model_call(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    h.graph.add(make_edge('e0', 'ana', 'acme', 'Ana works at Acme'))
    _, (_, result_2) = await commit_two(
        h,
        {'create_edges': [make_edge('e1', 'ana', 'bolt', 'Ana likes Bolt', Z)]},
        {
            # The analysis saw `e0` in the two passes.
            'create_edges': [make_edge('e2', 'ana', 'acme', 'Ana visits Acme', NEAR_X)],
            'seen_duplicate_candidates': {'e2': ['e0']},
            'seen_invalidation_candidates': {'e2': []},
        },
    )

    assert result_2.report['edge.checked'] == 1
    assert result_2.report['edge.collided'] == 0
    assert set(h.graph.edges) == {'e0', 'e1', 'e2'}
    h.llm.assert_not_awaited()


# --- C7a to C7c ------------------------------------------------------------------


def record_saga_writes(monkeypatch) -> list[tuple[str, str | None]]:
    """Record each saga write. A read of a saga node by UUID finds no node."""
    saved: list[tuple[str, str | None]] = []

    async def save_edge(self, driver):
        saved.append((type(self).__name__, self.uuid))

    async def save_saga(self, driver):
        saved.append(('SagaNode', self.last_episode_uuid))

    monkeypatch.setattr(NextEpisodeEdge, 'save', save_edge)
    monkeypatch.setattr(HasEpisodeEdge, 'save', save_edge)
    monkeypatch.setattr(SagaNode, 'save', save_saga)
    monkeypatch.setattr(SagaNode, 'get_by_uuid', AsyncMock(side_effect=NodeNotFoundError('saga')))
    return saved


async def test_c7a_saga_links_are_in_the_final_change_set(monkeypatch):
    h = Harness(monkeypatch)
    saga_node = SagaNode(uuid='saga1', name='thread', group_id=GROUP)
    h.graphiti._find_saga = AsyncMock(return_value=saga_node)
    h.graphiti._saga_get_previous_episode_uuid = AsyncMock(return_value='ep0')
    saved = record_saga_writes(monkeypatch)

    unit = await h.admit('u1')
    change_set = await h.analyze(unit)
    result = await h.commit(unit, change_set, saga='thread')

    (link,) = result.change_set.saga_links
    assert (link.saga_uuid, link.episode_uuid, link.previous_episode_uuid) == (
        'saga1',
        'ep-u1',
        'ep0',
    )
    assert result.change_set.create_saga is False
    assert saved == [
        ('NextEpisodeEdge', link.next_episode_edge_uuid),
        ('HasEpisodeEdge', link.has_episode_edge_uuid),
        ('SagaNode', 'ep-u1'),
    ]


async def test_c7a_new_saga_is_written_in_the_apply_and_not_before(monkeypatch):
    events: list[str] = []

    async def hook(change_set: ChangeSet) -> ChangeSet:
        events.append('hook')
        return change_set

    h = Harness(monkeypatch, before_commit_write=hook)
    h.graphiti._find_saga = AsyncMock(return_value=None)
    h.graphiti._saga_get_previous_episode_uuid = AsyncMock(
        side_effect=AssertionError('a new saga has no previous episode')
    )
    saved = record_saga_writes(monkeypatch)
    real_save_final = h.store.save_final

    async def save_final(*args):
        events.append(f'save_final after {len(saved)} saga writes')
        await real_save_final(*args)

    monkeypatch.setattr(h.store, 'save_final', save_final)
    unit = await h.admit('u1')
    change_set = await h.analyze(unit)
    result = await h.commit(unit, change_set, saga='thread')

    final = result.change_set
    (link,) = final.saga_links
    assert events == ['hook', 'save_final after 0 saga writes']
    assert final.create_saga is True
    assert final.saga_node.name == 'thread'
    assert link.saga_uuid == final.saga_node.uuid
    # The apply creates the saga node, then the edge, then it records the episode.
    assert saved == [
        ('SagaNode', None),
        ('HasEpisodeEdge', link.has_episode_edge_uuid),
        ('SagaNode', 'ep-u1'),
    ]


async def test_c7a_rejected_unit_leaves_no_saga_node(monkeypatch):
    async def reject(change_set: ChangeSet) -> ChangeSet:
        raise ChangeSetRejected('policy')

    h = Harness(monkeypatch, before_commit_write=reject)
    h.graphiti._find_saga = AsyncMock(return_value=None)
    saved = record_saga_writes(monkeypatch)
    h.graphiti._get_or_create_saga = AsyncMock(
        side_effect=lambda *args, **kwargs: saved.append(('created before the hook', None))
    )
    unit = await h.admit('u1')
    change_set = await h.analyze(unit)

    with pytest.raises(ChangeSetRejected):
        await h.commit(unit, change_set, saga='thread')

    assert saved == []
    assert 'write:bulk' not in h.graph.log


async def test_c7a_saga_object_that_is_not_in_the_graph_is_saved_as_in_serial_ingestion(
    monkeypatch,
):
    h = Harness(monkeypatch)
    h.graphiti._saga_get_previous_episode_uuid = AsyncMock(return_value=None)
    saved = record_saga_writes(monkeypatch)
    saga_node = SagaNode(uuid='saga-object', name='thread', group_id=GROUP)

    unit = await h.admit('u1')
    change_set = await h.analyze(unit)
    result = await h.commit(unit, change_set, saga=saga_node)

    final = result.change_set
    (link,) = final.saga_links
    assert link.saga_uuid == 'saga-object'
    assert final.create_saga is False
    # Serial ingestion saves the edge and then the object that the caller gave.
    assert saved == [('HasEpisodeEdge', link.has_episode_edge_uuid), ('SagaNode', 'ep-u1')]
    assert result.report['apply.attempts'] == 1


@pytest.mark.parametrize('saga_input', ['name', 'object'])
async def test_c7a_append_keeps_the_stored_fields_of_the_saga_node(monkeypatch, saga_input):
    h = Harness(monkeypatch)
    stored = SagaNode(
        uuid='saga1',
        name='thread',
        group_id=GROUP,
        created_at=T0,
        summary='The summary of the thread.',
        first_episode_uuid='ep0',
        last_episode_uuid='ep0',
        last_summarized_at=T1,
        last_summarized_episode_valid_at=T0,
    )
    saved_nodes: list[SagaNode] = []

    async def save_edge(self, driver):
        return None

    async def save_saga(self, driver):
        saved_nodes.append(self.model_copy())

    monkeypatch.setattr(NextEpisodeEdge, 'save', save_edge)
    monkeypatch.setattr(HasEpisodeEdge, 'save', save_edge)
    monkeypatch.setattr(SagaNode, 'save', save_saga)
    monkeypatch.setattr(
        SagaNode, 'get_by_uuid', AsyncMock(side_effect=lambda driver, uuid: stored.model_copy())
    )
    # The lookup by name returns no summary field, as the query of `_find_saga` does.
    h.graphiti._find_saga = AsyncMock(
        return_value=SagaNode(
            uuid='saga1',
            name='thread',
            group_id=GROUP,
            created_at=T0,
            first_episode_uuid='ep0',
            last_episode_uuid='ep0',
        )
    )
    h.graphiti._saga_get_previous_episode_uuid = AsyncMock(return_value='ep0')
    # A caller can give a saga object that does not have the stored fields.
    saga = 'thread' if saga_input == 'name' else SagaNode(uuid='saga1', name='x', group_id=GROUP)

    unit = await h.admit('u1')
    change_set = await h.analyze(unit)
    await h.commit(unit, change_set, saga=saga)

    (written,) = saved_nodes
    expected = stored.model_dump()
    expected['last_episode_uuid'] = 'ep-u1'
    assert written.model_dump() == expected


async def test_c7b_hook_output_is_the_stored_final_change_set(monkeypatch):
    seen_by_hook: list[int] = []

    async def drop_edges(change_set: ChangeSet) -> ChangeSet:
        seen_by_hook.append(len(change_set.create_edges))
        change_set.create_edges = []
        return change_set

    h = Harness(monkeypatch, before_commit_write=drop_edges)
    stored_people(h)
    unit, result = await h.run(
        'u1', create_edges=[make_edge('e1', 'ana', 'acme', 'Ana works at Acme')]
    )

    assert seen_by_hook == [1]
    assert result.change_set.create_edges == []
    assert h.graph.edges == {}
    stored_final = ChangeSet.from_bytes(h.store._groups[GROUP].units[unit.seq].payload)
    assert stored_final.create_edges == []


async def test_c7b_hook_can_reject_the_change_set(monkeypatch):
    async def reject(change_set: ChangeSet) -> ChangeSet:
        if change_set.unit_id == 'u1':
            raise ChangeSetRejected('not permitted')
        return change_set

    h = Harness(monkeypatch, before_commit_write=reject)
    unit_1 = await h.admit('u1')
    change_set_1 = await h.analyze(unit_1, create_nodes=[make_node('a1', 'Acme')])
    with pytest.raises(ChangeSetRejected):
        await h.commit(unit_1, change_set_1)

    assert h.store._groups[GROUP].units[unit_1.seq].status == STATUS_FAILED
    assert h.graph.nodes == {}
    assert 'write:bulk' not in h.graph.log
    # The position of the unit is released.
    _, result_2 = await h.run('u2', create_nodes=[make_node('a2', 'Acme')])
    assert result_2.report['delta.unit_count'] == 0
    assert set(h.graph.nodes) == {'a2'}


async def test_c7c_final_change_set_is_stored_before_the_first_graph_write(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('ana', 'Ana'), make_node('acme', 'Acme', Y))
    h.graph.add(make_edge('e0', 'ana', 'acme', 'Ana works at Acme'))
    await h.run(
        'u1',
        create_nodes=[make_node('a1', 'Zed', Z)],
        operations=[AddEpisodesToEdge(edge_uuid='e0', episode_uuids=['ep-u1'])],
    )

    writes = [entry for entry in h.graph.log if entry.startswith(('write:', 'store:'))]
    assert writes == ['store:save_final', 'write:bulk', 'write:edge_update', 'store:finish_commit']


# --- C8 --------------------------------------------------------------------------


async def test_c8_operations_apply_to_the_current_state_of_a_stored_edge(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    h.graph.add(
        make_edge(
            'e0', 'ana', 'acme', 'Ana works at Acme', episodes=['ep0'], attributes={'role': 'a'}
        )
    )
    await commit_two(
        h,
        {
            'operations': [
                EndEdge(edge_uuid='e0', invalid_at=T1, expired_at=T1),
                SetEdgeAttributes(
                    edge_uuid='e0', base_attributes={'role': 'a'}, attributes={'role': 'c'}
                ),
            ]
        },
        {
            # The analysis of unit 2 read the edge before the commit of unit 1.
            'operations': [
                EndEdge(edge_uuid='e0', invalid_at=T2, expired_at=T2),
                AddEpisodesToEdge(edge_uuid='e0', episode_uuids=['ep-u2', 'ep0']),
                SetEdgeAttributes(
                    edge_uuid='e0',
                    base_attributes={'role': 'a'},
                    attributes={'role': 'b', 'level': 1},
                ),
            ]
        },
    )

    edge = h.graph.edges['e0']
    # The earlier end dates of unit 1 stay.
    assert edge.invalid_at == T1
    assert edge.expired_at == T1
    assert edge.episodes == ['ep0', 'ep-u2']
    # The key that unit 1 changed keeps its value. The key that only unit 2 set applies.
    assert edge.attributes == {'role': 'c', 'level': 1}


async def test_c8_only_if_open_does_not_end_an_edge_that_is_ended(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    h.graph.add(make_edge('e0', 'ana', 'acme', 'Ana works at Acme', expired_at=T0, invalid_at=T0))
    _, result = await h.run(
        'u1',
        operations=[EndEdge(edge_uuid='e0', invalid_at=T1, expired_at=T1, only_if_open=True)],
    )

    assert h.graph.edges['e0'].expired_at == T0
    assert result.edges == []
    assert 'write:edge_update' not in h.graph.log


async def test_c8_drops_an_operation_on_an_edge_that_does_not_exist(monkeypatch):
    h = Harness(monkeypatch)
    _, result = await h.run(
        'u1', operations=[AddEpisodesToEdge(edge_uuid='deleted', episode_uuids=['ep-u1'])]
    )

    assert result.report['operation.dropped'] == 1
    assert result.report['operation.applied'] == 0
    assert h.graph.edges == {}


# --- C9 --------------------------------------------------------------------------


async def test_c9_each_write_has_a_time_limit_below_the_lease(monkeypatch):
    h = Harness(monkeypatch, lease_seconds=30.0, write_margin_seconds=4.0)
    stored_people(h)
    h.graph.add(make_edge('e0', 'ana', 'acme', 'Ana works at Acme'))
    await h.run('u1', operations=[AddEpisodesToEdge(edge_uuid='e0', episode_uuids=['ep-u1'])])

    assert len(h.graph.write_timeouts) == 2
    assert all(timeout is not None and 0 < timeout <= 26.0 for timeout in h.graph.write_timeouts)


async def test_c9_update_does_not_create_again_an_edge_that_a_writer_deleted(monkeypatch):
    h = Harness(monkeypatch)
    stored_people(h)
    h.graph.add(make_edge('e0', 'ana', 'acme', 'Ana works at Acme'))
    # A writer outside the sequence deletes the edge between the read and the write.
    h.graph.before_edge_update = lambda: h.graph.edges.pop('e0')
    await h.run('u1', operations=[AddEpisodesToEdge(edge_uuid='e0', episode_uuids=['ep-u1'])])

    assert 'write:edge_update' in h.graph.log
    assert h.graph.edges == {}


async def test_c9_check_of_the_turn_renews_the_deadline_before_a_write(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock, lease_seconds=30.0, write_margin_seconds=4.0)
    unit = await h.admit('u1')
    change_set = await h.analyze(unit, create_nodes=[make_node('a1', 'Acme')])
    lease = UnitLease(h.store, h.config, GROUP, unit.seq, clock=clock)

    class SlowReader:
        def __getattr__(self, name):
            return getattr(h.graph, name)

        async def node_candidates(self, nodes, identity_properties=None):
            clock.advance(27.0)
            return await h.graph.node_candidates(nodes, identity_properties)

    # The unit has the turn. The heartbeat with its fence gives the write a new deadline.
    await h.commit(unit, change_set, lease=lease, reader=SlowReader())

    assert set(h.graph.nodes) == {'a1'}
    assert lease.fence == 1


async def test_c9_no_write_starts_after_the_local_deadline(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock, lease_seconds=30.0, write_margin_seconds=4.0)
    unit = await h.admit('u1')
    change_set = await h.analyze(unit, create_nodes=[make_node('a1', 'Acme')])
    lease = UnitLease(h.store, h.config, GROUP, unit.seq, clock=clock)

    class SlowReader:
        """The repair takes more time than the lease."""

        def __getattr__(self, name):
            return getattr(h.graph, name)

        async def node_candidates(self, nodes, identity_properties=None):
            clock.advance(27.0)
            return await h.graph.node_candidates(nodes, identity_properties)

    # The store gives no answer to the check of the turn before the write. The local
    # deadline then stops the write.
    monkeypatch.setattr(h.store, 'heartbeat', AsyncMock(side_effect=RuntimeError('no answer')))
    with pytest.raises(UnitExpiredError):
        await h.commit(unit, change_set, lease=lease, reader=SlowReader())

    assert 'write:bulk' not in h.graph.log
    assert h.graph.nodes == {}
    # The final change set is stored. A later unit applies it after the deadline.
    assert h.store._groups[GROUP].units[unit.seq].status == STATUS_APPLYING


# --- C10 -------------------------------------------------------------------------


async def test_c10_commit_moves_the_watermark_and_registers_the_nodes(monkeypatch):
    h = Harness(monkeypatch)
    unit, result = await h.run(
        'u1', create_nodes=[make_node('a1', 'Acme'), make_node('b1', 'Bo', Y)]
    )

    assert h.store._groups[GROUP].watermark == unit.seq
    assert await h.store.ready_enrich(GROUP, unit.seq) == ['a1', 'b1']
    assert h.graph.log[-1] == 'store:finish_commit'
    assert h.graph.episodes['ep-u1'].entity_edges == []
    (span,) = h.tracer.named('ingest.commit')
    assert span.attributes['ingest.seq'] == unit.seq
    for key, value in result.report.items():
        assert span.attributes[key] == value
