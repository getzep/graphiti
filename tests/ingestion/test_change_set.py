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

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.ingestion.change_set import (
    AddEpisodesToEdge,
    ChangeSet,
    EndEdge,
    PatchNode,
    ResolutionTrace,
    SagaLink,
    SetEdgeAttributes,
    apply_edge_operation,
    merge_edge_attributes,
)
from graphiti_core.ingestion.config import ConcurrentIngestionConfig
from graphiti_core.ingestion.state_store import InMemoryIngestionStateStore
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_edge(**values) -> EntityEdge:
    defaults = {
        'uuid': 'e1',
        'group_id': 'g',
        'source_node_uuid': 'a',
        'target_node_uuid': 'b',
        'name': 'WORKS_AT',
        'fact': 'Ana works at Acme',
        'created_at': T0,
        'episodes': ['ep0'],
    }
    return EntityEdge(**{**defaults, **values})


def test_change_set_round_trip_keeps_embeddings_and_datetimes():
    episode = EpisodicNode(
        uuid='ep1',
        name='episode',
        group_id='g',
        source=EpisodeType.text,
        source_description='test',
        content='Ana works at Acme',
        valid_at=T0,
        created_at=T0,
        entity_edges=['e1'],
    )
    node = EntityNode(
        uuid='n1',
        name='Ana',
        group_id='g',
        labels=['Entity', 'Person'],
        created_at=T0,
        name_embedding=[0.1, 0.2, 0.3],
        attributes={'email': 'ana@example.com'},
    )
    edge = make_edge(fact_embedding=[0.4, 0.5], valid_at=T0, attributes={'role': 'engineer'})
    change_set = ChangeSet(
        group_id='g',
        unit_id='u1',
        seq=3,
        snapshot_seq=1,
        analysis_started_at=12.5,
        episodes=[episode],
        create_nodes=[node],
        create_edges=[edge],
        episodic_edges=[
            EpisodicEdge(source_node_uuid='ep1', target_node_uuid='n1', group_id='g', created_at=T0)
        ],
        operations=[
            AddEpisodesToEdge(edge_uuid='e9', episode_uuids=['ep1']),
            EndEdge(edge_uuid='e8', invalid_at=T0, expired_at=T0 + timedelta(days=1)),
            SetEdgeAttributes(edge_uuid='e7', base_attributes={'a': 1}, attributes={'a': 2}),
            PatchNode(node_uuid='n9', labels=['Entity', 'Person'], attributes={'email': 'x'}),
        ],
        node_uuids=['n1', 'n9'],
        node_first_episode={'n1': 'ep1', 'n9': 'ep1'},
        seen_node_candidates={'n1': ['n9']},
        seen_duplicate_candidates={'e1': ['e9']},
        seen_invalidation_candidates={'e1': ['e8']},
        uuid_map={'x': 'n9'},
        saga_links=[
            SagaLink(saga_uuid='s1', episode_uuid='ep1', has_episode_edge_uuid='h1', created_at=T0)
        ],
    )

    payload = change_set.to_bytes()
    restored = ChangeSet.from_bytes(payload)

    assert isinstance(payload, bytes)
    # `Edge.__eq__` does not compare edges, so the test compares the dumps.
    assert restored.model_dump() == change_set.model_dump()
    assert restored.create_nodes[0].name_embedding == [0.1, 0.2, 0.3]
    assert restored.create_edges[0].fact_embedding == [0.4, 0.5]
    assert restored.create_edges[0].valid_at == T0
    assert [type(operation) for operation in restored.operations] == [
        AddEpisodesToEdge,
        EndEdge,
        SetEdgeAttributes,
        PatchNode,
    ]
    assert list(restored.patches_by_node()) == ['n9']
    assert len(restored.edge_operations()) == 3


def apply_twice(edge: EntityEdge, operation) -> tuple[bool, bool]:
    first = apply_edge_operation(edge, operation)
    state = edge.model_dump()
    second = apply_edge_operation(edge, operation)
    assert edge.model_dump() == state
    return first, second


def test_add_episodes_keeps_order_and_adds_no_duplicate():
    edge = make_edge(episodes=['ep0', 'ep1'])
    assert apply_twice(edge, AddEpisodesToEdge(edge_uuid='e1', episode_uuids=['ep1', 'ep2'])) == (
        True,
        False,
    )
    assert edge.episodes == ['ep0', 'ep1', 'ep2']


def test_end_edge_sets_dates_on_open_edge():
    edge = make_edge()
    operation = EndEdge(edge_uuid='e1', invalid_at=T0, expired_at=T0 + timedelta(days=2))
    assert apply_twice(edge, operation) == (True, False)
    assert (edge.invalid_at, edge.expired_at) == (T0, T0 + timedelta(days=2))


def test_end_edge_keeps_earlier_end_date():
    edge = make_edge(invalid_at=T0, expired_at=T0 + timedelta(days=1))
    later = EndEdge(
        edge_uuid='e1', invalid_at=T0 + timedelta(days=5), expired_at=T0 + timedelta(days=6)
    )
    assert apply_twice(edge, later) == (False, False)
    assert (edge.invalid_at, edge.expired_at) == (T0, T0 + timedelta(days=1))


def test_end_edge_replaces_later_end_date():
    edge = make_edge(invalid_at=T0 + timedelta(days=5), expired_at=T0 + timedelta(days=6))
    earlier = EndEdge(edge_uuid='e1', invalid_at=T0, expired_at=T0 + timedelta(days=9))
    assert apply_twice(edge, earlier) == (True, False)
    # `expired_at` has a value, so it stays.
    assert (edge.invalid_at, edge.expired_at) == (T0, T0 + timedelta(days=6))


def test_end_edge_only_if_open_does_nothing_on_expired_edge():
    edge = make_edge(expired_at=T0 + timedelta(days=1))
    operation = EndEdge(
        edge_uuid='e1', invalid_at=T0, expired_at=T0 + timedelta(days=2), only_if_open=True
    )
    assert apply_twice(edge, operation) == (False, False)
    assert edge.invalid_at is None


def test_end_edge_with_no_invalid_at_sets_expired_at_only():
    edge = make_edge(invalid_at=T0)
    operation = EndEdge(edge_uuid='e1', expired_at=T0 + timedelta(days=2), only_if_open=True)
    assert apply_twice(edge, operation) == (True, False)
    assert (edge.invalid_at, edge.expired_at) == (T0, T0 + timedelta(days=2))


def test_merge_edge_attributes_by_key():
    base = {'a': 1, 'b': 1, 'd': 1}
    new = {'a': 2, 'b': 2, 'c': 2}
    # A different writer changed `a` and added `e` after the analysis read the edge.
    current = {'a': 9, 'b': 1, 'd': 1, 'e': 9}
    merged = merge_edge_attributes(base, new, current)
    # `a`: changed by the other writer, stays. `b`: not changed, gets the new value.
    # `c`: new key. `d`: removed by the unit. `e`: added by the other writer, stays.
    assert merged == {'a': 9, 'b': 2, 'c': 2, 'e': 9}


def test_merge_edge_attributes_is_a_replace_when_no_writer_changed_the_edge():
    base = {'a': 1, 'b': 1}
    new = {'a': 2, 'c': 3}
    assert merge_edge_attributes(base, new, dict(base)) == new


def test_set_edge_attributes_two_times_gives_same_record():
    edge = make_edge(attributes={'a': 9, 'b': 1})
    operation = SetEdgeAttributes(
        edge_uuid='e1', base_attributes={'a': 1, 'b': 1}, attributes={'a': 2, 'b': 2}
    )
    assert apply_twice(edge, operation) == (True, False)
    assert edge.attributes == {'a': 9, 'b': 2}


def test_trace_makes_operations_from_changes_to_stored_edges():
    duplicate = make_edge(uuid='dup', attributes={'role': 'old'})
    contradicted = make_edge(uuid='old', valid_at=T0)
    resolved_stored = make_edge(uuid='res', invalid_at=T0 + timedelta(days=3))
    untouched = make_edge(uuid='same')
    trace = ResolutionTrace()
    trace.watch_stored_edges([duplicate, contradicted, resolved_stored, untouched])

    duplicate.episodes.append('ep1')
    duplicate.attributes = {'role': 'new'}
    contradicted.invalid_at = T0 + timedelta(days=1)
    contradicted.expired_at = T0 + timedelta(days=9)
    trace.mark_ended_as_candidate([contradicted])
    resolved_stored.expired_at = T0 + timedelta(days=9)

    operations = trace.collect_edge_operations()

    assert operations == [
        AddEpisodesToEdge(edge_uuid='dup', episode_uuids=['ep1']),
        SetEdgeAttributes(
            edge_uuid='dup', base_attributes={'role': 'old'}, attributes={'role': 'new'}
        ),
        EndEdge(
            edge_uuid='old',
            invalid_at=T0 + timedelta(days=1),
            expired_at=T0 + timedelta(days=9),
            only_if_open=False,
        ),
        EndEdge(
            edge_uuid='res', invalid_at=None, expired_at=T0 + timedelta(days=9), only_if_open=True
        ),
    ]
    assert trace.operations == operations
    assert trace.collect_edge_operations() == []


def test_trace_records_node_resolution():
    extracted = EntityNode(
        uuid='new', name='Ana', group_id='g', labels=['Entity', 'Person'], attributes={'k': 'v'}
    )
    stored = EntityNode(uuid='stored', name='Ana', group_id='g', labels=['Entity'])
    trace = ResolutionTrace()
    trace.record_node_candidates('new', [stored])
    trace.record_node_resolution(extracted, 'new')
    assert trace.operations == []
    trace.record_node_resolution(extracted, 'stored')
    assert trace.seen_node_candidates == {'new': ['stored']}
    assert trace.stored_node_uuids == {'stored'}
    assert trace.operations == [
        PatchNode(node_uuid='stored', labels=['Entity', 'Person'], attributes={'k': 'v'})
    ]


def test_config_defaults_and_validation():
    config = ConcurrentIngestionConfig(window=4)
    assert (config.lease_seconds, config.write_margin_seconds) == (60.0, 5.0)
    assert (config.poll_interval_seconds, config.index_lag_seconds) == (0.25, 300.0)
    assert config.max_apply_attempts == 5
    store = config.get_state_store()
    assert isinstance(store, InMemoryIngestionStateStore)
    assert config.get_state_store() is store

    with pytest.raises(ValidationError):
        ConcurrentIngestionConfig()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ConcurrentIngestionConfig(window=0)
    with pytest.raises(ValidationError):
        ConcurrentIngestionConfig(window=1, lease_seconds=15.0, write_margin_seconds=5.0)
    with pytest.raises(ValidationError):
        ConcurrentIngestionConfig(window=1, state_store=object())
