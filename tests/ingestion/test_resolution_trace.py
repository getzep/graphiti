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

# Tests for the resolution refactors: the new parameters keep the behavior of today
# at their defaults, and `ResolutionTrace` records what the resolution saw and changed.

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from graphiti_core.edges import EntityEdge
from graphiti_core.ingestion.change_set import (
    AddEpisodesToEdge,
    EndEdge,
    PatchNode,
    ResolutionTrace,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations as edge_ops
from graphiti_core.utils.maintenance import node_operations as node_ops
from graphiti_core.utils.maintenance.edge_operations import (
    find_exact_duplicate_edge,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.node_operations import (
    _semantic_candidate_search,
    resolve_extracted_nodes,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=10)


def make_edge(uuid: str, source: str, target: str, fact: str, **values) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id='g',
        source_node_uuid=source,
        target_node_uuid=target,
        name='RELATES',
        fact=fact,
        created_at=T0,
        fact_embedding=[0.1, 0.2],
        **values,
    )


def make_episode() -> EpisodicNode:
    return EpisodicNode(
        uuid='ep1',
        name='episode',
        group_id='g',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=T1,
        created_at=T1,
    )


async def immediate_gather(*aws, max_coroutines=None):
    return [await aw for aw in aws]


async def run_edge_resolution(monkeypatch, trace: ResolutionTrace | None):
    """One extracted edge. The model marks candidate 0 as a duplicate and candidate 1
    as contradicted."""
    duplicate = make_edge('dup', 'a', 'b', 'Ana is employed by Acme', valid_at=T1, episodes=['ep0'])
    contradicted = make_edge('old', 'a', 'c', 'Ana works at Bolt', valid_at=T0, episodes=['ep0'])
    extracted = make_edge('new', 'a', 'b', 'Ana works at Acme', valid_at=T1)

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))
    # The first search is the duplicate search. The second search is graph-wide.
    monkeypatch.setattr(
        edge_ops,
        'search',
        AsyncMock(
            side_effect=[
                SearchResults(edges=[duplicate]),
                SearchResults(edges=[duplicate, contradicted]),
            ]
        ),
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [0], 'contradicted_facts': [1]}
    )
    clients = SimpleNamespace(
        driver=MagicMock(), llm_client=llm_client, embedder=MagicMock(), cross_encoder=MagicMock()
    )
    nodes = [EntityNode(uuid=uuid, name=uuid, group_id='g') for uuid in ('a', 'b', 'c')]
    kwargs = {'trace': trace} if trace is not None else {}

    resolved, invalidated, new = await resolve_extracted_edges(
        clients, [extracted], make_episode(), nodes, {}, {('Entity', 'Entity'): []}, **kwargs
    )
    return resolved, invalidated, new, llm_client


def dump_edges(edges: list[EntityEdge]) -> list[dict]:
    # `expired_at` is the time of the call, so the comparison uses only if it is set.
    return [
        {**edge.model_dump(exclude={'expired_at'}), 'expired': edge.expired_at is not None}
        for edge in edges
    ]


async def test_edge_resolution_returns_same_values_with_and_without_trace(monkeypatch):
    plain = await run_edge_resolution(monkeypatch, None)
    traced = await run_edge_resolution(monkeypatch, ResolutionTrace())

    for plain_edges, traced_edges in zip(plain[:3], traced[:3], strict=True):
        assert dump_edges(plain_edges) == dump_edges(traced_edges)
    assert plain[3].generate_response.await_count == traced[3].generate_response.await_count == 1
    assert [edge.uuid for edge in plain[0]] == ['dup']
    assert [edge.uuid for edge in plain[1]] == ['old']
    assert plain[2] == []


async def test_edge_resolution_trace_records_candidates_and_operations(monkeypatch):
    trace = ResolutionTrace()
    _, invalidated, _, _ = await run_edge_resolution(monkeypatch, trace)

    assert trace.seen_duplicate_candidates == {'new': ['dup']}
    assert trace.seen_invalidation_candidates == {'new': ['old']}
    assert trace.operations == [
        AddEpisodesToEdge(edge_uuid='dup', episode_uuids=['ep1']),
        EndEdge(
            edge_uuid='old',
            invalid_at=T1,
            expired_at=invalidated[0].expired_at,
            only_if_open=False,
        ),
    ]


def test_serial_date_rule_ends_a_fact_at_the_earliest_later_contradiction():
    """The date rule of serial ingestion, for a fact with two contradictions that start later."""
    fact = make_edge('new', 'a', 'b', 'Ana works at Acme', valid_at=T0)
    late = make_edge('late', 'a', 'c', 'Ana works at Bolt', valid_at=T1 + timedelta(days=10))
    early = make_edge('early', 'a', 'd', 'Ana works at Core', valid_at=T1)

    edge_ops._apply_temporal_invalidation(fact, [late, early])

    assert fact.invalid_at == T1
    assert fact.expired_at is not None


def test_find_exact_duplicate_edge_rules():
    seen: dict = {}
    first = make_edge('e1', 'a', 'b', 'Ana works at Acme')
    assert find_exact_duplicate_edge(first, seen) is None
    same = make_edge('e2', 'a', 'b', ' ana works at  ACME ')
    assert find_exact_duplicate_edge(same, seen) is first
    reverse_same_name = make_edge('e3', 'b', 'a', 'Ana works at Acme')
    assert find_exact_duplicate_edge(reverse_same_name, seen) is first
    reverse_other_name = make_edge('e4', 'b', 'a', 'Ana works at Acme')
    reverse_other_name.name = 'EMPLOYS'
    assert find_exact_duplicate_edge(reverse_other_name, seen) is None
    other_fact = make_edge('e5', 'a', 'b', 'Ana left Acme')
    assert find_exact_duplicate_edge(other_fact, seen) is None


async def run_node_resolution(monkeypatch, trace: ResolutionTrace | None):
    stored = EntityNode(uuid='stored', name='Ana Lopez', group_id='g', labels=['Entity'])
    other = EntityNode(uuid='other', name='Bolt', group_id='g', labels=['Entity'])
    extracted = [
        EntityNode(
            uuid='x1',
            name='Ana Lopez',
            group_id='g',
            labels=['Entity', 'Person'],
            attributes={'email': 'ana@example.com'},
        ),
        EntityNode(uuid='x2', name='Acme', group_id='g', labels=['Entity']),
    ]
    monkeypatch.setattr(
        node_ops, '_collect_candidate_nodes', AsyncMock(return_value=[[stored, other], []])
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock()
    clients = SimpleNamespace(
        driver=MagicMock(), llm_client=llm_client, embedder=MagicMock(), cross_encoder=MagicMock()
    )
    kwargs = {'trace': trace} if trace is not None else {}
    nodes, uuid_map, pairs = await resolve_extracted_nodes(
        clients, extracted, make_episode(), [], None, **kwargs
    )
    return nodes, uuid_map, pairs, llm_client


async def test_node_resolution_returns_same_values_with_and_without_trace(monkeypatch):
    plain = await run_node_resolution(monkeypatch, None)
    traced = await run_node_resolution(monkeypatch, ResolutionTrace())

    # `created_at` has a default of the current time, so the comparison leaves it out.
    assert [node.model_dump(exclude={'created_at'}) for node in plain[0]] == [
        node.model_dump(exclude={'created_at'}) for node in traced[0]
    ]
    assert plain[1] == traced[1] == {'x1': 'stored', 'x2': 'x2'}
    assert plain[3].generate_response.await_count == traced[3].generate_response.await_count == 0
    # The stored node gets the label and the attribute of the extracted node.
    assert plain[0][0].labels == ['Entity', 'Person']
    assert plain[0][0].attributes == {'email': 'ana@example.com'}


async def test_node_resolution_trace_records_candidates_and_patch(monkeypatch):
    trace = ResolutionTrace()
    await run_node_resolution(monkeypatch, trace)

    assert trace.seen_node_candidates == {'x1': ['stored', 'other'], 'x2': []}
    assert trace.stored_node_uuids == {'stored'}
    # The trace keeps the labels of each candidate at the time of the analysis.
    assert trace.seen_node_labels == {'stored': ['Entity'], 'other': ['Entity']}
    assert trace.operations == [
        PatchNode(
            node_uuid='stored',
            labels=['Entity', 'Person'],
            attributes={'email': 'ana@example.com'},
        )
    ]


async def test_semantic_search_reuses_name_embeddings_only_when_requested(monkeypatch):
    monkeypatch.setattr(node_ops, 'node_similarity_search', AsyncMock(return_value=[]))
    with_embedding = EntityNode(uuid='n1', name='Ana', group_id='g', name_embedding=[1.0, 0.0])
    without_embedding = EntityNode(uuid='n2', name='Acme', group_id='g')

    def make_clients():
        embedder = MagicMock()
        embedder.create_batch = AsyncMock(side_effect=lambda queries: [[0.5, 0.5]] * len(queries))
        return SimpleNamespace(driver=MagicMock(), embedder=embedder)

    # Default: the embedder gets each name, as before the change.
    clients = make_clients()
    await _semantic_candidate_search(clients, [with_embedding, without_embedding])
    clients.embedder.create_batch.assert_awaited_once_with(['Ana', 'Acme'])
    assert node_ops.node_similarity_search.await_args_list[0].args[1] == [0.5, 0.5]

    node_ops.node_similarity_search.reset_mock()
    clients = make_clients()
    await _semantic_candidate_search(
        clients, [with_embedding, without_embedding], reuse_name_embeddings=True
    )
    clients.embedder.create_batch.assert_awaited_once_with(['Acme'])
    vectors = [call.args[1] for call in node_ops.node_similarity_search.await_args_list]
    assert vectors == [[1.0, 0.0], [0.5, 0.5]]

    # No embedder call when each node has an embedding.
    clients = make_clients()
    await _semantic_candidate_search(clients, [with_embedding], reuse_name_embeddings=True)
    clients.embedder.create_batch.assert_not_awaited()
