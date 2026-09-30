"""Regression tests for fact invalidation when a new episode negates a prior fact.

Issue: https://github.com/getzep/graphiti/issues/1841
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts.dedupe_edges import resolve_edge
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance.edge_operations import (
    merge_edge_invalidation_candidates,
    resolve_extracted_edge,
    resolve_extracted_edges,
)

ASSIGNED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
RELEASED_AT = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _edge(
    *,
    uuid: str,
    fact: str,
    name: str,
    valid_at: datetime | None,
    invalid_at: datetime | None = None,
    source_node_uuid: str = 'kiran',
    target_node_uuid: str = 'payments',
    expired_at: datetime | None = None,
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid=source_node_uuid,
        target_node_uuid=target_node_uuid,
        name=name,
        group_id='group_1',
        fact=fact,
        episodes=['episode-1'],
        created_at=valid_at or ASSIGNED_AT,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


def _episode(valid_at: datetime, content: str) -> EpisodicNode:
    return EpisodicNode(
        uuid='episode-2',
        name='Release',
        group_id='group_1',
        source='message',
        source_description='test',
        content=content,
        valid_at=valid_at,
    )


def _assigned_edge() -> EntityEdge:
    return _edge(
        uuid='assigned',
        name='ASSIGNED',
        fact='Kiran is assigned to the Payments Project',
        valid_at=ASSIGNED_AT,
    )


def _coffee_edge() -> EntityEdge:
    return _edge(
        uuid='coffee',
        name='LIKES',
        fact='Kiran likes coffee',
        valid_at=ASSIGNED_AT,
        target_node_uuid='coffee',
    )


@pytest.mark.asyncio
async def test_negating_fact_invalidates_prior_edge_and_leaves_unrelated_fact():
    """An end-only release contradicts the open assignment and does not touch other facts.

    The release fact carries invalid_at and no valid_at, which is how a termination
    is extracted. The prior assignment must close at that invalid_at.
    """
    assigned = _assigned_edge()
    coffee = _coffee_edge()
    released = _edge(
        uuid='released',
        name='RELEASED',
        fact='Kiran was released from the Payments Project',
        valid_at=None,
        invalid_at=RELEASED_AT,
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': [0]}
    )

    resolved, invalidated, duplicates = await resolve_extracted_edge(
        llm_client,
        released,
        [assigned, coffee],
        [],
        _episode(RELEASED_AT, released.fact),
    )

    assert resolved.uuid == 'released'
    assert duplicates == []
    assert invalidated == [assigned]
    assert assigned.invalid_at == RELEASED_AT
    assert assigned.expired_at is not None
    assert coffee.invalid_at is None
    assert coffee.expired_at is None


@pytest.mark.asyncio
async def test_unrelated_fact_does_not_invalidate_prior_edge():
    """A new fact the model does not mark as a contradiction leaves existing facts open."""
    assigned = _assigned_edge()
    coffee = _edge(
        uuid='coffee-new',
        name='LIKES',
        fact='Kiran likes coffee',
        valid_at=RELEASED_AT,
        source_node_uuid='kiran',
        target_node_uuid='coffee',
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )

    resolved, invalidated, duplicates = await resolve_extracted_edge(
        llm_client,
        coffee,
        [assigned],
        [],
        _episode(RELEASED_AT, coffee.fact),
    )

    assert resolved.uuid == 'coffee-new'
    assert invalidated == []
    assert duplicates == []
    assert assigned.invalid_at is None
    assert assigned.expired_at is None


@pytest.mark.asyncio
async def test_later_valid_fact_still_invalidates_at_its_start():
    """The existing valid_at path is unchanged when the new fact has a later start."""
    assigned = _assigned_edge()
    moved = _edge(
        uuid='lived',
        name='LIVED_IN',
        fact='Kiran lived in Whitefield',
        valid_at=RELEASED_AT,
        invalid_at=RELEASED_AT + timedelta(days=1),
        target_node_uuid='whitefield',
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': [0]}
    )

    _, invalidated, _ = await resolve_extracted_edge(
        llm_client,
        moved,
        [assigned],
        [],
        _episode(RELEASED_AT, moved.fact),
    )

    assert invalidated == [assigned]
    assert assigned.invalid_at == RELEASED_AT
    assert assigned.expired_at is not None


@pytest.mark.asyncio
async def test_end_only_negation_does_not_extend_an_earlier_end():
    """A later negation must not push an already-recorded end further out."""
    ended_at = ASSIGNED_AT + timedelta(days=10)
    assigned = _edge(
        uuid='assigned',
        name='ASSIGNED',
        fact='Kiran is assigned to the Payments Project',
        valid_at=ASSIGNED_AT,
        invalid_at=ended_at,
        expired_at=ended_at,
    )
    released = _edge(
        uuid='released',
        name='RELEASED',
        fact='Kiran was released from the Payments Project',
        valid_at=None,
        invalid_at=RELEASED_AT,
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': [0]}
    )

    _, invalidated, _ = await resolve_extracted_edge(
        llm_client,
        released,
        [assigned],
        [],
        _episode(RELEASED_AT, released.fact),
    )

    assert invalidated == []
    assert assigned.invalid_at == ended_at
    assert assigned.expired_at == ended_at


@pytest.mark.asyncio
async def test_end_only_negation_does_not_invalidate_fact_that_starts_later():
    """A negation cannot close a fact that did not begin until after the negation."""
    later = _edge(
        uuid='later-assignment',
        name='ASSIGNED',
        fact='Kiran is assigned to the Payments Project',
        valid_at=RELEASED_AT + timedelta(days=10),
    )
    released = _edge(
        uuid='released',
        name='RELEASED',
        fact='Kiran was released from the Payments Project',
        valid_at=None,
        invalid_at=RELEASED_AT,
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': [0]}
    )

    _, invalidated, _ = await resolve_extracted_edge(
        llm_client,
        released,
        [later],
        [],
        _episode(RELEASED_AT, released.fact),
    )

    assert invalidated == []
    assert later.invalid_at is None
    assert later.expired_at is None


def test_merge_includes_open_same_endpoint_edges_search_missed():
    assigned = _assigned_edge()
    ended = _edge(
        uuid='ended',
        name='ASSIGNED',
        fact='Kiran was assigned to the Payments Project until last year',
        valid_at=ASSIGNED_AT - timedelta(days=400),
        invalid_at=ASSIGNED_AT - timedelta(days=30),
    )
    already_related = _edge(
        uuid='related',
        name='WORKS_ON',
        fact='Kiran works on the Payments Project',
        valid_at=ASSIGNED_AT,
    )
    searched = _edge(
        uuid='searched',
        name='MENTIONED',
        fact='Kiran mentioned the Payments Project',
        valid_at=ASSIGNED_AT,
    )

    merged = merge_edge_invalidation_candidates(
        [already_related],
        [searched, already_related],
        [assigned, ended, already_related],
    )

    assert [edge.uuid for edge in merged] == ['searched', 'assigned']


@pytest.mark.asyncio
async def test_search_miss_still_invalidates_same_endpoint_negation(monkeypatch):
    """Open same-endpoint facts stay in the contradiction pool when hybrid search misses them."""
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    assigned = _assigned_edge()
    chess = _edge(
        uuid='chess',
        name='PLAYS',
        fact='Kiran plays chess',
        valid_at=ASSIGNED_AT,
    )
    released = _edge(
        uuid='released',
        name='RELEASED',
        fact='Kiran was released from the Payments Project',
        valid_at=None,
        invalid_at=RELEASED_AT,
    )

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[assigned, chess]))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))

    async def generate_response(prompt, *args, **kwargs):
        content = prompt[1].content
        assert 'Kiran is assigned to the Payments Project' in content
        assert 'Kiran plays chess' in content
        assert kwargs['prompt_name'] == 'dedupe_edges.resolve_edge'
        return {'duplicate_facts': [], 'contradicted_facts': [0]}

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=generate_response)
    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )
    source = EntityNode(uuid='kiran', name='Kiran', group_id='group_1', labels=['Entity'])
    target = EntityNode(
        uuid='payments', name='Payments Project', group_id='group_1', labels=['Entity']
    )

    resolved_edges, invalidated, new_edges = await resolve_extracted_edges(
        clients,
        [released],
        _episode(RELEASED_AT, released.fact),
        [source, target],
        {},
        {},
    )

    assert [edge.uuid for edge in resolved_edges] == ['released']
    assert [edge.uuid for edge in new_edges] == ['released']
    assert invalidated == [assigned]
    assert assigned.invalid_at == RELEASED_AT
    assert assigned.expired_at is not None
    assert chess.invalid_at is None
    assert chess.expired_at is None
    llm_client.generate_response.assert_awaited_once()


def test_resolve_edge_prompt_treats_removal_as_contradiction_and_ignores_unrelated_facts():
    messages = resolve_edge(
        {
            'existing_edges': [{'idx': 0, 'fact': 'Kiran is assigned to project A'}],
            'new_edge': 'Kiran was removed from project A',
            'edge_invalidation_candidates': [],
        }
    )
    content = '\n'.join(message.content for message in messages)

    assert 'Kiran was removed from project A' in content
    assert 'contradicted_facts=[3]' in content
    assert 'Kiran likes coffee' in content
    assert 'do not contradict the assignment' in content
