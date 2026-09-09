"""Repeated wording must not erase a distinct occurrence of a fact."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations as ops

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
FACT = 'Kiran is assigned to Payments.'


def fact(start, end=None, *, wording=FACT, reference=None):
    return EntityEdge(
        source_node_uuid='person',
        target_node_uuid='project',
        name='ASSIGNED',
        group_id='temporal-test',
        fact=wording,
        episodes=['source-episode'],
        created_at=START + timedelta(days=30),
        valid_at=START + timedelta(days=start) if start is not None else None,
        invalid_at=START + timedelta(days=end) if end is not None else None,
        expired_at=START + timedelta(days=20) if end is not None else None,
        reference_time=reference,
    )


@pytest.fixture
def episode():
    return EpisodicNode(
        uuid='new-episode',
        name='Temporal evidence',
        content='Supplied temporal fixture.',
        group_id='temporal-test',
        source='message',
        source_description='unit test',
        valid_at=START + timedelta(days=30),
    )


@pytest.mark.parametrize(
    'old_window,new_window',
    [
        ((0, 5), (10, None)),
        ((10, None), (0, 5)),
        ((0, 10), (10, None)),
        ((10, None), (0, 10)),
    ],
)
@pytest.mark.parametrize('exact_wording', [False, True])
async def test_duplicate_decision_cannot_erase_disjoint_occurrence(
    episode,
    old_window,
    new_window,
    exact_wording,
):
    old = fact(*old_window)
    incoming = fact(*new_window, wording=FACT if exact_wording else 'Kiran works on Payments.')
    before = old.model_dump()
    client = MagicMock()
    client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [0], 'contradicted_facts': []}
    )

    resolved, invalidated, duplicates = await ops.resolve_extracted_edge(
        client,
        incoming,
        [old],
        [],
        episode,
    )

    assert resolved is incoming
    assert old.model_dump() == before
    assert resolved.valid_at == incoming.valid_at
    assert resolved.invalid_at == incoming.invalid_at
    assert invalidated == duplicates == []


@pytest.mark.parametrize(
    'old_window,new_window',
    [
        ((0, None), (5, None)),
        ((0, 10), (5, 8)),
        ((None, None), (None, None)),
    ],
)
async def test_overlapping_restatement_retains_existing_fast_path(episode, old_window, new_window):
    old, incoming = fact(*old_window), fact(*new_window)
    client = MagicMock()
    client.generate_response = AsyncMock(side_effect=AssertionError('Unexpected model call'))

    resolved, invalidated, duplicates = await ops.resolve_extracted_edge(
        client,
        incoming,
        [old],
        [],
        episode,
    )

    assert resolved is old
    assert resolved.episodes.count(episode.uuid) == 1
    assert invalidated == duplicates == []
    client.generate_response.assert_not_awaited()


@pytest.mark.parametrize('windows', [((0, 5), (10, None)), ((0, 10), (10, None))])
@pytest.mark.parametrize('reverse', [False, True])
async def test_batch_retains_separate_validity_periods(monkeypatch, episode, windows, reverse):
    edges = [fact(*window) for window in windows]
    if reverse:
        edges.reverse()
    result = await run_batch(monkeypatch, episode, edges)
    assert [edge.uuid for edge in result[0]] == [edge.uuid for edge in edges]
    assert [edge.uuid for edge in result[2]] == [edge.uuid for edge in edges]
    assert result[1] == []


@pytest.mark.parametrize('dated', [False, True])
async def test_batch_preserves_reference_until_dates_are_known(monkeypatch, episode, dated):
    # Combined extraction may leave dates unresolved while retaining the source
    # episode's reference clock. Different clocks must reach timestamp extraction.
    start, end = (0, 5) if dated else (None, None)
    edges = [
        fact(start, end, reference=START),
        fact(start, end, reference=START + timedelta(days=10)),
    ]
    resolved, invalidated, new = await run_batch(monkeypatch, episode, edges)
    assert len(resolved) == len(new) == (1 if dated else 2)
    assert invalidated == []


async def run_batch(monkeypatch, episode, edges):
    client = MagicMock()
    client.generate_response = AsyncMock(return_value={'valid_at': None, 'invalid_at': None})
    clients = SimpleNamespace(driver=MagicMock(), llm_client=client, embedder=MagicMock())
    nodes = [
        EntityNode(uuid='person', name='Kiran', group_id='temporal-test'),
        EntityNode(uuid='project', name='Payments', group_id='temporal-test'),
    ]
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))
    monkeypatch.setattr(ops, 'search', AsyncMock(return_value=SearchResults()))
    monkeypatch.setattr(ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    return await ops.resolve_extracted_edges(clients, edges, episode, nodes, {}, {})
