"""Temporal bookkeeping regressions with deterministic, already classified facts."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.utils.maintenance import edge_operations

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = START + timedelta(days=10)


def make_edge(fact, *, valid_at=START, invalid_at=None, expired_at=None):
    return EntityEdge(
        source_node_uuid='person',
        target_node_uuid='project',
        name='ASSIGNED_TO',
        group_id='temporal-test',
        fact=fact,
        created_at=NOW,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


@pytest.fixture
def episode():
    return EpisodicNode(
        uuid='episode',
        name='Retrospective update',
        content='Known dates are supplied directly by the regression fixture.',
        group_id='temporal-test',
        source='message',
        source_description='unit test',
        valid_at=NOW,
    )


@pytest.fixture
def llm_client():
    client = MagicMock()
    client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )
    return client


@pytest.mark.parametrize('with_context', [False, True])
@pytest.mark.parametrize('previous_expiration', [None, START + timedelta(days=7)])
async def test_ended_fact_expiration_does_not_depend_on_unrelated_context(
    monkeypatch, episode, llm_client, with_context, previous_expiration
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    end = START + timedelta(days=5)
    incoming = make_edge('The assignment ended.', invalid_at=end, expired_at=previous_expiration)
    context = [make_edge('The project has a website.')] if with_context else []

    resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
        llm_client, incoming, [], context, episode
    )

    assert resolved is incoming
    assert resolved.valid_at == START
    assert resolved.invalid_at == end
    assert resolved.expired_at == (previous_expiration or NOW)
    assert invalidated == []
    assert duplicates == []
    assert llm_client.generate_response.await_count == int(with_context)


@pytest.mark.parametrize('previous_expiration', [None, START + timedelta(days=7)])
async def test_finite_older_fact_is_clipped_at_known_later_contradiction(
    monkeypatch, episode, llm_client, previous_expiration
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    incoming = make_edge(
        'Kiran was assigned to Payments.',
        invalid_at=START + timedelta(days=5),
        expired_at=previous_expiration,
    )
    replacement = make_edge('Kiran was released from Payments.', valid_at=START + timedelta(days=3))
    llm_client.generate_response.return_value['contradicted_facts'] = [0]

    resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
        llm_client, incoming, [], [replacement], episode
    )

    assert resolved is incoming
    assert resolved.valid_at == START
    assert resolved.invalid_at == replacement.valid_at
    assert resolved.expired_at == (previous_expiration or NOW)
    assert replacement.invalid_at is None
    assert replacement.expired_at is None
    assert invalidated == []
    assert duplicates == []
    llm_client.generate_response.assert_awaited_once()


@pytest.mark.parametrize(
    'replacement_start', [START + timedelta(days=5), START + timedelta(days=6)]
)
async def test_later_contradiction_cannot_extend_a_closed_interval(
    monkeypatch, episode, llm_client, replacement_start
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    end = START + timedelta(days=5)
    incoming = make_edge('Kiran was assigned to Payments.', invalid_at=end)
    replacement = make_edge('Kiran was released from Payments.', valid_at=replacement_start)
    llm_client.generate_response.return_value['contradicted_facts'] = [0]

    resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
        llm_client, incoming, [], [replacement], episode
    )

    assert resolved.invalid_at == end
    assert resolved.expired_at == NOW
    assert replacement.invalid_at is None
    assert replacement.expired_at is None
    assert invalidated == []
    assert duplicates == []


async def test_unknown_later_fact_start_does_not_supply_an_invalidation_boundary(
    monkeypatch, episode, llm_client
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    end = START + timedelta(days=5)
    incoming = make_edge('Kiran was assigned to Payments.', invalid_at=end)
    unknown_start = make_edge(
        'Kiran was released from Payments.', valid_at=None, invalid_at=START + timedelta(days=3)
    )
    llm_client.generate_response.return_value['contradicted_facts'] = [0]

    resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
        llm_client, incoming, [], [unknown_start], episode
    )

    assert resolved.invalid_at == end
    assert resolved.expired_at == NOW
    assert unknown_start.valid_at is None
    assert unknown_start.invalid_at == START + timedelta(days=3)
    assert invalidated == []
    assert duplicates == []


@pytest.mark.parametrize('same_wording', [False, True])
async def test_timestamp_fallback_can_confirm_an_existing_historical_occurrence(
    monkeypatch, episode, llm_client, same_wording
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    end = START + timedelta(days=5)
    existing = make_edge('Kiran was assigned to Payments.', invalid_at=end, expired_at=NOW)
    incoming = make_edge(
        existing.fact if same_wording else 'Kiran worked on Payments.', valid_at=None
    )
    calls = []

    async def respond(*args, **kwargs):
        calls.append(kwargs['prompt_name'])
        if kwargs['prompt_name'] == 'extract_edges.extract_timestamps':
            return {'valid_at': START.isoformat(), 'invalid_at': end.isoformat()}
        assert kwargs['prompt_name'] == 'dedupe_edges.resolve_edge'
        return {'duplicate_facts': [0], 'contradicted_facts': []}

    llm_client.generate_response.side_effect = respond

    resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
        llm_client, incoming, [existing], [], episode
    )

    assert resolved is existing
    assert resolved.valid_at == START
    assert resolved.invalid_at == end
    assert resolved.expired_at == NOW
    assert invalidated == []
    assert duplicates == ([] if same_wording else [existing])
    assert calls.count('extract_edges.extract_timestamps') == 1
    assert calls.count('dedupe_edges.resolve_edge') == int(not same_wording)


async def test_unresolved_timestamp_fallback_is_not_repeated_or_treated_as_overlap(
    episode, llm_client
):
    end = START + timedelta(days=5)
    existing = make_edge('Kiran was assigned to Payments.', invalid_at=end, expired_at=NOW)
    incoming = make_edge(existing.fact, valid_at=None)
    calls = []

    async def respond(*args, **kwargs):
        calls.append(kwargs['prompt_name'])
        if kwargs['prompt_name'] == 'extract_edges.extract_timestamps':
            return {'valid_at': None, 'invalid_at': None}
        assert kwargs['prompt_name'] == 'dedupe_edges.resolve_edge'
        return {'duplicate_facts': [0], 'contradicted_facts': []}

    llm_client.generate_response.side_effect = respond

    resolved, invalidated, duplicates = await edge_operations.resolve_extracted_edge(
        llm_client, incoming, [existing], [], episode
    )

    assert resolved is incoming
    assert resolved.valid_at is None
    assert resolved.invalid_at is None
    assert resolved.expired_at is None
    assert existing.invalid_at == end
    assert existing.expired_at == NOW
    assert invalidated == []
    assert duplicates == []
    assert calls.count('extract_edges.extract_timestamps') == 1
    assert calls.count('dedupe_edges.resolve_edge') == 1


@pytest.mark.parametrize(
    ('edge_reference', 'with_episode', 'expected_reference'),
    [
        (START + timedelta(days=2), True, START + timedelta(days=2)),
        (None, True, NOW),
        (START + timedelta(days=2), False, START + timedelta(days=2)),
        (None, False, None),
    ],
)
async def test_timestamp_fallback_uses_the_attributed_episode_reference(
    episode, llm_client, edge_reference, with_episode, expected_reference
):
    incoming = make_edge('Kiran is assigned to Payments.', valid_at=None)
    incoming.reference_time = edge_reference
    llm_client.generate_response.return_value = {'valid_at': None, 'invalid_at': None}

    await edge_operations._extract_edge_timestamps(
        llm_client, incoming, episode if with_episode else None
    )

    assert incoming.valid_at is None
    assert incoming.invalid_at is None
    if expected_reference is None:
        llm_client.generate_response.assert_not_awaited()
    else:
        llm_client.generate_response.assert_awaited_once()
        actual_prompt = str(llm_client.generate_response.await_args.args[0])
        assert expected_reference.isoformat() in actual_prompt
        if expected_reference != NOW:
            assert NOW.isoformat() not in actual_prompt
