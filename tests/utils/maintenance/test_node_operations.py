import logging
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _cached_shingles,
    _has_high_entropy,
    _hash_shingle,
    _jaccard_similarity,
    _lsh_bands,
    _minhash_signature,
    _name_entropy,
    _normalize_name_for_fuzzy,
    _normalize_string_exact,
    _resolve_with_similarity,
    _shingles,
)
from graphiti_core.utils.maintenance.node_operations import (
    _collect_candidate_nodes,
    _extract_entity_summaries_batch,
    _resolve_with_llm,
    extract_attributes_from_nodes,
    resolve_extracted_nodes,
)


def _make_clients():
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_generate = AsyncMock()
    llm_client.generate_response = llm_generate

    clients = GraphitiClients.model_construct(  # bypass validation to allow test doubles
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )

    return clients, llm_generate


def _make_episode(group_id: str = 'group'):
    return EpisodicNode(
        name='episode',
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test',
        content='content',
        valid_at=utc_now(),
    )


def _semantic_candidates(candidate_groups: list[list[EntityNode]]):
    async def fake_search(*_, **__):
        return candidate_groups

    return fake_search


@pytest.mark.asyncio
async def test_resolve_nodes_exact_match_skips_llm(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_exact_match_promotes_generic_candidate_type(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Audrey', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Audrey', group_id='group', labels=['Entity', 'Person'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert set(resolved[0].labels) == {'Entity', 'Person'}
    assert set(candidate.labels) == {'Entity', 'Person'}
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_low_entropy_uses_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'name': 'Joe',
                'duplicate_candidate_id': -1,
            }
        ]
    }

    candidate = EntityNode(name='Joseph', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted.uuid
    assert uuid_map[extracted.uuid] == extracted.uuid
    llm_generate.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_short_name_exact_match_skips_llm(monkeypatch):
    """Short names with a unique exact candidate match should resolve deterministically."""
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Java', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_fuzzy_match(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe-Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_candidate_nodes_dedupes_and_merges_override(monkeypatch):
    clients, _ = _make_clients()

    candidate = EntityNode(name='Alice', group_id='group', labels=['Entity'])
    override_duplicate = EntityNode(
        uuid=candidate.uuid,
        name='Alice Alt',
        group_id='group',
        labels=['Entity'],
    )
    extracted = EntityNode(name='Alice', group_id='group', labels=['Entity'])

    semantic_search_mock = AsyncMock(return_value=[[candidate]])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        semantic_search_mock,
    )

    result = await _collect_candidate_nodes(
        clients,
        [extracted],
        existing_nodes_override=[override_duplicate],
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].uuid == candidate.uuid
    semantic_search_mock.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_semantic_miss_keeps_node_without_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    extracted = EntityNode(name='Completely New Thing', group_id='group', labels=['Entity'])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[]]),
    )

    resolved, uuid_map, duplicates = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted.uuid
    assert uuid_map[extracted.uuid] == extracted.uuid
    assert duplicates == []
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_multiple_exact_matches_use_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'name': 'Java',
                'duplicate_candidate_id': 0,
            }
        ]
    }
    candidate1 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Java', group_id='group', labels=['Entity'])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate1, candidate2]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate1.uuid
    assert uuid_map[extracted.uuid] == candidate1.uuid
    llm_generate.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_batches_unresolved_nodes_into_one_llm_call(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'name': 'Joe',
                'duplicate_candidate_id': -1,
            },
            {
                'id': 1,
                'name': 'Java',
                'duplicate_candidate_id': 1,
            },
        ]
    }

    low_entropy_candidate = EntityNode(name='Joseph', group_id='group', labels=['Entity'])
    java_candidate_1 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    java_candidate_2 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted_nodes = [
        EntityNode(name='Joe', group_id='group', labels=['Entity']),
        EntityNode(name='Java', group_id='group', labels=['Entity']),
    ]

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[low_entropy_candidate], [java_candidate_1, java_candidate_2]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        extracted_nodes,
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted_nodes[0].uuid
    assert resolved[1].uuid == java_candidate_1.uuid
    assert uuid_map[extracted_nodes[0].uuid] == extracted_nodes[0].uuid
    assert uuid_map[extracted_nodes[1].uuid] == java_candidate_1.uuid
    assert llm_generate.await_count == 1


def test_build_candidate_indexes_populates_structures():
    candidate = EntityNode(name='Bob Dylan', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])

    normalized_key = candidate.name.lower()
    assert indexes.normalized_existing[normalized_key][0].uuid == candidate.uuid
    assert indexes.nodes_by_uuid[candidate.uuid] is candidate
    assert candidate.uuid in indexes.shingles_by_candidate
    assert any(candidate.uuid in bucket for bucket in indexes.lsh_buckets.values())


def test_normalize_helpers():
    assert _normalize_string_exact('  Alice   Smith ') == 'alice smith'
    assert _normalize_name_for_fuzzy('Alice-Smith!') == 'alice smith'


def test_name_entropy_variants():
    assert _name_entropy('alice') > _name_entropy('aaaaa')
    assert _name_entropy('') == 0.0


def test_has_high_entropy_rules():
    assert _has_high_entropy('meaningful name') is True
    assert _has_high_entropy('aa') is False


def test_shingles_and_cache():
    raw = 'alice'
    shingle_set = _shingles(raw)
    assert shingle_set == {'ali', 'lic', 'ice'}
    assert _cached_shingles(raw) == shingle_set
    assert _cached_shingles(raw) is _cached_shingles(raw)


def test_hash_minhash_and_lsh():
    shingles = {'abc', 'bcd', 'cde'}
    signature = _minhash_signature(shingles)
    assert len(signature) == 32
    bands = _lsh_bands(signature)
    assert all(len(band) == 4 for band in bands)
    hashed = {_hash_shingle(s, 0) for s in shingles}
    assert len(hashed) == len(shingles)


def test_jaccard_similarity_edges():
    a = {'a', 'b'}
    b = {'a', 'c'}
    assert _jaccard_similarity(a, b) == pytest.approx(1 / 3)
    assert _jaccard_similarity(set(), set()) == 1.0
    assert _jaccard_similarity(a, set()) == 0.0


def test_resolve_with_similarity_exact_match_updates_state():
    candidate = EntityNode(name='Charlie Parker', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Charlie Parker', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == [(extracted, candidate)]


def test_resolve_with_similarity_short_name_exact_match_resolves_deterministically():
    """Short names like 'Nate' should resolve via exact match without hitting the LLM."""
    candidate = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])
    extracted = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == [(extracted, candidate)]


def test_resolve_with_similarity_short_name_no_candidate_defers_to_llm():
    """Short names with no exact match should still reach the LLM for resolution."""
    extracted = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.uuid_map == {}
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_short_name_multiple_candidates_defers_to_llm():
    """Short names with multiple exact matches should escalate to LLM."""
    candidate1 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Java', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate1, candidate2])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.uuid_map == {}
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_low_entropy_defers_resolution():
    extracted = EntityNode(name='Bob', group_id='group', labels=['Entity'])
    indexes = DedupCandidateIndexes(
        existing_nodes=[],
        nodes_by_uuid={},
        normalized_existing=defaultdict(list),
        shingles_by_candidate={},
        lsh_buckets=defaultdict(list),
    )
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_multiple_exact_matches_defers_to_llm():
    candidate1 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate1, candidate2])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


@pytest.mark.asyncio
async def test_resolve_with_llm_candidate_attributes_cannot_overwrite_candidate_id(monkeypatch):
    """Ensure candidate.attributes with a 'candidate_id' key cannot corrupt the LLM context."""
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])
    candidate.attributes = {'candidate_id': 999, 'genre': 'jazz'}

    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    captured_context = {}

    def fake_prompt_nodes(context):
        captured_context.update(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {'id': 0, 'name': 'Dizzy Gillespie', 'duplicate_candidate_id': 0}
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    # candidate_id must be the positional index (0), not the adversarial attribute (999)
    assert captured_context['existing_nodes'][0]['candidate_id'] == 0
    # non-colliding attributes should still be present
    assert captured_context['existing_nodes'][0]['genre'] == 'jazz'
    assert state.resolved_nodes[0].uuid == candidate.uuid


@pytest.mark.asyncio
async def test_resolve_with_llm_updates_unresolved(monkeypatch):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    captured_context = {}

    def fake_prompt_nodes(context):
        captured_context.update(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )

    async def fake_generate_response(*_, **__):
        return {
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Dizzy Gillespie',
                    'duplicate_candidate_id': 0,
                }
            ]
        }

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=fake_generate_response)

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert isinstance(captured_context['existing_nodes'], list)
    assert captured_context['existing_nodes'][0]['candidate_id'] == 0
    assert (
        captured_context['extracted_nodes'][0]['entity_type_description'] == 'Default Entity Type'
    )
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_promotes_generic_candidate_type(monkeypatch):
    extracted = EntityNode(name='Audrey', group_id='group', labels=['Entity', 'Person'])
    candidate = EntityNode(name='Audrey', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Audrey',
                    'duplicate_candidate_id': 0,
                }
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert set(state.resolved_nodes[0].labels) == {'Entity', 'Person'}
    assert set(candidate.labels) == {'Entity', 'Person'}
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_ignores_out_of_range_relative_ids(monkeypatch, caplog):
    extracted = EntityNode(name='Dexter', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 5,
                    'name': 'Dexter',
                    'duplicate_candidate_id': -1,
                }
            ]
        }
    )

    with caplog.at_level(logging.WARNING):
        await _resolve_with_llm(
            llm_client,
            [extracted],
            indexes,
            state,
            episode=_make_episode(),
            previous_episodes=[],
            entity_types=None,
        )

    assert state.resolved_nodes[0] is None
    assert 'Skipping invalid LLM dedupe id 5' in caplog.text


@pytest.mark.asyncio
async def test_resolve_with_llm_ignores_duplicate_relative_ids(monkeypatch):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Dizzy Gillespie',
                    'duplicate_candidate_id': 0,
                },
                {
                    'id': 0,
                    'name': 'Dizzy',
                    'duplicate_candidate_id': -1,
                },
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_invalid_candidate_id_defaults_to_extracted(monkeypatch):
    extracted = EntityNode(name='Dexter', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Dexter',
                    'duplicate_candidate_id': 999,
                }
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0] == extracted
    assert state.uuid_map[extracted.uuid] == extracted.uuid
    assert state.duplicate_pairs == []


@pytest.mark.asyncio
async def test_batch_summaries_short_summary_no_llm():
    """Test that short summaries are kept as-is without LLM call (optimization)."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summaries': [{'name': 'Test Node', 'summary': 'Generated summary'}]}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    await _extract_entity_summaries_batch(
        llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=None,
        edges_by_node={},
    )

    # Short summary should be kept as-is without LLM call
    assert node.summary == 'Old summary'
    # LLM should NOT have been called (summary is short enough)
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_summaries_callback_skip_summary():
    """Test that summary is NOT regenerated when callback returns False."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summaries': [{'name': 'Test Node', 'summary': 'This should not be used'}]}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    # Callback that always returns False (skip summary generation)
    async def skip_summary_filter(n: EntityNode) -> bool:
        return False

    await _extract_entity_summaries_batch(
        llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=skip_summary_filter,
        edges_by_node={},
    )

    # Summary should remain unchanged
    assert node.summary == 'Old summary'
    # LLM should NOT have been called for summary
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_summaries_selective_callback():
    """Test callback that selectively skips summaries based on node properties."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'summaries': []})

    user_node = EntityNode(name='User', group_id='group', labels=['Entity', 'User'], summary='Old')
    topic_node = EntityNode(
        name='Topic', group_id='group', labels=['Entity', 'Topic'], summary='Old'
    )

    episode = _make_episode()

    # Callback that skips User nodes but generates for others
    async def selective_filter(n: EntityNode) -> bool:
        return 'User' not in n.labels

    await _extract_entity_summaries_batch(
        llm_client,
        [user_node, topic_node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=selective_filter,
        edges_by_node={},
    )

    # User summary should remain unchanged (callback returned False)
    assert user_node.summary == 'Old'
    # Topic summary should also remain unchanged (short summary optimization)
    assert topic_node.summary == 'Old'
    # LLM should NOT have been called (summaries are short enough)
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_extract_attributes_from_nodes_with_callback():
    """Test that callback is properly passed through extract_attributes_from_nodes."""
    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity', 'User'], summary='Old1')
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity', 'Topic'], summary='Old2')

    episode = _make_episode()

    call_tracker = []

    # Callback that tracks which nodes it's called with
    async def tracking_filter(n: EntityNode) -> bool:
        call_tracker.append(n.name)
        return 'User' not in n.labels

    results = await extract_attributes_from_nodes(
        clients,
        [node1, node2],
        episode=episode,
        previous_episodes=[],
        entity_types=None,
        should_summarize_node=tracking_filter,
    )

    # Callback should have been called for both nodes
    assert len(call_tracker) == 2
    assert 'Node1' in call_tracker
    assert 'Node2' in call_tracker

    # Both nodes should keep old summaries (short summary optimization skips LLM)
    node1_result = next(n for n in results if n.name == 'Node1')
    node2_result = next(n for n in results if n.name == 'Node2')

    assert node1_result.summary == 'Old1'
    assert node2_result.summary == 'Old2'


@pytest.mark.asyncio
async def test_batch_summaries_calls_llm_for_long_summary():
    """Test that LLM is called when summary exceeds character limit."""
    from graphiti_core.edges import EntityEdge
    from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summaries': [{'name': 'Test Node', 'summary': 'Condensed summary'}]}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Short')
    episode = _make_episode()

    # Create edges with long facts that exceed the threshold
    long_fact = 'x' * (MAX_SUMMARY_CHARS * 2)
    edge = EntityEdge(
        uuid='edge1',
        group_id='group',
        source_node_uuid=node.uuid,
        target_node_uuid='other-uuid',
        name='test_edge',
        fact=long_fact,
        created_at=utc_now(),
    )

    edges_by_node = {node.uuid: [edge, edge]}  # Multiple long edges

    await _extract_entity_summaries_batch(
        llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=None,
        edges_by_node=edges_by_node,
    )

    # LLM should have been called to condense the long summary
    llm_client.generate_response.assert_awaited_once()
    assert node.summary == 'Condensed summary'
