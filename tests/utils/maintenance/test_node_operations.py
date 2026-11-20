import asyncio
import logging
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, create_model

from graphiti_core.edges import EntityEdge
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
    _identity_key_for_node,
    _identity_type_for_node,
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
    DEDUPE_SUMMARY_MAX_CHARS,
    _build_existing_nodes_context,
    _build_extracted_nodes_context,
    _collect_candidate_nodes,
    _collect_exact_name_candidate_nodes,
    _collect_identity_property_candidate_nodes,
    _extract_entity_summaries_batch,
    _resolve_with_llm,
    _semantic_candidate_search,
    extract_attributes_from_nodes,
    resolve_extracted_nodes,
)


def _make_clients():
    driver = MagicMock()
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], None, None))
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_client.max_tokens = 32768
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


NEIGHBOR_UUID = 'episode-neighbor'


def _allow_exact(_node: EntityNode, _candidate: EntityNode) -> bool:
    return True


def _refuse_exact(_node: EntityNode, _candidate: EntityNode) -> bool:
    return False


def _patch_node_edges(monkeypatch, neighbors_by_node: dict[str, list[str]]) -> list[str]:
    """Make EntityEdge.get_by_node_uuid return one edge to each listed neighbor."""
    reads: list[str] = []

    async def fake_get_by_node_uuid(_driver, node_uuid):
        reads.append(node_uuid)
        return [
            EntityEdge(
                group_id='group',
                source_node_uuid=node_uuid,
                target_node_uuid=neighbor_uuid,
                name='RELATES_TO',
                fact='fact',
                created_at=utc_now(),
            )
            for neighbor_uuid in neighbors_by_node.get(node_uuid, [])
        ]

    monkeypatch.setattr(EntityEdge, 'get_by_node_uuid', fake_get_by_node_uuid)
    return reads


def _entity_type(name: str) -> type[BaseModel]:
    return create_model(name)


@pytest.mark.asyncio
async def test_semantic_candidate_search_skips_blank_names(monkeypatch):
    clients, _ = _make_clients()
    queries = []

    async def create_batch(input_data):
        queries.extend(input_data)
        return [[0.1, 0.2, 0.3] for _ in input_data]

    clients.embedder.create_batch = create_batch
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.node_similarity_search',
        AsyncMock(return_value=[object()]),
    )
    nodes = [
        EntityNode(name='Alice', group_id='group', labels=['Entity']),
        EntityNode(name='', group_id='group', labels=['Entity']),
        EntityNode(name='Bob', group_id='group', labels=['Entity']),
    ]

    result = await _semantic_candidate_search(clients, nodes)

    assert len(result) == 3
    assert result[1] == []
    assert queries == ['Alice', 'Bob']


@pytest.mark.asyncio
async def test_semantic_candidate_search_raises_on_vector_count_mismatch():
    clients, _ = _make_clients()
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    nodes = [
        EntityNode(name='Alice', group_id='group', labels=['Entity']),
        EntityNode(name='Bob', group_id='group', labels=['Entity']),
    ]

    with pytest.raises(ValueError, match='vectors'):
        await _semantic_candidate_search(clients, nodes)


@pytest.mark.asyncio
async def test_resolve_nodes_exact_match_skips_llm(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    _patch_node_edges(monkeypatch, {candidate.uuid: [NEIGHBOR_UUID]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        episode_neighbor_uuids={NEIGHBOR_UUID},
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_extracted_nodes_can_disable_shared_neighbor_name_gate(monkeypatch):
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
        require_shared_neighbor_for_name_match=False,
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

    _patch_node_edges(monkeypatch, {candidate.uuid: [NEIGHBOR_UUID]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        episode_neighbor_uuids={NEIGHBOR_UUID},
    )

    assert resolved[0].uuid == candidate.uuid
    assert set(resolved[0].labels) == {'Entity', 'Person'}
    assert set(candidate.labels) == {'Entity', 'Person'}
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


def test_identity_type_for_node_uses_label_with_identity_properties():
    node = EntityNode(name='Acme', group_id='group', labels=['Entity', 'Party', 'Company'])
    identity_properties = {'Party': ['party_id'], 'Company': ['company_id']}

    assert _identity_type_for_node(node, identity_properties) == 'Company'


def test_identity_type_for_node_uses_matching_label_when_other_labels_do_not_match():
    node = EntityNode(name='Acme', group_id='group', labels=['Entity', 'Party', 'Company'])

    assert _identity_type_for_node(node, {'Party': ['party_id']}) == 'Party'


@pytest.mark.asyncio
async def test_resolve_nodes_identity_properties_match_before_llm(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(
        name='ACME',
        group_id='group',
        labels=['Entity', 'Company'],
        attributes={'domain': 'acme.example', 'size': 'enterprise'},
    )
    extracted = EntityNode(
        name='ACME Incorporated',
        group_id='group',
        labels=['Entity', 'Company'],
        attributes={'domain': 'acme.example', 'size': None, 'region': 'us-west'},
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[]]),
    )

    resolved, uuid_map, duplicates = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        existing_nodes_override=[candidate],
        identity_properties={'Company': ['domain']},
    )

    assert resolved[0].uuid == candidate.uuid
    assert resolved[0].attributes == {
        'domain': 'acme.example',
        'size': 'enterprise',
        'region': 'us-west',
    }
    assert uuid_map[extracted.uuid] == candidate.uuid
    assert duplicates == [(extracted, candidate)]
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_identity_properties_ignore_case_and_whitespace(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(
        name='CITIZENS FINANCIAL GROUP, INC.',
        group_id='group',
        labels=['Entity', 'Company'],
        attributes={'company_name': 'CITIZENS FINANCIAL GROUP, INC.'},
    )
    extracted = EntityNode(
        name='Citizens Financial Group',
        group_id='group',
        labels=['Entity', 'Company'],
        attributes={'company_name': 'Citizens  Financial Group, Inc.'},
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[]]),
    )

    resolved, uuid_map, duplicates = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        existing_nodes_override=[candidate],
        identity_properties={'Company': ['company_name']},
    )

    assert resolved[0].uuid == candidate.uuid
    assert resolved[0].attributes['company_name'] == 'CITIZENS FINANCIAL GROUP, INC.'
    assert candidate.attributes['company_name'] == 'CITIZENS FINANCIAL GROUP, INC.'
    assert uuid_map[extracted.uuid] == candidate.uuid
    assert duplicates == [(extracted, candidate)]
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_identity_lookup_requests_normalized_string_match():
    clients, _ = _make_clients()
    candidate = EntityNode(
        name='CITIZENS FINANCIAL GROUP, INC.',
        group_id='group',
        labels=['Entity', 'Company'],
        attributes={'company_name': 'CITIZENS FINANCIAL GROUP, INC.'},
    )
    extracted = EntityNode(
        name='Citizens Financial Group',
        group_id='group',
        labels=['Entity', 'Company'],
        attributes={'company_name': ' Citizens  Financial Group, Inc. '},
    )
    lookup = AsyncMock(return_value=[candidate])
    clients.driver.graph_operations_interface = MagicMock(node_get_by_attribute_values=lookup)

    results = await _collect_identity_property_candidate_nodes(
        clients,
        [extracted],
        {'Company': ['company_name']},
    )

    assert results == [[candidate]]
    lookup.assert_awaited_once()
    args, kwargs = lookup.call_args
    assert args[2:] == (
        'group',
        {'company_name': 'Citizens  Financial Group, Inc.'},
        ['Company'],
        args[5],
    )
    assert kwargs == {'normalize_strings': True}


def test_identity_key_keeps_non_string_rules():
    identity_properties = {'Company': ['company_id']}

    def key(value: object) -> object:
        node = EntityNode(
            name='Company',
            group_id='group',
            labels=['Entity', 'Company'],
            attributes={'company_id': value},
        )
        return _identity_key_for_node(node, identity_properties)

    assert key(42) != key('42')
    assert key(42) == key(42.0)
    assert key(True) != key('true')
    assert key(float('nan')) is None
    assert key('   ') is None


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

    _patch_node_edges(monkeypatch, {candidate.uuid: [NEIGHBOR_UUID]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        episode_neighbor_uuids={NEIGHBOR_UUID},
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_fuzzy_match_with_shared_neighbor_skips_llm(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe-Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    employer = EntityNode(name='Acme Corp', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate], []]),
    )
    _patch_node_edges(monkeypatch, {candidate.uuid: [employer.uuid]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted, employer],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_generic_fuzzy_name_of_other_company_goes_to_llm(monkeypatch):
    """A near-same generic name ("consumer's auto loans") needs a shared neighbor too."""
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {'id': 0, 'name': 'Westlake Services', 'duplicate_candidate_id': 0},
            {'id': 1, 'name': "consumer's auto loans", 'duplicate_candidate_id': -1},
        ]
    }
    westlake_existing, credit_acceptance, loan_existing, westlake, loan = _cfpb_nodes()
    loans = loan.model_copy(update={'name': "consumer's auto loans"})
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[westlake_existing], [loan_existing]]),
    )
    reads = _patch_node_edges(monkeypatch, {loan_existing.uuid: [credit_acceptance.uuid]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [westlake, loans],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert loan_existing.uuid in reads
    llm_generate.assert_awaited_once()
    assert uuid_map[loans.uuid] == loans.uuid
    assert resolved[1].uuid == loans.uuid


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
async def test_collect_candidate_nodes_prepends_exact_name_candidates(monkeypatch):
    clients, _ = _make_clients()

    exact_candidate = EntityNode(name='Alice', group_id='group', labels=['Entity'])
    semantic_candidate = EntityNode(name='Alice Smith', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Alice', group_id='group', labels=['Entity'])

    exact_lookup_mock = AsyncMock(return_value=[[exact_candidate]])
    semantic_search_mock = AsyncMock(return_value=[[semantic_candidate]])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._collect_exact_name_candidate_nodes',
        exact_lookup_mock,
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        semantic_search_mock,
    )

    result = await _collect_candidate_nodes(
        clients,
        [extracted],
        existing_nodes_override=None,
    )

    assert len(result) == 1
    assert [node.uuid for node in result[0]] == [exact_candidate.uuid, semantic_candidate.uuid]
    exact_lookup_mock.assert_awaited_once()
    semantic_search_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_collect_exact_name_candidate_nodes_scopes_by_group(monkeypatch):
    clients, _ = _make_clients()

    group_a_candidate = EntityNode(name='Alice', group_id='group-a', labels=['Entity'])
    group_b_candidate = EntityNode(name='Alice', group_id='group-b', labels=['Entity'])
    extracted_nodes = [
        EntityNode(name='Alice', group_id='group-a', labels=['Entity']),
        EntityNode(name='Alice', group_id='group-b', labels=['Entity']),
    ]

    get_by_names_mock = AsyncMock(return_value=[group_a_candidate, group_b_candidate])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.EntityNode.get_by_names',
        get_by_names_mock,
    )

    result = await _collect_exact_name_candidate_nodes(clients, extracted_nodes)

    assert result == [[group_a_candidate], [group_b_candidate]]
    get_by_names_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_nodes_exact_name_lookup_handles_semantic_miss(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Audrey', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Audrey', group_id='group', labels=['Entity', 'Person'])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._collect_exact_name_candidate_nodes',
        AsyncMock(return_value=[[candidate]]),
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[]]),
    )
    candidates_by_node = {}

    _patch_node_edges(monkeypatch, {candidate.uuid: [NEIGHBOR_UUID]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        episode_neighbor_uuids={NEIGHBOR_UUID},
        candidates_by_node=candidates_by_node,
    )

    assert resolved[0].uuid == candidate.uuid
    assert set(candidate.labels) == {'Entity', 'Person'}
    assert uuid_map[extracted.uuid] == candidate.uuid
    assert candidates_by_node[extracted.uuid][0].uuid == candidate.uuid
    assert candidates_by_node[extracted.uuid][0].labels == ['Entity']
    llm_generate.assert_not_awaited()


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

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_allow_exact)

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

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_allow_exact)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == [(extracted, candidate)]


def test_resolve_with_similarity_short_name_no_candidate_defers_to_llm():
    """Short names with no exact match should still reach the LLM for resolution."""
    extracted = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_allow_exact)

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

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_allow_exact)

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

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_allow_exact)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_multiple_exact_matches_defers_to_llm():
    candidate1 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate1, candidate2])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_allow_exact)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_refused_exact_match_defers_to_llm():
    candidate = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    extracted = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_refuse_exact)

    assert state.resolved_nodes[0] is None
    assert state.uuid_map == {}
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_same_uuid_exact_match_skips_gate():
    """A node that already has the candidate UUID (for example the user node) needs no neighbor."""
    candidate = EntityNode(name='Jane Doe', group_id='group', labels=['Entity'])
    extracted = candidate.model_copy()

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state, name_match_gate=_refuse_exact)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == []


def _cfpb_nodes():
    westlake_existing = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    credit_acceptance = EntityNode(name='Credit Acceptance', group_id='group', labels=['Company'])
    loan_existing = EntityNode(
        name="consumer's auto loan", group_id='group', labels=['FinancialAccount']
    )
    westlake = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    loan = EntityNode(name="consumer's auto loan", group_id='group', labels=['FinancialAccount'])
    return westlake_existing, credit_acceptance, loan_existing, westlake, loan


@pytest.mark.asyncio
async def test_resolve_nodes_generic_exact_name_of_other_company_goes_to_llm(monkeypatch):
    """The Westlake loan has the same name as a Credit Acceptance loan but no shared neighbor."""
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {'id': 0, 'name': 'Westlake Services', 'duplicate_candidate_id': 0},
            {'id': 1, 'name': "consumer's auto loan", 'duplicate_candidate_id': -1},
        ]
    }
    westlake_existing, credit_acceptance, loan_existing, westlake, loan = _cfpb_nodes()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[westlake_existing], [loan_existing]]),
    )
    _patch_node_edges(monkeypatch, {loan_existing.uuid: [credit_acceptance.uuid]})

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [westlake, loan],
        episode=_make_episode(),
        previous_episodes=[],
    )

    llm_generate.assert_awaited_once()
    user_prompt = llm_generate.await_args.args[0][1].content
    assert user_prompt.count("consumer's auto loan") >= 2
    assert uuid_map[loan.uuid] == loan.uuid
    assert resolved[1].uuid == loan.uuid
    assert uuid_map[westlake.uuid] == westlake_existing.uuid


@pytest.mark.asyncio
async def test_resolve_nodes_exact_name_with_shared_neighbor_skips_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    westlake_existing, _, loan_existing, westlake, loan = _cfpb_nodes()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[westlake_existing], [loan_existing]]),
    )
    _patch_node_edges(
        monkeypatch,
        {
            loan_existing.uuid: [westlake_existing.uuid],
            westlake_existing.uuid: [loan_existing.uuid],
        },
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [westlake, loan],
        episode=_make_episode(),
        previous_episodes=[],
    )

    llm_generate.assert_not_awaited()
    assert uuid_map[loan.uuid] == loan_existing.uuid
    assert uuid_map[westlake.uuid] == westlake_existing.uuid
    assert [node.uuid for node in resolved] == [westlake_existing.uuid, loan_existing.uuid]


@pytest.mark.asyncio
async def test_resolve_nodes_identity_resolved_neighbor_ignores_its_exact_name_candidate(
    monkeypatch,
):
    """An episode entity that identity matching resolved elsewhere is not its name twin."""
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {'id': 1, 'name': "consumer's auto loan", 'duplicate_candidate_id': -1}
        ]
    }
    westlake_by_domain = EntityNode(
        name='Westlake Financial',
        group_id='group',
        labels=['Company'],
        attributes={'domain': 'w.example'},
    )
    westlake_by_name = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    loan_existing = EntityNode(
        name="consumer's auto loan", group_id='group', labels=['FinancialAccount']
    )
    westlake = EntityNode(
        name='Westlake Services',
        group_id='group',
        labels=['Company'],
        attributes={'domain': 'w.example'},
    )
    loan = EntityNode(name="consumer's auto loan", group_id='group', labels=['FinancialAccount'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[westlake_by_domain, westlake_by_name], [loan_existing]]),
    )
    _patch_node_edges(monkeypatch, {loan_existing.uuid: [westlake_by_name.uuid]})

    _, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [westlake, loan],
        episode=_make_episode(),
        previous_episodes=[],
        entity_types={'Company': _entity_type('Company')},
        identity_properties={'Company': ['domain']},
    )

    assert uuid_map[westlake.uuid] == westlake_by_domain.uuid
    llm_generate.assert_awaited_once()
    assert uuid_map[loan.uuid] == loan.uuid


@pytest.mark.asyncio
async def test_resolve_nodes_exact_name_edge_read_failure_goes_to_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [{'id': 0, 'name': 'Joe Michaels', 'duplicate_candidate_id': 0}]
    }
    candidate = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )
    monkeypatch.setattr(
        EntityEdge, 'get_by_node_uuid', AsyncMock(side_effect=RuntimeError('graph down'))
    )

    _, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
        episode_neighbor_uuids={NEIGHBOR_UUID},
    )

    llm_generate.assert_awaited_once()
    assert uuid_map[extracted.uuid] == candidate.uuid


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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
@pytest.mark.parametrize(
    'client_max_tokens, expected_max_tokens',
    [(32768, 16384), (4096, 4096), (None, 16384)],
)
async def test_resolve_with_llm_caps_completion_budget(
    monkeypatch, client_max_tokens, expected_max_tokens
):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.max_tokens = client_max_tokens
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

    assert llm_client.generate_response.await_args.kwargs['max_tokens'] == expected_max_tokens


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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
    llm_client.max_tokens = 32768
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
async def test_extract_attributes_from_nodes_keeps_initial_nulls_from_response():
    class Patient(BaseModel):
        first_name: str | None = None
        last_name: str | None = None
        favorite_color: str | None = None

    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'first_name': 'John',
        'last_name': None,
        'favorite_color': None,
    }
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node = EntityNode(
        name='John Smith',
        group_id='group',
        labels=['Entity', 'Patient'],
        attributes={'last_name': 'Smith'},
        summary='Existing summary',
    )

    async def skip_summary(_node: EntityNode) -> bool:
        return False

    results = await extract_attributes_from_nodes(
        clients,
        [node],
        episode=_make_episode(),
        previous_episodes=[],
        entity_types={'Patient': Patient},
        should_summarize_node=skip_summary,
    )

    assert results[0].attributes == {
        'first_name': 'John',
        'last_name': 'Smith',
        'favorite_color': None,
    }


@pytest.mark.asyncio
async def test_batch_summaries_calls_llm_for_long_summary():
    """Test that LLM is called when summary exceeds character limit."""
    from graphiti_core.edges import EntityEdge
    from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

    llm_client = MagicMock()
    llm_client.max_tokens = 32768
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


@pytest.mark.asyncio
async def test_extract_attributes_preserves_prior_attributes_when_entity_types_none():
    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node = EntityNode(
        name='Alice',
        group_id='group',
        labels=['Entity', 'Person'],
        summary='Existing summary',
        attributes={'age': 30, 'city': 'New York'},
    )

    results = await extract_attributes_from_nodes(
        clients,
        [node],
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert results[0].attributes == {'age': 30, 'city': 'New York'}


@pytest.mark.asyncio
async def test_extract_attributes_preserves_prior_attributes_when_label_not_in_entity_types():
    class Organization(BaseModel):
        industry: str | None = None

    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node = EntityNode(
        name='Alice',
        group_id='group',
        labels=['Entity', 'Person'],
        summary='Existing summary',
        attributes={'age': 30, 'city': 'New York'},
    )

    results = await extract_attributes_from_nodes(
        clients,
        [node],
        episode=_make_episode(),
        previous_episodes=[],
        entity_types={'Organization': Organization},
    )

    assert results[0].attributes == {'age': 30, 'city': 'New York'}


def test_build_existing_nodes_context_sends_full_summary():
    short = EntityNode(name='Short', group_id='group', labels=['Entity'], summary='A ' * 200)
    long_summary = 'word ' * 150
    long = EntityNode(name='Long', group_id='group', labels=['Entity'], summary=long_summary)
    long.attributes = {'account_kind': 'auto loan'}

    context = _build_existing_nodes_context([short, long])

    assert context[0]['summary'] == short.summary
    assert len(context[1]['summary']) <= DEDUPE_SUMMARY_MAX_CHARS
    assert long_summary.startswith(context[1]['summary'])
    assert context[1]['summary'].endswith('word')
    assert context[1]['account_kind'] == 'auto loan'
    assert context[1]['candidate_id'] == 1


def test_build_extracted_nodes_context_sends_attributes():
    node = EntityNode(name="consumer's auto loan", group_id='group', labels=['FinancialAccount'])
    node.attributes = {'account_kind': 'auto loan', 'id': 99}

    context = _build_extracted_nodes_context([node], {})

    assert context[0]['account_kind'] == 'auto loan'
    assert context[0]['id'] == 0
    assert context[0]['name'] == "consumer's auto loan"


CANDIDATE_SUMMARY = (
    "Credit Acceptance placed the consumer's auto loan on payment arrangements. " * 3
).strip()


async def _captured_dedupe_contexts(monkeypatch, *, bulk: bool) -> dict:
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {'id': 0, 'name': "consumer's auto loan account", 'duplicate_candidate_id': -1}
        ]
    }
    candidate = EntityNode(
        name="consumer's auto loan",
        group_id='group',
        labels=['FinancialAccount'],
        summary=CANDIDATE_SUMMARY,
    )
    extracted = EntityNode(
        name="consumer's auto loan account", group_id='group', labels=['FinancialAccount']
    )
    extracted.attributes = {'account_kind': 'auto loan'}
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    captured: dict = {}

    def fake_prompt_nodes(context):
        captured.update(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )

    if bulk:
        resolved, _, _ = await resolve_extracted_nodes(
            clients, [extracted], _make_episode(), None, None, candidates_by_node={}
        )
    else:
        resolved, _, _ = await resolve_extracted_nodes(
            clients, [extracted], episode=_make_episode(), previous_episodes=[_make_episode()]
        )

    assert resolved[0].uuid == extracted.uuid
    return captured


@pytest.mark.asyncio
async def test_resolve_with_llm_sends_summary_and_extracted_attributes(monkeypatch):
    context = await _captured_dedupe_contexts(monkeypatch, bulk=False)

    assert 'related_facts' not in context['existing_nodes'][0]
    assert context['existing_nodes'][0]['summary'] == CANDIDATE_SUMMARY
    assert context['extracted_nodes'][0]['account_kind'] == 'auto loan'


@pytest.mark.asyncio
async def test_batch_path_builds_same_dedupe_context_shape(monkeypatch):
    single = await _captured_dedupe_contexts(monkeypatch, bulk=False)
    batch = await _captured_dedupe_contexts(monkeypatch, bulk=True)

    assert batch['previous_episodes'] == []
    assert set(batch) == set(single)
    assert set(batch['existing_nodes'][0]) == set(single['existing_nodes'][0])
    assert set(batch['extracted_nodes'][0]) == set(single['extracted_nodes'][0])


def _capture_dedupe_contexts(monkeypatch) -> list[dict]:
    contexts: list[dict] = []

    def fake_prompt_nodes(context):
        contexts.append(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )
    return contexts


def _two_nodes_with_disjoint_candidates():
    extracted = [
        EntityNode(name='Kestrel Analytics', group_id='group', labels=['Entity']),
        EntityNode(name='Juniper Ridge', group_id='group', labels=['Entity']),
    ]
    candidates = [
        EntityNode(name='Northwind Holdings', group_id='group', labels=['Entity']),
        EntityNode(name='Blue Harbor Trust', group_id='group', labels=['Entity']),
    ]
    return extracted, candidates


@pytest.mark.asyncio
async def test_resolve_nodes_per_entity_client_sends_only_own_candidates(monkeypatch):
    clients, llm_generate = _make_clients()
    clients.llm_client.dedupe_nodes_per_entity = True
    extracted, candidates = _two_nodes_with_disjoint_candidates()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidates[0]], [candidates[1]]]),
    )
    contexts = _capture_dedupe_contexts(monkeypatch)
    # Each response also resolves the sibling; only the focus resolution is applied.
    llm_generate.return_value = {
        'entity_resolutions': [
            {'id': 0, 'name': 'match', 'duplicate_candidate_id': 0},
            {'id': 1, 'name': 'match', 'duplicate_candidate_id': 0},
        ]
    }

    resolved, uuid_map, duplicates = await resolve_extracted_nodes(
        clients,
        extracted,
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert llm_generate.await_count == 2
    sent = sorted(
        (
            context['focus_entity_id'],
            [node['name'] for node in context['extracted_nodes']],
            [node['name'] for node in context['existing_nodes']],
        )
        for context in contexts
    )
    assert sent == [
        (0, ['Kestrel Analytics', 'Juniper Ridge'], ['Northwind Holdings']),
        (1, ['Kestrel Analytics', 'Juniper Ridge'], ['Blue Harbor Trust']),
    ]
    assert [node.uuid for node in resolved] == [candidate.uuid for candidate in candidates]
    assert uuid_map == {
        extracted[0].uuid: candidates[0].uuid,
        extracted[1].uuid: candidates[1].uuid,
    }
    assert duplicates == [(extracted[0], candidates[0]), (extracted[1], candidates[1])]


@pytest.mark.asyncio
async def test_resolve_nodes_per_entity_applies_in_node_order(monkeypatch):
    clients, llm_generate = _make_clients()
    clients.llm_client.dedupe_nodes_per_entity = True
    extracted = [
        EntityNode(
            name='Kestrel Analytics', group_id='group', labels=['Entity'], attributes={'hq': 'Oslo'}
        ),
        EntityNode(
            name='Kestrel', group_id='group', labels=['Entity'], attributes={'hq': 'Bergen'}
        ),
    ]
    candidate = EntityNode(name='Kestrel Analytics AS', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate], [candidate]]),
    )
    _capture_dedupe_contexts(monkeypatch)
    calls = 0

    async def respond_first_call_last(*_, **__):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.01)
        return {
            'entity_resolutions': [
                {'id': 0, 'name': 'match', 'duplicate_candidate_id': 0},
                {'id': 1, 'name': 'match', 'duplicate_candidate_id': 0},
            ]
        }

    llm_generate.side_effect = respond_first_call_last

    resolved, _, duplicates = await resolve_extracted_nodes(
        clients,
        extracted,
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert [node.uuid for node in resolved] == [candidate.uuid, candidate.uuid]
    assert candidate.attributes == {'hq': 'Bergen'}
    assert duplicates == [(extracted[0], candidate), (extracted[1], candidate)]


@pytest.mark.asyncio
async def test_resolve_nodes_per_entity_obeys_max_coroutines(monkeypatch):
    clients, llm_generate = _make_clients()
    clients.llm_client.dedupe_nodes_per_entity = True
    clients.max_coroutines = 1
    extracted, candidates = _two_nodes_with_disjoint_candidates()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidates[0]], [candidates[1]]]),
    )
    _capture_dedupe_contexts(monkeypatch)
    in_flight = 0
    max_in_flight = 0

    async def track_concurrency(*_, **__):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return {'entity_resolutions': [{'id': 0, 'name': 'match', 'duplicate_candidate_id': -1}]}

    llm_generate.side_effect = track_concurrency

    await resolve_extracted_nodes(
        clients,
        extracted,
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert llm_generate.await_count == 2
    assert max_in_flight == 1


@pytest.mark.asyncio
async def test_resolve_nodes_default_client_sends_candidate_union(monkeypatch):
    clients, llm_generate = _make_clients()
    extracted, candidates = _two_nodes_with_disjoint_candidates()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidates[0]], [candidates[1]]]),
    )
    contexts = _capture_dedupe_contexts(monkeypatch)
    llm_generate.return_value = {
        'entity_resolutions': [
            {'id': 0, 'name': 'match', 'duplicate_candidate_id': 0},
            {'id': 1, 'name': 'match', 'duplicate_candidate_id': -1},
        ]
    }

    resolved, _, duplicates = await resolve_extracted_nodes(
        clients,
        extracted,
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert llm_generate.await_count == 1
    assert [node['name'] for node in contexts[0]['existing_nodes']] == [
        'Northwind Holdings',
        'Blue Harbor Trust',
    ]
    assert resolved[0].uuid == candidates[0].uuid
    assert resolved[1].uuid == extracted[1].uuid
    assert duplicates == [(extracted[0], candidates[0])]
