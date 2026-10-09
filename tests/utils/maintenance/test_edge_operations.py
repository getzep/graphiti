from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import (
    ExtractedEdges,
)
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations as edge_ops
from graphiti_core.utils.maintenance.edge_operations import (
    _pair_search_filter,
    _same_pair_edges,
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
)


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.generate_response = AsyncMock()
    return client


@pytest.fixture
def mock_extracted_edge():
    return EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='Test fact',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )


@pytest.fixture
def mock_related_edges():
    return [
        EntityEdge(
            source_node_uuid='source_uuid_2',
            target_node_uuid='target_uuid_2',
            name='related_edge',
            group_id='group_1',
            fact='Related fact',
            episodes=['episode_2'],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            valid_at=datetime.now(timezone.utc) - timedelta(days=1),
            invalid_at=None,
        )
    ]


@pytest.fixture
def mock_existing_edges():
    return [
        EntityEdge(
            source_node_uuid='source_uuid_3',
            target_node_uuid='target_uuid_3',
            name='existing_edge',
            group_id='group_1',
            fact='Existing fact',
            episodes=['episode_3'],
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
            valid_at=datetime.now(timezone.utc) - timedelta(days=2),
            invalid_at=None,
        )
    ]


@pytest.fixture
def mock_current_episode():
    return EpisodicNode(
        uuid='episode_1',
        content='Current episode content',
        valid_at=datetime.now(timezone.utc),
        name='Current Episode',
        group_id='group_1',
        source='message',
        source_description='Test source description',
    )


@pytest.fixture
def mock_previous_episodes():
    return [
        EpisodicNode(
            uuid='episode_2',
            content='Previous episode content',
            valid_at=datetime.now(timezone.utc) - timedelta(days=1),
            name='Previous Episode',
            group_id='group_1',
            source='message',
            source_description='Test source description',
        )
    ]


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])


@pytest.mark.asyncio
async def test_resolve_extracted_edge_exact_fact_short_circuit(
    mock_llm_client,
    mock_existing_edges,
    mock_current_episode,
):
    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='Related fact',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    related_edges = [
        EntityEdge(
            source_node_uuid='source_uuid',
            target_node_uuid='target_uuid',
            name='related_edge',
            group_id='group_1',
            fact=' related FACT  ',
            episodes=['episode_2'],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            valid_at=None,
            invalid_at=None,
        )
    ]

    resolved_edge, duplicate_edges, invalidated = await resolve_extracted_edge(
        mock_llm_client,
        extracted,
        related_edges,
        mock_existing_edges,
        mock_current_episode,
        edge_type_candidates=None,
    )

    assert resolved_edge is related_edges[0]
    assert resolved_edge.episodes.count(mock_current_episode.uuid) == 1
    assert duplicate_edges == []
    assert invalidated == []
    mock_llm_client.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_extracted_edge_keeps_caller_attributes_without_schema(
    mock_llm_client,
    mock_existing_edges,
    mock_current_episode,
    monkeypatch,
):
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))
    mock_llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )

    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='VIEWED',
        group_id='group_1',
        fact='Customer viewed listing 42',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
        attributes={'event_name': 'car_viewed', 'event_id': '42'},
    )

    resolved_edge, duplicate_edges, invalidated = await resolve_extracted_edge(
        mock_llm_client,
        extracted,
        related_edges=[],
        existing_edges=mock_existing_edges,
        episode=mock_current_episode,
        edge_type_candidates=None,
    )

    assert resolved_edge.attributes == {'event_name': 'car_viewed', 'event_id': '42'}
    assert duplicate_edges == []
    assert invalidated == []


@pytest.mark.asyncio
async def test_resolve_extracted_edge_clears_attributes_when_schema_omits_type(
    mock_llm_client,
    mock_existing_edges,
    mock_current_episode,
    monkeypatch,
):
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))
    mock_llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )

    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='VIEWED',
        group_id='group_1',
        fact='Customer viewed listing 42',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
        attributes={'event_name': 'car_viewed'},
    )

    resolved_edge, _, _ = await resolve_extracted_edge(
        mock_llm_client,
        extracted,
        related_edges=[],
        existing_edges=mock_existing_edges,
        episode=mock_current_episode,
        edge_type_candidates={},
    )

    assert resolved_edge.attributes == {}


class OccurredAtEdge(BaseModel):
    """Edge model stub for OCCURRED_AT."""


@pytest.mark.asyncio
async def test_resolve_extracted_edges_keeps_unknown_names(monkeypatch):
    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'duplicate_facts': [],
            'contradicted_facts': [],
        }
    )

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    source_node = EntityNode(
        uuid='source_uuid',
        name='User Node',
        group_id='group_1',
        labels=['User'],
    )
    target_node = EntityNode(
        uuid='target_uuid',
        name='Topic Node',
        group_id='group_1',
        labels=['Topic'],
    )

    extracted_edge = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='INTERACTED_WITH',
        group_id='group_1',
        fact='User interacted with topic',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    edge_types = {'OCCURRED_AT': OccurredAtEdge}
    edge_type_map = {('Event', 'Entity'): ['OCCURRED_AT']}

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        clients,
        [extracted_edge],
        episode,
        [source_node, target_node],
        edge_types,
        edge_type_map,
    )

    assert resolved_edges[0].name == 'INTERACTED_WITH'
    assert invalidated_edges == []
    assert new_edges == resolved_edges  # No duplicates, so all edges are new


class WorksInEdge(BaseModel):
    """A person works in a domain."""


class MeansEdge(BaseModel):
    """A term means something."""


class MentionsEdge(BaseModel):
    """Any entity mentions any other entity. No signature is declared."""


_SIGNATURE_EDGE_TYPES: dict[str, type[BaseModel]] = {
    'WORKS_IN': WorksInEdge,
    'MEANS': MeansEdge,
    'MENTIONS': MentionsEdge,
}
_SIGNATURE_EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ('Person', 'Domain'): ['WORKS_IN'],
    ('User', 'Domain'): ['WORKS_IN'],
    ('Term', 'Rule'): ['MEANS'],
    ('Term', 'Domain'): ['MEANS'],
}


def _signature_node(uuid: str, label: str) -> EntityNode:
    return EntityNode(uuid=uuid, name=uuid, group_id='group_1', labels=[label])


def _signature_edge(name: str, source: EntityNode, target: EntityNode) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name=name,
        group_id='group_1',
        fact=f'{source.name} {name} {target.name}',
        attributes={'note': 'x'},
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )


def _enforce(edges: list[EntityEdge], nodes: list[EntityNode], strict: bool):
    uuid_entity_map = {node.uuid: node for node in nodes}
    edge_types_lst = [
        edge_ops._edge_types_for_endpoints(
            edge, uuid_entity_map, _SIGNATURE_EDGE_TYPES, _SIGNATURE_EDGE_TYPE_MAP
        )
        for edge in edges
    ]
    return edge_ops.enforce_edge_type_signatures(
        edges, edge_types_lst, _SIGNATURE_EDGE_TYPE_MAP, strict
    )


def test_enforce_edge_type_signatures_keeps_matching_custom_edge():
    person = _signature_node('person', 'Person')
    domain = _signature_node('domain', 'Domain')
    edge = _signature_edge('WORKS_IN', person, domain)

    for strict in (True, False):
        kept, kept_types = _enforce([edge], [person, domain], strict)
        assert kept == [edge]
        assert edge.name == 'WORKS_IN'
        assert edge.attributes == {'note': 'x'}
        assert kept_types == [{'WORKS_IN': WorksInEdge}]


def test_enforce_edge_type_signatures_accepts_any_declared_signature():
    user = _signature_node('user', 'User')
    term = _signature_node('term', 'Term')
    domain = _signature_node('domain', 'Domain')
    edges = [
        _signature_edge('WORKS_IN', user, domain),
        _signature_edge('MEANS', term, domain),
    ]

    kept, _ = _enforce(edges, [user, term, domain], strict=True)

    assert [edge.name for edge in kept] == ['WORKS_IN', 'MEANS']


def test_enforce_edge_type_signatures_strict_drops_mismatched_custom_edge():
    person = _signature_node('person', 'Person')
    org = _signature_node('org', 'Organization')
    valid_domain = _signature_node('domain', 'Domain')
    wrong = _signature_edge('WORKS_IN', person, org)
    valid = _signature_edge('WORKS_IN', person, valid_domain)

    kept, kept_types = _enforce([wrong, valid], [person, org, valid_domain], strict=True)

    assert kept == [valid]
    assert kept_types == [{'WORKS_IN': WorksInEdge}]


def test_enforce_edge_type_signatures_non_strict_renames_mismatched_custom_edge():
    person = _signature_node('person', 'Person')
    org = _signature_node('org', 'Organization')
    wrong = _signature_edge('WORKS_IN', person, org)

    kept, kept_types = _enforce([wrong], [person, org], strict=False)

    assert kept == [wrong]
    assert wrong.name == edge_ops.DEFAULT_EDGE_NAME
    assert wrong.attributes == {}
    assert kept_types == [{}]


def test_enforce_edge_type_signatures_ignores_non_custom_names():
    person = _signature_node('person', 'Person')
    org = _signature_node('org', 'Organization')
    edge = _signature_edge('EMPLOYED_BY', person, org)

    for strict in (True, False):
        kept, _ = _enforce([edge], [person, org], strict)
        assert kept == [edge]
        assert edge.name == 'EMPLOYED_BY'


def test_enforce_edge_type_signatures_ignores_custom_type_without_signature():
    person = _signature_node('person', 'Person')
    org = _signature_node('org', 'Organization')
    edge = _signature_edge('MENTIONS', person, org)

    for strict in (True, False):
        kept, kept_types = _enforce([edge], [person, org], strict)
        assert kept == [edge]
        assert edge.name == 'MENTIONS'
        assert edge.attributes == {'note': 'x'}
        assert kept_types == [{}]


def test_enforce_edge_type_signatures_treats_unknown_endpoint_as_generic_entity():
    person = _signature_node('person', 'Person')
    missing = _signature_node('missing', 'Domain')
    edge = _signature_edge('WORKS_IN', person, missing)

    kept, _ = _enforce([edge], [person], strict=True)

    assert kept == []


@pytest.mark.asyncio
async def test_resolve_extracted_edges_applies_signature_before_dedupe(monkeypatch):
    embed = AsyncMock(return_value=None)
    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', embed)
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )
    clients = SimpleNamespace(
        driver=MagicMock(), llm_client=llm_client, embedder=MagicMock(), cross_encoder=MagicMock()
    )

    person = _signature_node('person', 'Person')
    org = _signature_node('org', 'Organization')
    domain = _signature_node('domain', 'Domain')
    wrong = _signature_edge('WORKS_IN', person, org)
    valid = _signature_edge('WORKS_IN', person, domain)
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    resolved, _, new_edges = await resolve_extracted_edges(
        clients,
        [wrong, valid],
        episode,
        [person, org, domain],
        _SIGNATURE_EDGE_TYPES,
        _SIGNATURE_EDGE_TYPE_MAP,
        strict_edge_types=True,
    )

    assert [edge.uuid for edge in resolved] == [valid.uuid]
    assert new_edges == resolved
    assert [edge.uuid for edge in embed.await_args_list[0].args[1]] == [valid.uuid]

    renamed = _signature_edge('WORKS_IN', person, org)
    resolved, _, _ = await resolve_extracted_edges(
        clients,
        [renamed],
        episode,
        [person, org],
        _SIGNATURE_EDGE_TYPES,
        _SIGNATURE_EDGE_TYPE_MAP,
        strict_edge_types=False,
    )

    assert [edge.name for edge in resolved] == [edge_ops.DEFAULT_EDGE_NAME]

    # Two edges with the same endpoints and fact but different names: the
    # mismatched name is dropped first, so the exact-fact dedupe keeps the valid one.
    term = _signature_node('term', 'Term')
    same_fact_wrong = _signature_edge('WORKS_IN', term, domain)
    same_fact_valid = _signature_edge('MEANS', term, domain)
    same_fact_valid.fact = same_fact_wrong.fact
    resolved, _, _ = await resolve_extracted_edges(
        clients,
        [same_fact_wrong, same_fact_valid],
        episode,
        [term, domain],
        _SIGNATURE_EDGE_TYPES,
        _SIGNATURE_EDGE_TYPE_MAP,
        strict_edge_types=True,
    )

    assert [(edge.uuid, edge.name) for edge in resolved] == [(same_fact_valid.uuid, 'MEANS')]


@pytest.mark.asyncio
async def test_resolve_extracted_edge_uses_integer_indices_for_duplicates(mock_llm_client):
    """Test that resolve_extracted_edge correctly uses integer indices for LLM duplicate detection."""
    # Mock LLM to return duplicate_facts with integer indices
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [0, 1],  # LLM identifies first two related edges as duplicates
        'contradicted_facts': [],
    }

    extracted_edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User likes yoga',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    # Create multiple related edges - LLM should receive these with integer indices
    related_edge_0 = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User enjoys yoga',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        valid_at=None,
        invalid_at=None,
    )

    related_edge_1 = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User practices yoga',
        episodes=['episode_2'],
        created_at=datetime.now(timezone.utc) - timedelta(days=2),
        valid_at=None,
        invalid_at=None,
    )

    related_edge_2 = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User loves swimming',
        episodes=['episode_3'],
        created_at=datetime.now(timezone.utc) - timedelta(days=3),
        valid_at=None,
        invalid_at=None,
    )

    related_edges = [related_edge_0, related_edge_1, related_edge_2]

    resolved_edge, invalidated, duplicates = await resolve_extracted_edge(
        mock_llm_client,
        extracted_edge,
        related_edges,
        [],
        episode,
        edge_type_candidates=None,
    )

    # Verify LLM was called
    mock_llm_client.generate_response.assert_called_once()

    # Verify the system correctly identified duplicates using integer indices
    # The LLM returned [0, 1], so related_edge_0 and related_edge_1 should be marked as duplicates
    assert len(duplicates) == 2
    assert related_edge_0 in duplicates
    assert related_edge_1 in duplicates
    assert invalidated == []

    # Verify that the resolved edge is one of the duplicates (the first one found)
    # Check UUID since the episode list gets modified
    assert resolved_edge.uuid == related_edge_0.uuid
    assert episode.uuid in resolved_edge.episodes


@pytest.mark.asyncio
async def test_resolve_extracted_edges_fast_path_deduplication(monkeypatch):
    """Test that resolve_extracted_edges deduplicates exact matches before parallel processing."""
    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))

    # Track how many times resolve_extracted_edge is called
    resolve_call_count = 0

    async def mock_resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
    ):
        nonlocal resolve_call_count
        resolve_call_count += 1
        return extracted_edge, [], None

    # Mock semaphore_gather to execute awaitable immediately
    async def immediate_gather(*aws, max_coroutines=None):
        results = []
        for aw in aws:
            results.append(await aw)
        return results

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))
    monkeypatch.setattr(edge_ops, '_dedupe_extracted_edge', mock_resolve_extracted_edge)

    llm_client = MagicMock()
    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    source_node = EntityNode(
        uuid='source_uuid',
        name='Assistant',
        group_id='group_1',
        labels=['Entity'],
    )
    target_node = EntityNode(
        uuid='target_uuid',
        name='User',
        group_id='group_1',
        labels=['Entity'],
    )

    # Create 3 identical edges
    edge1 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='assistant recommends yoga poses',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    edge2 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='  Assistant Recommends YOGA Poses  ',  # Different whitespace/case
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    edge3 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='assistant recommends yoga poses',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        clients,
        [edge1, edge2, edge3],
        episode,
        [source_node, target_node],
        {},
        {},
    )

    # Fast path should have deduplicated the 3 identical edges to 1
    # So resolve_extracted_edge should only be called once
    assert resolve_call_count == 1
    assert len(resolved_edges) == 1
    assert invalidated_edges == []
    assert new_edges == resolved_edges  # All edges are new (no graph duplicates)


@pytest.mark.parametrize(
    ('reverse_fact', 'reverse_name', 'expected_edge_uuids'),
    [
        ('  Assistant Recommends YOGA Poses  ', 'recommends', ['edge-1']),
        ('assistant likes yoga poses', 'recommends', ['edge-1', 'edge-2']),
        ('  Assistant Recommends YOGA Poses  ', 'recommended_by', ['edge-1', 'edge-2']),
    ],
)
async def test_resolve_extracted_edges_fast_path_deduplicates_reverse_direction_by_fact(
    monkeypatch, reverse_fact, reverse_name, expected_edge_uuids
):
    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))

    resolve_call_count = 0

    async def mock_resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
    ):
        nonlocal resolve_call_count
        resolve_call_count += 1
        return extracted_edge, [], None

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    search_mock = AsyncMock(return_value=SearchResults())
    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', search_mock)
    monkeypatch.setattr(edge_ops, '_dedupe_extracted_edge', mock_resolve_extracted_edge)

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )
    source_node = EntityNode(
        uuid='source_uuid',
        name='Assistant',
        group_id='group_1',
        labels=['Entity'],
    )
    target_node = EntityNode(
        uuid='target_uuid',
        name='User',
        group_id='group_1',
        labels=['Entity'],
    )
    edge1 = EntityEdge(
        uuid='edge-1',
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='assistant recommends yoga poses',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    edge2 = EntityEdge(
        uuid='edge-2',
        source_node_uuid=target_node.uuid,
        target_node_uuid=source_node.uuid,
        name=reverse_name,
        group_id='group_1',
        fact=reverse_fact,
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        clients,
        [edge1, edge2],
        episode,
        [source_node, target_node],
        {},
        {},
    )

    assert [edge.uuid for edge in resolved_edges] == expected_edge_uuids
    assert resolved_edges[0] is edge1
    assert resolve_call_count == len(expected_edge_uuids)
    assert search_mock.await_count == 2 * len(expected_edge_uuids)
    assert invalidated_edges == []
    assert new_edges == resolved_edges


class InterpersonalRelationship(BaseModel):
    """A relationship between two people."""


class LocatedIn(BaseModel):
    """A relationship indicating something is located in a place."""


def test_edge_type_signatures_map_preserves_multiple_signatures():
    """Test that edge types used across multiple node type pairs preserve all signatures.

    This tests the fix for the bug where dict comprehension would overwrite
    previous signatures when the same edge type appeared in multiple node pairs.
    """
    # Edge type map where the same edge type is used for multiple node pair signatures
    # This is the scenario that was broken before the fix
    edge_type_map: dict[tuple[str, str], list[str]] = {
        ('Person', 'Person'): ['InterpersonalRelationship'],
        ('Person', 'Entity'): ['InterpersonalRelationship'],  # Same type, different signature
        ('Person', 'City'): ['LocatedIn'],
        ('Entity', 'City'): ['LocatedIn'],  # Same type, different signature
    }

    edge_types: dict[str, type[BaseModel]] = {
        'InterpersonalRelationship': InterpersonalRelationship,
        'LocatedIn': LocatedIn,
    }

    # Build the mapping the same way as in extract_edges (the fixed implementation)
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    # Verify InterpersonalRelationship has BOTH signatures preserved
    assert 'InterpersonalRelationship' in edge_type_signatures_map
    interpersonal_signatures = edge_type_signatures_map['InterpersonalRelationship']
    assert len(interpersonal_signatures) == 2
    assert ('Person', 'Person') in interpersonal_signatures
    assert ('Person', 'Entity') in interpersonal_signatures

    # Verify LocatedIn has BOTH signatures preserved
    assert 'LocatedIn' in edge_type_signatures_map
    located_signatures = edge_type_signatures_map['LocatedIn']
    assert len(located_signatures) == 2
    assert ('Person', 'City') in located_signatures
    assert ('Entity', 'City') in located_signatures

    # Verify the edge_types_context structure
    edge_types_context = [
        {
            'fact_type_name': type_name,
            'fact_type_signatures': edge_type_signatures_map.get(type_name, [('Entity', 'Entity')]),
            'fact_type_description': type_model.__doc__,
        }
        for type_name, type_model in edge_types.items()
    ]

    # Verify the context has the correct structure with plural 'fact_type_signatures'
    for ctx in edge_types_context:
        assert 'fact_type_signatures' in ctx
        assert isinstance(ctx['fact_type_signatures'], list)
        assert len(ctx['fact_type_signatures']) == 2  # Each type has 2 signatures


def test_edge_type_signatures_map_single_signature_still_works():
    """Test that edge types with a single signature still work correctly."""
    edge_type_map: dict[tuple[str, str], list[str]] = {
        ('Person', 'Organization'): ['WorksAt'],
        ('Person', 'City'): ['LivesIn'],
    }

    edge_types: dict[str, type[BaseModel]] = {
        'WorksAt': BaseModel,
        'LivesIn': BaseModel,
    }

    # Build the mapping
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    # Verify each edge type has exactly one signature
    assert len(edge_type_signatures_map['WorksAt']) == 1
    assert ('Person', 'Organization') in edge_type_signatures_map['WorksAt']

    assert len(edge_type_signatures_map['LivesIn']) == 1
    assert ('Person', 'City') in edge_type_signatures_map['LivesIn']

    # Verify the context structure
    edge_types_context = [
        {
            'fact_type_name': type_name,
            'fact_type_signatures': edge_type_signatures_map.get(type_name, [('Entity', 'Entity')]),
            'fact_type_description': type_model.__doc__,
        }
        for type_name, type_model in edge_types.items()
    ]

    for ctx in edge_types_context:
        assert 'fact_type_signatures' in ctx
        assert isinstance(ctx['fact_type_signatures'], list)
        assert len(ctx['fact_type_signatures']) == 1


@pytest.mark.asyncio
async def test_extract_edges_keeps_self_edges(monkeypatch):
    """Self-edges (source == target) are kept during extraction."""
    from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
    from graphiti_core.prompts.extract_edges import ExtractedEdges

    alice = EntityNode(
        uuid='alice_uuid',
        name='Alice',
        group_id='group_1',
        labels=['Person'],
    )
    bob = EntityNode(
        uuid='bob_uuid',
        name='Bob',
        group_id='group_1',
        labels=['Person'],
    )

    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='Alice',
                target_entity_name='Bob',
                relation_type='CONGRATULATED',
                fact='Alice congratulated Bob',
                valid_at=None,
                invalid_at=None,
            ),
            ExtractedEdge(
                source_entity_name='Alice',
                target_entity_name='Alice',
                relation_type='FEELS_HAPPY',
                fact='Alice feels happy',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Alice congratulated Bob. Alice feels happy.',
        valid_at=datetime.now(timezone.utc),
    )

    edges, _materialized = await extract_edges(
        clients,
        episode,
        [alice, bob],
        [],
        {},
        group_id='group_1',
    )

    assert {(e.source_node_uuid, e.target_node_uuid, e.name) for e in edges} == {
        ('alice_uuid', 'bob_uuid', 'CONGRATULATED'),
        ('alice_uuid', 'alice_uuid', 'FEELS_HAPPY'),
    }


@pytest.mark.asyncio
async def test_extract_edges_keeps_valid_edges_with_same_name_different_nodes(monkeypatch):
    """Edges between different nodes that happen to share a name are NOT self-edges."""
    alice = EntityNode(
        uuid='alice_uuid',
        name='Alice',
        group_id='group_1',
        labels=['Person'],
    )
    paris = EntityNode(
        uuid='paris_uuid',
        name='Paris',
        group_id='group_1',
        labels=['City'],
    )

    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='Alice',
                target_entity_name='Paris',
                relation_type='LIVES_IN',
                fact='Alice lives in Paris',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Alice lives in Paris.',
        valid_at=datetime.now(timezone.utc),
    )

    edges, _materialized = await extract_edges(
        clients,
        episode,
        [alice, paris],
        [],
        {},
        group_id='group_1',
    )

    assert len(edges) == 1
    assert edges[0].source_node_uuid == 'alice_uuid'
    assert edges[0].target_node_uuid == 'paris_uuid'


@pytest.mark.asyncio
async def test_extract_edges_materializes_missing_target_endpoint():
    """Dangling target names are materialized onto the nodes list; the edge is kept."""
    ami = EntityNode(
        uuid='ami_uuid',
        name='AMI',
        group_id='group_1',
        labels=['Entity'],
    )
    nodes = [ami]

    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='AMI',
                target_entity_name='version 2.18.1',
                relation_type='PINNED_AT',
                fact='AMI is pinned at version 2.18.1.',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='AMI is pinned at version 2.18.1.',
        valid_at=datetime.now(timezone.utc),
    )

    edges, materialized = await extract_edges(
        clients,
        episode,
        nodes,
        [],
        {},
        group_id='group_1',
    )

    assert len(edges) == 1
    assert edges[0].name == 'PINNED_AT'
    assert edges[0].source_node_uuid == 'ami_uuid'
    assert {n.name for n in nodes} == {'AMI', 'version 2.18.1'}
    assert len(materialized) == 1
    assert materialized[0].name == 'version 2.18.1'
    assert edges[0].target_node_uuid == materialized[0].uuid
    assert edges[0].source_node_uuid != edges[0].target_node_uuid


@pytest.mark.asyncio
async def test_extract_edges_does_not_keep_orphans_when_edge_filtered():
    """Materialized endpoints are removed when the edge is later dropped."""

    class PINNED_AT(BaseModel):
        """Software is pinned at a version."""

        pass

    ami = EntityNode(
        uuid='ami_uuid',
        name='AMI',
        group_id='group_1',
        labels=['Entity'],
    )
    nodes = [ami]

    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='AMI',
                target_entity_name='version 2.18.1',
                relation_type='NOT_IN_ONTOLOGY',
                fact='AMI is pinned at version 2.18.1.',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)
    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )
    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='AMI is pinned at version 2.18.1.',
        valid_at=datetime.now(timezone.utc),
    )

    edges, materialized = await extract_edges(
        clients,
        episode,
        nodes,
        [],
        {('Entity', 'Entity'): ['PINNED_AT']},
        group_id='group_1',
        edge_types={'PINNED_AT': PINNED_AT},
        strict_edge_types=True,
    )

    assert edges == []
    assert materialized == []
    assert [n.name for n in nodes] == ['AMI']


@pytest.mark.asyncio
async def test_extract_edges_strict_edge_types_filters_derived_relations():
    """Strict edge extraction should only keep configured edge ontology types."""

    class INTEGRATES_WITH(BaseModel):
        """One product integrates with another product."""

        pass

    atlas = EntityNode(
        uuid='atlas_uuid',
        name='Atlas Home Robot',
        group_id='group_1',
        labels=['Entity', 'Product'],
    )
    hub = EntityNode(
        uuid='hub_uuid',
        name='Acme Smart Hub',
        group_id='group_1',
        labels=['Entity', 'Product'],
    )

    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='Atlas Home Robot',
                target_entity_name='Acme Smart Hub',
                relation_type='INTEGRATES_WITH',
                fact='Atlas Home Robot integrates with Acme Smart Hub.',
                valid_at=None,
                invalid_at=None,
            ),
            ExtractedEdge(
                source_entity_name='Atlas Home Robot',
                target_entity_name='Acme Smart Hub',
                relation_type='SOLD_WITH',
                fact='Atlas Home Robot is sold with Acme Smart Hub.',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)
    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Atlas Home Robot integrates with Acme Smart Hub.',
        valid_at=datetime.now(timezone.utc),
    )

    edges, _materialized = await extract_edges(
        clients,
        episode,
        [atlas, hub],
        [],
        {('Product', 'Product'): ['INTEGRATES_WITH']},
        group_id='group_1',
        edge_types={'INTEGRATES_WITH': INTEGRATES_WITH},
        strict_edge_types=True,
    )

    assert [edge.name for edge in edges] == ['INTEGRATES_WITH']
    prompt_text = '\n'.join(
        message.content for message in mock_llm.generate_response.call_args.args[0]
    )
    assert 'If a relationship does not match any FACT_TYPE, skip it' in prompt_text
    assert 'Do not derive a new relation_type' in prompt_text


class _EmploymentEdge(BaseModel):
    """Edge attribute schema mirroring a customer's EMPLOYMENT relation."""

    title: str | None = None
    is_current: str | None = None


@pytest.mark.asyncio
async def test_resolve_extracted_edge_overcap_attribute_preserves_prior(monkeypatch):
    """End-to-end: when the LLM bleeds meta-reasoning into one attribute, the
    cap drops that field and the merge logic falls back to the prior on-edge
    value — without affecting fields the LLM legitimately updated."""
    # Avoid the timestamps LLM call that follows attribute extraction.
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))

    # The LLM returns a clean update for `title` but bleeds 9 KB of meta-reasoning into
    # `is_current` — the kind of failure the customer reported.
    bleed = (
        'true (implied by context, but no new information explicitly stated. '
        'I should ideally remove it or set it to null. However, the instruction '
        'is to preserve existing values...) '
    ) * 30
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'title': 'Senior Engineer', 'is_current': bleed}
    )

    extracted_edge = EntityEdge(
        source_node_uuid='person_uuid',
        target_node_uuid='company_uuid',
        name='EMPLOYMENT',
        group_id='group_1',
        fact='Sam was promoted to Senior Engineer at Northwind',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
        attributes={'title': 'Engineer', 'is_current': 'true'},
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Sam was promoted to Senior Engineer.',
        valid_at=datetime.now(timezone.utc),
    )

    resolved, dupes, invalidated = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges=[],
        existing_edges=[],
        episode=episode,
        edge_type_candidates={'EMPLOYMENT': _EmploymentEdge},
    )

    # Title should reflect the legitimate LLM update.
    assert resolved.attributes['title'] == 'Senior Engineer'
    # is_current was bleed-dropped, so the prior value must be preserved (not the
    # 9 KB rant, not None, not absent).
    assert resolved.attributes['is_current'] == 'true'
    assert dupes == []
    assert invalidated == []


@pytest.mark.asyncio
async def test_resolve_extracted_edge_keeps_initial_null_attributes(monkeypatch):
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'title': 'Senior Engineer', 'is_current': None}
    )

    extracted_edge = EntityEdge(
        source_node_uuid='person_uuid',
        target_node_uuid='company_uuid',
        name='EMPLOYMENT',
        group_id='group_1',
        fact='Sam is a Senior Engineer at Northwind',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Sam is a Senior Engineer at Northwind.',
        valid_at=datetime.now(timezone.utc),
    )

    resolved, dupes, invalidated = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges=[],
        existing_edges=[],
        episode=episode,
        edge_type_candidates={'EMPLOYMENT': _EmploymentEdge},
    )

    assert resolved.attributes == {'title': 'Senior Engineer', 'is_current': None}
    assert dupes == []
    assert invalidated == []


def _candidate_edge(idx: int, fact: str, embedding: list[float] | None, age_minutes: int = 0):
    return EntityEdge(
        uuid=f'candidate_{idx}',
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='relates_to',
        group_id='group_1',
        fact=fact,
        fact_embedding=embedding,
        episodes=[],
        created_at=datetime.now(timezone.utc) - timedelta(minutes=age_minutes),
        valid_at=None,
        invalid_at=None,
    )


@pytest.mark.asyncio
async def test_resolve_extracted_edges_dedupes_against_reverse_direction(monkeypatch):
    """An extracted (A -> B) edge must see an existing (B -> A) edge as a
    duplicate candidate; the directed exact-fact fast path can't claim it, so
    the LLM judges and the extracted edge resolves onto the reverse edge."""
    reverse_edge = EntityEdge(
        uuid='reverse_0',
        source_node_uuid='target_uuid',
        target_node_uuid='source_uuid',
        name='liked_by',
        group_id='group_1',
        fact='Yoga is liked by the user',
        episodes=[],
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        valid_at=None,
        invalid_at=None,
    )

    async def fake_search(clients, query, group_ids=None, config=None, search_filter=None, **kw):
        # The pair-scoped duplicate search returns the reverse-orientation
        # edge (its endpoints are within the {a,b} filter sets).
        if getattr(search_filter, 'edge_source_node_uuids', None):
            return SearchResults(edges=[reverse_edge])
        return SearchResults()

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(side_effect=fake_search))

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [0], 'contradicted_facts': []}
    )

    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='likes',
        group_id='group_1',
        fact='User likes yoga',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )
    nodes = [
        EntityNode(uuid='source_uuid', name='User', group_id='group_1', labels=['Entity']),
        EntityNode(uuid='target_uuid', name='Yoga', group_id='group_1', labels=['Entity']),
    ]

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        clients=SimpleNamespace(
            driver=MagicMock(),
            llm_client=llm_client,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
        ),
        extracted_edges=[extracted],
        episode=episode,
        entities=nodes,
        edge_types={},
        edge_type_map={},
    )

    assert resolved_edges[0].uuid == reverse_edge.uuid
    assert episode.uuid in resolved_edges[0].episodes
    assert invalidated_edges == []
    assert new_edges == []


def test_same_pair_edges_keeps_both_orientations_drops_strays():
    extracted = _candidate_edge(0, 'fact', None)
    extracted.source_node_uuid = 'a'
    extracted.target_node_uuid = 'b'

    forward = _candidate_edge(1, 'forward', None)
    forward.source_node_uuid, forward.target_node_uuid = 'a', 'b'
    reverse = _candidate_edge(2, 'reverse', None)
    reverse.source_node_uuid, reverse.target_node_uuid = 'b', 'a'
    self_loop = _candidate_edge(3, 'self loop', None)
    self_loop.source_node_uuid, self_loop.target_node_uuid = 'a', 'a'
    stray = _candidate_edge(4, 'stray', None)
    stray.source_node_uuid, stray.target_node_uuid = 'a', 'c'

    result = _same_pair_edges(extracted, [forward, reverse, self_loop, stray])

    assert [e.uuid for e in result] == ['candidate_1', 'candidate_2']


def test_same_pair_edges_self_loop_pair():
    extracted = _candidate_edge(0, 'fact', None)
    extracted.source_node_uuid = 'a'
    extracted.target_node_uuid = 'a'
    self_loop = _candidate_edge(1, 'self loop', None)
    self_loop.source_node_uuid, self_loop.target_node_uuid = 'a', 'a'
    other = _candidate_edge(2, 'other', None)
    other.source_node_uuid, other.target_node_uuid = 'a', 'b'

    result = _same_pair_edges(extracted, [self_loop, other])

    assert [e.uuid for e in result] == ['candidate_1']


def test_pair_search_filter_sets_both_endpoint_lists(mock_extracted_edge):
    search_filter = _pair_search_filter(mock_extracted_edge)

    assert search_filter.edge_source_node_uuids == ['source_uuid', 'target_uuid']
    assert search_filter.edge_target_node_uuids == ['source_uuid', 'target_uuid']
    assert search_filter.edge_uuids is None


@pytest.mark.asyncio
async def test_resolve_extracted_edges_pair_scopes_duplicate_search(monkeypatch):
    """The duplicate-candidate search must be endpoint-pre-filtered (never a
    graph-wide retrieval); the invalidation search stays unfiltered."""
    existing = _candidate_edge(0, 'User likes yoga', None)
    existing.source_node_uuid = 'source_uuid'
    existing.target_node_uuid = 'target_uuid'

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)

    captured_filters = []

    async def fake_search(clients, query, group_ids=None, config=None, search_filter=None, **kw):
        captured_filters.append(search_filter)
        if getattr(search_filter, 'edge_source_node_uuids', None):
            return SearchResults(edges=[existing])
        return SearchResults()

    monkeypatch.setattr(edge_ops, 'search', AsyncMock(side_effect=fake_search))

    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='likes',
        group_id='group_1',
        fact='  user LIKES yoga ',  # normalizes to the existing fact
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )
    nodes = [
        EntityNode(uuid='source_uuid', name='User', group_id='group_1', labels=['Entity']),
        EntityNode(uuid='target_uuid', name='Yoga', group_id='group_1', labels=['Entity']),
    ]

    resolved_edges, invalidated_edges, _ = await resolve_extracted_edges(
        clients=SimpleNamespace(
            driver=MagicMock(),
            llm_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
        ),
        extracted_edges=[extracted],
        episode=episode,
        entities=nodes,
        edge_types={},
        edge_type_map={},
    )

    # Exactly two searches: the pair-scoped duplicate search, then the
    # graph-wide invalidation search.
    assert len(captured_filters) == 2
    pair_filter, invalidation_filter = captured_filters
    assert pair_filter.edge_source_node_uuids == ['source_uuid', 'target_uuid']
    assert pair_filter.edge_target_node_uuids == ['source_uuid', 'target_uuid']
    assert invalidation_filter.edge_source_node_uuids is None
    assert invalidation_filter.edge_uuids is None
    # The pair edge resolved via the exact-fact fast path.
    assert resolved_edges[0].uuid == existing.uuid
    assert invalidated_edges == []


@pytest.mark.asyncio
async def test_resolve_extracted_edges_merges_override_candidates(monkeypatch):
    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))

    seen_related: list[list[EntityEdge]] = []

    async def capture_resolve(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
        **kwargs,
    ):
        seen_related.append(related_edges)
        return extracted_edge, [], None

    monkeypatch.setattr(edge_ops, '_dedupe_extracted_edge', capture_resolve)

    override = EntityEdge(
        uuid='override_edge',
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='likes',
        group_id='group_1',
        fact='User likes yoga',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='likes',
        group_id='group_1',
        fact='User likes yoga',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    nodes = [
        EntityNode(uuid='source_uuid', name='User', group_id='group_1', labels=['Entity']),
        EntityNode(uuid='target_uuid', name='Yoga', group_id='group_1', labels=['Entity']),
    ]
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    await resolve_extracted_edges(
        clients=SimpleNamespace(
            driver=MagicMock(),
            llm_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
        ),
        extracted_edges=[extracted],
        episode=episode,
        entities=nodes,
        edge_types={},
        edge_type_map={},
        existing_edges_override=[override],
    )

    assert len(seen_related) == 1
    assert [edge.uuid for edge in seen_related[0]] == ['override_edge']


@pytest.mark.asyncio
async def test_resolve_extracted_edges_reuses_fact_embedding_for_searches(monkeypatch):
    """Both per-edge searches must receive the precomputed fact embedding as
    query_vector, so the search pipeline never re-embeds the same text."""
    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)

    captured_vectors = []

    async def fake_search(
        clients, query, group_ids=None, config=None, search_filter=None, query_vector=None, **kw
    ):
        captured_vectors.append(query_vector)
        return SearchResults()

    monkeypatch.setattr(edge_ops, 'search', AsyncMock(side_effect=fake_search))

    sentinel_embedding = [0.1, 0.2, 0.3]
    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='likes',
        group_id='group_1',
        fact='User likes yoga',
        fact_embedding=sentinel_embedding,
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )
    nodes = [
        EntityNode(uuid='source_uuid', name='User', group_id='group_1', labels=['Entity']),
        EntityNode(uuid='target_uuid', name='Yoga', group_id='group_1', labels=['Entity']),
    ]
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )

    await resolve_extracted_edges(
        clients=SimpleNamespace(
            driver=MagicMock(),
            llm_client=llm_client,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
        ),
        extracted_edges=[extracted],
        episode=episode,
        entities=nodes,
        edge_types={},
        edge_type_map={},
    )

    # Duplicate search + invalidation search, both with the precomputed vector.
    assert captured_vectors == [sentinel_embedding, sentinel_embedding]


@pytest.mark.asyncio
async def test_resolve_extracted_edge_retries_timestamps_for_untagged_pairwise(monkeypatch):
    helper = AsyncMock(return_value=None)
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', helper)

    llm_client = MagicMock()
    extracted_edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='KNOWS',
        group_id='group_1',
        fact='Alice knows Bob.',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Alice knows Bob.',
        valid_at=datetime.now(timezone.utc),
    )

    await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges=[],
        existing_edges=[],
        episode=episode,
    )

    helper.assert_awaited_once_with(llm_client, extracted_edge, episode)


@pytest.mark.asyncio
async def test_extract_edge_timestamps_prefers_edge_reference_time(monkeypatch):
    edge_reference_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    episode_reference_time = datetime(2024, 2, 1, tzinfo=timezone.utc)
    edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='KNOWS',
        group_id='group_1',
        fact='Alice knows Bob.',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        reference_time=edge_reference_time,
    )
    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content=edge.fact,
        valid_at=episode_reference_time,
    )
    prompt = MagicMock(return_value=[])
    monkeypatch.setattr(edge_ops.prompt_library.extract_edges, 'extract_timestamps', prompt)
    apply_timestamps = MagicMock()
    monkeypatch.setattr(edge_ops, 'apply_extracted_timestamps', apply_timestamps)
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'valid_at': '2024-01-02T00:00:00Z', 'invalid_at': None}
    )

    await edge_ops._extract_edge_timestamps(llm_client, edge, episode)

    prompt.assert_called_once_with(
        {
            'fact': edge.fact,
            'reference_time': edge_reference_time.isoformat(),
        }
    )
    apply_timestamps.assert_called_once_with(
        edge,
        '2024-01-02T00:00:00Z',
        None,
        edge_reference_time,
    )
