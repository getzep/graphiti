from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance.edge_operations import (
    DEFAULT_EDGE_NAME,
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


class OccurredAtEdge(BaseModel):
    """Edge model stub for OCCURRED_AT."""


@pytest.mark.asyncio
async def test_resolve_extracted_edges_resets_unmapped_names(monkeypatch):
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

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
            'fact_type': 'DEFAULT',
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
        name='Document Node',
        group_id='group_1',
        labels=['Document'],
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
        name='OCCURRED_AT',
        group_id='group_1',
        fact='Document occurred at somewhere',
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

    resolved_edges, invalidated_edges = await resolve_extracted_edges(
        clients,
        [extracted_edge],
        episode,
        [source_node, target_node],
        edge_types,
        edge_type_map,
    )

    assert resolved_edges[0].name == DEFAULT_EDGE_NAME
    assert invalidated_edges == []


@pytest.mark.asyncio
async def test_resolve_extracted_edges_keeps_unknown_names(monkeypatch):
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

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
            'fact_type': 'DEFAULT',
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

    resolved_edges, invalidated_edges = await resolve_extracted_edges(
        clients,
        [extracted_edge],
        episode,
        [source_node, target_node],
        edge_types,
        edge_type_map,
    )

    assert resolved_edges[0].name == 'INTERACTED_WITH'
    assert invalidated_edges == []


@pytest.mark.asyncio
async def test_resolve_extracted_edge_rejects_unmapped_fact_type(mock_llm_client):
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [],
        'contradicted_facts': [],
        'fact_type': 'OCCURRED_AT',
    }

    extracted_edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='OCCURRED_AT',
        group_id='group_1',
        fact='Document occurred at somewhere',
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

    related_edge = EntityEdge(
        source_node_uuid='alt_source',
        target_node_uuid='alt_target',
        name='OTHER',
        group_id='group_1',
        fact='Different fact',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    resolved_edge, duplicates, invalidated = await resolve_extracted_edge(
        mock_llm_client,
        extracted_edge,
        [related_edge],
        [],
        episode,
        edge_type_candidates={},
        custom_edge_type_names={'OCCURRED_AT'},
    )

    assert resolved_edge.name == DEFAULT_EDGE_NAME
    assert duplicates == []
    assert invalidated == []


@pytest.mark.asyncio
async def test_resolve_extracted_edge_accepts_unknown_fact_type(mock_llm_client):
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [],
        'contradicted_facts': [],
        'fact_type': 'INTERACTED_WITH',
    }

    extracted_edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='DEFAULT',
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

    related_edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='DEFAULT',
        group_id='group_1',
        fact='User mentioned a topic',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    resolved_edge, duplicates, invalidated = await resolve_extracted_edge(
        mock_llm_client,
        extracted_edge,
        [related_edge],
        [],
        episode,
        edge_type_candidates={'OCCURRED_AT': OccurredAtEdge},
        custom_edge_type_names={'OCCURRED_AT'},
    )

    assert resolved_edge.name == 'INTERACTED_WITH'
    assert resolved_edge.attributes == {}
    assert duplicates == []
    assert invalidated == []


@pytest.mark.asyncio
async def test_resolve_extracted_edge_uses_integer_indices_for_duplicates(mock_llm_client):
    """Test that resolve_extracted_edge correctly uses integer indices for LLM duplicate detection."""
    # Mock LLM to return duplicate_facts with integer indices
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [0, 1],  # LLM identifies first two related edges as duplicates
        'contradicted_facts': [],
        'fact_type': 'DEFAULT',
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
        custom_edge_type_names=set(),
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
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

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
        custom_edge_type_names=None,
    ):
        nonlocal resolve_call_count
        resolve_call_count += 1
        return extracted_edge, [], []

    # Mock semaphore_gather to execute awaitable immediately
    async def immediate_gather(*aws, max_coroutines=None):
        results = []
        for aw in aws:
            results.append(await aw)
        return results

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))
    monkeypatch.setattr(edge_ops, 'resolve_extracted_edge', mock_resolve_extracted_edge)

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

    resolved_edges, invalidated_edges = await resolve_extracted_edges(
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
