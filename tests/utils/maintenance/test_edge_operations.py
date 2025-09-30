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
        ensure_ascii=True,
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
        ensure_ascii=True,
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
        ensure_ascii=True,
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
        ensure_ascii=True,
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
        ensure_ascii=True,
    )

    assert resolved_edge.name == 'INTERACTED_WITH'
    assert resolved_edge.attributes == {}
    assert duplicates == []
    assert invalidated == []
