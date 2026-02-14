"""Unit tests for graphiti_core.vector_store.milvus_graph_operations."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.vector_store.milvus_graph_operations import MilvusGraphOperationsInterface


@pytest.fixture
def mock_vs_client():
    """Create a mock VectorStoreClient."""
    client = MagicMock()
    client.collection_name = lambda suffix: f'test_{suffix}'
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.query = AsyncMock(return_value=[])
    client.reset_collections = AsyncMock()
    return client


@pytest.fixture
def graph_ops(mock_vs_client):
    """Create a MilvusGraphOperationsInterface with a mock VectorStoreClient."""
    return MilvusGraphOperationsInterface(vs_client=mock_vs_client)


def _make_entity_node(**kwargs):
    node = MagicMock()
    node.uuid = kwargs.get('uuid', 'node-1')
    node.group_id = kwargs.get('group_id', 'g1')
    node.name = kwargs.get('name', 'Alice')
    node.summary = kwargs.get('summary', 'Summary')
    node.labels = kwargs.get('labels', ['Person'])
    node.created_at = kwargs.get('created_at', datetime(2024, 1, 15, tzinfo=timezone.utc))
    node.attributes = kwargs.get('attributes', {})
    node.name_embedding = kwargs.get('name_embedding', [0.1] * 128)
    return node


def _make_entity_edge(**kwargs):
    edge = MagicMock()
    edge.uuid = kwargs.get('uuid', 'edge-1')
    edge.group_id = kwargs.get('group_id', 'g1')
    edge.source_node_uuid = kwargs.get('source_node_uuid', 'src-1')
    edge.target_node_uuid = kwargs.get('target_node_uuid', 'tgt-1')
    edge.name = kwargs.get('name', 'WORKS_AT')
    edge.fact = kwargs.get('fact', 'Alice works at Acme')
    edge.episodes = kwargs.get('episodes', ['ep1'])
    edge.created_at = kwargs.get('created_at', datetime(2024, 1, 15, tzinfo=timezone.utc))
    edge.expired_at = kwargs.get('expired_at')
    edge.valid_at = kwargs.get('valid_at')
    edge.invalid_at = kwargs.get('invalid_at')
    edge.attributes = kwargs.get('attributes', {})
    edge.fact_embedding = kwargs.get('fact_embedding', [0.2] * 128)
    return edge


def _make_episodic_node(**kwargs):
    node = MagicMock()
    node.uuid = kwargs.get('uuid', 'ep-1')
    node.group_id = kwargs.get('group_id', 'g1')
    node.name = kwargs.get('name', 'episode1')
    node.content = kwargs.get('content', 'Alice met Bob')
    node.source.value = kwargs.get('source', 'text')
    node.source_description = kwargs.get('source_description', 'conversation')
    node.created_at = kwargs.get('created_at', datetime(2024, 1, 15, tzinfo=timezone.utc))
    node.valid_at = kwargs.get('valid_at', datetime(2024, 1, 15, tzinfo=timezone.utc))
    node.entity_edges = kwargs.get('entity_edges', ['edge1'])
    return node


def _make_community_node(**kwargs):
    node = MagicMock()
    node.uuid = kwargs.get('uuid', 'comm-1')
    node.group_id = kwargs.get('group_id', 'g1')
    node.name = kwargs.get('name', 'Tech Community')
    node.summary = kwargs.get('summary', 'A tech community')
    node.created_at = kwargs.get('created_at', datetime(2024, 1, 15, tzinfo=timezone.utc))
    node.name_embedding = kwargs.get('name_embedding', [0.3] * 128)
    return node


# ---- Node Operations ----


class TestNodeSave:
    @pytest.mark.asyncio
    async def test_upserts_to_entity_nodes(self, graph_ops, mock_vs_client):
        node = _make_entity_node()

        await graph_ops.node_save(node, driver=MagicMock())

        mock_vs_client.upsert.assert_called_once()
        call_kwargs = mock_vs_client.upsert.call_args.kwargs
        assert call_kwargs['collection_name'] == 'test_entity_nodes'
        assert call_kwargs['data'][0]['uuid'] == 'node-1'


class TestNodeSaveBulk:
    @pytest.mark.asyncio
    async def test_upserts_batch(self, graph_ops, mock_vs_client):
        nodes = [_make_entity_node(uuid=f'n{i}') for i in range(3)]

        await graph_ops.node_save_bulk(None, MagicMock(), None, nodes)

        mock_vs_client.upsert.assert_called_once()
        data = mock_vs_client.upsert.call_args.kwargs['data']
        assert len(data) == 3

    @pytest.mark.asyncio
    async def test_empty_nodes_skips(self, graph_ops, mock_vs_client):
        await graph_ops.node_save_bulk(None, MagicMock(), None, [])
        mock_vs_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_batch_size(self, graph_ops, mock_vs_client):
        nodes = [_make_entity_node(uuid=f'n{i}') for i in range(5)]

        await graph_ops.node_save_bulk(None, MagicMock(), None, nodes, batch_size=2)

        assert mock_vs_client.upsert.call_count == 3  # 2 + 2 + 1


class TestNodeDelete:
    @pytest.mark.asyncio
    async def test_deletes_by_uuid(self, graph_ops, mock_vs_client):
        node = _make_entity_node(uuid='del-1')

        await graph_ops.node_delete(node, driver=MagicMock())

        call_kwargs = mock_vs_client.delete.call_args.kwargs
        assert 'uuid == "del-1"' in call_kwargs['filter_expr']


class TestNodeDeleteByUuids:
    @pytest.mark.asyncio
    async def test_deletes_multiple(self, graph_ops, mock_vs_client):
        await graph_ops.node_delete_by_uuids(None, MagicMock(), ['u1', 'u2'])

        call_kwargs = mock_vs_client.delete.call_args.kwargs
        assert 'uuid in ["u1", "u2"]' in call_kwargs['filter_expr']

    @pytest.mark.asyncio
    async def test_empty_uuids_skips(self, graph_ops, mock_vs_client):
        await graph_ops.node_delete_by_uuids(None, MagicMock(), [])
        mock_vs_client.delete.assert_not_called()


class TestNodeDeleteByGroupId:
    @pytest.mark.asyncio
    async def test_deletes_by_group(self, graph_ops, mock_vs_client):
        await graph_ops.node_delete_by_group_id(None, MagicMock(), 'g1')

        call_kwargs = mock_vs_client.delete.call_args.kwargs
        assert 'group_id == "g1"' in call_kwargs['filter_expr']


# ---- Node Embeddings ----


class TestNodeLoadEmbeddings:
    @pytest.mark.asyncio
    async def test_loads_embedding(self, graph_ops, mock_vs_client):
        mock_vs_client.query.return_value = [{'name_embedding': [0.5, 0.6]}]
        node = MagicMock()
        node.uuid = 'n1'

        await graph_ops.node_load_embeddings(node, driver=MagicMock())
        assert node.name_embedding == [0.5, 0.6]


class TestNodeLoadEmbeddingsBulk:
    @pytest.mark.asyncio
    async def test_returns_dict(self, graph_ops, mock_vs_client):
        mock_vs_client.query.return_value = [
            {'uuid': 'n1', 'name_embedding': [0.1, 0.2]},
            {'uuid': 'n2', 'name_embedding': [0.3, 0.4]},
        ]
        nodes = [MagicMock(uuid='n1'), MagicMock(uuid='n2')]

        result = await graph_ops.node_load_embeddings_bulk(MagicMock(), nodes)
        assert result == {'n1': [0.1, 0.2], 'n2': [0.3, 0.4]}

    @pytest.mark.asyncio
    async def test_empty_nodes_returns_empty(self, graph_ops, mock_vs_client):
        result = await graph_ops.node_load_embeddings_bulk(MagicMock(), [])
        assert result == {}


# ---- Edge Operations ----


class TestEdgeSave:
    @pytest.mark.asyncio
    async def test_upserts_to_entity_edges(self, graph_ops, mock_vs_client):
        edge = _make_entity_edge()

        await graph_ops.edge_save(edge, driver=MagicMock())

        call_kwargs = mock_vs_client.upsert.call_args.kwargs
        assert call_kwargs['collection_name'] == 'test_entity_edges'
        assert call_kwargs['data'][0]['uuid'] == 'edge-1'


class TestEdgeSaveBulk:
    @pytest.mark.asyncio
    async def test_upserts_batch(self, graph_ops, mock_vs_client):
        edges = [_make_entity_edge(uuid=f'e{i}') for i in range(3)]

        await graph_ops.edge_save_bulk(None, MagicMock(), None, edges)
        assert mock_vs_client.upsert.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_edges_skips(self, graph_ops, mock_vs_client):
        await graph_ops.edge_save_bulk(None, MagicMock(), None, [])
        mock_vs_client.upsert.assert_not_called()


class TestEdgeDelete:
    @pytest.mark.asyncio
    async def test_deletes_by_uuid(self, graph_ops, mock_vs_client):
        edge = _make_entity_edge(uuid='del-e1')

        await graph_ops.edge_delete(edge, driver=MagicMock())
        assert 'uuid == "del-e1"' in mock_vs_client.delete.call_args.kwargs['filter_expr']


class TestEdgeLoadEmbeddingsBulk:
    @pytest.mark.asyncio
    async def test_returns_dict(self, graph_ops, mock_vs_client):
        mock_vs_client.query.return_value = [
            {'uuid': 'e1', 'fact_embedding': [0.5, 0.6]},
        ]
        edges = [MagicMock(uuid='e1')]

        result = await graph_ops.edge_load_embeddings_bulk(MagicMock(), edges)
        assert result == {'e1': [0.5, 0.6]}


# ---- Episodic Node Operations ----


class TestEpisodicNodeSave:
    @pytest.mark.asyncio
    async def test_upserts_to_episodic_nodes(self, graph_ops, mock_vs_client):
        node = _make_episodic_node()

        await graph_ops.episodic_node_save(node, driver=MagicMock())

        call_kwargs = mock_vs_client.upsert.call_args.kwargs
        assert call_kwargs['collection_name'] == 'test_episodic_nodes'


class TestEpisodicNodeDeleteByGroupId:
    @pytest.mark.asyncio
    async def test_deletes_by_group(self, graph_ops, mock_vs_client):
        await graph_ops.episodic_node_delete_by_group_id(None, MagicMock(), 'g1')

        call_kwargs = mock_vs_client.delete.call_args.kwargs
        assert call_kwargs['collection_name'] == 'test_episodic_nodes'
        assert 'group_id == "g1"' in call_kwargs['filter_expr']


# ---- Community Node Operations ----


class TestCommunityNodeSave:
    @pytest.mark.asyncio
    async def test_upserts_to_community_nodes(self, graph_ops, mock_vs_client):
        node = _make_community_node()

        await graph_ops.community_node_save(node, driver=MagicMock())

        call_kwargs = mock_vs_client.upsert.call_args.kwargs
        assert call_kwargs['collection_name'] == 'test_community_nodes'


class TestCommunityNodeLoadNameEmbedding:
    @pytest.mark.asyncio
    async def test_loads_embedding(self, graph_ops, mock_vs_client):
        mock_vs_client.query.return_value = [{'name_embedding': [0.7, 0.8]}]
        node = MagicMock()
        node.uuid = 'c1'

        await graph_ops.community_node_load_name_embedding(node, driver=MagicMock())
        assert node.name_embedding == [0.7, 0.8]


# ---- Clear Data ----


class TestClearData:
    @pytest.mark.asyncio
    async def test_clear_by_group_ids(self, graph_ops, mock_vs_client):
        await graph_ops.clear_data(driver=MagicMock(), group_ids=['g1'])

        assert mock_vs_client.delete.call_count == 4
        for call in mock_vs_client.delete.call_args_list:
            assert 'group_id == "g1"' in call.kwargs['filter_expr']

    @pytest.mark.asyncio
    async def test_clear_all_calls_reset(self, graph_ops, mock_vs_client):
        await graph_ops.clear_data(driver=MagicMock())

        mock_vs_client.reset_collections.assert_called_once()


# ---- NotImplementedError fallback ----


class TestNotImplementedFallbacks:
    @pytest.mark.asyncio
    async def test_node_get_by_uuid_raises(self, graph_ops):
        with pytest.raises(NotImplementedError):
            await graph_ops.node_get_by_uuid(None, MagicMock(), 'uuid')

    @pytest.mark.asyncio
    async def test_edge_get_by_uuid_raises(self, graph_ops):
        with pytest.raises(NotImplementedError):
            await graph_ops.edge_get_by_uuid(None, MagicMock(), 'uuid')

    @pytest.mark.asyncio
    async def test_episodic_node_get_by_uuid_raises(self, graph_ops):
        with pytest.raises(NotImplementedError):
            await graph_ops.episodic_node_get_by_uuid(None, MagicMock(), 'uuid')

    @pytest.mark.asyncio
    async def test_get_community_clusters_raises(self, graph_ops):
        with pytest.raises(NotImplementedError):
            await graph_ops.get_community_clusters(MagicMock(), None)

    @pytest.mark.asyncio
    async def test_retrieve_episodes_raises(self, graph_ops):
        with pytest.raises(NotImplementedError):
            await graph_ops.retrieve_episodes(MagicMock(), datetime.now(tz=timezone.utc))
