"""Unit tests for graphiti_core.vector_store.milvus_client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.vector_store.milvus_client import (
    MilvusVectorStoreClient,
    MilvusVectorStoreConfig,
)


@pytest.fixture
def config():
    return MilvusVectorStoreConfig(
        uri='http://test:19530',
        token='test-token',
        db_name='test_db',
        embedding_dim=128,
        collection_prefix='test',
    )


@pytest.fixture
def vs_client(config):
    return MilvusVectorStoreClient(config=config)


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncMilvusClient instance."""
    client = AsyncMock()
    client.has_collection = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.query = AsyncMock(return_value=[])
    client.search = AsyncMock(return_value=[[]])
    client.drop_collection = AsyncMock()
    client.close = AsyncMock()
    return client


# ---- collection_name ----


class TestCollectionName:
    def test_returns_prefixed_name(self, vs_client):
        assert vs_client.collection_name('entity_nodes') == 'test_entity_nodes'

    def test_custom_prefix(self):
        cfg = MilvusVectorStoreConfig(collection_prefix='myapp')
        client = MilvusVectorStoreClient(config=cfg)
        assert client.collection_name('edges') == 'myapp_edges'


# ---- ensure_ready ----


class TestEnsureReady:
    @pytest.mark.asyncio
    async def test_creates_client_on_first_call(self, vs_client):
        with patch('pymilvus.AsyncMilvusClient') as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.has_collection = AsyncMock(return_value=True)
            mock_cls.return_value = mock_instance

            await vs_client.ensure_ready()

            mock_cls.assert_called_once_with(
                uri='http://test:19530', token='test-token', db_name='test_db'
            )

    @pytest.mark.asyncio
    async def test_idempotent(self, vs_client):
        with patch('pymilvus.AsyncMilvusClient') as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.has_collection = AsyncMock(return_value=True)
            mock_cls.return_value = mock_instance

            await vs_client.ensure_ready()
            await vs_client.ensure_ready()

            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_missing_collections(self, vs_client):
        with patch('pymilvus.AsyncMilvusClient') as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.has_collection = AsyncMock(return_value=False)
            mock_instance.create_collection = AsyncMock()
            mock_cls.return_value = mock_instance

            with patch('pymilvus.MilvusClient') as mock_sync_cls:
                mock_sync_cls.create_schema.return_value = MagicMock()
                mock_sync_cls.prepare_index_params.return_value = MagicMock()

                await vs_client.ensure_ready()

            assert mock_instance.create_collection.call_count == 4


# ---- CRUD delegation ----


class TestUpsert:
    @pytest.mark.asyncio
    async def test_delegates_to_client(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.upsert('test_col', [{'uuid': '1'}])
        mock_async_client.upsert.assert_called_once_with(
            collection_name='test_col', data=[{'uuid': '1'}]
        )


class TestDelete:
    @pytest.mark.asyncio
    async def test_delegates_to_client(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.delete('test_col', 'uuid == "1"')
        mock_async_client.delete.assert_called_once_with(
            collection_name='test_col', filter='uuid == "1"'
        )


class TestQuery:
    @pytest.mark.asyncio
    async def test_delegates_to_client(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        mock_async_client.query.return_value = [{'uuid': '1'}]

        result = await vs_client.query('test_col', 'uuid == "1"', ['uuid'])

        mock_async_client.query.assert_called_once_with(
            collection_name='test_col', filter='uuid == "1"', output_fields=['uuid']
        )
        assert result == [{'uuid': '1'}]


class TestSearch:
    @pytest.mark.asyncio
    async def test_delegates_to_client(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        mock_async_client.search.return_value = [[{'distance': 0.9, 'entity': {}}]]

        result = await vs_client.search(
            collection_name='test_col',
            data=[[0.1, 0.2]],
            anns_field='embedding',
            search_params={'metric_type': 'COSINE'},
            filter_expr='group_id == "g1"',
            output_fields=['uuid'],
            limit=10,
        )

        mock_async_client.search.assert_called_once_with(
            collection_name='test_col',
            data=[[0.1, 0.2]],
            anns_field='embedding',
            search_params={'metric_type': 'COSINE'},
            filter='group_id == "g1"',
            output_fields=['uuid'],
            limit=10,
        )
        assert len(result) == 1


# ---- reset_collections ----


class TestResetCollections:
    @pytest.mark.asyncio
    async def test_drops_and_recreates(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        mock_async_client.has_collection.return_value = True

        with patch('pymilvus.MilvusClient') as mock_sync_cls:
            mock_sync_cls.create_schema.return_value = MagicMock()
            mock_sync_cls.prepare_index_params.return_value = MagicMock()

            await vs_client.reset_collections()

        assert mock_async_client.drop_collection.call_count == 4
        # After drop, _ensure_collections recreates (has_collection still True
        # from mock, so no new create_collection calls in this mock setup)


# ---- close ----


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_client(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.close()
        mock_async_client.close.assert_called_once()
        assert vs_client._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self, vs_client):
        # Should not raise
        await vs_client.close()


# ---- Domain-aware save methods ----


class TestSaveEntityNodes:
    @pytest.mark.asyncio
    async def test_upserts_serialized_nodes(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        node = MagicMock()
        node.uuid = 'n1'
        node.group_id = 'g1'
        node.name = 'Alice'
        node.summary = 'A person'
        node.labels = ['Person']
        node.created_at = None
        node.attributes = {}
        node.name_embedding = [0.1, 0.2]

        await vs_client.save_entity_nodes([node])

        mock_async_client.upsert.assert_called_once()
        call_args = mock_async_client.upsert.call_args
        assert call_args.kwargs['collection_name'] == 'test_entity_nodes'
        assert len(call_args.kwargs['data']) == 1
        assert call_args.kwargs['data'][0]['uuid'] == 'n1'

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.save_entity_nodes([])
        mock_async_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensures_ready(self, vs_client):
        with patch('pymilvus.AsyncMilvusClient') as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.has_collection = AsyncMock(return_value=True)
            mock_instance.upsert = AsyncMock()
            mock_cls.return_value = mock_instance

            node = MagicMock()
            node.uuid = 'n1'
            node.group_id = 'g1'
            node.name = 'Alice'
            node.summary = ''
            node.labels = []
            node.created_at = None
            node.attributes = {}
            node.name_embedding = None

            await vs_client.save_entity_nodes([node])
            mock_cls.assert_called_once()


class TestSaveEntityEdges:
    @pytest.mark.asyncio
    async def test_upserts_serialized_edges(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        edge = MagicMock()
        edge.uuid = 'e1'
        edge.group_id = 'g1'
        edge.source_node_uuid = 'n1'
        edge.target_node_uuid = 'n2'
        edge.name = 'WORKS_AT'
        edge.fact = 'Alice works at Acme'
        edge.episodes = []
        edge.created_at = None
        edge.expired_at = None
        edge.valid_at = None
        edge.invalid_at = None
        edge.attributes = {}
        edge.fact_embedding = [0.1]

        await vs_client.save_entity_edges([edge])

        mock_async_client.upsert.assert_called_once()
        call_args = mock_async_client.upsert.call_args
        assert call_args.kwargs['collection_name'] == 'test_entity_edges'
        assert call_args.kwargs['data'][0]['uuid'] == 'e1'

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.save_entity_edges([])
        mock_async_client.upsert.assert_not_called()


class TestSaveEpisodicNodes:
    @pytest.mark.asyncio
    async def test_upserts_serialized_episodes(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        ep = MagicMock()
        ep.uuid = 'ep1'
        ep.group_id = 'g1'
        ep.name = 'episode-1'
        ep.content = 'Hello world'
        ep.source = MagicMock(value='message')
        ep.source_description = 'chat'
        ep.created_at = None
        ep.valid_at = None
        ep.entity_edges = []

        await vs_client.save_episodic_nodes([ep])

        mock_async_client.upsert.assert_called_once()
        call_args = mock_async_client.upsert.call_args
        assert call_args.kwargs['collection_name'] == 'test_episodic_nodes'
        assert call_args.kwargs['data'][0]['uuid'] == 'ep1'

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.save_episodic_nodes([])
        mock_async_client.upsert.assert_not_called()


class TestSaveCommunityNodes:
    @pytest.mark.asyncio
    async def test_upserts_serialized_community(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        cn = MagicMock()
        cn.uuid = 'c1'
        cn.group_id = 'g1'
        cn.name = 'Community A'
        cn.summary = 'A community'
        cn.created_at = None
        cn.name_embedding = [0.1]

        await vs_client.save_community_nodes([cn])

        mock_async_client.upsert.assert_called_once()
        call_args = mock_async_client.upsert.call_args
        assert call_args.kwargs['collection_name'] == 'test_community_nodes'
        assert call_args.kwargs['data'][0]['uuid'] == 'c1'

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.save_community_nodes([])
        mock_async_client.upsert.assert_not_called()


# ---- Domain-aware delete methods ----


class TestDeleteEntityNodes:
    @pytest.mark.asyncio
    async def test_deletes_by_uuids(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client

        await vs_client.delete_entity_nodes(['n1', 'n2'])

        mock_async_client.delete.assert_called_once()
        call_args = mock_async_client.delete.call_args
        assert call_args.kwargs['collection_name'] == 'test_entity_nodes'
        assert '"n1"' in call_args.kwargs['filter']
        assert '"n2"' in call_args.kwargs['filter']

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.delete_entity_nodes([])
        mock_async_client.delete.assert_not_called()


class TestDeleteEntityEdges:
    @pytest.mark.asyncio
    async def test_deletes_by_uuids(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client

        await vs_client.delete_entity_edges(['e1'])

        mock_async_client.delete.assert_called_once()
        call_args = mock_async_client.delete.call_args
        assert call_args.kwargs['collection_name'] == 'test_entity_edges'
        assert '"e1"' in call_args.kwargs['filter']

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.delete_entity_edges([])
        mock_async_client.delete.assert_not_called()


class TestDeleteByGroupIds:
    @pytest.mark.asyncio
    async def test_deletes_from_all_collections(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client

        await vs_client.delete_by_group_ids(['g1', 'g2'])

        # Should delete from all 4 collections
        assert mock_async_client.delete.call_count == 4
        collection_names = [
            call.kwargs['collection_name'] for call in mock_async_client.delete.call_args_list
        ]
        assert 'test_entity_nodes' in collection_names
        assert 'test_entity_edges' in collection_names
        assert 'test_episodic_nodes' in collection_names
        assert 'test_community_nodes' in collection_names
        # Check filter contains group IDs
        for call in mock_async_client.delete.call_args_list:
            assert '"g1"' in call.kwargs['filter']
            assert '"g2"' in call.kwargs['filter']

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.delete_by_group_ids([])
        mock_async_client.delete.assert_not_called()


class TestDeleteNodesByUuids:
    @pytest.mark.asyncio
    async def test_deletes_from_three_node_collections(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client

        await vs_client.delete_nodes_by_uuids(['n1', 'n2'])

        # Should delete from entity_nodes, episodic_nodes, community_nodes (3 collections)
        assert mock_async_client.delete.call_count == 3
        collection_names = [
            call.kwargs['collection_name'] for call in mock_async_client.delete.call_args_list
        ]
        assert 'test_entity_nodes' in collection_names
        assert 'test_episodic_nodes' in collection_names
        assert 'test_community_nodes' in collection_names
        # Should NOT include entity_edges
        assert 'test_entity_edges' not in collection_names

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.delete_nodes_by_uuids([])
        mock_async_client.delete.assert_not_called()


class TestDeleteCommunityNodes:
    @pytest.mark.asyncio
    async def test_deletes_all_when_no_uuids(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client

        await vs_client.delete_community_nodes()

        mock_async_client.delete.assert_called_once()
        call_args = mock_async_client.delete.call_args
        assert call_args.kwargs['collection_name'] == 'test_community_nodes'
        assert 'uuid != ""' in call_args.kwargs['filter']

    @pytest.mark.asyncio
    async def test_deletes_by_uuids(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client

        await vs_client.delete_community_nodes(['c1', 'c2'])

        mock_async_client.delete.assert_called_once()
        call_args = mock_async_client.delete.call_args
        assert call_args.kwargs['collection_name'] == 'test_community_nodes'
        assert '"c1"' in call_args.kwargs['filter']
        assert '"c2"' in call_args.kwargs['filter']

    @pytest.mark.asyncio
    async def test_empty_list_no_op(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        await vs_client.delete_community_nodes([])
        mock_async_client.delete.assert_not_called()


class TestClearAll:
    @pytest.mark.asyncio
    async def test_delegates_to_reset_collections(self, vs_client, mock_async_client):
        vs_client._client = mock_async_client
        mock_async_client.has_collection.return_value = True

        with patch('pymilvus.MilvusClient') as mock_sync_cls:
            mock_sync_cls.create_schema.return_value = MagicMock()
            mock_sync_cls.prepare_index_params.return_value = MagicMock()

            await vs_client.clear_all()

        # clear_all() delegates to reset_collections() which drops all 4
        assert mock_async_client.drop_collection.call_count == 4
