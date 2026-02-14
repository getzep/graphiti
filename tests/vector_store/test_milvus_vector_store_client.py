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
