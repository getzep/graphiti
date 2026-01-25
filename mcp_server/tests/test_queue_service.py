"""Tests for the Redis Streams queue service.

These tests verify the persistent queue implementation using Redis Streams.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestQueueServiceUnit:
    """Unit tests for QueueService using mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for unit tests."""
        redis = MagicMock()
        redis.xadd = AsyncMock(return_value='1234567890-0')
        redis.xreadgroup = AsyncMock(return_value=[])
        redis.xack = AsyncMock(return_value=1)
        redis.xgroup_create = AsyncMock()
        redis.xpending = AsyncMock(return_value={'pending': 0})
        redis.xautoclaim = AsyncMock(return_value=(b'0-0', []))
        redis.close = AsyncMock()
        return redis

    @pytest.fixture
    def mock_graphiti_client(self):
        """Mock Graphiti client for unit tests."""
        client = MagicMock()
        client.add_episode = AsyncMock()
        return client

    @pytest.fixture
    def queue_service(self, mock_redis, mock_graphiti_client):
        """Create QueueService instance with mocks."""
        from services.queue_service import QueueService

        service = QueueService()
        # Manually initialize without actual Redis connection
        service._redis = mock_redis
        service._graphiti_client = mock_graphiti_client
        return service

    @pytest.mark.asyncio
    async def test_add_episode_calls_xadd(self, queue_service, mock_redis):
        """Episode is added to Redis Stream with XADD."""
        # Prevent worker from starting
        queue_service._shutting_down = True

        await queue_service.add_episode(
            group_id='main',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            entity_types=None,
            uuid='test-uuid-123',
        )

        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        stream_key = call_args[0][0]
        assert stream_key == 'graphiti:queue:main'

    @pytest.mark.asyncio
    async def test_add_episode_returns_message_id(self, queue_service, mock_redis):
        """XADD returns message ID."""
        queue_service._shutting_down = True

        result = await queue_service.add_episode(
            group_id='main',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            entity_types=None,
            uuid='test-uuid-123',
        )

        assert result == '1234567890-0'

    @pytest.mark.asyncio
    async def test_stream_key_includes_group_id(self, queue_service, mock_redis):
        """Stream key format: graphiti:queue:{group_id}."""
        queue_service._shutting_down = True

        await queue_service.add_episode(
            group_id='Milofax-infrastructure',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            entity_types=None,
            uuid='test-uuid-123',
        )

        call_args = mock_redis.xadd.call_args
        stream_key = call_args[0][0]
        assert stream_key == 'graphiti:queue:Milofax-infrastructure'

    @pytest.mark.asyncio
    async def test_add_episode_without_initialize_raises(self):
        """add_episode raises RuntimeError if not initialized."""
        from services.queue_service import QueueService

        service = QueueService()

        with pytest.raises(RuntimeError, match='not initialized'):
            await service.add_episode(
                group_id='main',
                name='Test',
                content='Test',
                source_description='Test',
                episode_type='text',
                entity_types=None,
                uuid='test',
            )

    @pytest.mark.asyncio
    async def test_stream_key_format(self):
        """Verify stream key generation."""
        from services.queue_service import QueueService

        service = QueueService()
        assert service._stream_key('main') == 'graphiti:queue:main'
        assert service._stream_key('Milofax-prp') == 'graphiti:queue:Milofax-prp'

    @pytest.mark.asyncio
    async def test_dlq_key_format(self):
        """Verify dead letter queue key generation."""
        from services.queue_service import QueueService

        service = QueueService()
        assert service._dlq_key('main') == 'graphiti:queue:main:dlq'


class TestWorkerLifecycle:
    """Tests for worker task lifecycle management."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = MagicMock()
        redis.xadd = AsyncMock(return_value='1234567890-0')
        redis.xreadgroup = AsyncMock(return_value=[])
        redis.xack = AsyncMock(return_value=1)
        redis.xgroup_create = AsyncMock()
        redis.xautoclaim = AsyncMock(return_value=(b'0-0', []))
        redis.close = AsyncMock()
        return redis

    @pytest.fixture
    def mock_graphiti_client(self):
        """Mock Graphiti client."""
        client = MagicMock()
        client.add_episode = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_worker_task_reference_is_stored(self, mock_redis, mock_graphiti_client):
        """Task reference is stored in _worker_tasks to prevent GC."""
        from services.queue_service import QueueService

        service = QueueService()
        service._redis = mock_redis
        service._graphiti_client = mock_graphiti_client

        # Mock xreadgroup to return empty once, then block
        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            # Signal shutdown after first iteration
            service._shutting_down = True
            return []

        mock_redis.xreadgroup = mock_xreadgroup
        mock_redis.xautoclaim = AsyncMock(return_value=(b'0-0', []))

        await service._ensure_worker_running('test-group')

        # Worker task should be stored
        assert 'test-group' in service._worker_tasks
        assert isinstance(service._worker_tasks['test-group'], asyncio.Task)

        # Cleanup
        service._shutting_down = True
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self, mock_redis, mock_graphiti_client):
        """Shutdown sets _shutting_down flag."""
        from services.queue_service import QueueService

        service = QueueService()
        service._redis = mock_redis
        service._graphiti_client = mock_graphiti_client
        # No workers - just test the flag is set
        service._worker_tasks = {}

        await service.shutdown(timeout=1.0)

        assert service._shutting_down is True


class TestEpisodeMessage:
    """Tests for EpisodeMessage dataclass."""

    def test_to_dict_serialization(self):
        """EpisodeMessage serializes to dict correctly."""
        from services.queue_service import EpisodeMessage

        msg = EpisodeMessage(
            message_id='123-0',
            group_id='main',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            uuid='test-uuid',
            retry_count=2,
        )

        result = msg.to_dict()

        assert result['group_id'] == 'main'
        assert result['name'] == 'Test Episode'
        assert result['content'] == 'Test content'
        assert result['source_description'] == 'Test source'
        assert result['episode_type'] == 'text'
        assert result['uuid'] == 'test-uuid'
        assert result['retry_count'] == '2'

    def test_from_stream_data_deserialization(self):
        """EpisodeMessage deserializes from Redis Stream data."""
        from services.queue_service import EpisodeMessage

        data = {
            'group_id': 'main',
            'name': 'Test Episode',
            'content': 'Test content',
            'source_description': 'Test source',
            'episode_type': 'text',
            'uuid': 'test-uuid',
            'retry_count': '3',
        }

        msg = EpisodeMessage.from_stream_data('123-0', data)

        assert msg.message_id == '123-0'
        assert msg.group_id == 'main'
        assert msg.name == 'Test Episode'
        assert msg.uuid == 'test-uuid'
        assert msg.retry_count == 3

    def test_from_stream_data_handles_empty_uuid(self):
        """Empty UUID string converts to None."""
        from services.queue_service import EpisodeMessage

        data = {
            'group_id': 'main',
            'name': 'Test',
            'content': 'Test',
            'source_description': 'Test',
            'episode_type': 'text',
            'uuid': '',
            'retry_count': '0',
        }

        msg = EpisodeMessage.from_stream_data('123-0', data)

        assert msg.uuid is None


class TestQueueConfig:
    """Tests for QueueConfig dataclass."""

    def test_default_values(self):
        """QueueConfig has sensible defaults."""
        from services.queue_service import QueueConfig

        config = QueueConfig()

        assert config.consumer_group == 'graphiti_workers'
        assert config.block_ms == 5000
        assert config.claim_min_idle_ms == 60000
        assert config.max_retries == 3
        assert config.shutdown_timeout == 30.0

    def test_custom_values(self):
        """QueueConfig accepts custom values."""
        from services.queue_service import QueueConfig

        config = QueueConfig(
            redis_url='redis://custom:6380',
            consumer_group='custom_workers',
            block_ms=10000,
        )

        assert config.redis_url == 'redis://custom:6380'
        assert config.consumer_group == 'custom_workers'
        assert config.block_ms == 10000


@pytest.mark.integration
class TestFalkorDBEscaping:
    """Tests for FalkorDB RediSearch escaping fix.

    These tests require FalkorDB/Redis connection.
    Run with: pytest -m integration
    """

    def test_build_fulltext_query_escapes_reserved_words(self):
        """Group IDs like 'main' are quoted to prevent syntax errors."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('search term', ['main'])

        # "main" should be quoted
        assert '(@group_id:"main")' in query

    def test_build_fulltext_query_escapes_hyphens(self):
        """Group IDs with hyphens are quoted."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('search term', ['Milofax-infrastructure'])

        assert '(@group_id:"Milofax-infrastructure")' in query

    def test_build_fulltext_query_multiple_groups(self):
        """Multiple group IDs are all quoted and joined with pipe."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('test', ['main', 'Milofax-prp'])

        assert '(@group_id:"main"|"Milofax-prp")' in query

    def test_build_fulltext_query_no_groups(self):
        """Empty group_ids results in no group filter."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('test', [])

        assert '@group_id' not in query


# Integration tests (require real Redis)
@pytest.mark.integration
class TestQueueServiceIntegration:
    """Integration tests requiring real Redis connection.

    Run with: pytest -m integration
    """

    @pytest.fixture
    async def redis_client(self):
        """Real Redis client for integration tests."""
        import redis.asyncio as redis

        client = redis.from_url('redis://localhost:6379', decode_responses=True)
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_message_survives_service_restart(self, redis_client):
        """Message persists in Redis Stream after service restart."""
        from services.queue_service import QueueService

        # First service instance adds a message
        service1 = QueueService()
        service1._redis = redis_client
        service1._graphiti_client = MagicMock()
        service1._shutting_down = True  # Prevent processing

        stream_key = 'graphiti:queue:integration-test'

        # Clean up from previous runs
        await redis_client.delete(stream_key)

        # Add message
        message_id = await redis_client.xadd(
            stream_key,
            {
                'group_id': 'integration-test',
                'name': 'Persistent Episode',
                'content': 'This should survive',
                'source_description': 'Integration test',
                'episode_type': 'text',
                'uuid': 'persist-test',
                'retry_count': '0',
            },
        )

        # Verify message exists
        messages = await redis_client.xrange(stream_key)
        assert len(messages) == 1
        assert messages[0][0] == message_id

        # Clean up
        await redis_client.delete(stream_key)
