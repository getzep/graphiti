"""Queue service using Redis Streams for persistent episode processing.

This replaces the previous in-memory asyncio.Queue implementation with Redis Streams
for crash-resilient, persistent message processing.

Key improvements:
- Messages persist in Redis (survives service restarts)
- Consumer Groups with XACK for guaranteed delivery
- xautoclaim for recovering abandoned messages after crashes
- Task references stored to prevent Python GC from collecting workers
"""

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from graphiti_core.nodes import EpisodeType

logger = logging.getLogger(__name__)


@dataclass
class QueueConfig:
    """Configuration for the queue service."""

    redis_url: str = 'redis://localhost:6379'
    consumer_group: str = 'graphiti_workers'
    block_ms: int = 5000  # 5 seconds blocking read
    claim_min_idle_ms: int = 60000  # Reclaim messages idle for 60 seconds
    max_retries: int = 3
    shutdown_timeout: float = 30.0


@dataclass
class EpisodeMessage:
    """Episode data for queue processing."""

    message_id: str
    group_id: str
    name: str
    content: str
    source_description: str
    episode_type: str
    uuid: str | None
    retry_count: int = 0

    def to_dict(self) -> dict[str, str]:
        """Serialize for Redis Stream."""
        return {
            'group_id': self.group_id,
            'name': self.name,
            'content': self.content,
            'source_description': self.source_description,
            'episode_type': self.episode_type,
            'uuid': self.uuid or '',
            'retry_count': str(self.retry_count),
        }

    @classmethod
    def from_stream_data(cls, message_id: str, data: dict) -> 'EpisodeMessage':
        """Deserialize from Redis Stream."""
        return cls(
            message_id=message_id,
            group_id=data['group_id'],
            name=data['name'],
            content=data['content'],
            source_description=data['source_description'],
            episode_type=data['episode_type'],
            uuid=data['uuid'] or None,
            retry_count=int(data.get('retry_count', 0)),
        )


class QueueService:
    """Persistent queue service using Redis Streams.

    This service manages episode processing queues per group_id using Redis Streams.
    Each group_id gets its own stream (graphiti:queue:{group_id}).

    Features:
    - Persistent: Messages survive service restarts
    - Guaranteed delivery: Consumer Groups with acknowledgment
    - Crash recovery: xautoclaim recovers abandoned messages
    - No GC issues: Worker task references are stored
    """

    def __init__(self, config: QueueConfig | None = None):
        self._config = config or QueueConfig()
        self._redis: redis.Redis | None = None
        self._graphiti_client: Any = None

        # FIX: Task references stored to prevent GC collection
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._worker_running: dict[str, bool] = {}
        self._shutting_down: bool = False

        # Unique consumer name per instance
        self._consumer_name = f'worker_{socket.gethostname()}_{os.getpid()}'

    def _stream_key(self, group_id: str) -> str:
        """Get Redis Stream key for a group_id."""
        return f'graphiti:queue:{group_id}'

    def _dlq_key(self, group_id: str) -> str:
        """Get Dead Letter Queue key for a group_id."""
        return f'graphiti:queue:{group_id}:dlq'

    async def initialize(
        self, graphiti_client: Any, redis_client: redis.Redis | None = None
    ) -> None:
        """Initialize with graphiti client and Redis connection.

        Args:
            graphiti_client: The Graphiti client instance for processing episodes
            redis_client: Optional Redis client (creates new if not provided)
        """
        self._graphiti_client = graphiti_client

        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis.from_url(self._config.redis_url, decode_responses=True)

        logger.info(f'Queue service initialized with consumer: {self._consumer_name}')

        # Eager startup: resume workers for streams that have unprocessed messages
        await self._resume_pending_streams()

    async def _resume_pending_streams(self) -> None:
        """Resume workers for any streams that have unprocessed messages.

        This runs at startup to ensure no episodes are lost after a crash or restart.
        Scans all graphiti:queue:* streams and starts workers for any with lag > 0
        or pending (unacknowledged) messages.
        """
        if self._redis is None:
            return

        try:
            # Find all queue streams
            keys: list[str] = []
            async for key in self._redis.scan_iter(match='graphiti:queue:*', count=100):
                # Skip DLQ keys
                if ':dlq' not in key:
                    keys.append(key)

            if not keys:
                return

            resumed = 0
            for stream_key in keys:
                group_id = stream_key.removeprefix('graphiti:queue:')

                try:
                    # Check if consumer group exists and has lag or pending
                    groups = await self._redis.xinfo_groups(stream_key)
                    for group in groups:
                        if group.get('name') != self._config.consumer_group:
                            continue
                        lag = group.get('lag', 0) or 0
                        pending = group.get('pending', 0) or 0
                        if lag > 0 or pending > 0:
                            logger.info(
                                f'Resuming worker for {group_id}: '
                                f'lag={lag}, pending={pending}'
                            )
                            await self._ensure_worker_running(group_id)
                            resumed += 1
                except redis.ResponseError:
                    # Stream exists but no consumer group yet — skip
                    pass

            if resumed > 0:
                logger.info(f'Resumed {resumed} workers for streams with pending messages')
            else:
                logger.info('No pending streams found at startup')

        except redis.ConnectionError as e:
            logger.error(f'Redis connection error during startup scan: {e}')
        except Exception as e:
            logger.warning(f'Error scanning pending streams: {e}')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
    ) -> str:
        """Add episode to Redis Stream for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode (EpisodeType enum or string)
            entity_types: Entity types for extraction (stored but not used in queue)
            uuid: Episode UUID

        Returns:
            The Redis Stream message ID
        """
        if self._redis is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        stream_key = self._stream_key(group_id)

        message = EpisodeMessage(
            message_id='',  # Will be set by Redis
            group_id=group_id,
            name=name,
            content=content,
            source_description=source_description,
            episode_type=str(episode_type.value if hasattr(episode_type, 'value') else episode_type),
            uuid=uuid,
        )

        # XADD - immediately persisted by Redis
        message_id: str = await self._redis.xadd(stream_key, message.to_dict())  # type: ignore[arg-type]

        logger.info(f'Queued episode {uuid} for group {group_id}: {message_id}')

        # Ensure worker is running for this group
        await self._ensure_worker_running(group_id)

        return message_id

    async def _ensure_worker_running(self, group_id: str) -> None:
        """Start worker for group_id if not already running."""
        if self._shutting_down:
            return

        if self._worker_running.get(group_id, False):
            return

        if self._redis is None:
            return

        stream_key = self._stream_key(group_id)

        # Create consumer group if not exists
        try:
            await self._redis.xgroup_create(
                stream_key, self._config.consumer_group, id='0', mkstream=True
            )
            logger.info(f'Created consumer group for {group_id}')
        except redis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise
            # Group already exists - that's fine

        # FIX: Store task reference to prevent GC!
        task = asyncio.create_task(self._process_stream(group_id))
        self._worker_tasks[group_id] = task

        # Cleanup callback when task completes
        def on_done(t: asyncio.Task):
            self._worker_tasks.pop(group_id, None)
            self._worker_running[group_id] = False
            if t.exception() and not self._shutting_down:
                logger.error(f'Worker for {group_id} crashed: {t.exception()}')

        task.add_done_callback(on_done)

    async def _process_stream(self, group_id: str) -> None:
        """Process messages from Redis Stream for a group_id."""
        if self._redis is None:
            raise RuntimeError('Queue service not initialized')

        stream_key = self._stream_key(group_id)
        logger.info(f'Starting stream worker for {group_id}')
        self._worker_running[group_id] = True

        try:
            # First: Claim any abandoned messages from previous crashes
            await self._claim_abandoned(group_id)

            while not self._shutting_down:
                try:
                    messages = await self._redis.xreadgroup(
                        groupname=self._config.consumer_group,
                        consumername=self._consumer_name,
                        streams={stream_key: '>'},  # '>' = only new messages
                        count=1,
                        block=self._config.block_ms,
                    )
                except redis.ConnectionError as e:
                    logger.error(f'Redis connection error for {group_id}: {e}')
                    await asyncio.sleep(5)  # Wait before retry
                    continue

                if not messages:
                    continue

                for _stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        await self._process_message(group_id, message_id, data)

        except asyncio.CancelledError:
            logger.info(f'Worker for {group_id} cancelled')
        except Exception as e:
            logger.error(f'Worker error for {group_id}: {e}')
            raise
        finally:
            self._worker_running[group_id] = False
            logger.info(f'Stopped worker for {group_id}')

    async def _process_message(self, group_id: str, message_id: str, data: dict) -> None:
        """Process a single message from the stream."""
        if self._redis is None:
            raise RuntimeError('Queue service not initialized')

        episode = EpisodeMessage.from_stream_data(message_id, data)

        try:
            logger.info(f'Processing episode {episode.uuid} for {group_id}')

            await self._graphiti_client.add_episode(
                name=episode.name,
                episode_body=episode.content,
                source_description=episode.source_description,
                source=EpisodeType.from_str(episode.episode_type),
                group_id=group_id,
                reference_time=datetime.now(timezone.utc),
                uuid=episode.uuid,
            )

            # Success: Acknowledge the message
            await self._redis.xack(
                self._stream_key(group_id), self._config.consumer_group, message_id
            )
            logger.info(f'Successfully processed episode {episode.uuid}')

        except Exception as e:
            logger.error(f'Failed to process {message_id} (uuid={episode.uuid}): {e}')
            # No XACK → message stays pending → will be reclaimed by xautoclaim
            # Could add DLQ logic here for max_retries exceeded

    async def _claim_abandoned(self, group_id: str) -> None:
        """Claim and reprocess abandoned messages from previous crashes.

        Loops until all abandoned messages are claimed (not just 10).
        """
        if self._redis is None:
            return

        stream_key = self._stream_key(group_id)
        total_claimed = 0
        cursor = '0'

        try:
            while True:
                result = await self._redis.xautoclaim(
                    stream_key,
                    self._config.consumer_group,
                    self._consumer_name,
                    min_idle_time=self._config.claim_min_idle_ms,
                    start_id=cursor,
                    count=50,
                )

                if not result or len(result) < 2 or not result[1]:
                    break

                next_cursor = result[0]
                claimed_messages = result[1]

                for message_id, data in claimed_messages:
                    await self._process_message(group_id, message_id, data)
                    total_claimed += 1

                # If cursor is '0-0', we've scanned everything
                if next_cursor == '0-0' or next_cursor == '0':
                    break
                cursor = next_cursor

            if total_claimed > 0:
                logger.info(f'Claimed and processed {total_claimed} abandoned messages for {group_id}')

        except redis.ResponseError as e:
            # XAUTOCLAIM might fail if stream doesn't exist yet
            if 'NOGROUP' not in str(e):
                logger.warning(f'Error claiming abandoned messages for {group_id}: {e}')

    async def shutdown(self, timeout: float | None = None) -> None:
        """Graceful shutdown - wait for in-flight processing to complete."""
        timeout = timeout or self._config.shutdown_timeout
        self._shutting_down = True

        if not self._worker_tasks:
            logger.info('No workers to shut down')
            return

        logger.info(f'Shutting down {len(self._worker_tasks)} workers...')

        # Cancel all workers
        for group_id, task in self._worker_tasks.items():
            task.cancel()
            logger.debug(f'Cancelled worker for {group_id}')

        # Wait with timeout for graceful completion
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._worker_tasks.values(), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f'Shutdown timeout ({timeout}s) reached, forcing termination')

        # Close Redis connection if we created it
        if self._redis is not None:
            await self._redis.close()

        logger.info('Queue service shutdown complete')

    def get_queue_size(self, group_id: str) -> int:
        """Get pending message count for a group_id.

        Note: This is approximate - use XPENDING for accurate count.
        """
        # For backwards compatibility, return 0 (actual count requires async call)
        return 0

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._worker_running.get(group_id, False)

    # Legacy method for backwards compatibility
    async def add_episode_task(
        self, group_id: str, process_func: Any
    ) -> int:
        """Legacy method - redirects to add_episode pattern.

        Deprecated: Use add_episode() directly instead.
        """
        logger.warning('add_episode_task is deprecated, use add_episode() instead')
        # Can't easily support arbitrary callables with Redis Streams
        # This exists only for backwards compatibility during transition
        await process_func()
        return 0
