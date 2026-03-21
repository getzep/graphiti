"""Queue service for managing episode processing."""

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EpisodeIngestStatus:
    episode_uuid: str
    group_id: str
    state: str
    queued_at: datetime
    started_at: datetime | None = None
    processed_at: datetime | None = None
    last_error: str | None = None
    queue_position: int | None = None


@dataclass
class _QueuedEpisodeTask:
    episode_uuid: str
    process_func: Callable[[], Awaitable[None]]


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Track queued episode UUIDs so queue positions can be surfaced externally.
        self._pending_episode_uuids: dict[str, deque[str]] = {}
        # Track ingest lifecycle by episode UUID.
        self._episode_statuses: dict[str, EpisodeIngestStatus] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None

    async def add_episode_task(
        self,
        group_id: str,
        episode_uuid: str,
        process_func: Callable[[], Awaitable[None]],
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode

        Returns:
            The position in the queue
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()
            self._pending_episode_uuids[group_id] = deque()

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(
            _QueuedEpisodeTask(episode_uuid=episode_uuid, process_func=process_func)
        )
        self._pending_episode_uuids[group_id].append(episode_uuid)

        queue_position = len(self._pending_episode_uuids[group_id])
        self._episode_statuses[episode_uuid] = EpisodeIngestStatus(
            episode_uuid=episode_uuid,
            group_id=group_id,
            state='queued',
            queued_at=datetime.now(timezone.utc),
            queue_position=queue_position,
        )

        # Start a worker for this queue if one isn't already running
        if not self._queue_workers.get(group_id, False):
            asyncio.create_task(self._process_episode_queue(group_id))

        return queue_position

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        self._queue_workers[group_id] = True

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                queued_task = await self._episode_queues[group_id].get()
                process_func = queued_task.process_func
                episode_uuid = queued_task.episode_uuid
                pending_queue = self._pending_episode_uuids.get(group_id)
                if pending_queue and pending_queue and pending_queue[0] == episode_uuid:
                    pending_queue.popleft()
                elif pending_queue and episode_uuid in pending_queue:
                    pending_queue.remove(episode_uuid)

                status = self._episode_statuses.get(episode_uuid)
                if status is not None:
                    status.state = 'processing'
                    status.started_at = datetime.now(timezone.utc)
                    status.queue_position = None

                try:
                    # Process the episode
                    await process_func()
                    if status is not None:
                        status.state = 'completed'
                        status.processed_at = datetime.now(timezone.utc)
                except Exception as e:
                    if status is not None:
                        status.state = 'failed'
                        status.last_error = str(e)
                        status.processed_at = datetime.now(timezone.utc)
                    logger.error(
                        f'Error processing queued episode for group_id {group_id}: {str(e)}'
                    )
                finally:
                    # Mark the task as done regardless of success/failure
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._queue_workers.get(group_id, False)

    def get_queue_position(self, episode_uuid: str) -> int | None:
        """Return the current queue position for a queued episode UUID."""
        status = self._episode_statuses.get(episode_uuid)
        if status is None:
            return None

        pending_queue = self._pending_episode_uuids.get(status.group_id)
        if pending_queue is None:
            return None

        for index, pending_uuid in enumerate(pending_queue, start=1):
            if pending_uuid == episode_uuid:
                return index

        return None

    def get_episode_status(self, episode_uuid: str) -> EpisodeIngestStatus | None:
        """Return the latest ingest lifecycle state for an episode UUID."""
        status = self._episode_statuses.get(episode_uuid)
        if status is None:
            return None

        if status.state == 'queued':
            status.queue_position = self.get_queue_position(episode_uuid)

        return status

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
        """
        self._graphiti_client = graphiti_client
        logger.info('Queue service initialized with graphiti client')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
    ) -> int:
        """Add an episode for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Entity types for extraction
            uuid: Episode UUID

        Returns:
            The position in the queue
        """
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')
        if uuid is None:
            raise ValueError('uuid is required for queue-backed episode tracking')

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                # Process the episode using the graphiti client
                await self._graphiti_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, uuid, process_episode)
