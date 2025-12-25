"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
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

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already running
        if not self._queue_workers.get(group_id, False):
            asyncio.create_task(self._process_episode_queue(group_id))

        return self._episode_queues[group_id].qsize()

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
                process_func = await self._episode_queues[group_id].get()

                try:
                    # Process the episode
                    await process_func()
                except Exception as e:
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
        return await self.add_episode_task(group_id, process_episode)
