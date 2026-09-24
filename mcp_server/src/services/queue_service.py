"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Bounded retries for transient episode failures. Validation failures are
# deterministic and are never retried; see is_retryable_failure.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0


def retry_delay(base_delay: float, attempt: int) -> float:
    """Return the seconds to wait after `attempt` failed attempts (1-based)."""
    return base_delay * (2 ** (attempt - 1))


def is_retryable_failure(error: BaseException) -> bool:
    """Return whether a failed episode is worth retrying.

    Schema/validation failures are deterministic: the same episode produces the
    same malformed extraction on every attempt, so retrying only delays the
    episodes queued behind it. Everything else — empty LLM responses, rate
    limits, query timeouts, driver errors — is treated as transient.
    """
    return not isinstance(error, ValidationError)


@dataclass
class QueueStats:
    """Counters for a single group_id queue.

    Attributes:
        processed: Episodes that completed successfully.
        retried: Retry attempts made (one per failed attempt that was retried).
        dropped: Episodes abandoned after exhausting their attempts or failing
            permanently. Their facts are not in the graph.
    """

    processed: int = 0
    retried: int = 0
    dropped: int = 0


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ):
        """Initialize the queue service.

        Args:
            max_attempts: Total attempts per episode, including the first.
            retry_base_delay: Seconds to wait before the first retry. Doubles on
                each subsequent attempt.
        """
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Per-group counters for episodes processed, retried, and dropped
        self._queue_stats: dict[str, QueueStats] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay = max(0.0, retry_base_delay)

    async def add_episode_task(
        self,
        group_id: str,
        process_func: Callable[[], Awaitable[None]],
        item_id: str | None = None,
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode
            item_id: Optional identifier (usually the episode UUID) used in the
                worker's retry and drop logs

        Returns:
            The position in the queue
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()
            self._queue_stats[group_id] = QueueStats()

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put((process_func, item_id))

        # Start a worker for this queue if one isn't already running. The flag is
        # set here, before the worker task gets a chance to run, so that calls
        # arriving back to back cannot each spawn a worker for the same queue and
        # process that group's episodes out of order.
        if not self._queue_workers.get(group_id, False):
            self._queue_workers[group_id] = True
            asyncio.create_task(self._process_episode_queue(group_id))

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time. A failing episode is retried in place, so
        ordering within the group is preserved; once it is dropped, the worker
        moves on to the next episode. The running flag is set by add_episode_task.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func, item_id = await self._episode_queues[group_id].get()

                try:
                    await self._run_with_retries(group_id, process_func, item_id)
                finally:
                    # Mark the task as done once it has been processed or dropped
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    async def _run_with_retries(
        self,
        group_id: str,
        process_func: Callable[[], Awaitable[None]],
        item_id: str | None,
    ) -> None:
        """Run one queued episode, retrying transient failures with backoff.

        Retries happen in place, so the queue stays sequential per group_id. An
        episode that exhausts max_attempts, or fails permanently, is counted as
        dropped and logged with the counters for that group.
        """
        stats = self._queue_stats[group_id]
        label = item_id or 'unknown'
        attempt = 1

        while True:
            try:
                await process_func()
            except Exception as e:
                if attempt >= self._max_attempts or not is_retryable_failure(e):
                    stats.dropped += 1
                    logger.error(
                        f'Dropping episode {label} for group_id {group_id} after '
                        f'{attempt} attempt(s): {type(e).__name__}: {str(e)} '
                        f'(processed={stats.processed}, retried={stats.retried}, '
                        f'dropped={stats.dropped})'
                    )
                    return

                delay = retry_delay(self._retry_base_delay, attempt)
                stats.retried += 1
                logger.warning(
                    f'Episode {label} for group_id {group_id} failed on attempt '
                    f'{attempt}/{self._max_attempts} ({type(e).__name__}: {str(e)}); '
                    f'retrying in {delay:.1f}s'
                )
                attempt += 1
                await asyncio.sleep(delay)
            else:
                stats.processed += 1
                return

    def get_stats(self, group_id: str) -> QueueStats:
        """Get the live processing counters for a group_id's queue."""
        if group_id not in self._queue_stats:
            return QueueStats()
        return self._queue_stats[group_id]

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
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
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
            reference_time: Event occurrence time for the episode. Defaults to
                the current UTC time when not provided (bi-temporal model).
            edge_types: Optional mapping of edge (fact) type name to Pydantic model
            edge_type_map: Optional mapping of (source, target) entity type pairs to
                allowed edge type names
            excluded_entity_types: Optional list of entity type names to exclude
                from extraction
            previous_episode_uuids: Optional explicit list of prior episode UUIDs to
                use as context (overrides automatic retrieval)
            custom_extraction_instructions: Optional extra natural-language
                instructions for the extraction LLM
            update_communities: Whether to incrementally update communities after
                ingestion
            saga: Optional saga name/id to attach this episode to
            saga_previous_episode_uuid: Optional UUID of the prior episode in the saga

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
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode, item_id=uuid)
