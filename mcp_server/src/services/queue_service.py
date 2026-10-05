"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class QueueService:
    """Service for managing sequential episode processing queues by group_id with global concurrency control."""

    def __init__(self, max_concurrent: int = 5, max_queue_size_per_group: int = 0):
        """Initialize the queue service.

        Args:
            max_concurrent: Maximum number of episodes to process concurrently across all group_ids.
                          - 1-3: Strict limit for low-tier APIs (OpenAI Free, Anthropic Free)
                          - 5-8: Balanced for mid-tier APIs (OpenAI Tier 2-3, default: 5)
                          - 10-20: High concurrency for high-tier APIs (OpenAI Tier 4+)
            max_queue_size_per_group: Maximum queued episodes per group_id before new
                          ones are rejected. 0 means unbounded.
        """
        self.max_queue_size_per_group = max_queue_size_per_group
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Strong references to spawned worker tasks (CPython may otherwise
        # garbage-collect them mid-execution).
        self._worker_tasks: set[asyncio.Task] = set()
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None
        # Global semaphore to limit concurrent processing across all group_ids
        self._max_concurrent = max_concurrent
        self._process_semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f'QueueService initialized with max_concurrent={max_concurrent}')

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
            self._episode_queues[group_id] = asyncio.Queue(
                maxsize=self.max_queue_size_per_group if self.max_queue_size_per_group > 0 else 0
            )

        # Add the episode processing function to the queue. put_nowait on a
        # full BOUNDED queue raises QueueFull immediately - the caller reports
        # the rejection BEFORE add_memory returns success (await put() would
        # block forever instead).
        self._episode_queues[group_id].put_nowait(process_func)

        # Start a worker for this queue if one isn't already running.
        # Keep a strong reference: CPython may garbage-collect unreferenced
        # tasks mid-execution (asyncio docs: save a reference).
        if not self._queue_workers.get(group_id, False):
            task = asyncio.create_task(self._process_episode_queue(group_id))
            self._worker_tasks.add(task)
            task.add_done_callback(self._worker_tasks.discard)

        return self._episode_queues[group_id].qsize()

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait for queued and in-flight episodes to finish (shutdown hook).

        Cancels workers still idle-waiting after the timeout. Episodes still
        processing when the timeout hits are abandoned with a warning.
        """
        import contextlib

        for group_id, q in self._episode_queues.items():
            if q.qsize():
                logger.info(f'Draining {q.qsize()} queued episodes for group {group_id}')
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(
                    *[t for t in self._worker_tasks if not t.done()], return_exceptions=True
                ),
                timeout=timeout,
            )
        pending = [t for t in self._worker_tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            logger.warning(f'drain timeout: {len(pending)} worker task(s) cancelled')

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time. A global semaphore limits concurrent
        processing across all group_ids to prevent LLM API rate limit errors.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        self._queue_workers[group_id] = True

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func = await self._episode_queues[group_id].get()

                try:
                    # Acquire semaphore before processing
                    # This limits concurrent processing across ALL group_ids
                    async with self._process_semaphore:
                        # Calculate active concurrent tasks
                        active = self._max_concurrent - self._process_semaphore._value
                        logger.debug(
                            f'Processing episode for {group_id} '
                            f'(active: {active}/{self._max_concurrent}, queued: {self._episode_queues[group_id].qsize()})'
                        )
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
        return await self.add_episode_task(group_id, process_episode)
