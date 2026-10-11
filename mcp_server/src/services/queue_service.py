"""Queue service for managing episode processing."""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.rate_limit_retry import is_rate_limit_error

logger = logging.getLogger(__name__)

DEFAULT_DEAD_LETTER_PATH = 'episode_dead_letter.jsonl'


def _type_names(types: Any) -> list[str] | None:
    """Return the names of a mapping/list of entity or edge types, for logging."""
    if not types:
        return None
    return sorted(str(name) for name in types)


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self, dead_letter_path: str | os.PathLike[str] | None = None):
        """Initialize the queue service.

        Args:
            dead_letter_path: JSONL file that receives the payload of every episode that
                fails to process, so it can be replayed. Defaults to the
                ``EPISODE_DEAD_LETTER_PATH`` environment variable, then
                ``episode_dead_letter.jsonl`` in the working directory.
        """
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None
        self._dead_letter_path = Path(
            dead_letter_path or os.getenv('EPISODE_DEAD_LETTER_PATH') or DEFAULT_DEAD_LETTER_PATH
        )
        self._dead_letter_lock = asyncio.Lock()

    @property
    def dead_letter_path(self) -> Path:
        """Path of the JSONL file that records episodes that failed to process."""
        return self._dead_letter_path

    async def _write_dead_letter(self, record: dict[str, Any]) -> bool:
        """Append one record to the dead-letter file. Returns True on success."""
        line = json.dumps(record, default=str, ensure_ascii=False) + '\n'

        def _append() -> None:
            self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            with self._dead_letter_path.open('a', encoding='utf-8') as f:
                f.write(line)

        try:
            async with self._dead_letter_lock:
                await asyncio.to_thread(_append)
            return True
        except Exception as e:
            logger.error(
                f'Could not write dead-letter record to {self._dead_letter_path}: {e}. '
                f'Episode payload follows so it is not lost: {line.strip()}'
            )
            return False

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
                # The provider clients already retried rate-limit errors with backoff (see
                # services.rate_limit_retry), so this failure is final. Nothing re-queues
                # the episode: record its payload so it can be replayed instead of lost.
                logger.error(
                    f'EPISODE DROPPED: failed to process episode {uuid or name!r} for group '
                    f'{group_id}: {type(e).__name__}: {e}. Payload written to dead-letter '
                    f'file {self._dead_letter_path}'
                )
                await self._write_dead_letter(
                    {
                        'failed_at': datetime.now(timezone.utc).isoformat(),
                        'error_type': type(e).__name__,
                        'error': str(e),
                        'rate_limited': is_rate_limit_error(e),
                        'group_id': group_id,
                        'uuid': uuid,
                        'name': name,
                        'episode_body': content,
                        'source': getattr(episode_type, 'value', str(episode_type)),
                        'source_description': source_description,
                        'reference_time': reference_time.isoformat() if reference_time else None,
                        'excluded_entity_types': excluded_entity_types,
                        'previous_episode_uuids': previous_episode_uuids,
                        'custom_extraction_instructions': custom_extraction_instructions,
                        'update_communities': update_communities,
                        'saga': saga,
                        'saga_previous_episode_uuid': saga_previous_episode_uuid,
                        'entity_types': _type_names(entity_types),
                        'edge_types': _type_names(edge_types),
                    }
                )
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode)
