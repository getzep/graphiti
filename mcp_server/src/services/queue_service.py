"""Queue service for managing episode processing."""

import asyncio
import json
import logging
import random
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import httpx
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Bounded retries for transient episode failures. Validation failures and other
# permanently broken inputs are deterministic and are never retried; see
# is_retryable_failure.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0

# Log an aggregate counter line for a group every this many episodes handled
# (processed + dropped), so the counters show up in the server logs at a steady,
# bounded rate instead of one extra line per episode.
STATS_LOG_INTERVAL = 100

# Errors that are genuinely transient and worth retrying: the classes the core
# LLM client already retries (graphiti_core.llm_client.client.is_server_or_retry_error)
# plus transient IO. This is an allowlist: everything outside it — a
# ValidationError, a bad uuid raising NodeNotFoundError, an HTTP 4xx, a
# KeyError/TypeError from malformed input — is treated as permanent. Retrying
# those repeats their side effects and delays the episodes queued behind them.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    EmptyResponseError,
    RateLimitError,
    json.JSONDecodeError,
    # TimeoutError is an OSError subclass; ConnectionError likewise. They are
    # listed explicitly for readability.
    TimeoutError,
    ConnectionError,
    OSError,
    # On Python 3.10 (the MCP server's floor) asyncio.TimeoutError is a distinct
    # class from the builtin TimeoutError; on 3.11+ it is an alias of it, so this
    # entry is a no-op there.
    asyncio.TimeoutError,
    # httpx transport failures (connect/read/write timeouts, resets) are the
    # transient IO errors an LLM/embedder endpoint produces; they are unrelated to
    # OSError, so an OSError-family match alone would miss them.
    httpx.TransportError,
)


def retry_delay(base_delay: float, attempt: int) -> float:
    """Return the deterministic minimum backoff after `attempt` failed attempts (1-based)."""
    return base_delay * (2 ** (attempt - 1))


def retry_delay_with_jitter(base_delay: float, attempt: int) -> float:
    """Return the backoff actually slept after `attempt` failed attempts (1-based).

    Exponential backoff with full jitter, in the style of the tenacity
    `wait_random_exponential` used by graphiti_core/llm_client/client.py: the wait is
    uniform in [minimum, 2 * minimum), where `retry_delay` is the minimum. Randomising
    the wait stops a batch of episodes failing together (a provider 429, a restart)
    from retrying in lockstep, while the floor keeps the backoff from collapsing to
    near-zero sleeps the way unfloored full jitter can.
    """
    minimum = retry_delay(base_delay, attempt)
    return minimum + random.uniform(0.0, minimum)


def is_retryable_failure(error: BaseException) -> bool:
    """Return whether a failed episode is worth retrying.

    This is an allowlist of genuinely transient failures, not a denylist. Schema or
    validation failures are deterministic: the same episode produces the same
    malformed extraction on every attempt. So is a bad episode uuid (a node that does
    not exist raises `NodeNotFoundError: node ... not found`), an HTTP 401/403, and a
    KeyError/TypeError from malformed input. Retrying those just burns the attempt cap
    while repeating side effects.
    """
    if isinstance(error, ValidationError):
        # Permanently non-retryable, and pydantic's ValidationError is also a
        # ValueError subclass, so keep this check explicit and ahead of the allowlist.
        return False
    if isinstance(error, _TRANSIENT_ERRORS):
        return True
    # A 5xx from an LLM, embedder, or database endpoint is transient; a 4xx is not.
    return isinstance(error, httpx.HTTPStatusError) and 500 <= error.response.status_code < 600


def new_correlation_id() -> str:
    """Return a log-only correlation id for an episode queued without a uuid."""
    return f'ep-{secrets.token_hex(4)}'


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
                each subsequent attempt, with full jitter on top (see
                retry_delay_with_jitter).
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
            item_id: Optional identifier used in the worker's retry and drop logs.
                This is the episode UUID when the caller supplied one, otherwise a
                queue-time correlation id.

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
        dropped and logged with that group's counters plus the totals across all
        groups. The counters are also logged periodically (every
        STATS_LOG_INTERVAL episodes handled) so drops are visible without one log
        line per episode.
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
                        f'({self._format_stats(stats)}; all groups: '
                        f'{self._format_stats(self._totals())})'
                    )
                    return

                delay = retry_delay_with_jitter(self._retry_base_delay, attempt)
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
                if (stats.processed + stats.dropped) % STATS_LOG_INTERVAL == 0:
                    logger.info(
                        f'Episode queue counters for group_id {group_id}: '
                        f'{self._format_stats(stats)}; all groups: '
                        f'{self._format_stats(self._totals())}'
                    )
                return

    def _totals(self) -> QueueStats:
        """Aggregate the counters of every group_id queue into a fresh QueueStats."""
        totals = QueueStats()
        for stats in self._queue_stats.values():
            totals.processed += stats.processed
            totals.retried += stats.retried
            totals.dropped += stats.dropped
        return totals

    @staticmethod
    def _format_stats(stats: QueueStats) -> str:
        """Format counters for a log line."""
        return f'processed={stats.processed}, retried={stats.retried}, dropped={stats.dropped}'

    def get_stats(self, group_id: str) -> QueueStats:
        """Get a snapshot copy of the processing counters for a group_id's queue.

        Returns a copy: mutating it does not affect the service's own counters, and
        the counters keep advancing as the worker runs.
        """
        if group_id not in self._queue_stats:
            return QueueStats()
        return replace(self._queue_stats[group_id])

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

        # Correlation id for the worker's retry/drop logs. When the caller supplies no
        # uuid (the default add_memory path) one is generated here, at queue time, and
        # used for logging ONLY: it must never be handed to Graphiti.add_episode. A
        # uuid that does not exist yet short-circuits through
        # EpisodicNode.get_by_uuid and fails with 'node not found', so inventing one
        # would break the very path it is meant to make observable.
        log_label = uuid or new_correlation_id()

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {log_label} for group {group_id}')

                # Process the episode using the graphiti client
                episode = await self._graphiti_client.add_episode(
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

            except Exception as e:
                logger.error(
                    f'Failed to process episode {log_label} for group {group_id}: {str(e)}'
                )
                raise

            # The client assigns the uuid when the caller did not supply one; log the
            # episode's own uuid so the success line correlates with the graph.
            episode_uuid = getattr(episode, 'uuid', None)
            queued_as = (
                f' (queued as {log_label})' if episode_uuid and episode_uuid != log_label else ''
            )
            logger.info(
                f'Successfully processed episode {episode_uuid or log_label}{queued_as} '
                f'for group {group_id}'
            )

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode, item_id=log_label)
