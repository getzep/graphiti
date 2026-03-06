"""Queue service for managing episode processing."""

import asyncio
import logging
import os
import random
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_GROUP_ID_RE = re.compile(r'^[a-zA-Z0-9_]+$')
_MAX_QUEUE_SIZE = 1000

# ---------------------------------------------------------------------------
# Configurable episode body size cap
# ---------------------------------------------------------------------------
# Truncates oversized bodies before they reach the LLM to avoid context
# window errors.  Range-clamped: minimum 2 000, maximum 200 000 characters.
# Override via environment variable: GRAPHITI_MAX_EPISODE_BODY_CHARS
_MAX_EPISODE_BODY_CHARS: int = max(
    2_000,
    min(200_000, int(os.environ.get('GRAPHITI_MAX_EPISODE_BODY_CHARS', '60000'))),
)

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
# On transient database timeouts, retry with exponential backoff + jitter.
# Note: retries may cause duplicate edges if the upstream operation is not
# fully idempotent; we prefer eventual completion + later dedup over
# silently dropping episodes.
#
# Override via:
#   GRAPHITI_TIMEOUT_MAX_RETRIES  (default: 5)
#   GRAPHITI_CONTEXT_MAX_RETRIES  (default: 3)
_TIMEOUT_MAX_RETRIES: int = int(os.environ.get('GRAPHITI_TIMEOUT_MAX_RETRIES', '5'))
_CONTEXT_MAX_RETRIES: int = int(os.environ.get('GRAPHITI_CONTEXT_MAX_RETRIES', '3'))
_TIMEOUT_BACKOFF_BASE_S: float = 1.0
_TIMEOUT_BACKOFF_CAP_S: float = 30.0


def _sanitize_episode_body(body: str) -> str:
    """Truncate episode bodies that exceed the configured size cap.

    Uses a head+tail strategy to preserve both opening and closing context.
    The cap is controlled by the ``GRAPHITI_MAX_EPISODE_BODY_CHARS``
    environment variable (default: 60 000 characters).

    Args:
        body: Raw episode body text.

    Returns:
        The body, possibly truncated with a truncation notice inserted.
    """
    if len(body) <= _MAX_EPISODE_BODY_CHARS:
        return body

    # Keep 70 % from the head, 30 % from the tail so the summary / conclusion
    # context is retained alongside the opening.
    head_n = int(_MAX_EPISODE_BODY_CHARS * 0.7)
    tail_n = _MAX_EPISODE_BODY_CHARS - head_n
    return (
        body[:head_n]
        + f'\n\n[... TRUNCATED {len(body) - _MAX_EPISODE_BODY_CHARS} chars to fit extraction budget ...]\n\n'
        + body[-tail_n:]
    )


def build_om_candidate_rows(
    om_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build candidate-bridge rows from OM fact payloads.

    Emits only rows with the required provenance contract fields.
    """
    rows: list[dict[str, Any]] = []

    for fact in om_facts:
        source_node_id = str(
            fact.get('source_node_id') or fact.get('source_node_uuid') or ''
        ).strip()
        source_event_id = str(fact.get('uuid') or '').strip()
        source_group_id = str(fact.get('group_id') or '').strip()
        created_at = str(fact.get('created_at') or '').strip()

        if not source_node_id or not source_event_id or not source_group_id or not created_at:
            continue

        evidence_refs = [
            {
                'evidence_id': source_event_id,
                'source_key': f'om:{source_group_id}:{source_node_id}',
                'scope': source_group_id,
            }
        ]

        rows.append(
            {
                'source_lane': source_group_id,
                'source_node_id': source_node_id,
                'source_event_id': source_event_id,
                'source_group_id': source_group_id,
                'evidence_refs': evidence_refs,
                'created_at': created_at,
            }
        )

    return rows


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Worker tasks keyed by group_id — stored to prevent GC and enable
        # orderly shutdown / liveness checks.
        self._queue_workers: dict[str, asyncio.Task] = {}
        # Legacy single Graphiti client (fallback)
        self._graphiti_client: Any = None
        # Optional per-group client resolver
        self._graphiti_client_resolver: Callable[[str], Awaitable[Any]] | None = None
        # Optional per-group ontology resolver.
        # New-style v3: returns (entity_types, intent_guidance, edge_types, extraction_mode) 4-tuple.
        # New-style v2: returns (entity_types, extraction_emphasis, edge_types) 3-tuple.
        # New-style v1: returns (entity_types, extraction_emphasis) 2-tuple.
        # Legacy: returns entity_types dict or None.
        self._ontology_resolver: Callable[[str], tuple[dict | None, str, dict | None, str] | tuple[dict | None, str, dict | None] | tuple[dict | None, str] | dict | None] | None = None

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
            self._episode_queues[group_id] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already running.
        # Store the Task immediately to prevent both GC and duplicate spawns.
        existing = self._queue_workers.get(group_id)
        if existing is None or existing.done():
            task = asyncio.create_task(self._process_episode_queue(group_id))
            self._queue_workers[group_id] = task

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id with configurable concurrency.

        Runs as a long-lived task.  Uses a semaphore to limit concurrent
        ``add_episode`` calls.  The default concurrency of 1 preserves legacy
        serial behaviour; set ``GRAPHITI_QUEUE_CONCURRENCY`` to a higher value
        to allow parallel processing within a single group.

        On unexpected worker-loop errors the loop restarts after a 1 s delay
        rather than dying silently.
        """
        logger.info('Starting episode queue worker for group_id: %s', group_id)

        # Concurrency limit — default 1 matches legacy serial behaviour.
        concurrency = int(os.environ.get('GRAPHITI_QUEUE_CONCURRENCY', '1'))
        sem = asyncio.Semaphore(concurrency)
        logger.info('Episode queue concurrency set to %d for group %s', concurrency, group_id)

        async def run_task(func: Callable[[], Awaitable[None]]) -> None:
            """Execute a queued task, then release the semaphore slot."""
            try:
                await func()
            except Exception as exc:
                logger.error(
                    'Error processing queued episode for group_id %s: %s',
                    group_id, type(exc).__name__,
                )
            finally:
                sem.release()
                self._episode_queues[group_id].task_done()

        while True:
            try:
                # Fetch next work item (blocks until available)
                process_func = await self._episode_queues[group_id].get()

                # Acquire a concurrency slot (blocks when at the configured limit)
                await sem.acquire()

                # Dispatch execution in background; loop immediately fetches next item
                asyncio.create_task(run_task(process_func))

            except asyncio.CancelledError:
                logger.info('Episode queue worker for group_id %s was cancelled', group_id)
                return  # honour cancellation — do not restart

            except Exception as exc:
                logger.error(
                    'Queue worker loop for group_id %s crashed (%s). Restarting in 1s...',
                    group_id, type(exc).__name__,
                )
                await asyncio.sleep(1.0)

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker task is alive for a group_id."""
        task = self._queue_workers.get(group_id)
        return task is not None and not task.done()

    async def initialize(
        self,
        graphiti_client: Any | None = None,
        client_resolver: Callable[[str], Awaitable[Any]] | None = None,
        ontology_resolver: Callable[[str], tuple[dict | None, str, dict | None, str] | tuple[dict | None, str, dict | None] | tuple[dict | None, str] | dict | None] | None = None,
    ) -> None:
        """Initialize the queue service with client and ontology routing.

        Args:
            graphiti_client: Optional single Graphiti client (legacy fallback)
            client_resolver: Optional async resolver returning a Graphiti client
                for a given group_id.
            ontology_resolver: Optional callable returning per-group ontology.
                New-style v3: returns ``(entity_types, intent_guidance, edge_types, extraction_mode)`` 4-tuple.
                New-style v2: returns ``(entity_types, extraction_emphasis, edge_types)`` 3-tuple.
                New-style v1: returns ``(entity_types, extraction_emphasis)`` 2-tuple.
                Legacy: returns ``entity_types`` dict or None.
                When provided, ``add_episode`` uses it to resolve lane-specific
                entity types, intent guidance (passed as
                ``custom_extraction_instructions`` to Graphiti Core), edge
                types (passed as ``edge_types`` to Graphiti Core), and
                extraction mode (passed as ``extraction_mode`` to Graphiti Core).
        """
        if graphiti_client is None and client_resolver is None:
            raise RuntimeError(
                'Queue service initialize() requires graphiti_client or client_resolver.'
            )

        self._graphiti_client = graphiti_client
        self._graphiti_client_resolver = client_resolver
        self._ontology_resolver = ontology_resolver

        if client_resolver is not None:
            logger.info('Queue service initialized with per-group client resolver')
        else:
            logger.info('Queue service initialized with single graphiti client (legacy mode)')

        if ontology_resolver is not None:
            logger.info('Queue service initialized with per-group ontology resolver')
        else:
            logger.info('Queue service will use global entity types for all groups')

    async def _get_client_for_group(self, group_id: str) -> Any:
        """Resolve the Graphiti client for a group."""
        if self._graphiti_client_resolver is not None:
            return await self._graphiti_client_resolver(group_id)

        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        return self._graphiti_client

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        uuid: str | None,
        *,
        fallback_entity_types: Any = None,
    ) -> int:
        """Add an episode for processing.

        Entity types and extraction emphasis are resolved internally via the
        ontology resolver when configured.  ``fallback_entity_types`` is used
        only when no resolver is configured or the resolver returns ``None``
        for this group_id.

        **Production hardening (all configurable via environment variables):**

        * **Body size cap** — bodies exceeding ``GRAPHITI_MAX_EPISODE_BODY_CHARS``
          (default: 60 000) are truncated using a head+tail strategy before
          ingestion.
        * **Concurrency** — the queue worker limits parallel ``add_episode``
          calls via ``GRAPHITI_QUEUE_CONCURRENCY`` (default: 1, serial).
        * **Timeout retries** — transient database timeouts are retried with
          exponential backoff + jitter, up to ``GRAPHITI_TIMEOUT_MAX_RETRIES``
          attempts (default: 5).
        * **Context-overflow retries** — LLM context window errors trigger
          progressive body shrinking and up to ``GRAPHITI_CONTEXT_MAX_RETRIES``
          additional attempts (default: 3).

        Args:
            group_id: The group ID for the episode.  Must match
                ``^[a-zA-Z0-9_]+$``.
            name: Name / title of the episode.
            content: Episode body text.
            source_description: Human-readable description of the episode source.
            episode_type: Graphiti ``EpisodeType`` value.
            uuid: Optional deterministic UUID for the episode.
            fallback_entity_types: Entity types to use when the ontology
                resolver has no profile for this group_id (keyword-only).

        Returns:
            The position in the queue at the time of enqueue.

        Raises:
            RuntimeError: If the service has not been initialized.
            ValueError: If ``group_id`` contains unsafe characters.
        """
        if self._graphiti_client is None and self._graphiti_client_resolver is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        if not _SAFE_GROUP_ID_RE.match(group_id):
            raise ValueError(f'Invalid group_id: {group_id!r}')

        # Resolve lane-specific entity types, intent guidance, edge types, and extraction mode via
        # ontology resolver. Falls back to caller-supplied fallback_entity_types (global default).
        resolved_entity_types = fallback_entity_types
        resolved_extraction_emphasis: str = ''
        resolved_edge_types: dict | None = None
        resolved_extraction_mode: str = 'permissive'
        if self._ontology_resolver is not None:
            resolver_result = self._ontology_resolver(group_id)
            # v3: returns (entity_types, intent_guidance, edge_types, extraction_mode) 4-tuple.
            # v2: returns (entity_types, extraction_emphasis, edge_types) 3-tuple.
            # v1: returns (entity_types, extraction_emphasis) 2-tuple.
            # Legacy: returns entity_types dict or None.
            if isinstance(resolver_result, tuple):
                if len(resolver_result) >= 4:
                    per_group, resolved_extraction_emphasis, resolved_edge_types, resolved_extraction_mode = resolver_result
                elif len(resolver_result) >= 3:
                    per_group, resolved_extraction_emphasis, resolved_edge_types = resolver_result
                else:
                    per_group, resolved_extraction_emphasis = resolver_result
            else:
                per_group = resolver_result
            if per_group is not None:
                resolved_entity_types = per_group
                logger.info(
                    'Using lane-specific ontology for group %s: %s (guidance: %d chars, edge_types: %s, mode: %s)',
                    group_id,
                    list(per_group.keys()),
                    len(resolved_extraction_emphasis),
                    list(resolved_edge_types.keys()) if resolved_edge_types else None,
                    resolved_extraction_mode,
                )

        def _is_transient_timeout(exc: Exception) -> bool:
            """Return True if the exception looks like a transient database timeout."""
            msg = str(exc).lower()
            if 'query timed out' in msg or 'timed out' in msg or 'timeout' in msg:
                return True
            # Detect FalkorDB / redis-py error classes without a hard import.
            return exc.__class__.__name__ in {'ResponseError', 'TimeoutError'}

        def _is_context_overflow(exc: Exception) -> bool:
            """Return True if the exception indicates an LLM context window overflow."""
            msg = str(exc).lower()
            return (
                'context_length_exceeded' in msg
                or 'exceeds the context window' in msg
                or 'your input exceeds the context window' in msg
            )

        async def process_episode() -> None:
            """Process the episode using the graphiti client.

            Retries transient database timeouts with exponential backoff + jitter.
            For LLM context overflows, progressively shrinks the episode body and
            retries up to ``_CONTEXT_MAX_RETRIES`` times.
            """
            timeout_attempt = 0
            context_attempt = 0

            # Apply the body size cap before any network / LLM calls.
            episode_body_full = _sanitize_episode_body(content)
            if len(episode_body_full) < len(content):
                logger.warning(
                    'Episode body for group %s (name=%s) truncated from %d to %d chars',
                    group_id, name, len(content), len(episode_body_full),
                )

            # Generic progressive shrink schedule for context-overflow retries.
            # Each step cuts the body roughly in half, down to a safe floor of 500 chars.
            shrink_steps = [
                len(episode_body_full),
                max(len(episode_body_full) // 2, 1_000),
                max(len(episode_body_full) // 4, 500),
                500,
            ]

            while True:
                try:
                    logger.info('Processing episode %s for group %s', uuid, group_id)

                    client = await self._get_client_for_group(group_id)

                    max_chars = shrink_steps[min(context_attempt, len(shrink_steps) - 1)]
                    episode_body = (
                        episode_body_full
                        if max_chars >= len(episode_body_full)
                        else episode_body_full[:max_chars]
                    )

                    await client.add_episode(
                        name=name,
                        episode_body=episode_body,
                        source_description=source_description,
                        source=episode_type,
                        group_id=group_id,
                        reference_time=datetime.now(timezone.utc),
                        entity_types=resolved_entity_types,
                        edge_types=resolved_edge_types,
                        uuid=uuid,
                        custom_extraction_instructions=resolved_extraction_emphasis or None,
                        extraction_mode=resolved_extraction_mode,
                    )

                    logger.info('Successfully processed episode %s for group %s', uuid, group_id)
                    return

                except Exception as exc:
                    if _is_context_overflow(exc) and context_attempt < _CONTEXT_MAX_RETRIES:
                        context_attempt += 1
                        next_size = shrink_steps[min(context_attempt, len(shrink_steps) - 1)]
                        logger.warning(
                            'LLM context overflow for episode %s group %s (%s). '
                            'Retry %d/%d with body <= %d chars',
                            uuid, group_id, exc.__class__.__name__,
                            context_attempt, _CONTEXT_MAX_RETRIES, next_size,
                        )
                        # Short random jitter to avoid thundering-herd on shared LLM endpoints.
                        await asyncio.sleep(0.2 + random.random() * 0.3)
                        continue

                    if _is_transient_timeout(exc) and timeout_attempt < _TIMEOUT_MAX_RETRIES:
                        delay = min(
                            _TIMEOUT_BACKOFF_BASE_S * (2 ** timeout_attempt),
                            _TIMEOUT_BACKOFF_CAP_S,
                        ) + random.random() * 0.5
                        timeout_attempt += 1
                        logger.warning(
                            'Transient timeout for episode %s group %s (%s). '
                            'Retry %d/%d in %.1fs',
                            uuid, group_id, exc.__class__.__name__,
                            timeout_attempt, _TIMEOUT_MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.error(
                        'Failed to process episode %s for group %s: %s',
                        uuid, group_id, type(exc).__name__,
                    )
                    raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode)
