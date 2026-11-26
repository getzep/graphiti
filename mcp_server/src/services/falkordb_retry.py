"""Retry FalkorDB queries that fail with transient connection errors.

graphiti-core's ``FalkorDriver`` sends every query through ``AsyncGraph.query`` /
``ro_query`` once and re-raises any error. A single Redis timeout or dropped connection
during background ingestion therefore propagates out of ``Graphiti.add_episode`` and the
episode goes straight to the dead-letter file.

The retry is installed on the shared ``FalkorDB`` client, so it covers the base driver and
every per-group clone (``FalkorDriver.clone`` reuses the same client). Only connection-level
failures are retried: timeouts, dropped or refused connections, a server still loading its
dataset, and ``TRYAGAIN``. Query errors (``ResponseError``) and authentication failures are
not, because repeating them cannot succeed. Neither are ``ReadOnlyError`` or cluster
redirection errors, which are ``ResponseError`` subclasses.

The retry applies to every query on the client, so searches and startup index builds also
retry before failing. redis-py 6+ can retry connection errors internally as well (5.x does
not by default); if a client is configured that way the two schedules stack.

A timeout can arrive after the server already applied a write. Re-sending is safe for
graphiti's writes, which ``MERGE`` on ``uuid`` (or ``SET`` properties) rather than ``CREATE``,
so a repeated query updates the same node or edge instead of duplicating it.
"""

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from services.rate_limit_retry import RateLimitRetryPolicy, retry_with_backoff

logger = logging.getLogger(__name__)

_MAX_CAUSE_DEPTH = 10

# 3 retries after 1s, 3s and 9s: long enough to ride out a FalkorDB restart or a brief
# network drop, short enough not to stall the per-group queue.
DEFAULT_FALKORDB_RETRY_POLICY = RateLimitRetryPolicy(
    max_retries=3, base_delay=1.0, multiplier=3.0, max_delay=30.0
)


def _transient_types() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    """Return (retryable, never-retry) exception types; redis is an optional dependency."""
    try:
        from redis import exceptions as redis_errors
    except ImportError:  # pragma: no cover - falkordb always installs redis
        return (), ()
    retryable = (
        redis_errors.TimeoutError,
        redis_errors.ConnectionError,  # includes BusyLoadingError, MaxConnectionsError
        redis_errors.TryAgainError,
    )
    # AuthenticationError (and its subclasses) inherit from ConnectionError, but a bad
    # password will not fix itself.
    never = (redis_errors.AuthenticationError, redis_errors.AuthorizationError)
    return retryable, never


def is_transient_falkordb_error(exc: BaseException) -> bool:
    """Return True if ``exc`` or an explicit cause (``raise ... from``) is a transient Redis error.

    Only ``__cause__`` is followed, not the implicit ``__context__``: an unrelated error
    raised while handling a timeout (say a query error on reconnect) must not be retried.
    """
    retryable, never = _transient_types()
    if not retryable:
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and id(current) not in seen and depth < _MAX_CAUSE_DEPTH:
        seen.add(id(current))
        if isinstance(current, never):
            return False
        if isinstance(current, retryable):
            return True
        current = current.__cause__
        depth += 1
    return False


def _wrap_query(
    method: Callable[..., Awaitable[Any]],
    policy: RateLimitRetryPolicy,
    operation: str,
    sleep: Callable[[float], Awaitable[Any]] | None,
) -> Callable[..., Awaitable[Any]]:
    if getattr(method, '_transient_retry_wrapped', False):
        return method

    @functools.wraps(method)
    async def query(*args: Any, **kwargs: Any) -> Any:
        extra = {} if sleep is None else {'sleep': sleep}
        return await retry_with_backoff(
            lambda: method(*args, **kwargs),
            policy=policy,
            operation=operation,
            should_retry=is_transient_falkordb_error,
            reason='transient FalkorDB error',
            **extra,
        )

    query._transient_retry_wrapped = True  # type: ignore[attr-defined]
    return query


def add_falkordb_transient_retry(
    falkor_client: Any,
    policy: RateLimitRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[Any]] | None = None,
) -> Any:
    """Make graphs selected from ``falkor_client`` retry transient query failures.

    ``falkor_client`` is the ``falkordb.asyncio.FalkorDB`` instance shared by a
    ``FalkorDriver`` and its clones (``driver.client``). Its ``select_graph`` is patched so
    each returned graph's ``query`` and ``ro_query`` retry with backoff. Policy comes from
    ``FALKORDB_MAX_RETRIES``, ``FALKORDB_RETRY_BASE_DELAY``, ``FALKORDB_RETRY_MULTIPLIER``
    and ``FALKORDB_RETRY_MAX_DELAY`` (defaults 3, 1, 3, 30).
    """
    if getattr(falkor_client, '_transient_retry_installed', False):
        return falkor_client

    policy = policy or RateLimitRetryPolicy.from_env('FALKORDB', DEFAULT_FALKORDB_RETRY_POLICY)
    original_select_graph = falkor_client.select_graph

    @functools.wraps(original_select_graph)
    def select_graph(graph_id: str, *args: Any, **kwargs: Any) -> Any:
        graph = original_select_graph(graph_id, *args, **kwargs)
        for name in ('query', 'ro_query'):
            method = getattr(graph, name, None)
            if method is not None:
                setattr(
                    graph, name, _wrap_query(method, policy, f'FalkorDB {name} ({graph_id})', sleep)
                )
        return graph

    falkor_client.select_graph = select_graph
    falkor_client._transient_retry_installed = True
    return falkor_client
