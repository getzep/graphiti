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
import math
import os
import re
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


def falkordb_socket_timeout() -> float | None:
    """Return ``FALKORDB_SOCKET_TIMEOUT`` in seconds, or None when unset or invalid.

    Unset by default: without a socket timeout a stalled server hangs a query rather than
    raising ``TimeoutError``, but too short a timeout would cut off slow legitimate queries.
    """
    raw = os.getenv('FALKORDB_SOCKET_TIMEOUT')
    if raw is None or raw.strip() == '':
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f'Ignoring invalid FALKORDB_SOCKET_TIMEOUT={raw!r}')
        return None
    if not math.isfinite(value) or value <= 0:
        logger.warning(f'Ignoring non-positive or non-finite FALKORDB_SOCKET_TIMEOUT={raw!r}')
        return None
    return value


_FALKORDB_DRIVER_LOGGER = 'graphiti_core.driver.falkordb_driver'
_QUERY_ERROR_PREFIX = 'Error executing FalkorDB query:'
_EXCEPTION_NAME = re.compile(r'([A-Z]\w*)?(Error|Exception)')


class _RedactFalkorDBQueryErrors(logging.Filter):
    """Strip error text and params from graphiti-core's FalkorDB query-failure log line.

    graphiti-core releases up to and including the one this server pins log
    ``Error executing FalkorDB query: {e}\\n{query}\\n{params}`` at ERROR. The params (and
    FalkorDB error text, which can inline them) carry episode bodies, entity names and
    facts. The exception itself still propagates, and its message is kept in the 0o600
    dead-letter record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith(_QUERY_ERROR_PREFIX):
            first_line = message.split('\n', 1)[0][len(_QUERY_ERROR_PREFIX) :].strip()
            # Newer graphiti-core logs only the exception type: keep that, drop the rest.
            kind = first_line if _EXCEPTION_NAME.fullmatch(first_line) else 'error'
            record.msg = (
                f'{_QUERY_ERROR_PREFIX} {kind} (query text, params and error message withheld)'
            )
            record.args = ()
        return True


def install_falkordb_log_redaction() -> None:
    """Install the redaction filter on graphiti-core's FalkorDB driver logger (idempotent)."""
    driver_logger = logging.getLogger(_FALKORDB_DRIVER_LOGGER)
    if not any(isinstance(f, _RedactFalkorDBQueryErrors) for f in driver_logger.filters):
        driver_logger.addFilter(_RedactFalkorDBQueryErrors())


def build_falkordb_client(db_config: dict[str, Any]) -> Any | None:
    """Return a ``FalkorDB`` client with socket timeouts, or None to use the driver default.

    Only built when ``FALKORDB_SOCKET_TIMEOUT`` is set; the result is passed to
    ``FalkorDriver(falkor_db=...)``.
    """
    socket_timeout = falkordb_socket_timeout()
    if socket_timeout is None:
        return None
    from falkordb.asyncio import FalkorDB

    return FalkorDB(
        host=db_config['host'],
        port=db_config['port'],
        username=db_config.get('username'),
        password=db_config['password'],
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
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
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < _MAX_CAUSE_DEPTH:
        chain.append(current)
        current = current.__cause__
    # An auth failure anywhere in the chain wins, even when it is wrapped in a
    # ConnectionError: a bad password will not fix itself.
    if any(isinstance(e, never) for e in chain):
        return False
    return any(isinstance(e, retryable) for e in chain)


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
