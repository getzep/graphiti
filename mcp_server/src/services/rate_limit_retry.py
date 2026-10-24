"""Retry provider calls that fail with transient rate-limit errors.

graphiti-core's provider clients fail fast on rate limits: ``AnthropicClient``,
``GeminiClient`` and ``BaseOpenAIClient`` re-raise ``RateLimitError`` without retrying,
and ``GeminiEmbedder`` has no retry at all. Inside the MCP server a single 429 during
background ingestion therefore propagates out of ``Graphiti.add_episode`` and the queued
episode is lost.

HTTP 429 / ``RESOURCE_EXHAUSTED`` is also returned for exhausted daily or billing quotas,
which waiting will not fix. Those cannot be told apart from per-minute limits, so they use
the same schedule and then fall through to the dead-letter file.

The retry is applied at the provider-call level (one embedding request, one LLM
request) rather than by re-running ``add_episode``. A provider call has no side
effects on the graph, so repeating it is idempotent; re-running a half-finished
``add_episode`` would not be.
"""

import asyncio
import copy
import functools
import logging
import os
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.errors import RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Substrings that identify a rate-limit error when no typed status code is available.
_RATE_LIMIT_MARKERS = ('resource_exhausted', 'rate limit', 'ratelimit', 'too many requests')
_MAX_CAUSE_DEPTH = 10


def _env_number(name: str, default: float, cast: Callable[[str], Any] = float) -> Any:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning(f'Ignoring invalid {name}={raw!r}; using default {default}')
        return default
    if value < 0:
        logger.warning(f'Ignoring negative {name}={raw!r}; using default {default}')
        return default
    return value


@dataclass(frozen=True)
class RateLimitRetryPolicy:
    """Exponential backoff schedule for rate-limited provider calls.

    With the defaults, a call is retried 3 times, after 10s, 30s and 90s.
    """

    max_retries: int = 3
    base_delay: float = 10.0
    multiplier: float = 3.0
    max_delay: float = 300.0

    @classmethod
    def from_env(cls) -> 'RateLimitRetryPolicy':
        """Build a policy from environment variables, falling back to the defaults.

        - ``RATE_LIMIT_MAX_RETRIES`` (default 3; 0 disables retrying)
        - ``RATE_LIMIT_RETRY_BASE_DELAY`` seconds (default 10)
        - ``RATE_LIMIT_RETRY_MULTIPLIER`` (default 3)
        - ``RATE_LIMIT_RETRY_MAX_DELAY`` seconds (default 300)
        """
        defaults = cls()
        return cls(
            max_retries=_env_number('RATE_LIMIT_MAX_RETRIES', defaults.max_retries, int),
            base_delay=_env_number('RATE_LIMIT_RETRY_BASE_DELAY', defaults.base_delay),
            multiplier=_env_number('RATE_LIMIT_RETRY_MULTIPLIER', defaults.multiplier),
            max_delay=_env_number('RATE_LIMIT_RETRY_MAX_DELAY', defaults.max_delay),
        )

    def delays(self) -> list[float]:
        """Return the sleep before each retry, in order."""
        return [
            min(self.base_delay * (self.multiplier**attempt), self.max_delay)
            for attempt in range(self.max_retries)
        ]


def _status_code(exc: BaseException) -> int | None:
    for attr in ('status_code', 'code', 'status'):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, 'response', None)
    value = getattr(response, 'status_code', None)
    return value if isinstance(value, int) else None


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if ``exc`` (or anything in its cause chain) is a rate-limit error.

    Recognises graphiti-core's ``RateLimitError``, provider SDK ``RateLimitError``
    classes (Anthropic, OpenAI, Voyage, ...), HTTP 429 responses (httpx, google-genai
    ``ClientError``), and Google's ``RESOURCE_EXHAUSTED`` status.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and id(current) not in seen and depth < _MAX_CAUSE_DEPTH:
        seen.add(id(current))
        if isinstance(current, RateLimitError):
            return True
        if type(current).__name__ == 'RateLimitError':
            return True
        if _status_code(current) == 429:
            return True
        status = getattr(current, 'status', None)
        if isinstance(status, str) and status.upper() == 'RESOURCE_EXHAUSTED':
            return True
        # Untyped errors: require an explicit 429 next to a rate-limit marker, so messages
        # such as "connection pool resource_exhausted" are not mistaken for rate limits.
        message = str(current).lower()
        if '429' in message and any(marker in message for marker in _RATE_LIMIT_MARKERS):
            return True
        current = current.__cause__ or current.__context__
        depth += 1
    return False


async def retry_on_rate_limit(
    func: Callable[[], Awaitable[T]],
    *,
    policy: RateLimitRetryPolicy,
    operation: str,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> T:
    """Await ``func()``, retrying with backoff while it raises rate-limit errors.

    Non-rate-limit errors are re-raised immediately. When the retries are used up the
    last rate-limit error is re-raised.
    """
    delays = policy.delays()
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as e:
            if not is_rate_limit_error(e):
                raise
            if attempt >= len(delays):
                logger.error(
                    f'{operation}: rate limited, giving up after {attempt} '
                    f'retr{"y" if attempt == 1 else "ies"}: {e}'
                )
                raise
            delay = delays[attempt]
            attempt += 1
            logger.warning(
                f'{operation}: rate limited ({type(e).__name__}: {e}); '
                f'retry {attempt}/{len(delays)} in {delay:g}s'
            )
            await sleep(delay)


class RateLimitRetryEmbedder(EmbedderClient):
    """Embedder wrapper that retries rate-limited embedding requests.

    Every other attribute is delegated to the wrapped embedder.
    """

    def __init__(
        self,
        inner: EmbedderClient,
        policy: RateLimitRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ):
        self._inner = inner
        self._policy = policy or RateLimitRetryPolicy.from_env()
        self._sleep = sleep

    @property
    def inner(self) -> EmbedderClient:
        return self._inner

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, e.g. ``embedder.config`` / ``.client``.
        if name == '_inner':
            raise AttributeError(name)
        return getattr(self._inner, name)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        return await retry_on_rate_limit(
            lambda: self._inner.create(input_data),
            policy=self._policy,
            operation=f'{type(self._inner).__name__}.create',
            sleep=self._sleep,
        )

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await retry_on_rate_limit(
            lambda: self._inner.create_batch(input_data_list),
            policy=self._policy,
            operation=f'{type(self._inner).__name__}.create_batch',
            sleep=self._sleep,
        )


def add_llm_rate_limit_retry(
    llm_client: LLMClient,
    policy: RateLimitRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> LLMClient:
    """Make ``llm_client.generate_response`` retry rate-limited requests in place.

    The client object is patched rather than wrapped so that ``isinstance`` checks and
    attributes such as ``model``, ``token_tracker`` and ``tracer`` keep working.

    Provider clients append correction messages to ``messages`` during their own
    validation retries, so each attempt gets a fresh copy of the messages as they were
    when the call was made.

    Clients that go through ``LLMClient._generate_response_with_retry`` (for example
    ``OpenAIGenericClient``) already retry ``RateLimitError`` with tenacity; for those the
    two schedules stack and an episode can wait considerably longer before it is dead-lettered.
    """
    if getattr(llm_client, '_rate_limit_retry_installed', False):
        return llm_client

    policy = policy or RateLimitRetryPolicy.from_env()
    original = llm_client.generate_response
    operation = f'{type(llm_client).__name__}.generate_response'

    @functools.wraps(original)
    async def generate_response(messages, *args, **kwargs):
        snapshot = copy.deepcopy(messages)

        async def call():
            # Every attempt, including the first, works on its own copy, so a failed
            # attempt never leaves the caller's list half-mutated.
            return await original(copy.deepcopy(snapshot), *args, **kwargs)

        return await retry_on_rate_limit(call, policy=policy, operation=operation, sleep=sleep)

    llm_client.generate_response = generate_response  # type: ignore[method-assign]
    llm_client._rate_limit_retry_installed = True  # type: ignore[attr-defined]
    return llm_client
