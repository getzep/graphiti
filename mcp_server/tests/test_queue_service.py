#!/usr/bin/env python3
"""Unit tests for the QueueService episode worker.

These drive the worker directly with fake process functions, so they need no
database, LLM, or network. They cover the failure handling behind an episode
being stored with no edges: only genuinely transient failures are retried in
place with backoff, permanent ones (schema/validation failures, an unknown uuid,
HTTP 4xx, deterministic bugs) are not, and every outcome is counted so a dropped
episode is visible instead of silent.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from pydantic import BaseModel, ValidationError

# Add the src directory to the path (mirrors the other service tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import services.queue_service as queue_service_module  # noqa: E402
from services.queue_service import (  # noqa: E402
    QueueService,
    QueueStats,
    is_retryable_failure,
    retry_delay,
    retry_delay_with_jitter,
)

GROUP_ID = 'test-group'


class _Schema(BaseModel):
    """Stand-in for an extraction response model."""

    value: int


def _validation_error() -> ValidationError:
    """Build the ValidationError a malformed extraction produces."""
    try:
        _Schema.model_validate({'value': 'not-an-int'})
    except ValidationError as e:
        return e
    raise AssertionError('expected a ValidationError')


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build the error an HTTP error response raises."""
    request = httpx.Request('POST', 'https://example.invalid/v1/chat/completions')
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f'Server error {status_code}', request=request, response=response)


class _Item:
    """A fake process function that fails a scripted number of times."""

    def __init__(self, label: str, failures: int = 0, error_factory=None, log: list | None = None):
        self.label = label
        self.failures = failures
        self.error_factory = error_factory or (lambda: OSError('query timed out'))
        self.calls = 0
        self.log = log

    async def __call__(self) -> None:
        self.calls += 1
        if self.log is not None:
            self.log.append(self.label)
        if self.calls <= self.failures:
            raise self.error_factory()


class _FakeGraphitiClient:
    """Records the kwargs QueueService passes to Graphiti.add_episode.

    Returns an object carrying the uuid the real client would have assigned, so the
    success logging can be checked without a database.
    """

    def __init__(self, uuid: str = 'server-assigned-uuid', error: BaseException | None = None):
        self.uuid = uuid
        self.error = error
        self.calls = 0
        self.kwargs: dict | None = None

    async def add_episode(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(uuid=self.uuid)


async def _drain(queue_service: QueueService, group_id: str, timeout: float = 5.0) -> None:
    """Wait until every queued item has been processed or dropped.

    A sentinel is queued behind the items under test; the worker is sequential,
    so once the sentinel runs, nothing queued before it is still outstanding.
    """
    done = asyncio.Event()

    async def sentinel() -> None:
        done.set()

    await queue_service.add_episode_task(group_id, sentinel, item_id='sentinel')
    await asyncio.wait_for(done.wait(), timeout=timeout)


async def _run(
    queue_service: QueueService, group_id: str, *items, timeout: float = 5.0
) -> QueueStats:
    """Queue items, wait for them to settle, and return this batch's counters.

    The drain sentinel also succeeds, so its own count is removed from the
    returned counters.
    """
    before = queue_service.get_stats(group_id)

    for item in items:
        await queue_service.add_episode_task(group_id, item, item_id=item.label)
    await _drain(queue_service, group_id, timeout=timeout)

    after = queue_service.get_stats(group_id)
    return QueueStats(
        processed=after.processed - before.processed - 1,
        retried=after.retried - before.retried,
        dropped=after.dropped - before.dropped,
    )


class TestFailureClassification:
    """Transient vs permanent failure classification."""

    @pytest.mark.parametrize(
        'error',
        [
            OSError('query timed out'),
            TimeoutError(),
            asyncio.TimeoutError(),
            ConnectionError('connection reset by peer'),
            httpx.ConnectError('connection reset by peer'),
            httpx.ReadTimeout('timed out'),
            EmptyResponseError('LLM returned an empty response'),
            RateLimitError(),
            json.JSONDecodeError('Expecting value', 'not json', 0),
            _http_status_error(500),
            _http_status_error(503),
        ],
    )
    def test_transient_failures_are_retryable(self, error):
        assert is_retryable_failure(error) is True

    @pytest.mark.parametrize(
        'error',
        [
            # A malformed extraction: the same episode produces the same error.
            _validation_error(),
            # A uuid that does not exist yet: graphiti raises 'node not found'.
            NodeNotFoundError('00000000-0000-0000-0000-000000000000'),
            # The same failure matched on its message must not be retried either.
            RuntimeError('node not found'),
            # Deterministic bugs and 4xx responses are permanent.
            KeyError('uuid'),
            TypeError('unexpected keyword argument'),
            ValueError('not enough values to unpack'),
            _http_status_error(400),
            _http_status_error(401),
            _http_status_error(403),
            _http_status_error(404),
        ],
    )
    def test_permanent_failures_are_not_retryable(self, error):
        assert is_retryable_failure(error) is False

    def test_validation_error_is_not_retried_despite_being_a_value_error(self):
        error = _validation_error()

        assert isinstance(error, ValueError)
        assert is_retryable_failure(error) is False


class TestRetryDelay:
    """Backoff is exponential with full jitter, floored at the exponential minimum."""

    @pytest.mark.parametrize('attempt, expected', [(1, 1.0), (2, 2.0), (3, 4.0)])
    def test_retry_delay_doubles(self, attempt, expected):
        assert retry_delay(1.0, attempt) == expected

    @pytest.mark.parametrize('attempt, minimum', [(1, 1.0), (2, 2.0), (3, 4.0)])
    def test_jittered_delay_stays_inside_the_backoff_window(self, attempt, minimum):
        for _ in range(50):
            delay = retry_delay_with_jitter(1.0, attempt)
            assert minimum <= delay < 2 * minimum

    def test_jittered_delay_uses_random_uniform_over_the_window(self, monkeypatch):
        calls: list[tuple[float, float]] = []

        def fake_uniform(low: float, high: float) -> float:
            calls.append((low, high))
            return (high - low) / 2

        monkeypatch.setattr(queue_service_module.random, 'uniform', fake_uniform)

        assert retry_delay_with_jitter(1.0, 2) == 3.0
        assert calls == [(0.0, 2.0)]

    def test_a_zero_base_delay_stays_zero(self):
        for attempt in (1, 2, 3):
            assert retry_delay_with_jitter(0.0, attempt) == 0.0


class TestWorkerSuccess:
    """The happy path is unchanged."""

    async def test_episode_is_processed_once_and_counted(self):
        queue_service = QueueService(retry_base_delay=0)
        item = _Item('ep-1')

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 1
        assert (stats.processed, stats.retried, stats.dropped) == (1, 0, 0)
        assert queue_service.get_queue_size(GROUP_ID) == 0

    async def test_items_are_processed_in_fifo_order(self):
        queue_service = QueueService(retry_base_delay=0)
        log: list[str] = []
        items = [_Item(label, log=log) for label in ('a', 'b', 'c')]

        await _run(queue_service, GROUP_ID, *items)

        assert log == ['a', 'b', 'c']

    async def test_items_queued_back_to_back_run_one_at_a_time(self):
        """Guards the worker-start race: one worker per group_id, not one per call."""
        queue_service = QueueService(retry_base_delay=0)
        started: list[str] = []
        release = asyncio.Event()

        def gated(label: str):
            async def run() -> None:
                started.append(label)
                await release.wait()

            return run

        for label in ('a', 'b', 'c'):
            await queue_service.add_episode_task(GROUP_ID, gated(label), item_id=label)

        # Let the loop schedule whatever workers exist: only the first episode
        # may be in flight, because 'a' is still blocked.
        await asyncio.sleep(0)
        assert started == ['a']

        release.set()
        await _drain(queue_service, GROUP_ID)
        assert started == ['a', 'b', 'c']


class TestRetries:
    """Transient failures are retried in place."""

    async def test_transient_failure_is_retried_then_succeeds(self):
        queue_service = QueueService(max_attempts=3, retry_base_delay=0)
        item = _Item('ep-1', failures=2)

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 3
        assert (stats.processed, stats.retried, stats.dropped) == (1, 2, 0)

    async def test_worker_awaits_the_jittered_backoff(self, monkeypatch):
        queue_service = QueueService(max_attempts=3, retry_base_delay=1.0)
        item = _Item('ep-1', failures=2, error_factory=lambda: asyncio.TimeoutError())
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            # Record the scheduled delay instead of waiting for it.
            slept.append(delay)

        monkeypatch.setattr(queue_service_module.asyncio, 'sleep', fake_sleep)

        await _run(queue_service, GROUP_ID, item)

        assert item.calls == 3
        # Full jitter, but never below the deterministic exponential minimum.
        assert len(slept) == 2
        assert 1.0 <= slept[0] < 2.0
        assert 2.0 <= slept[1] < 4.0

    async def test_retries_do_not_reorder_the_queue(self):
        queue_service = QueueService(max_attempts=3, retry_base_delay=0)
        log: list[str] = []
        first = _Item('a', failures=1, log=log)

        stats = await _run(queue_service, GROUP_ID, first, _Item('b', log=log), _Item('c', log=log))

        assert log == ['a', 'a', 'b', 'c']
        assert (stats.processed, stats.retried, stats.dropped) == (3, 1, 0)

    async def test_attempts_are_exhausted_then_the_episode_is_dropped(self):
        queue_service = QueueService(max_attempts=3, retry_base_delay=0)
        failing = _Item('ep-fail', failures=99)
        following = _Item('ep-next')

        stats = await _run(queue_service, GROUP_ID, failing, following)

        assert failing.calls == 3
        # A dropped episode must not stall the episodes behind it.
        assert following.calls == 1
        assert (stats.processed, stats.retried, stats.dropped) == (1, 2, 1)

    async def test_drop_is_logged_with_the_episode_and_counters(self, caplog):
        queue_service = QueueService(max_attempts=2, retry_base_delay=0)
        item = _Item('ep-42', failures=99)

        with caplog.at_level('ERROR', logger='services.queue_service'):
            await _run(queue_service, GROUP_ID, item)

        drops = [r.message for r in caplog.records if 'Dropping episode' in r.message]
        assert len(drops) == 1
        assert 'ep-42' in drops[0]
        assert 'after 2 attempt(s)' in drops[0]
        assert 'dropped=1' in drops[0]
        # The drop line also carries the totals across every group.
        assert 'all groups:' in drops[0]


class TestPermanentFailures:
    """Deterministic failures are dropped instead of retried."""

    async def test_validation_failure_is_not_retried(self):
        queue_service = QueueService(max_attempts=5, retry_base_delay=0)
        item = _Item('ep-bad', failures=99, error_factory=_validation_error)

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 1
        assert (stats.processed, stats.retried, stats.dropped) == (0, 0, 1)

    async def test_validation_failure_does_not_block_the_queue(self):
        queue_service = QueueService(max_attempts=5, retry_base_delay=0)
        bad = _Item('ep-bad', failures=99, error_factory=_validation_error)
        good = _Item('ep-good')

        stats = await _run(queue_service, GROUP_ID, bad, good)

        assert bad.calls == 1
        assert good.calls == 1
        assert (stats.processed, stats.retried, stats.dropped) == (1, 0, 1)

    async def test_unknown_uuid_is_dropped_without_retrying(self):
        """A bad uuid raises 'node not found': retrying repeats the same failure."""
        queue_service = QueueService(max_attempts=5, retry_base_delay=0)
        item = _Item(
            'ep-unknown',
            failures=99,
            error_factory=lambda: NodeNotFoundError('00000000-0000-0000-0000-000000000000'),
        )

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 1
        assert (stats.processed, stats.retried, stats.dropped) == (0, 0, 1)

    async def test_http_403_is_dropped_without_retrying(self):
        queue_service = QueueService(max_attempts=5, retry_base_delay=0)
        item = _Item('ep-denied', failures=99, error_factory=lambda: _http_status_error(403))

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 1
        assert (stats.processed, stats.retried, stats.dropped) == (0, 0, 1)

    async def test_http_503_is_retried(self):
        queue_service = QueueService(max_attempts=3, retry_base_delay=0)
        item = _Item('ep-busy', failures=2, error_factory=lambda: _http_status_error(503))

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 3
        assert (stats.processed, stats.retried, stats.dropped) == (1, 2, 0)


class TestCounters:
    """Counters are readable snapshots and show up in the worker's logs."""

    async def test_get_stats_returns_a_copy_of_the_live_counters(self):
        queue_service = QueueService(retry_base_delay=0)

        await _run(queue_service, GROUP_ID, _Item('ep-1'))

        snapshot = queue_service.get_stats(GROUP_ID)
        # The drain sentinel also succeeded, so the group has two processed episodes.
        assert (snapshot.processed, snapshot.retried, snapshot.dropped) == (2, 0, 0)

        snapshot.processed = 999
        snapshot.dropped = 999
        assert queue_service.get_stats(GROUP_ID).processed == 2
        assert queue_service.get_stats(GROUP_ID).dropped == 0

    async def test_periodic_aggregate_counters_are_logged(self, caplog, monkeypatch):
        monkeypatch.setattr(queue_service_module, 'STATS_LOG_INTERVAL', 2)
        queue_service = QueueService(retry_base_delay=0)

        with caplog.at_level('INFO', logger='services.queue_service'):
            await _run(queue_service, GROUP_ID, _Item('a'), _Item('b'), _Item('c'))

        summaries = [
            r.message for r in caplog.records if 'Episode queue counters for group_id' in r.message
        ]
        # Two summaries: one after 2 episodes handled, one after the 4th (the sentinel).
        assert len(summaries) == 2
        assert 'processed=2' in summaries[0]
        assert 'all groups: processed=2' in summaries[0]
        assert 'all groups: processed=4' in summaries[1]


class TestEpisodeIdentity:
    """Episodes queued without a uuid are logged, never given an invented uuid."""

    async def test_a_generated_correlation_id_is_not_passed_to_the_client(self):
        """add_episode(uuid=X) fails with 'node not found' for an unknown uuid."""
        client = _FakeGraphitiClient()
        queue_service = QueueService(retry_base_delay=0)
        await queue_service.initialize(client)

        await queue_service.add_episode(
            group_id=GROUP_ID,
            name='episode',
            content='content',
            source_description='test',
            episode_type=None,
            entity_types=None,
            uuid=None,
        )
        await _drain(queue_service, GROUP_ID)

        assert client.calls == 1
        assert client.kwargs is not None
        assert client.kwargs['uuid'] is None

    async def test_caller_supplied_uuid_is_passed_through(self):
        client = _FakeGraphitiClient()
        queue_service = QueueService(retry_base_delay=0)
        await queue_service.initialize(client)

        await queue_service.add_episode(
            group_id=GROUP_ID,
            name='episode',
            content='content',
            source_description='test',
            episode_type=None,
            entity_types=None,
            uuid='caller-supplied-uuid',
        )
        await _drain(queue_service, GROUP_ID)

        assert client.kwargs is not None
        assert client.kwargs['uuid'] == 'caller-supplied-uuid'

    async def test_success_log_records_the_episodes_own_uuid(self, caplog):
        client = _FakeGraphitiClient(uuid='graph-assigned-uuid')
        queue_service = QueueService(retry_base_delay=0)
        await queue_service.initialize(client)

        with caplog.at_level('INFO', logger='services.queue_service'):
            await queue_service.add_episode(
                group_id=GROUP_ID,
                name='episode',
                content='content',
                source_description='test',
                episode_type=None,
                entity_types=None,
                uuid=None,
            )
            await _drain(queue_service, GROUP_ID)

        messages = [r.message for r in caplog.records]
        assert any('Successfully processed episode graph-assigned-uuid' in m for m in messages)
        assert not any('unknown' in m for m in messages)

    async def test_drop_log_uses_a_correlation_id_instead_of_unknown(self, caplog):
        client = _FakeGraphitiClient(error=OSError('query timed out'))
        queue_service = QueueService(max_attempts=2, retry_base_delay=0)
        await queue_service.initialize(client)

        with caplog.at_level('ERROR', logger='services.queue_service'):
            await queue_service.add_episode(
                group_id=GROUP_ID,
                name='episode',
                content='content',
                source_description='test',
                episode_type=None,
                entity_types=None,
                uuid=None,
            )
            await _drain(queue_service, GROUP_ID)

        drops = [r.message for r in caplog.records if 'Dropping episode' in r.message]
        assert len(drops) == 1
        assert re.search(r'Dropping episode ep-[0-9a-f]{8} for group_id', drops[0])
        assert 'unknown' not in drops[0]


class TestConfiguration:
    """Constructor inputs are clamped to sane values."""

    async def test_max_attempts_below_one_still_runs_the_episode_once(self):
        queue_service = QueueService(max_attempts=0, retry_base_delay=0)
        item = _Item('ep-1', failures=99)

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 1
        assert stats.dropped == 1

    async def test_negative_retry_delay_is_clamped(self):
        queue_service = QueueService(max_attempts=2, retry_base_delay=-5.0)
        item = _Item('ep-1', failures=1)

        stats = await _run(queue_service, GROUP_ID, item)

        assert item.calls == 2
        assert stats.processed == 1

    async def test_unknown_group_reports_empty_counters(self):
        queue_service = QueueService()

        stats = queue_service.get_stats('never-used')
        assert (stats.processed, stats.retried, stats.dropped) == (0, 0, 0)
        assert queue_service.get_queue_size('never-used') == 0
        assert queue_service.is_worker_running('never-used') is False
