#!/usr/bin/env python3
"""Unit tests for the QueueService episode worker.

These drive the worker directly with fake process functions, so they need no
database, LLM, or network. They cover the failure handling behind an episode
being stored with no edges: transient failures are retried in place with
backoff, permanent (schema/validation) failures are not, and every outcome is
counted so a dropped episode is visible instead of silent.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

# Add the src directory to the path (mirrors the other service tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import services.queue_service as queue_service_module  # noqa: E402
from services.queue_service import (  # noqa: E402
    QueueService,
    QueueStats,
    is_retryable_failure,
    retry_delay,
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
            asyncio.TimeoutError(),
            RuntimeError('LLM returned an empty response'),
            Exception('generic failure'),
        ],
    )
    def test_transient_failures_are_retryable(self, error):
        assert is_retryable_failure(error) is True

    def test_validation_failures_are_not_retryable(self):
        assert is_retryable_failure(_validation_error()) is False

    @pytest.mark.parametrize('attempt, expected', [(1, 1.0), (2, 2.0), (3, 4.0)])
    def test_retry_delay_doubles(self, attempt, expected):
        assert retry_delay(1.0, attempt) == expected


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

    async def test_worker_awaits_the_scheduled_backoff(self, monkeypatch):
        queue_service = QueueService(max_attempts=3, retry_base_delay=1.0)
        item = _Item('ep-1', failures=2, error_factory=lambda: asyncio.TimeoutError())
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            # Record the scheduled delay instead of waiting for it.
            slept.append(delay)

        monkeypatch.setattr(queue_service_module.asyncio, 'sleep', fake_sleep)

        await _run(queue_service, GROUP_ID, item)

        assert slept == [1.0, 2.0]
        assert item.calls == 3

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
