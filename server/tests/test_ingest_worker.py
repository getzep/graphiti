"""Unit tests for the ingestion ``AsyncWorker`` failure-resilience behavior.

A job that raises a non-cancellation exception must not kill the worker:
transient errors are retried with exponential backoff, permanent (bad-data)
errors are skipped immediately, and in every case the worker stays alive to
process the next job. The backoff delay is patched to zero so the retry tests
run instantly while still exercising the real ``asyncio.sleep`` yield.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from graph_service.routers.ingest import (
    MAX_RETRIES,
    AsyncWorker,
    _is_permanent_error,
)


@pytest.fixture(autouse=True)
def _instant_backoff():
    # Collapse the 5/10/20/40s backoff to 0s so retry tests don't actually sleep.
    with patch('graph_service.routers.ingest.BASE_BACKOFF_SECONDS', 0):
        yield


class TestIsPermanentError:
    def test_builtin_data_errors_are_permanent(self):
        assert _is_permanent_error(ValueError('bad'))
        assert _is_permanent_error(KeyError('missing'))
        assert _is_permanent_error(IndexError('range'))

    def test_provider_client_error_status_codes_are_permanent(self):
        class BadRequestError(Exception):
            status_code = 400

        assert _is_permanent_error(BadRequestError())

    def test_matched_by_class_name(self):
        class NotFoundError(Exception):  # no status_code attribute
            pass

        assert _is_permanent_error(NotFoundError())

    def test_rate_limit_and_network_errors_are_transient(self):
        # 429 (rate limit) and 408 (timeout) are retryable — must NOT be permanent.
        class RateLimitError(Exception):
            status_code = 429

        assert not _is_permanent_error(RateLimitError())
        assert not _is_permanent_error(ConnectionError('reset'))
        assert not _is_permanent_error(TimeoutError())


@pytest.mark.asyncio
async def test_transient_failure_is_retried_then_succeeds():
    worker = AsyncWorker()
    job = AsyncMock(side_effect=[ConnectionError('transient'), None])

    await worker._process_job(job)

    assert job.await_count == 2  # failed once, retried, then succeeded


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried():
    worker = AsyncWorker()
    job = AsyncMock(side_effect=ValueError('bad data'))

    await worker._process_job(job)

    assert job.await_count == 1  # permanent → skipped immediately, no retry


@pytest.mark.asyncio
async def test_persistent_transient_failure_gives_up_after_max_retries():
    worker = AsyncWorker()
    job = AsyncMock(side_effect=ConnectionError('always down'))

    await worker._process_job(job)

    assert job.await_count == MAX_RETRIES  # tried MAX_RETRIES times, then skipped


@pytest.mark.asyncio
async def test_cancellation_propagates():
    worker = AsyncWorker()
    job = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await worker._process_job(job)


@pytest.mark.asyncio
async def test_worker_survives_failing_job_and_processes_next():
    """End-to-end: a failing job must not stop the worker loop."""
    worker = AsyncWorker()
    bad_job = AsyncMock(side_effect=ValueError('bad'))
    good_job = AsyncMock()

    task = asyncio.create_task(worker.worker())
    await worker.queue.put(bad_job)
    await worker.queue.put(good_job)

    # Let the worker drain both jobs (real asyncio.sleep(0) yields the loop).
    for _ in range(100):
        if good_job.await_count and worker.queue.empty():
            break
        await asyncio.sleep(0)

    # worker() catches CancelledError internally and exits cleanly, so just
    # cancel and drain the task without expecting the exception to propagate.
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    bad_job.assert_awaited_once()
    good_job.assert_awaited_once()
