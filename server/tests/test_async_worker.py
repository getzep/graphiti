import asyncio

import pytest

from graph_service.routers.ingest import AsyncWorker


@pytest.mark.asyncio
async def test_async_worker_continues_after_failure():
    worker = AsyncWorker()
    await worker.start()

    ran_second_job = asyncio.Event()

    async def failing_job():
        raise RuntimeError('boom')

    async def ok_job():
        ran_second_job.set()

    await worker.queue.put(failing_job)
    await worker.queue.put(ok_job)

    await asyncio.wait_for(ran_second_job.wait(), timeout=2)

    await worker.stop()
