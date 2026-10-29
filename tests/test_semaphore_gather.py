import asyncio
import inspect
from unittest.mock import Mock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client.client import LLMClient


@pytest.mark.parametrize('error_type', [RuntimeError, asyncio.CancelledError])
async def test_search_drains_sibling_queries_on_failure(monkeypatch, error_type):
    sibling_started = asyncio.Event()
    sibling_closed = asyncio.Event()
    sibling_tasks = []
    error = error_type('query failed')

    async def failing_query(*args, **kwargs):
        await sibling_started.wait()
        raise error

    async def pending_query(*args, **kwargs):
        sibling_tasks.append(asyncio.current_task())
        sibling_started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            sibling_closed.set()

    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', failing_query)
    monkeypatch.setattr('graphiti_core.search.search.edge_similarity_search', pending_query)
    driver = Mock(spec=GraphDriver, provider=GraphProvider.NEO4J)
    embedder = Mock(spec=EmbedderClient)
    embedder.create.return_value = [0.1, 0.2]
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=embedder,
        cross_encoder=Mock(spec=CrossEncoderClient),
    )

    try:
        with pytest.raises(error_type) as exc_info:
            await graphiti.search('where does Alice work?')

        assert exc_info.value is error
        assert sibling_closed.is_set()
        assert all(task.done() for task in sibling_tasks)
    finally:
        for task in sibling_tasks:
            task.cancel()
        await asyncio.gather(*sibling_tasks, return_exceptions=True)


async def test_cancel_closes_coroutines_waiting_for_a_slot():
    started = asyncio.Event()
    closed = asyncio.Event()

    async def active():
        started.set()
        try:
            await asyncio.Future()
        finally:
            closed.set()

    async def queued():
        pytest.fail('The queued coroutine must not run')

    active_coro, queued_coro = active(), queued()
    task = asyncio.create_task(semaphore_gather(active_coro, queued_coro, max_coroutines=1))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert closed.is_set()
        assert inspect.getcoroutinestate(queued_coro) == inspect.CORO_CLOSED
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        active_coro.close()
        queued_coro.close()


async def test_results_preserve_order_and_concurrency_limit():
    active = 0
    peak = 0

    async def worker(index):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return index
        finally:
            active -= 1

    assert await semaphore_gather(*(worker(i) for i in range(5)), max_coroutines=2) == list(
        range(5)
    )
    assert peak == 2
    assert active == 0
