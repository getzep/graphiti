"""Tests for Neptune driver async behavior."""

from __future__ import annotations

import asyncio
import time

import pytest

from graphiti_core.driver.neptune_driver import NeptuneDriver


class TestNeptuneDriverAsyncBoundary:
    @pytest.mark.asyncio
    async def test_execute_query_does_not_block_event_loop(self):
        driver = object.__new__(NeptuneDriver)

        def slow_run_query(query, params):
            time.sleep(0.05)
            return [{"ok": True}], None, None

        driver._run_query = slow_run_query

        tick = asyncio.Event()

        async def mark_event_loop_progress():
            await asyncio.sleep(0.01)
            tick.set()

        query_task = asyncio.create_task(driver.execute_query("RETURN 1"))
        marker_task = asyncio.create_task(mark_event_loop_progress())

        await asyncio.wait_for(tick.wait(), timeout=0.03)
        assert not query_task.done()

        result, _, _ = await query_task
        await marker_task
        assert result == [{"ok": True}]

    @pytest.mark.asyncio
    async def test_execute_query_list_preserves_last_result(self):
        driver = object.__new__(NeptuneDriver)
        calls = []

        def run_query(query, params):
            calls.append((query, params))
            return [{"query": query}], None, None

        driver._run_query = run_query

        result, _, _ = await driver.execute_query([
            ("RETURN 1", {"first": True}),
            ("RETURN 2", {"second": True}),
        ])

        assert calls == [
            ("RETURN 1", {"first": True}),
            ("RETURN 2", {"second": True}),
        ]
        assert result == [{"query": "RETURN 2"}]
