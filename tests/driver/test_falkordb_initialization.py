"""FalkorDB index initialization is shared across request-scoped clones."""

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.falkordb_driver import FalkorDriver

pytestmark = pytest.mark.asyncio


def client():
    db = MagicMock()
    db.aclose = AsyncMock()
    result = MagicMock(header=[], result_set=[])
    db.select_graph.return_value.query = AsyncMock(return_value=result)
    return db


async def test_repeated_and_concurrent_clones_share_one_index_build_per_graph():
    db = client()
    started = Counter()
    gate = asyncio.Event()

    async def build(driver, delete_existing=False):
        started[driver._database] += 1
        if driver._database in {'alpha', 'beta'}:
            await gate.wait()

    with patch.object(FalkorDriver, 'build_indices_and_constraints', build):
        owner = FalkorDriver(falkor_db=db)
        first = owner.clone('alpha')
        second = owner.clone('alpha')
        nested = first.clone('alpha')
        other = first.clone('beta')
        await asyncio.sleep(0)
        assert started == Counter({'default_db': 1, 'alpha': 1, 'beta': 1})
        queries = [
            asyncio.create_task(driver.execute_query('RETURN 1'))
            for driver in (first, second, nested, other)
        ]
        await asyncio.sleep(0)
        assert db.select_graph.call_count == 0
        gate.set()
        await asyncio.gather(*queries)
        assert db.select_graph.call_count == 4
        await first.close()
        db.aclose.assert_not_awaited()
        await owner.close()
        db.aclose.assert_awaited_once()


async def test_failed_index_build_is_reported_to_every_scoped_query():
    db = client()
    attempts = Counter()

    async def build(driver, delete_existing=False):
        attempts[driver._database] += 1
        if driver._database == 'broken':
            raise RuntimeError('index failed')

    with patch.object(FalkorDriver, 'build_indices_and_constraints', build):
        owner = FalkorDriver(falkor_db=db)
        first = owner.clone('broken')
        second = owner.clone('broken')
        await asyncio.sleep(0)
        for driver in (first, second):
            with pytest.raises(RuntimeError, match='index failed'):
                await driver.execute_query('RETURN 1')
        assert attempts['broken'] == 1
        db.select_graph.assert_not_called()
        await owner.close()


async def test_scoped_session_waits_for_index_build():
    db = client()
    gate = asyncio.Event()

    async def build(driver, delete_existing=False):
        if driver._database == 'tenant':
            await gate.wait()

    with patch.object(FalkorDriver, 'build_indices_and_constraints', build):
        owner = FalkorDriver(falkor_db=db)
        clone = owner.clone('tenant')
        session = clone.session()
        read = asyncio.create_task(session.run('RETURN 1'))
        await asyncio.sleep(0)
        db.select_graph.return_value.query.assert_not_awaited()
        gate.set()
        await read
        db.select_graph.return_value.query.assert_awaited_once()
        await owner.close()


async def test_cancelled_waiter_does_not_cancel_shared_index_build():
    db = client()
    gate = asyncio.Event()
    builds = Counter()

    async def build(driver, delete_existing=False):
        builds[driver._database] += 1
        if driver._database == 'tenant':
            await gate.wait()

    with patch.object(FalkorDriver, 'build_indices_and_constraints', build):
        owner = FalkorDriver(falkor_db=db)
        clone = owner.clone('tenant')
        waiter = asyncio.create_task(clone.execute_query('RETURN 1'))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not owner._index_tasks['tenant'].done()
        gate.set()
        await clone.execute_query('RETURN 1')
        assert builds['tenant'] == 1
        await owner.close()
