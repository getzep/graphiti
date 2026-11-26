#!/usr/bin/env python3
"""Unit tests for retrying transient FalkorDB query failures."""

import sys
from pathlib import Path

import pytest

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.falkordb_retry import (
    DEFAULT_FALKORDB_RETRY_POLICY,
    add_falkordb_transient_retry,
    is_transient_falkordb_error,
)
from services.rate_limit_retry import RateLimitRetryPolicy

redis_errors = pytest.importorskip('redis.exceptions')


class RecordingSleep:
    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class FakeGraph:
    """Stands in for falkordb.asyncio.AsyncGraph."""

    def __init__(self, name: str, errors: list[Exception]):
        self.name = name
        self.errors = errors
        self.calls: list[tuple[str, str]] = []

    async def query(self, q, params=None, timeout=None):
        self.calls.append(('query', q))
        if self.errors:
            raise self.errors.pop(0)
        return f'result:{q}'

    async def ro_query(self, q, params=None, timeout=None):
        self.calls.append(('ro_query', q))
        if self.errors:
            raise self.errors.pop(0)
        return f'ro:{q}'


class FakeFalkorDB:
    """Stands in for falkordb.asyncio.FalkorDB: select_graph returns a new graph each call."""

    def __init__(self, errors: list[Exception] | None = None):
        self.errors = list(errors or [])
        self.graphs: list[FakeGraph] = []

    def select_graph(self, graph_id: str) -> FakeGraph:
        graph = FakeGraph(graph_id, self.errors)
        self.graphs.append(graph)
        return graph


class TestIsTransientFalkorDBError:
    @pytest.mark.parametrize(
        'exc',
        [
            redis_errors.TimeoutError('Timeout reading from socket'),
            redis_errors.ConnectionError('Connection reset by peer'),
            redis_errors.BusyLoadingError('Redis is loading the dataset in memory'),
            redis_errors.TryAgainError('TRYAGAIN'),
        ],
    )
    def test_transient(self, exc):
        assert is_transient_falkordb_error(exc)

    @pytest.mark.parametrize(
        'exc',
        [
            redis_errors.AuthenticationError('invalid password'),
            redis_errors.AuthorizationError('NOPERM'),
            redis_errors.ReadOnlyError("You can't write against a read only replica."),
            redis_errors.ResponseError('Invalid input at end of input'),
            ValueError('boom'),
        ],
    )
    def test_not_transient(self, exc):
        assert not is_transient_falkordb_error(exc)

    def test_transient_cause_is_found(self):
        try:
            try:
                raise redis_errors.TimeoutError('Timeout reading from socket')
            except redis_errors.TimeoutError as inner:
                raise RuntimeError('query failed') from inner
        except RuntimeError as outer:
            assert is_transient_falkordb_error(outer)

    def test_error_raised_while_handling_a_timeout_is_not_transient(self):
        # Implicit __context__ only: the query error, not the earlier timeout, is the failure.
        try:
            try:
                raise redis_errors.TimeoutError('Timeout reading from socket')
            except redis_errors.TimeoutError:
                raise redis_errors.ResponseError('Invalid input')  # noqa: B904
        except redis_errors.ResponseError as outer:
            assert not is_transient_falkordb_error(outer)


class TestAddFalkorDBTransientRetry:
    async def test_timeout_is_retried_then_succeeds(self):
        sleep = RecordingSleep()
        client = FakeFalkorDB([redis_errors.TimeoutError('Timeout reading from socket')])
        add_falkordb_transient_retry(client, DEFAULT_FALKORDB_RETRY_POLICY, sleep)

        graph = client.select_graph('lucy')
        assert await graph.query('MATCH (n) RETURN n', {'a': 1}) == 'result:MATCH (n) RETURN n'
        assert graph.calls == [('query', 'MATCH (n) RETURN n')] * 2
        assert sleep.delays == [1.0]

    async def test_ro_query_is_retried(self):
        sleep = RecordingSleep()
        client = FakeFalkorDB([redis_errors.ConnectionError('Connection closed by server.')])
        add_falkordb_transient_retry(client, DEFAULT_FALKORDB_RETRY_POLICY, sleep)

        assert await client.select_graph('g').ro_query('RETURN 1') == 'ro:RETURN 1'
        assert sleep.delays == [1.0]

    async def test_gives_up_after_policy(self):
        sleep = RecordingSleep()
        errors = [redis_errors.TimeoutError(f'timeout {i}') for i in range(10)]
        client = FakeFalkorDB(errors)
        add_falkordb_transient_retry(client, DEFAULT_FALKORDB_RETRY_POLICY, sleep)

        with pytest.raises(redis_errors.TimeoutError, match='timeout 3'):
            await client.select_graph('g').query('RETURN 1')
        assert sleep.delays == [1.0, 3.0, 9.0]

    async def test_query_errors_are_not_retried(self):
        sleep = RecordingSleep()
        client = FakeFalkorDB([redis_errors.ResponseError('Invalid input')])
        add_falkordb_transient_retry(client, DEFAULT_FALKORDB_RETRY_POLICY, sleep)

        with pytest.raises(redis_errors.ResponseError):
            await client.select_graph('g').query('BROKEN')
        assert sleep.delays == []

    async def test_every_selected_graph_is_wrapped(self):
        # FalkorDriver.clone() shares the client and selects a graph per query.
        sleep = RecordingSleep()
        client = FakeFalkorDB([redis_errors.TimeoutError('t1'), redis_errors.TimeoutError('t2')])
        add_falkordb_transient_retry(client, DEFAULT_FALKORDB_RETRY_POLICY, sleep)

        assert await client.select_graph('a').query('RETURN 1') == 'result:RETURN 1'
        assert await client.select_graph('b').query('RETURN 2') == 'result:RETURN 2'
        assert [g.name for g in client.graphs] == ['a', 'b']
        assert sleep.delays == [1.0, 3.0]

    async def test_install_is_idempotent(self):
        sleep = RecordingSleep()
        client = FakeFalkorDB([redis_errors.TimeoutError('t')] * 10)
        policy = RateLimitRetryPolicy(max_retries=1, base_delay=1.0)
        add_falkordb_transient_retry(client, policy, sleep)
        add_falkordb_transient_retry(client, policy, sleep)

        with pytest.raises(redis_errors.TimeoutError):
            await client.select_graph('g').query('RETURN 1')
        # Wrapped once: one retry, not one retry per installation.
        assert len(client.graphs[0].calls) == 2

    async def test_already_wrapped_methods_are_not_rewrapped(self):
        # If falkordb-py ever caches graphs, select_graph must not nest wrappers.
        sleep = RecordingSleep()
        client = FakeFalkorDB([redis_errors.TimeoutError('t')] * 10)
        cached = client.select_graph('g')
        client.select_graph = lambda graph_id: cached
        add_falkordb_transient_retry(client, RateLimitRetryPolicy(max_retries=1), sleep)
        client.select_graph('g')
        client.select_graph('g')

        with pytest.raises(redis_errors.TimeoutError):
            await cached.query('RETURN 1')
        assert len(cached.calls) == 2

    def test_policy_from_env(self, monkeypatch):
        monkeypatch.setenv('FALKORDB_MAX_RETRIES', '5')
        monkeypatch.setenv('FALKORDB_RETRY_BASE_DELAY', '0.5')
        policy = RateLimitRetryPolicy.from_env('FALKORDB', DEFAULT_FALKORDB_RETRY_POLICY)
        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
        assert policy.multiplier == DEFAULT_FALKORDB_RETRY_POLICY.multiplier
        assert policy.max_delay == DEFAULT_FALKORDB_RETRY_POLICY.max_delay


class FakeResult:
    def __init__(self, rows):
        self.header = [(1, 'uuid')]
        self.result_set = rows


class ResultGraph(FakeGraph):
    async def query(self, q, params=None, timeout=None):
        await super().query(q, params, timeout)
        return FakeResult([['u1']])


class ResultFalkorDB(FakeFalkorDB):
    def select_graph(self, graph_id: str) -> FakeGraph:
        graph = ResultGraph(graph_id, self.errors)
        self.graphs.append(graph)
        return graph


class TestWithRealDriver:
    """FalkorDriver obtains graphs from the shared client; clones and sessions must be covered."""

    async def test_execute_query_and_session_run_on_a_clone(self):
        falkordb_driver = pytest.importorskip('graphiti_core.driver.falkordb_driver')
        FalkorDriver = falkordb_driver.FalkorDriver

        sleep = RecordingSleep()
        client = ResultFalkorDB()
        add_falkordb_transient_retry(client, DEFAULT_FALKORDB_RETRY_POLICY, sleep)
        driver = FalkorDriver(falkor_db=client)
        clone = driver.clone('lucy')
        # Each driver builds indices in a background task at construction; let both finish
        # so the injected errors reach the queries under test.
        for d in (driver, clone):
            task = getattr(d, '_init_task', None)
            if task is not None:
                await task

        client.errors.append(redis_errors.TimeoutError('t1'))
        records, header, _ = await clone.execute_query(
            "MATCH (n {uuid: 'u1'}) RETURN n.uuid AS uuid"
        )
        assert records == [{'uuid': 'u1'}]
        assert header == ['uuid']

        client.errors.append(redis_errors.ConnectionError('Connection reset by peer'))
        async with clone.session() as session:
            await session.run("MERGE (n:Entity {uuid: 'u1'})")

        assert sleep.delays == [1.0, 1.0]
        assert {g.name for g in client.graphs if g.calls} >= {'lucy'}
