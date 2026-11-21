"""
Copyright 2026, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import count
from os import PathLike
from typing import Any, Self

import relate as _relate

from ._errors import (
    LeaseOwnershipError,
    ReadOnlyStoreError,
    StoreClosedError,
    TransactionOutcomeIndeterminateError,
    TransactionStateError,
)
from ._results import QueryResult, materialize_result


class StoreState(StrEnum):
    OPENING = 'opening'
    OPEN = 'open'
    CLOSING = 'closing'
    CLOSED = 'closed'
    FAILED = 'failed'


class _OwnerWorker:
    """One native owner thread with observed cancellation semantics."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

    async def run(self, function: Callable[..., Any], *args: Any) -> Any:
        if self._closed:
            raise RuntimeError('owner worker is closed')
        future: Future[Any] = self._executor.submit(function, *args)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            if future.cancel():
                raise
            outcome: BaseException | None = None
            while True:
                try:
                    await asyncio.shield(wrapped)
                except asyncio.CancelledError as repeated:
                    if wrapped.cancelled():
                        outcome = repeated
                        break
                    # Repeated cancellation only delays observation of native work.
                    continue
                except BaseException as error:  # Native outcome is retained as context.
                    outcome = error
                    break
                else:
                    break
            if outcome is not None:
                raise asyncio.CancelledError() from outcome
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, True)


@dataclass(slots=True)
class _Lease:
    token: int
    owner: asyncio.Task[Any] | None
    explicit: bool
    released: bool = False


class _LeaseGate:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tokens = count()
        self._lease: _Lease | None = None

    @property
    def owner(self) -> asyncio.Task[Any] | None:
        return None if self._lease is None or self._lease.released else self._lease.owner

    @property
    def explicit(self) -> bool:
        return self._lease is not None and not self._lease.released and self._lease.explicit

    async def acquire(self, *, explicit: bool = False) -> _Lease:
        current = asyncio.current_task()
        if self.owner is current and self._lease is not None:
            raise LeaseOwnershipError('the current task already holds this store lease')
        acquired = False
        try:
            await self._lock.acquire()
            acquired = True
            lease = _Lease(next(self._tokens), current, explicit)
            self._lease = lease
            return lease
        except BaseException:
            if acquired:
                self._lease = None
                self._lock.release()
            raise

    async def release(self, lease: _Lease) -> None:
        if lease.released:
            return
        if self._lease is not lease or lease.owner is not asyncio.current_task():
            raise LeaseOwnershipError('only the lease-owning task can release it')
        lease.released = True
        self._lease = None
        self._lock.release()


async def _observe_shared(task: asyncio.Task[None]) -> None:
    """Wait for shared cleanup through arbitrary repeated caller cancellation."""

    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
            continue
        except BaseException as error:
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
    if cancelled:
        raise asyncio.CancelledError


class RelateStore:
    def __init__(self) -> None:
        self._database = ''
        self._namespace = 'neo4j'
        self._read_only = False
        self._state = StoreState.OPENING
        self._worker = _OwnerWorker()
        self._gate = _LeaseGate()
        self._connection: Any = None
        self._close_task: asyncio.Task[None] | None = None
        self._native_close_calls = 0

    @classmethod
    async def open(
        cls,
        database: str | PathLike[str],
        *,
        namespace: str = 'neo4j',
        read_only: bool = False,
    ) -> Self:
        store = cls()
        store._database = os.fsdecode(os.fspath(database))
        store._namespace = namespace
        store._read_only = read_only
        try:
            await store._worker.run(store._open_native)
        except BaseException:
            if store._connection is not None:
                try:
                    await store._worker.run(store._close_native)
                except BaseException:
                    # Preserve the open/cancellation outcome. No handle escapes, and
                    # executor teardown still runs even if native cleanup also fails.
                    pass
            await store._worker.aclose()
            store._state = StoreState.FAILED
            raise
        store._state = StoreState.OPEN
        return store

    def _open_native(self) -> None:
        connection = _relate.connect(
            self._database,
            namespace=self._namespace,
            read_only=self._read_only,
        )
        self._connection = connection
        if not self._read_only:
            connection.autocommit = False

    @property
    def database(self) -> str:
        return self._database

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def state(self) -> StoreState:
        return self._state

    def _check_open(self) -> None:
        if self._state is not StoreState.OPEN:
            raise StoreClosedError('store does not accept new operations')

    async def execute(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        self._check_open()
        if self._gate.owner is asyncio.current_task() and self._gate.explicit:
            raise LeaseOwnershipError(
                'ordinary store execution is forbidden inside an explicit lease'
            )
        lease = await self._gate.acquire()
        try:
            return await self._worker.run(
                self._execute_native, cypher, parameters, not self._read_only
            )
        finally:
            await self._gate.release(lease)

    def _execute_native(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None,
        commit: bool,
    ) -> QueryResult:
        try:
            statement = self._execute_statement_native(cypher, parameters)
        except Exception:
            if not commit:
                raise
            try:
                self._rollback_native()
            except Exception as rollback_error:
                self._state = StoreState.FAILED
                raise TransactionStateError(
                    'ordinary statement rollback failed; the store must be closed'
                ) from rollback_error
            raise

        if not commit:
            return statement
        try:
            bookmark = self._commit_native()
        except Exception as commit_error:
            self._state = StoreState.FAILED
            raise TransactionOutcomeIndeterminateError(
                'ordinary write commit outcome is indeterminate; do not retry automatically'
            ) from commit_error
        return replace(statement, bookmark=bookmark)

    def _execute_statement_native(
        self, cypher: str, parameters: Mapping[str, Any] | None
    ) -> QueryResult:
        cursor = self._connection.cursor()
        cursor.execute(cypher, None if parameters is None else dict(parameters))
        return materialize_result(cursor)

    def _commit_native(self) -> str | None:
        return self._connection.commit()

    def _rollback_native(self) -> None:
        self._connection.rollback()

    def transaction(self) -> AbstractAsyncContextManager[RelateTransaction]:
        if self._read_only:
            raise ReadOnlyStoreError('read-only snapshots cannot open transactions')
        self._check_open()
        return RelateTransaction(self)

    async def close(self) -> None:
        if self._state is StoreState.CLOSED:
            return
        if self._state is StoreState.FAILED and self._close_task is not None:
            if self._close_task.done():
                return
        if self._close_task is None:
            if self._gate.owner is asyncio.current_task():
                raise LeaseOwnershipError('close cannot wait for its own active lease')
            self._state = StoreState.CLOSING
            self._close_task = asyncio.create_task(self._close_observed())
        await _observe_shared(self._close_task)

    def _close_native(self) -> None:
        self._native_close_calls += 1
        if self._connection is not None:
            self._connection.close()

    async def _close_observed(self) -> None:
        lease = await self._gate.acquire()
        try:
            await self._worker.run(self._close_native)
        except BaseException:
            self._state = StoreState.FAILED
            raise
        else:
            self._state = StoreState.CLOSED
        finally:
            await self._gate.release(lease)
            await self._worker.aclose()

    async def __aenter__(self) -> Self:
        self._check_open()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


from ._transaction import RelateTransaction  # noqa: E402
