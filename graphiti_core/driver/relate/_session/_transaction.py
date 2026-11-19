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
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

from ._errors import (
    LeaseOwnershipError,
    StoreClosedError,
    TransactionOutcomeIndeterminateError,
    TransactionStateError,
)
from ._results import QueryResult
from ._store import StoreState

if TYPE_CHECKING:
    from ._store import RelateStore, _Lease


class TransactionState(StrEnum):
    ACTIVE = 'active'
    POISONED = 'poisoned'
    COMMITTED = 'committed'
    ROLLED_BACK = 'rolled_back'
    INDETERMINATE = 'indeterminate'
    CLOSED = 'closed'


class RelateTransaction:
    def __init__(self, store: RelateStore) -> None:
        self._store = store
        self._state = TransactionState.CLOSED
        self._owner: asyncio.Task[Any] | None = None
        self._lease: _Lease | None = None
        self._entered = False
        self._failure: BaseException | None = None

    @property
    def state(self) -> TransactionState:
        return self._state

    async def __aenter__(self) -> Self:
        if self._entered:
            raise self._terminal_error()
        if self._store.read_only or self._store.state is not StoreState.OPEN:
            # transaction() normally guards this; this second gate closes the race
            # between constructing the context and actually acquiring its lease.
            self._store._check_open()
        owner = asyncio.current_task()
        lease = await self._store._gate.acquire(explicit=True)
        if self._store.state is not StoreState.OPEN:
            await self._store._gate.release(lease)
            raise StoreClosedError('store closing began before transaction acquisition')
        self._owner = owner
        self._lease = lease
        self._entered = True
        self._state = TransactionState.ACTIVE
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._check_owner()
        if self._state in {
            TransactionState.COMMITTED,
            TransactionState.ROLLED_BACK,
            TransactionState.CLOSED,
        }:
            return
        if self._state is not TransactionState.ACTIVE or exc_type is not None:
            self._failure = self._failure or exc
            await self.rollback()
            return
        await self.commit()

    def _check_owner(self) -> None:
        if self._owner is not asyncio.current_task():
            raise LeaseOwnershipError('transaction methods require the lease-owning task')

    def _check_store_connection(self) -> None:
        if self._store.state in {StoreState.CLOSED, StoreState.FAILED}:
            raise StoreClosedError('transaction store connection is unavailable')

    async def execute(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        self._check_owner()
        self._check_store_connection()
        if self._state is not TransactionState.ACTIVE:
            raise self._terminal_error()
        try:
            return await self._store._worker.run(
                self._store._execute_statement_native, cypher, parameters
            )
        except asyncio.CancelledError as cancelled:
            if cancelled.__cause__ is not None:
                self._state = TransactionState.POISONED
                self._failure = cancelled.__cause__
            raise
        except Exception as error:
            self._state = TransactionState.POISONED
            self._failure = error
            raise

    async def commit(self) -> str | None:
        self._check_owner()
        self._check_store_connection()
        if self._state is not TransactionState.ACTIVE:
            raise self._terminal_error()
        try:
            bookmark = await self._store._worker.run(self._store._commit_native)
        except asyncio.CancelledError as cancelled:
            if cancelled.__cause__ is not None:
                self._commit_indeterminate(cancelled.__cause__)
            self._state = TransactionState.COMMITTED
            await self._release()
            raise
        except Exception as error:
            self._commit_indeterminate(error)
        self._state = TransactionState.COMMITTED
        await self._release()
        return bookmark

    def _commit_indeterminate(self, error: BaseException) -> None:
        self._state = TransactionState.INDETERMINATE
        self._failure = error
        indeterminate = TransactionOutcomeIndeterminateError(
            'transaction commit outcome is indeterminate; do not retry automatically'
        )
        raise indeterminate from error

    async def rollback(self) -> None:
        self._check_owner()
        if self._state not in {
            TransactionState.ACTIVE,
            TransactionState.POISONED,
            TransactionState.INDETERMINATE,
        }:
            raise self._terminal_error()
        cancelled: asyncio.CancelledError | None = None
        rollback_error: BaseException | None = None
        try:
            await self._store._worker.run(self._store._rollback_native)
        except asyncio.CancelledError as error:
            if error.__cause__ is None:
                cancelled = error
            else:
                rollback_error = error.__cause__
        except Exception as error:
            rollback_error = error

        if rollback_error is not None:
            self._state = TransactionState.CLOSED
            self._store._state = StoreState.FAILED
            state_error = TransactionStateError(
                'transaction rollback failed; only store close is valid'
            )
            if self._failure is not None:
                # Raising outside the native handler preserves the earlier failure
                # as context, while the rollback failure remains the direct cause.
                state_error.__context__ = self._failure
            await self._release()
            raise state_error from rollback_error

        self._state = TransactionState.ROLLED_BACK
        self._failure = None
        await self._release()
        if cancelled is not None:
            raise cancelled

    async def _release(self) -> None:
        if self._lease is not None and not self._lease.released:
            await self._store._gate.release(self._lease)

    def _terminal_error(self) -> TransactionStateError:
        return TransactionStateError(f'transaction is not active (state={self._state.value})')
