# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, cast

from graphiti_core.driver.query_executor import QueryExecutor, Transaction

from ..errors import GroupScopeConflict


async def run_write(
    executor: QueryExecutor,
    tx: Transaction | None,
    statement: str,
    **binds: object,
) -> None:
    if tx is not None:
        await tx.run(statement, **binds)
    else:
        await executor.execute_query(statement, **binds)


async def run_read(
    executor: QueryExecutor,
    statement: str,
    **binds: object,
) -> list[dict[str, Any]]:
    records, _, _ = await executor.execute_query(statement, **binds)
    return records


def resolve_group_scope(executor: QueryExecutor, explicit: str | None = None) -> str | None:
    """Resolve one group without allowing an explicit value to escape a clone scope."""

    value = getattr(executor, '_scope_group_id', None)
    request_scope = str(value) if value is not None else None
    if request_scope is not None and explicit is not None and request_scope != explicit:
        raise GroupScopeConflict(
            f'explicit group {explicit} is outside request scope {request_scope}'
        )
    return request_scope if request_scope is not None else explicit


def resolve_group_ids(
    executor: QueryExecutor,
    explicit: list[str] | None,
) -> list[str] | None:
    """Intersect list filters with a request scope; ``None`` stays global only on base drivers."""

    request_scope = resolve_group_scope(executor)
    if request_scope is None:
        return explicit
    if explicit is None:
        return [request_scope]
    return [request_scope] if request_scope in explicit else []


def require_unscoped_admin(executor: QueryExecutor, *, operation: str) -> None:
    request_scope = resolve_group_scope(executor)
    if request_scope is not None:
        raise GroupScopeConflict(
            f'{operation} is a global administration operation and is unavailable '
            f'inside request scope {request_scope}'
        )


@asynccontextmanager
async def composite_transaction(
    executor: QueryExecutor,
    tx: Transaction | None,
) -> AsyncIterator[Transaction]:
    """Use the caller transaction or own one transaction for a composite mutation."""

    if tx is not None:
        yield tx
        return
    transaction_factory = getattr(executor, 'transaction', None)
    if not callable(transaction_factory):
        raise TypeError('composite Oracle mutation requires an executor transaction()')
    context = cast(AbstractAsyncContextManager[Transaction], transaction_factory())
    async with context as owned:
        yield owned


def validate_request_scope(executor: QueryExecutor, *, kind: str, group_id: str) -> None:
    request_scope = resolve_group_scope(executor)
    if request_scope is not None and request_scope != group_id:
        raise GroupScopeConflict(
            f'{kind} write for group {group_id} is outside request scope {request_scope}'
        )


async def validate_uuid_scope(
    executor: QueryExecutor,
    tx: Transaction | None,
    statement: str,
    *,
    uuid: str,
    kind: str,
    group_id: str,
) -> None:
    """Reject writes that would transfer an existing UUID across kind or group."""

    validate_request_scope(executor, kind=kind, group_id=group_id)
    if tx is None:
        records = await run_read(executor, statement, uuid=uuid)
    else:
        records, _, _ = await tx.run(statement, uuid=uuid)
    if not records:
        return
    actual_kind = str(records[0]['kind'])
    actual_group = str(records[0]['group_id'])
    if actual_kind != kind or actual_group != group_id:
        raise GroupScopeConflict(
            f'{kind} UUID {uuid} already belongs to group {actual_group} as {actual_kind}'
        )
