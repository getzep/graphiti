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

import logging
import re
from decimal import Decimal
from time import perf_counter
from typing import Any

import oracledb

from graphiti_core.driver.driver import GraphDriverSession, GraphProvider

from .errors import OracleQueryContractError, oracle_error_code
from .logging import operation_fields
from .protocols import AsyncConnectionProtocol, AsyncPoolProtocol

logger = logging.getLogger(__name__)

_ALLOWED_PREFIX = re.compile(
    r'^\s*(SELECT|INSERT|UPDATE|DELETE|MERGE|BEGIN|DECLARE|CREATE|DROP|ALTER)\b',
    re.IGNORECASE,
)
_FORBIDDEN_PREFIX = re.compile(
    r'^\s*(MATCH|OPTIONAL\s+MATCH|CALL\s+DB\.|UNWIND)\b',
    re.IGNORECASE,
)
_DOLLAR_BIND = re.compile(r'\$[A-Za-z_][A-Za-z0-9_]*')
_CLOB_BIND_NAMES = frozenset(
    {
        'summary',
        'content',
        'fact',
        'search_text',
        'labels',
        'properties',
        'entity_edge_uuids',
        'episode_uuids',
    }
)
# Persisted payload binds occur in both MERGE branches. Keep direct strings at
# Oracle's portable SQL VARCHAR2 byte limit; larger values use CLOB locators.
_MAX_DIRECT_PAYLOAD_BYTES = 4000


def validate_oracle_statement(statement: str) -> None:
    """Reject graph-language fallthrough while allowing SQL/PGQ MATCH clauses."""

    if _FORBIDDEN_PREFIX.search(statement):
        raise OracleQueryContractError('top-level Cypher is not accepted by Oracle26aiDriver')
    if _DOLLAR_BIND.search(statement):
        raise OracleQueryContractError('Oracle SQL binds must use :name, not $name')
    if not _ALLOWED_PREFIX.search(statement):
        raise OracleQueryContractError('statement must start with an allowlisted Oracle SQL verb')


def _column_name(column: Any) -> str:
    name = getattr(column, 'name', None)
    if name is None:
        name = column[0]
    return str(name).lower()


async def _materialize_oracle_value(value: Any) -> Any:
    if isinstance(value, oracledb.AsyncLOB):
        return await value.read()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {key: await _materialize_oracle_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [await _materialize_oracle_value(item) for item in value]
    return value


async def _prepare_oracle_binds(
    connection: AsyncConnectionProtocol,
    binds: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    prepared_binds = dict(binds)
    input_sizes: dict[str, object] = {}
    for name, value in binds.items():
        if name not in _CLOB_BIND_NAMES or not isinstance(value, str):
            continue
        if len(value.encode('utf-8')) > _MAX_DIRECT_PAYLOAD_BYTES:
            prepared_binds[name] = await connection.createlob(oracledb.DB_TYPE_CLOB, value)
        else:
            input_sizes[name] = oracledb.DB_TYPE_CLOB
    return prepared_binds, input_sizes


async def execute_on_connection(
    connection: AsyncConnectionProtocol,
    statement: str,
    binds: dict[str, object],
    *,
    commit_write: bool,
) -> tuple[list[dict[str, Any]], list[str], None]:
    validate_oracle_statement(statement)
    prepared_binds, input_sizes = await _prepare_oracle_binds(connection, binds)
    async with connection.cursor() as cursor:
        if input_sizes:
            cursor.setinputsizes(**input_sizes)
        await cursor.execute(statement, prepared_binds)
        if cursor.description is None:
            if commit_write:
                await connection.commit()
            return [], [], None

        columns = [_column_name(column) for column in cursor.description]
        rows = await cursor.fetchall()
        records = [
            dict(
                zip(
                    columns,
                    [await _materialize_oracle_value(value) for value in row],
                    strict=True,
                )
            )
            for row in rows
        ]
        return records, columns, None


class Oracle26aiSession(GraphDriverSession):
    provider = GraphProvider.ORACLE

    def __init__(self, pool: AsyncPoolProtocol) -> None:
        self._pool = pool

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def run(self, query: str, **kwargs: Any) -> tuple[list[dict[str, Any]], list[str], None]:
        started = perf_counter()
        result_count = 0
        error_code = None
        connection = await self._pool.acquire()
        try:
            result = await execute_on_connection(
                connection,
                query,
                dict(kwargs),
                commit_write=True,
            )
            result_count = len(result[0])
            return result
        except BaseException as error:
            error_code = oracle_error_code(error)
            await connection.rollback()
            raise
        finally:
            await self._pool.release(connection)
            logger.info(
                'oracle_graphiti_operation',
                extra=operation_fields(
                    operation='session.run',
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    result_count=result_count,
                    retry_count=0,
                    oracle_code=error_code,
                ),
            )

    async def close(self) -> None:
        return None

    async def execute_write(self, func, *args, **kwargs):
        # Graphiti bulk writers expect one transaction across episode, node and edge saves.
        from .transaction import Oracle26aiTransaction

        connection = await self._pool.acquire()
        try:
            result = await func(Oracle26aiTransaction(connection), *args, **kwargs)
            await connection.commit()
            return result
        except BaseException:
            await connection.rollback()
            raise
        finally:
            await self._pool.release(connection)
