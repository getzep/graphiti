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

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .migrations import MigrationRunner
from .sql.catalog import (
    MIGRATION_READINESS,
    PROPERTY_GRAPH_READINESS,
    TABLE_READINESS,
    TEXT_INDEX_READINESS,
    VECTOR_INDEX_READINESS,
)

EXPECTED_MIGRATIONS = 4


class QueryExecutorProtocol(Protocol):
    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any: ...


class SchemaReadiness(BaseModel):
    migrations_applied: int
    migration_in_progress: bool
    tables_ready: bool
    property_graph_ready: bool
    text_indexes_ready: bool
    vector_indexes_ready: bool
    ready: bool


async def _count(executor: QueryExecutorProtocol, statement: str) -> int:
    records, _, _ = await executor.execute_query(statement)
    return int(records[0]['object_count'])


class SchemaInspector:
    def __init__(self, executor: QueryExecutorProtocol) -> None:
        self._executor = executor

    async def readiness(self) -> SchemaReadiness:
        migration_records, _, _ = await self._executor.execute_query(MIGRATION_READINESS)
        migration_record = migration_records[0]
        migrations_applied = int(migration_record['applied_count'] or 0)
        migration_in_progress = int(migration_record['applying_count'] or 0) > 0
        tables_ready = await _count(self._executor, TABLE_READINESS) == 2
        property_graph_ready = await _count(self._executor, PROPERTY_GRAPH_READINESS) == 1
        text_indexes_ready = await _count(self._executor, TEXT_INDEX_READINESS) == 2
        vector_indexes_ready = await _count(self._executor, VECTOR_INDEX_READINESS) == 2
        ready = all(
            (
                migrations_applied == EXPECTED_MIGRATIONS,
                not migration_in_progress,
                tables_ready,
                property_graph_ready,
                text_indexes_ready,
                vector_indexes_ready,
            )
        )
        return SchemaReadiness(
            migrations_applied=migrations_applied,
            migration_in_progress=migration_in_progress,
            tables_ready=tables_ready,
            property_graph_ready=property_graph_ready,
            text_indexes_ready=text_indexes_ready,
            vector_indexes_ready=vector_indexes_ready,
            ready=ready,
        )


async def apply_schema(executor: QueryExecutorProtocol, directory: Path) -> SchemaReadiness:
    await MigrationRunner(executor, directory).apply()
    return await SchemaInspector(executor).readiness()
