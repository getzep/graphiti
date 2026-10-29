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

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class QueryExecutorProtocol(Protocol):
    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any: ...


class MigrationError(RuntimeError):
    """Base migration error."""


class MigrationDriftError(MigrationError):
    """Raised when applied SQL differs from the pinned migration."""


class MigrationInProgressError(MigrationError):
    """Raised when a previous migration stopped after entering APPLYING."""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    checksum: str
    statements: tuple[str, ...]


_MIGRATION_LOCK = asyncio.Lock()

CREATE_LEDGER = """
BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE graphiti_schema_migrations (
      version VARCHAR2(32 CHAR) PRIMARY KEY,
      name VARCHAR2(200 CHAR) NOT NULL,
      checksum VARCHAR2(64 CHAR) NOT NULL,
      state VARCHAR2(16 CHAR) NOT NULL,
      applied_at TIMESTAMP WITH TIME ZONE,
      CONSTRAINT graphiti_migration_state_ck CHECK (state IN (''APPLYING'', ''APPLIED''))
    )';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN
      RAISE;
    END IF;
END;
""".strip()

READ_LEDGER = """
SELECT version, checksum, state
FROM graphiti_schema_migrations
ORDER BY version
""".strip()

MARK_APPLYING = """
INSERT INTO graphiti_schema_migrations (version, name, checksum, state, applied_at)
VALUES (:version, :name, :checksum, 'APPLYING', NULL)
""".strip()

MARK_APPLIED = """
UPDATE graphiti_schema_migrations
SET state = 'APPLIED', applied_at = SYSTIMESTAMP
WHERE version = :version AND state = 'APPLYING'
""".strip()


def _split_statements(sql: str) -> tuple[str, ...]:
    return tuple(statement.strip() for statement in sql.split('-- @statement') if statement.strip())


def load_migrations(directory: Path) -> list[Migration]:
    migrations: list[Migration] = []
    for path in sorted(directory.glob('[0-9][0-9][0-9]_*.sql')):
        sql = path.read_text(encoding='utf-8')
        version, name = path.stem.split('_', maxsplit=1)
        migrations.append(
            Migration(
                version=version,
                name=name,
                checksum=hashlib.sha256(sql.encode('utf-8')).hexdigest(),
                statements=_split_statements(sql),
            )
        )
    return migrations


def verify_applied_checksum(migration: Migration, applied_checksum: str) -> None:
    if migration.checksum != applied_checksum:
        raise MigrationDriftError(f'checksum drift for migration {migration.version}')


class MigrationRunner:
    def __init__(self, executor: QueryExecutorProtocol, directory: Path) -> None:
        self._executor = executor
        self._migrations = load_migrations(directory)

    async def apply(self) -> None:
        async with _MIGRATION_LOCK:
            await self._executor.execute_query(CREATE_LEDGER)
            records, _, _ = await self._executor.execute_query(READ_LEDGER)
            applied = {str(record['version']): record for record in records}

            for migration in self._migrations:
                record = applied.get(migration.version)
                if record is not None:
                    verify_applied_checksum(migration, str(record['checksum']))
                    if record['state'] == 'APPLYING':
                        raise MigrationInProgressError(
                            f'migration {migration.version} is APPLYING; inspect before retry'
                        )
                    continue

                await self._executor.execute_query(
                    MARK_APPLYING,
                    version=migration.version,
                    name=migration.name,
                    checksum=migration.checksum,
                )
                for statement in migration.statements:
                    await self._executor.execute_query(statement)
                await self._executor.execute_query(MARK_APPLIED, version=migration.version)
