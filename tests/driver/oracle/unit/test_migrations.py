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

from pathlib import Path

import pytest

from graphiti_core.driver.oracle.migrations import (
    Migration,
    MigrationDriftError,
    MigrationInProgressError,
    MigrationRunner,
    load_migrations,
    verify_applied_checksum,
)


class FakeMigrationExecutor:
    def __init__(self, applied: list[dict] | None = None) -> None:
        self.applied = applied or []
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, statement: str, **binds: object):
        self.calls.append((statement, binds))
        if 'SELECT version, checksum, state' in statement:
            return self.applied, ['version', 'checksum', 'state'], None
        return [], [], None


def test_migration_checksum_drift_fails_closed() -> None:
    migration = Migration(version='001', name='base', checksum='new', statements=('select 1',))

    with pytest.raises(MigrationDriftError, match='checksum drift for migration 001'):
        verify_applied_checksum(migration, applied_checksum='old')


def test_load_migrations_is_ordered_and_splits_explicit_statements(tmp_path: Path) -> None:
    (tmp_path / '002_second.sql').write_text('select 2\n', encoding='utf-8')
    (tmp_path / '001_first.sql').write_text(
        "select 1\n-- @statement\nselect 'two' from dual\n",
        encoding='utf-8',
    )

    migrations = load_migrations(tmp_path)

    assert [migration.version for migration in migrations] == ['001', '002']
    assert migrations[0].statements == ('select 1', "select 'two' from dual")
    assert len(migrations[0].checksum) == 64


@pytest.mark.asyncio
async def test_runner_marks_each_migration_applying_then_applied(tmp_path: Path) -> None:
    (tmp_path / '001_base.sql').write_text(
        'create table example (id number)\n'
        '-- @statement\n'
        'create index example_idx on example(id)\n',
        encoding='utf-8',
    )
    executor = FakeMigrationExecutor()

    await MigrationRunner(executor, tmp_path).apply()

    call_text = '\n'.join(statement for statement, _ in executor.calls)
    assert "'APPLYING'" in call_text
    assert 'create table example' in call_text
    assert 'create index example_idx' in call_text
    assert "state = 'APPLIED'" in call_text


@pytest.mark.asyncio
async def test_runner_refuses_an_in_progress_migration(tmp_path: Path) -> None:
    migration_file = tmp_path / '001_base.sql'
    migration_file.write_text('create table example (id number)\n', encoding='utf-8')
    migration = load_migrations(tmp_path)[0]
    executor = FakeMigrationExecutor(
        [{'version': '001', 'checksum': migration.checksum, 'state': 'APPLYING'}]
    )

    with pytest.raises(MigrationInProgressError, match='migration 001 is APPLYING'):
        await MigrationRunner(executor, tmp_path).apply()


def test_packaged_migrations_are_complete() -> None:
    root = Path(__file__).parents[4]
    migrations = load_migrations(root / 'graphiti_core/driver/oracle/sql/migrations')
    assert [migration.version for migration in migrations] == ['001', '002', '003', '004']
