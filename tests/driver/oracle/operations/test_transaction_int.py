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

from uuid import uuid4

import pytest

from graphiti_core.driver.oracle.session import Oracle26aiSession

pytestmark = [pytest.mark.integration, pytest.mark.oracle]


@pytest.mark.oracle
async def test_live_transaction_rollback_removes_uncommitted_row(oracle_connection) -> None:
    assert oracle_connection is not None, 'Oracle capability configuration missing'
    fixture_uuid = str(uuid4())
    async with oracle_connection.cursor() as cursor:
        await cursor.execute(
            'INSERT INTO graphiti_nodes (uuid, kind, group_id, name, labels, properties, created_at) '
            "VALUES (:uuid, 'ENTITY', :group_id, 'rollback fixture', '[]', '{}', SYSTIMESTAMP)",
            uuid=fixture_uuid,
            group_id='oracle-rollback-' + fixture_uuid,
        )
        await oracle_connection.rollback()
        await cursor.execute(
            'SELECT COUNT(*) FROM graphiti_nodes WHERE uuid = :uuid', uuid=fixture_uuid
        )
        assert (await cursor.fetchone())[0] == 0


@pytest.mark.oracle
async def test_oracle_graphiti_bulk_session_rolls_back_earlier_writes(oracle_connection) -> None:
    assert oracle_connection is not None, 'Oracle capability configuration missing'
    fixture_uuid = str(uuid4())

    class OneConnectionPool:
        async def acquire(self):
            return oracle_connection

        async def release(self, connection):
            assert connection is oracle_connection

    async def interrupted_bulk(tx):
        await tx.run(
            'INSERT INTO graphiti_nodes (uuid, kind, group_id, name, labels, properties, created_at) '
            "VALUES (:uuid, 'ENTITY', :group_id, 'partial episode', '[]', '{}', SYSTIMESTAMP)",
            uuid=fixture_uuid,
            group_id='oracle-bulk-rollback-' + fixture_uuid,
        )
        raise RuntimeError('subsequent graph write failed')

    with pytest.raises(RuntimeError, match='subsequent graph write failed'):
        await Oracle26aiSession(OneConnectionPool()).execute_write(interrupted_bulk)

    async with oracle_connection.cursor() as cursor:
        await cursor.execute(
            'SELECT COUNT(*) FROM graphiti_nodes WHERE uuid = :uuid', uuid=fixture_uuid
        )
        assert (await cursor.fetchone())[0] == 0
