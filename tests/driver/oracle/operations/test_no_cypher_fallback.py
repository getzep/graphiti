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

from datetime import UTC, datetime

import pytest

from graphiti_core.driver.oracle.config import Oracle26aiConfig
from graphiti_core.driver.oracle.driver import Oracle26aiDriver
from graphiti_core.driver.oracle.session import validate_oracle_statement
from graphiti_core.nodes import EntityNode
from tests.driver.oracle.unit.fakes import FakeConnection, FakeCursor, FakePool


@pytest.mark.asyncio
async def test_legacy_entity_save_reaches_oracle_sql_not_cypher() -> None:
    cursor = FakeCursor()
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=FakePool(FakeConnection(cursor)),
    )
    node = EntityNode(
        uuid='entity-a',
        name='Alpha',
        group_id='group-a',
        labels=['Concept'],
        created_at=datetime(2026, 9, 21, tzinfo=UTC),
    )

    await node.save(driver)

    assert len(cursor.executions) == 2
    statement, _ = cursor.executions[-1]
    validate_oracle_statement(statement)
    assert statement.startswith('MERGE INTO graphiti_nodes')


@pytest.mark.asyncio
async def test_exact_name_lookup_reaches_oracle_sql_not_cypher() -> None:
    cursor = FakeCursor()
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=FakePool(FakeConnection(cursor)),
    )

    assert await EntityNode.get_by_names(driver, ['Alpha'], ['group-a']) == []

    statement, binds = cursor.executions[-1]
    validate_oracle_statement(statement)
    assert statement.startswith('SELECT')
    assert binds['names_json'] == '["Alpha"]'
    assert binds['group_ids_json'] == '["group-a"]'
