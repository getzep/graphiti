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

import pytest

from graphiti_core.driver.oracle.errors import GroupScopeConflict
from graphiti_core.driver.oracle.operations.graph import OracleGraphMaintenanceOperations
from tests.driver.oracle.operations.recording import RecordingExecutor


@pytest.mark.asyncio
async def test_group_scoped_clear_deletes_edges_before_nodes() -> None:
    executor = RecordingExecutor()

    await OracleGraphMaintenanceOperations().clear_data(executor, ['group-a', 'group-b'])

    assert len(executor.calls) == 2
    assert executor.calls[0][0].startswith('DELETE FROM graphiti_edges')
    assert executor.calls[1][0].startswith('DELETE FROM graphiti_nodes')
    assert executor.calls[0][1]['group_ids_json'] == '["group-a","group-b"]'


@pytest.mark.asyncio
async def test_clear_all_never_uses_an_unbounded_single_table_delete_first() -> None:
    executor = RecordingExecutor()

    await OracleGraphMaintenanceOperations().clear_data(executor)

    assert [statement for statement, _ in executor.calls] == [
        'DELETE FROM graphiti_edges',
        'DELETE FROM graphiti_nodes',
    ]


@pytest.mark.asyncio
async def test_scoped_clear_without_group_list_only_clears_request_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    await OracleGraphMaintenanceOperations().clear_data(executor)

    assert all('group_ids_json' in binds for _, binds in executor.calls)
    assert all(binds['group_ids_json'] == '["group-b"]' for _, binds in executor.calls)
    assert all(
        statement not in {'DELETE FROM graphiti_edges', 'DELETE FROM graphiti_nodes'}
        for statement, _ in executor.calls
    )


@pytest.mark.asyncio
async def test_scoped_mentioned_nodes_reject_foreign_episode() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'
    episode = type('Episode', (), {'uuid': 'episode-a', 'group_id': 'group-a'})()

    with pytest.raises(GroupScopeConflict, match='group-b'):
        await OracleGraphMaintenanceOperations().get_mentioned_nodes(executor, [episode])

    assert executor.calls == []
