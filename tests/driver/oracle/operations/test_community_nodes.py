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

from graphiti_core.driver.oracle.codecs import community_node_to_binds
from graphiti_core.driver.oracle.operations.community_nodes import OracleCommunityNodeOperations
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import CommunityNode
from tests.driver.oracle.operations.recording import RecordingExecutor


def community() -> CommunityNode:
    return CommunityNode(
        uuid='00000000-0000-0000-0000-000000000201',
        name='Neutral cluster',
        group_id='group-a',
        labels=[],
        summary='Alpha and Beta',
        name_embedding=[0.0] * 1024,
        created_at=datetime(2026, 9, 21, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_community_round_trip_preserves_summary_and_embedding() -> None:
    node = community()
    executor = RecordingExecutor([community_node_to_binds(node)])
    operations = OracleCommunityNodeOperations()

    await operations.save(executor, node)
    restored = await operations.get_by_uuid(executor, node.uuid)

    assert restored.model_dump() == node.model_dump()
    assert "kind = 'COMMUNITY'" in executor.calls[2][0]


@pytest.mark.asyncio
async def test_community_missing_uuid_maps_to_graphiti_error() -> None:
    with pytest.raises(NodeNotFoundError):
        await OracleCommunityNodeOperations().get_by_uuid(RecordingExecutor(), 'missing')


@pytest.mark.asyncio
async def test_community_group_read_binds_pagination() -> None:
    executor = RecordingExecutor()

    await OracleCommunityNodeOperations().get_by_group_ids(
        executor,
        ['group-a'],
        limit=5,
        uuid_cursor='cursor-uuid',
    )

    assert executor.calls[0][1] == {
        'group_ids_json': '["group-a"]',
        'uuid_cursor': 'cursor-uuid',
        'result_limit': 5,
    }
