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

from graphiti_core.driver.oracle.errors import GroupScopeConflict, InvalidEmbeddingDimension
from graphiti_core.driver.oracle.operations.entity_nodes import OracleEntityNodeOperations
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode
from tests.driver.oracle.operations.recording import RecordingExecutor, RecordingTransaction


def entity(group_id: str = 'group-a', *, vector_size: int = 1024) -> EntityNode:
    return EntityNode(
        uuid=f'00000000-0000-0000-0000-{group_id[-1] * 12}',
        name='Shared name',
        group_id=group_id,
        labels=['Concept'],
        summary='group-scoped',
        attributes={'source': group_id},
        name_embedding=[0.0] * vector_size,
        created_at=datetime(2026, 9, 21, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_save_uses_transaction_when_provided() -> None:
    executor = RecordingExecutor()
    transaction = RecordingTransaction()

    await OracleEntityNodeOperations().save(executor, entity(), tx=transaction)

    assert executor.calls == []
    assert len(transaction.calls) == 2
    assert transaction.calls[0][0].startswith('SELECT')
    assert transaction.calls[1][1]['group_id'] == 'group-a'
    assert 'MERGE INTO graphiti_nodes' in transaction.calls[1][0]


@pytest.mark.asyncio
async def test_group_queries_bind_each_group_without_shared_state() -> None:
    executor = RecordingExecutor()
    operations = OracleEntityNodeOperations()

    await operations.get_by_group_ids(executor, ['group-a'])
    await operations.get_by_group_ids(executor, ['group-b'])

    assert executor.calls[0][1]['group_ids_json'] == '["group-a"]'
    assert executor.calls[1][1]['group_ids_json'] == '["group-b"]'
    assert 'JSON_TABLE' in executor.calls[0][0]


@pytest.mark.asyncio
async def test_invalid_vector_does_not_call_executor() -> None:
    executor = RecordingExecutor()

    with pytest.raises(InvalidEmbeddingDimension):
        await OracleEntityNodeOperations().save(executor, entity(vector_size=3))

    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_known_uuid_read_binds_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    with pytest.raises(NodeNotFoundError):
        await OracleEntityNodeOperations().get_by_uuid(executor, entity('group-a').uuid)

    statement, binds = executor.calls[0]
    assert 'group_id = :scope_group_id' in statement
    assert binds['scope_group_id'] == 'group-b'


@pytest.mark.asyncio
async def test_node_uuid_cannot_be_reassigned_to_another_group() -> None:
    executor = RecordingExecutor([{'kind': 'ENTITY', 'group_id': 'group-a'}])

    with pytest.raises(GroupScopeConflict, match='group-a'):
        await OracleEntityNodeOperations().save(executor, entity('group-b'))

    assert len(executor.calls) == 1
    assert executor.calls[0][0].startswith('SELECT')


@pytest.mark.asyncio
async def test_scoped_driver_rejects_new_node_for_another_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    with pytest.raises(GroupScopeConflict, match='group-b'):
        await OracleEntityNodeOperations().save(executor, entity('group-a'))

    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_driver_rejects_model_delete_for_another_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    with pytest.raises(GroupScopeConflict, match='group-b'):
        await OracleEntityNodeOperations().delete(executor, entity('group-a'))

    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_driver_rejects_conflicting_explicit_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    with pytest.raises(GroupScopeConflict, match='group-b'):
        await OracleEntityNodeOperations().get_by_uuid(
            executor,
            entity('group-a').uuid,
            group_id='group-a',
        )

    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_group_list_is_intersected_with_request_scope() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'
    operations = OracleEntityNodeOperations()

    await operations.get_by_group_ids(executor, ['group-a', 'group-b'])
    missing = await operations.get_by_group_ids(executor, ['group-a'])

    assert executor.calls[0][1]['group_ids_json'] == '["group-b"]'
    assert missing == []
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_bulk_uuid_delete_binds_explicit_group() -> None:
    executor = RecordingExecutor()

    await OracleEntityNodeOperations().delete_by_uuids(
        executor,
        [entity('group-a').uuid],
        group_id='group-a',
    )

    assert executor.calls
    assert all(call[1]['scope_group_id'] == 'group-a' for call in executor.calls)


@pytest.mark.asyncio
async def test_bulk_embedding_reads_partition_nodes_by_group() -> None:
    node_a = entity('group-a')
    node_b = entity('group-b')
    executor = RecordingExecutor(
        responses=[
            [{**node_a.model_dump(), 'properties': '{}', 'labels': '["Concept"]'}],
            [{**node_b.model_dump(), 'properties': '{}', 'labels': '["Concept"]'}],
        ]
    )

    await OracleEntityNodeOperations().load_embeddings_bulk(executor, [node_a, node_b])

    assert [call[1]['scope_group_id'] for call in executor.calls] == ['group-a', 'group-b']
