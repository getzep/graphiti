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

from graphiti_core.driver.oracle.codecs import edge_to_binds
from graphiti_core.driver.oracle.errors import GroupScopeConflict, InvalidEdgeEndpoint
from graphiti_core.driver.oracle.operations.community_edges import OracleCommunityEdgeOperations
from graphiti_core.driver.oracle.operations.edge_base import validate_endpoints
from graphiti_core.driver.oracle.operations.entity_edges import OracleEntityEdgeOperations
from graphiti_core.driver.oracle.operations.episodic_edges import OracleEpisodicEdgeOperations
from graphiti_core.driver.oracle.operations.has_episode_edges import OracleHasEpisodeEdgeOperations
from graphiti_core.driver.oracle.operations.next_episode_edges import (
    OracleNextEpisodeEdgeOperations,
)
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.errors import EdgeNotFoundError
from tests.driver.oracle.operations.recording import RecordingExecutor, RecordingTransaction

NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)


def make_entity_edge() -> EntityEdge:
    return EntityEdge(
        uuid='edge-entity',
        source_node_uuid='entity-a',
        target_node_uuid='entity-b',
        group_id='group-a',
        name='RELATES_TO',
        fact='Alpha relates to Beta',
        fact_embedding=[0.0] * 1024,
        episodes=['episode-a'],
        created_at=NOW,
        valid_at=NOW,
        reference_time=NOW,
        attributes={'confidence': 1.0},
    )


def make_episodic_edge() -> EpisodicEdge:
    return EpisodicEdge(
        uuid='edge-mentions',
        source_node_uuid='episode-a',
        target_node_uuid='entity-a',
        group_id='group-a',
        created_at=NOW,
    )


def make_community_edge() -> CommunityEdge:
    return CommunityEdge(
        uuid='edge-member',
        source_node_uuid='community-a',
        target_node_uuid='entity-a',
        group_id='group-a',
        created_at=NOW,
    )


def make_has_episode_edge() -> HasEpisodeEdge:
    return HasEpisodeEdge(
        uuid='edge-has-episode',
        source_node_uuid='saga-a',
        target_node_uuid='episode-a',
        group_id='group-a',
        created_at=NOW,
    )


def make_next_episode_edge() -> NextEpisodeEdge:
    return NextEpisodeEdge(
        uuid='edge-next',
        source_node_uuid='episode-a',
        target_node_uuid='episode-b',
        group_id='group-a',
        created_at=NOW,
    )


CASES = (
    (OracleEntityEdgeOperations, make_entity_edge, 'RELATES_TO', 'ENTITY', 'ENTITY'),
    (OracleEpisodicEdgeOperations, make_episodic_edge, 'MENTIONS', 'EPISODIC', 'ENTITY'),
    (OracleCommunityEdgeOperations, make_community_edge, 'HAS_MEMBER', 'COMMUNITY', 'ENTITY'),
    (OracleHasEpisodeEdgeOperations, make_has_episode_edge, 'HAS_EPISODE', 'SAGA', 'EPISODIC'),
    (
        OracleNextEpisodeEdgeOperations,
        make_next_episode_edge,
        'NEXT_EPISODE',
        'EPISODIC',
        'EPISODIC',
    ),
)


def endpoint_rows(
    edge,
    source_kind: str,
    target_kind: str,
    *,
    source_group_id: str | None = None,
    target_group_id: str | None = None,
) -> list[dict[str, str]]:
    rows = {
        edge.source_node_uuid: {
            'uuid': edge.source_node_uuid,
            'kind': source_kind,
            'group_id': source_group_id or edge.group_id,
        },
        edge.target_node_uuid: {
            'uuid': edge.target_node_uuid,
            'kind': target_kind,
            'group_id': target_group_id or edge.group_id,
        },
    }
    return [rows[uuid] for uuid in sorted(rows)]


@pytest.mark.parametrize(('ops_type', 'edge_factory', 'kind', 'source_kind', 'target_kind'), CASES)
@pytest.mark.asyncio
async def test_edge_family_round_trip(
    ops_type,
    edge_factory,
    kind: str,
    source_kind: str,
    target_kind: str,
) -> None:
    edge = edge_factory()
    record = edge_to_binds(edge, kind=kind)
    executor = RecordingExecutor(
        responses=[
            endpoint_rows(edge, source_kind, target_kind),
            [],
            [],
            [record],
        ]
    )
    operations = ops_type()

    await operations.save(executor, edge)
    restored = await operations.get_by_uuid(executor, edge.uuid)

    assert restored.model_dump() == edge.model_dump()
    assert executor.calls[2][1]['kind'] == kind


@pytest.mark.asyncio
async def test_entity_edge_preserves_all_temporal_fields_and_attributes() -> None:
    edge = make_entity_edge()
    record = edge_to_binds(edge, kind='RELATES_TO')

    assert record['valid_at'] == NOW
    assert record['reference_time'] == NOW
    assert record['invalid_at'] is None
    assert record['expired_at'] is None
    assert record['properties'] == '{"confidence":1.0}'


@pytest.mark.asyncio
async def test_missing_endpoint_is_rejected_before_edge_write() -> None:
    edge = make_episodic_edge()
    executor = RecordingExecutor(
        responses=[
            [
                {
                    'uuid': edge.source_node_uuid,
                    'kind': 'EPISODIC',
                    'group_id': edge.group_id,
                }
            ]
        ]
    )

    with pytest.raises(InvalidEdgeEndpoint, match='MENTIONS requires EPISODIC -> ENTITY'):
        await OracleEpisodicEdgeOperations().save(executor, edge)

    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_cross_group_edge_is_rejected_before_write() -> None:
    edge = make_entity_edge()
    executor = RecordingExecutor(
        responses=[endpoint_rows(edge, 'ENTITY', 'ENTITY', target_group_id='group-b')]
    )

    with pytest.raises(InvalidEdgeEndpoint, match='one group'):
        await OracleEntityEdgeOperations().save(executor, edge)

    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_edge_validation_and_write_share_the_supplied_transaction() -> None:
    executor = RecordingExecutor()
    transaction = RecordingTransaction(
        responses=[endpoint_rows(make_entity_edge(), 'ENTITY', 'ENTITY')]
    )

    await OracleEntityEdgeOperations().save(executor, make_entity_edge(), tx=transaction)

    assert executor.calls == []
    assert len(transaction.calls) == 3
    assert transaction.calls[0][0].startswith('SELECT')
    assert transaction.calls[1][0].startswith('SELECT')
    assert transaction.calls[2][0].startswith('MERGE INTO graphiti_edges')


@pytest.mark.asyncio
async def test_endpoint_validation_requires_transactional_ordered_row_locks() -> None:
    executor = RecordingExecutor(
        responses=[
            [
                {
                    'source_kind': 'ENTITY',
                    'target_kind': 'ENTITY',
                    'source_group_id': 'group-a',
                    'target_group_id': 'group-a',
                }
            ]
        ]
    )

    with pytest.raises(TypeError, match='transaction'):
        await validate_endpoints(
            executor,
            make_entity_edge(),
            kind='RELATES_TO',
            source_kind='ENTITY',
            target_kind='ENTITY',
        )

    transaction = RecordingTransaction(
        responses=[endpoint_rows(make_entity_edge(), 'ENTITY', 'ENTITY')]
    )
    await validate_endpoints(
        executor,
        make_entity_edge(),
        kind='RELATES_TO',
        source_kind='ENTITY',
        target_kind='ENTITY',
        tx=transaction,
    )

    statement, binds = transaction.calls[0]
    assert executor.calls == []
    assert 'ORDER BY uuid' in statement
    assert 'FOR UPDATE' in statement
    assert binds == {'source_uuid': 'entity-a', 'target_uuid': 'entity-b'}


@pytest.mark.asyncio
async def test_edge_uuid_cannot_be_reassigned_to_another_group() -> None:
    edge = make_entity_edge().model_copy(update={'group_id': 'group-b'})
    executor = RecordingExecutor(
        responses=[
            endpoint_rows(edge, 'ENTITY', 'ENTITY'),
            [{'kind': 'RELATES_TO', 'group_id': 'group-a'}],
        ]
    )

    with pytest.raises(GroupScopeConflict, match='group-a'):
        await OracleEntityEdgeOperations().save(executor, edge)

    assert len(executor.calls) == 2
    assert all(statement.startswith('SELECT') for statement, _ in executor.calls)


@pytest.mark.asyncio
async def test_same_uuid_endpoint_is_locked_and_validated_once() -> None:
    edge = make_entity_edge().model_copy(
        update={'source_node_uuid': 'entity-a', 'target_node_uuid': 'entity-a'}
    )
    transaction = RecordingTransaction(responses=[endpoint_rows(edge, 'ENTITY', 'ENTITY'), [], []])

    await OracleEntityEdgeOperations().save(
        RecordingExecutor(),
        edge,
        tx=transaction,
    )

    assert len(transaction.calls) == 3
    assert transaction.calls[0][1] == {
        'source_uuid': 'entity-a',
        'target_uuid': 'entity-a',
    }


@pytest.mark.asyncio
async def test_scoped_driver_rejects_new_edge_for_another_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    with pytest.raises(GroupScopeConflict, match='group-b'):
        await OracleEntityEdgeOperations().save(executor, make_entity_edge())

    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_known_edge_read_and_delete_bind_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'
    operations = OracleEntityEdgeOperations()

    with pytest.raises(EdgeNotFoundError):
        await operations.get_by_uuid(executor, make_entity_edge().uuid)
    await operations.delete_by_uuids(
        executor,
        [make_entity_edge().uuid],
        group_id='group-b',
    )

    assert executor.calls[0][1]['scope_group_id'] == 'group-b'
    assert executor.calls[1][1]['scope_group_id'] == 'group-b'


@pytest.mark.asyncio
async def test_bulk_embedding_reads_partition_edges_by_group() -> None:
    edge_a = make_entity_edge()
    edge_b = edge_a.model_copy(update={'uuid': 'edge-b', 'group_id': 'group-b'})
    executor = RecordingExecutor(
        responses=[
            [edge_to_binds(edge_a, kind='RELATES_TO')],
            [edge_to_binds(edge_b, kind='RELATES_TO')],
        ]
    )

    await OracleEntityEdgeOperations().load_embeddings_bulk(executor, [edge_a, edge_b])

    assert [call[1]['scope_group_id'] for call in executor.calls] == ['group-a', 'group-b']
