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

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from graphiti_core.driver.oracle.config import Oracle26aiConfig
from graphiti_core.driver.oracle.driver import Oracle26aiDriver
from graphiti_core.driver.oracle.errors import InvalidEdgeEndpoint
from graphiti_core.driver.oracle.operations.community_edges import OracleCommunityEdgeOperations
from graphiti_core.driver.oracle.operations.community_nodes import OracleCommunityNodeOperations
from graphiti_core.driver.oracle.operations.entity_edges import OracleEntityEdgeOperations
from graphiti_core.driver.oracle.operations.entity_nodes import OracleEntityNodeOperations
from graphiti_core.driver.oracle.operations.episode_nodes import OracleEpisodeNodeOperations
from graphiti_core.driver.oracle.operations.episodic_edges import OracleEpisodicEdgeOperations
from graphiti_core.driver.oracle.operations.has_episode_edges import OracleHasEpisodeEdgeOperations
from graphiti_core.driver.oracle.operations.next_episode_edges import (
    OracleNextEpisodeEdgeOperations,
)
from graphiti_core.driver.oracle.operations.saga_nodes import OracleSagaNodeOperations
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from tests.driver.oracle.unit.fakes import FakeConnection, FakeCursor, FakePool

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def make_driver(connection: FakeConnection) -> Oracle26aiDriver:
    return Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=FakePool(connection),
    )


@pytest.mark.asyncio
async def test_transaction_commits_on_clean_exit() -> None:
    connection = FakeConnection()
    driver = make_driver(connection)

    async with driver.transaction() as transaction:
        await transaction.run('insert into graphiti_probe_nodes (id) values (:id)', id='one')

    assert connection.commits == 1
    assert connection.rollbacks == 0


@pytest.mark.asyncio
async def test_transaction_rolls_back_on_error() -> None:
    connection = FakeConnection()
    driver = make_driver(connection)

    with pytest.raises(RuntimeError, match='stop'):
        async with driver.transaction() as transaction:
            await transaction.run('delete from graphiti_probe_nodes')
            raise RuntimeError('stop')

    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_transaction_rolls_back_on_cancellation() -> None:
    connection = FakeConnection()
    driver = make_driver(connection)

    with pytest.raises(asyncio.CancelledError):
        async with driver.transaction():
            raise asyncio.CancelledError

    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_transaction_commit_failure_rolls_back_before_releasing_connection() -> None:
    class FailingCommitConnection(FakeConnection):
        async def commit(self) -> None:
            self.commits += 1
            raise RuntimeError('commit failed')

    connection = FailingCommitConnection()
    driver = make_driver(connection)

    with pytest.raises(RuntimeError, match='commit failed'):
        async with driver.transaction() as transaction:
            await transaction.run('DELETE FROM graphiti_edges WHERE uuid = :uuid', uuid='own')

    assert connection.commits == 1
    assert connection.rollbacks == 1
    assert driver._pool.acquired == driver._pool.released == 1


class FailOnSecondMergeCursor(FakeCursor):
    def __init__(self) -> None:
        super().__init__()
        self.merge_count = 0

    async def execute(self, statement: str, parameters: dict[str, object]) -> None:
        await super().execute(statement, parameters)
        if statement.startswith('MERGE INTO graphiti_nodes'):
            self.merge_count += 1
            if self.merge_count == 2:
                raise RuntimeError('second merge failed')


def _bulk_entity(uuid: str) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=uuid,
        group_id='group-a',
        labels=['Concept'],
        summary='atomic bulk fixture',
        name_embedding=[0.0] * 1024,
    )


def _episode() -> EpisodicNode:
    return EpisodicNode(
        uuid='episode-a',
        name='Episode A',
        group_id='group-a',
        labels=[],
        source=EpisodeType.json,
        source_description='atomic fixture',
        content='{"event":"atomic"}',
        created_at=NOW,
        valid_at=NOW,
    )


def _community() -> CommunityNode:
    return CommunityNode(
        uuid='community-a',
        name='Community A',
        group_id='group-a',
        labels=[],
        summary='atomic fixture',
        created_at=NOW,
    )


def _saga() -> SagaNode:
    return SagaNode(
        uuid='saga-a',
        name='Saga A',
        group_id='group-a',
        labels=[],
        created_at=NOW,
    )


NODE_SAVE_CASES = (
    (OracleEntityNodeOperations, _bulk_entity('entity-a')),
    (OracleEpisodeNodeOperations, _episode()),
    (OracleCommunityNodeOperations, _community()),
    (OracleSagaNodeOperations, _saga()),
)


EDGE_SAVE_CASES = (
    (
        OracleEntityEdgeOperations,
        EntityEdge(
            uuid='edge-entity',
            source_node_uuid='entity-a',
            target_node_uuid='entity-b',
            group_id='group-a',
            name='RELATES_TO',
            fact='A relates to B',
            created_at=NOW,
        ),
        'ENTITY',
        'ENTITY',
    ),
    (
        OracleEpisodicEdgeOperations,
        EpisodicEdge(
            uuid='edge-mention',
            source_node_uuid='episode-a',
            target_node_uuid='entity-a',
            group_id='group-a',
            created_at=NOW,
        ),
        'EPISODIC',
        'ENTITY',
    ),
    (
        OracleCommunityEdgeOperations,
        CommunityEdge(
            uuid='edge-member',
            source_node_uuid='community-a',
            target_node_uuid='entity-a',
            group_id='group-a',
            created_at=NOW,
        ),
        'COMMUNITY',
        'ENTITY',
    ),
    (
        OracleHasEpisodeEdgeOperations,
        HasEpisodeEdge(
            uuid='edge-has-episode',
            source_node_uuid='saga-a',
            target_node_uuid='episode-a',
            group_id='group-a',
            created_at=NOW,
        ),
        'SAGA',
        'EPISODIC',
    ),
    (
        OracleNextEpisodeEdgeOperations,
        NextEpisodeEdge(
            uuid='edge-next',
            source_node_uuid='episode-a',
            target_node_uuid='episode-b',
            group_id='group-a',
            created_at=NOW,
        ),
        'EPISODIC',
        'EPISODIC',
    ),
)


class EdgeSaveCursor(FakeCursor):
    def __init__(self, source_kind: str, target_kind: str) -> None:
        super().__init__()
        self.source_kind = source_kind
        self.target_kind = target_kind

    async def execute(self, statement: str, parameters: dict[str, object]) -> None:
        self.executions.append((statement, parameters))
        if 'FOR UPDATE' in statement:
            source_uuid = str(parameters['source_uuid'])
            target_uuid = str(parameters['target_uuid'])
            values = {
                source_uuid: (source_uuid, self.source_kind, 'group-a'),
                target_uuid: (target_uuid, self.target_kind, 'group-a'),
            }
            self.rows = [values[uuid] for uuid in sorted(values)]
            self.description = [SimpleNamespace(name=name) for name in ('UUID', 'KIND', 'GROUP_ID')]
        elif statement.startswith('SELECT kind, group_id'):
            self.rows = []
            self.description = [SimpleNamespace(name='KIND'), SimpleNamespace(name='GROUP_ID')]
        else:
            self.rows = []
            self.description = None


class FailOnFirstMergeCursor(FakeCursor):
    async def execute(self, statement: str, parameters: dict[str, object]) -> None:
        await super().execute(statement, parameters)
        if statement.startswith('MERGE INTO graphiti_nodes'):
            raise RuntimeError('merge failed')


@pytest.mark.parametrize(('ops_type', 'node'), NODE_SAVE_CASES)
@pytest.mark.asyncio
async def test_standalone_node_save_owns_one_transaction(ops_type, node) -> None:
    connection = FakeConnection()
    pool = FakePool(connection)
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=pool,
    )

    await ops_type().save(driver, node)

    assert pool.acquired == pool.released == 1
    assert connection.commits == 1
    assert connection.rollbacks == 0


@pytest.mark.parametrize(('ops_type', 'edge', 'source_kind', 'target_kind'), EDGE_SAVE_CASES)
@pytest.mark.asyncio
async def test_standalone_edge_save_owns_one_transaction(
    ops_type,
    edge,
    source_kind: str,
    target_kind: str,
) -> None:
    connection = FakeConnection(EdgeSaveCursor(source_kind, target_kind))
    pool = FakePool(connection)
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=pool,
    )

    await ops_type().save(driver, edge)

    assert pool.acquired == pool.released == 1
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert 'ORDER BY uuid' in connection.fake_cursor.executions[0][0]
    assert 'FOR UPDATE' in connection.fake_cursor.executions[0][0]


@pytest.mark.asyncio
async def test_endpoint_validation_failure_rolls_back_and_releases_locks() -> None:
    connection = FakeConnection(EdgeSaveCursor('SAGA', 'ENTITY'))
    pool = FakePool(connection)
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=pool,
    )

    with pytest.raises(InvalidEdgeEndpoint, match='RELATES_TO requires ENTITY -> ENTITY'):
        await OracleEntityEdgeOperations().save(driver, EDGE_SAVE_CASES[0][1])

    assert pool.acquired == pool.released == 1
    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_standalone_save_rolls_back_validation_and_merge_together() -> None:
    connection = FakeConnection(FailOnFirstMergeCursor())
    pool = FakePool(connection)
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=pool,
    )

    with pytest.raises(RuntimeError, match='merge failed'):
        await OracleEntityNodeOperations().save(driver, _bulk_entity('entity-failing'))

    assert pool.acquired == pool.released == 1
    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_default_bulk_is_one_transaction_and_rolls_back_second_merge_failure() -> None:
    connection = FakeConnection(FailOnSecondMergeCursor())
    pool = FakePool(connection)
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=pool,
    )

    with pytest.raises(RuntimeError, match='second merge failed'):
        await OracleEntityNodeOperations().save_bulk(
            driver,
            [_bulk_entity('entity-a'), _bulk_entity('entity-b')],
        )

    assert pool.acquired == 1
    assert pool.released == 1
    assert connection.commits == 0
    assert connection.rollbacks == 1
