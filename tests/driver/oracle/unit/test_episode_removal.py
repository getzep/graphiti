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

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.oracle.codecs import (
    edge_to_binds,
    entity_node_to_binds,
    episode_node_to_binds,
)
from graphiti_core.driver.oracle.errors import EpisodeRemovalConflict, EpisodeSummaryRebuildError
from graphiti_core.driver.oracle.operations.graph import OracleGraphMaintenanceOperations
from graphiti_core.driver.oracle.sql.episode_removal import (
    GET_REMOVAL_EDGES,
    GET_REMOVAL_NODES,
    LOCK_REMOVAL_EDGES,
    LOCK_REMOVAL_NODES,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

NOW = datetime(2026, 10, 22, tzinfo=UTC)


class RemovalExecutor:
    def __init__(self):
        self._scope_group_id = 'group-a'
        self.active_transaction = False
        self.commits = 0
        self.rollbacks = 0
        self.locks = []
        self.episodes = [
            EpisodicNode(
                uuid=f'episode-{letter}',
                name=f'Source {letter}',
                group_id='group-a',
                source=EpisodeType.text,
                source_description='unit fixture',
                content=f'Evidence {letter}',
                entity_edges=['shared-fact'],
                created_at=NOW,
                valid_at=NOW,
            )
            for letter in ('a', 'b')
        ]
        self.entities = [
            EntityNode(
                uuid=f'entity-{letter}',
                name=f'Entity {letter}',
                group_id='group-a',
                summary='old summary',
                attributes={'preserved': True},
                created_at=NOW,
            )
            for letter in ('a', 'b')
        ]
        self.fact = EntityEdge(
            uuid='shared-fact',
            name='RELATION',
            group_id='group-a',
            source_node_uuid='entity-a',
            target_node_uuid='entity-b',
            fact='Shared fact',
            episodes=['episode-a', 'episode-b'],
            created_at=NOW,
        )
        self.node_records = [entity_node_to_binds(node) for node in self.entities] + [
            episode_node_to_binds(episode) for episode in self.episodes
        ]
        self.edge_records = [edge_to_binds(self.fact, kind='RELATES_TO')]
        for episode in self.episodes:
            for entity in self.entities:
                self.edge_records.append(
                    {'kind': 'MENTIONS', 'source_uuid': episode.uuid, 'target_uuid': entity.uuid}
                )
        self.episode_node_ops = SimpleNamespace(
            get_by_uuid=AsyncMock(return_value=self.episodes[0]), delete=AsyncMock()
        )
        self.entity_node_ops = SimpleNamespace(save=AsyncMock(), delete=AsyncMock())
        self.entity_edge_ops = SimpleNamespace(save=AsyncMock(), delete=AsyncMock())

    async def execute_query(self, statement, **binds):
        assert binds['group_id'] == 'group-a'
        if statement == GET_REMOVAL_NODES:
            return deepcopy(self.node_records), [], None
        if statement == GET_REMOVAL_EDGES:
            return deepcopy(self.edge_records), [], None
        raise AssertionError('unexpected SQL')

    async def run(self, statement, **binds):
        assert self.active_transaction
        if statement in (LOCK_REMOVAL_NODES, LOCK_REMOVAL_EDGES):
            self.locks.append(statement)
            return [], [], None
        return await self.execute_query(statement, **binds)

    @asynccontextmanager
    async def transaction(self):
        self.active_transaction = True
        try:
            yield self
            self.commits += 1
        except BaseException:
            self.rollbacks += 1
            raise
        finally:
            self.active_transaction = False


@pytest.mark.asyncio
async def test_removal_prepares_only_surviving_sources_before_locks_and_preserves_other_fields():
    executor = RemovalExecutor()

    async def prepare(node, episodes):
        assert not executor.active_transaction
        assert [episode.uuid for episode in episodes] == ['episode-b']
        node.summary = episodes[0].content
        node.name = 'callback must not change persisted identity'
        node.attributes = {}

    await OracleGraphMaintenanceOperations().remove_episode(executor, 'episode-a', prepare)

    assert executor.locks == [LOCK_REMOVAL_NODES, LOCK_REMOVAL_EDGES]
    assert executor.commits == 1
    assert executor.rollbacks == 0
    saved_fact = executor.entity_edge_ops.save.await_args.args[1]
    assert saved_fact.episodes == ['episode-b']
    assert executor.entity_edge_ops.save.await_args.kwargs['tx'] is executor
    for call in executor.entity_node_ops.save.await_args_list:
        node = call.args[1]
        assert node.summary == 'Evidence b'
        assert node.name.startswith('Entity ')
        assert node.attributes == {'preserved': True}
        assert call.kwargs['tx'] is executor
    assert executor.entities[0].summary == 'old summary'
    assert executor.fact.episodes == ['episode-a', 'episode-b']


@pytest.mark.asyncio
async def test_removal_summary_failure_never_enters_a_mutating_transaction():
    executor = RemovalExecutor()

    async def fail(*_args):
        raise RuntimeError('model unavailable')

    with pytest.raises(EpisodeSummaryRebuildError, match='summary preparation'):
        await OracleGraphMaintenanceOperations().remove_episode(executor, 'episode-a', fail)

    assert executor.commits == executor.rollbacks == 0
    assert executor.locks == []
    executor.entity_edge_ops.save.assert_not_awaited()
    executor.episode_node_ops.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_removal_snapshot_change_aborts_before_any_mutation():
    executor = RemovalExecutor()

    async def change(node, _episodes):
        node.summary = 'new summary'
        executor.node_records[-1]['content'] = 'a concurrent new revision'

    with pytest.raises(EpisodeRemovalConflict, match='changed'):
        await OracleGraphMaintenanceOperations().remove_episode(executor, 'episode-a', change)

    assert executor.commits == 0
    assert executor.rollbacks == 1
    executor.entity_edge_ops.save.assert_not_awaited()
    executor.episode_node_ops.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_removal_refuses_missing_surviving_provenance():
    executor = RemovalExecutor()
    executor.edge_records[0]['episode_uuids'] = '["episode-a","absent-source"]'
    prepare = AsyncMock()

    with pytest.raises(EpisodeRemovalConflict, match='missing surviving provenance'):
        await OracleGraphMaintenanceOperations().remove_episode(executor, 'episode-a', prepare)

    prepare.assert_not_awaited()
    assert executor.locks == []


@pytest.mark.asyncio
async def test_removal_refuses_saga_membership_without_a_reconciliation_contract():
    executor = RemovalExecutor()
    executor.edge_records.append(
        {'kind': 'HAS_EPISODE', 'source_uuid': 'saga-a', 'target_uuid': 'episode-a'}
    )
    prepare = AsyncMock()

    with pytest.raises(EpisodeRemovalConflict, match='Saga'):
        await OracleGraphMaintenanceOperations().remove_episode(executor, 'episode-a', prepare)

    prepare.assert_not_awaited()
    assert executor.locks == []


@pytest.mark.asyncio
async def test_removal_refuses_unavailable_remaining_source_content():
    executor = RemovalExecutor()
    executor.node_records[-1]['content'] = ''
    prepare = AsyncMock()

    with pytest.raises(EpisodeRemovalConflict, match='content'):
        await OracleGraphMaintenanceOperations().remove_episode(executor, 'episode-a', prepare)

    prepare.assert_not_awaited()
    assert executor.locks == []
