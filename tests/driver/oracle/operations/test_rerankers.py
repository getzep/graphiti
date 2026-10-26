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

from graphiti_core.driver.oracle.operations.search import OracleSearchOperations
from tests.driver.oracle.operations.recording import RecordingExecutor


@pytest.mark.asyncio
async def test_distance_reranker_preserves_center_and_uses_uuid_ties() -> None:
    executor = RecordingExecutor(
        responses=[
            [
                {'from_uuid': 'center', 'to_uuid': 'entity-b'},
                {'from_uuid': 'center', 'to_uuid': 'entity-a'},
            ],
            [
                {'from_uuid': 'entity-a', 'to_uuid': 'center'},
                {'from_uuid': 'entity-b', 'to_uuid': 'entity-c'},
            ],
            [],
        ]
    )

    uuids, scores = await OracleSearchOperations().node_distance_reranker(
        executor,
        ['center', 'entity-c', 'entity-b', 'entity-a', 'disconnected'],
        'center',
        min_score=0.1,
    )

    statement, binds = executor.calls[0]
    assert statement.startswith('SELECT ')
    assert uuids == ['center', 'entity-a', 'entity-b', 'entity-c']
    assert scores == [0.1, 1.0, 1.0, 0.5]
    assert 'GRAPH_TABLE' not in statement
    assert 'graphiti_edges' in statement
    assert 'e.group_id = center.group_id' in statement
    assert binds['center_node_uuid'] == 'center'
    assert binds['frontier_uuids_json'] == '["center"]'
    assert len(executor.calls) == 3
    assert executor.calls[1][1]['frontier_uuids_json'] == '["entity-a","entity-b"]'


@pytest.mark.asyncio
async def test_distance_reranker_stops_when_all_candidates_are_found() -> None:
    executor = RecordingExecutor(responses=[[{'from_uuid': 'center', 'to_uuid': 'entity-a'}]])

    uuids, scores = await OracleSearchOperations().node_distance_reranker(
        executor, ['entity-a'], 'center'
    )

    assert (uuids, scores) == (['entity-a'], [1.0])
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_mentions_reranker_uses_distinct_episodes_after_rrf() -> None:
    executor = RecordingExecutor(
        [
            {'uuid': 'entity-b', 'mention_count': 2},
            {'uuid': 'entity-a', 'mention_count': 2},
        ]
    )

    uuids, scores = await OracleSearchOperations().episode_mentions_reranker(
        executor,
        [['entity-b', 'entity-a'], ['entity-a', 'entity-c']],
    )

    statement, binds = executor.calls[0]
    assert uuids == ['entity-a', 'entity-b', 'entity-c']
    assert scores == [2.0, 2.0, 0.0]
    assert 'COUNT(DISTINCT episode_uuid) AS mention_count' in statement
    assert 'GRAPH_TABLE' in statement
    assert binds['candidate_uuids_json'] == '["entity-a","entity-b","entity-c"]'


@pytest.mark.asyncio
async def test_empty_reranker_candidates_short_circuit() -> None:
    executor = RecordingExecutor()

    assert await OracleSearchOperations().episode_mentions_reranker(executor, []) == ([], [])
    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_rerankers_bind_request_group() -> None:
    distance_executor = RecordingExecutor([])
    distance_executor._scope_group_id = 'group-b'
    mentions_executor = RecordingExecutor([])
    mentions_executor._scope_group_id = 'group-b'
    operations = OracleSearchOperations()

    await operations.node_distance_reranker(
        distance_executor,
        ['entity-a'],
        'center',
    )
    await operations.episode_mentions_reranker(mentions_executor, ['entity-a'])

    assert distance_executor.calls[0][1]['scope_group_id'] == 'group-b'
    assert ':scope_group_id IS NULL' in distance_executor.calls[0][0]
    assert mentions_executor.calls[0][1]['scope_group_id'] == 'group-b'
    assert ':scope_group_id IS NULL' in mentions_executor.calls[0][0]


@pytest.mark.asyncio
async def test_scoped_distance_reranker_drops_foreign_center() -> None:
    executor = RecordingExecutor([])
    executor._scope_group_id = 'group-b'

    uuids, scores = await OracleSearchOperations().node_distance_reranker(
        executor,
        ['foreign-center'],
        'foreign-center',
    )

    assert (uuids, scores) == ([], [])
    assert executor.calls[0][1]['scope_group_id'] == 'group-b'


@pytest.mark.asyncio
async def test_scoped_mentions_reranker_drops_foreign_zero_score_candidate() -> None:
    executor = RecordingExecutor([])
    executor._scope_group_id = 'group-b'

    uuids, scores = await OracleSearchOperations().episode_mentions_reranker(
        executor,
        ['foreign-entity'],
    )

    assert (uuids, scores) == ([], [])
    assert executor.calls[0][1]['scope_group_id'] == 'group-b'


@pytest.mark.asyncio
async def test_scoped_rerankers_keep_existing_disconnected_entities() -> None:
    distance_executor = RecordingExecutor(
        responses=[
            [{'uuid': 'center'}, {'uuid': 'disconnected'}],
            [],
        ]
    )
    distance_executor._scope_group_id = 'group-b'
    mentions_executor = RecordingExecutor(
        responses=[
            [{'uuid': 'disconnected'}],
            [],
        ]
    )
    mentions_executor._scope_group_id = 'group-b'
    operations = OracleSearchOperations()

    distance = await operations.node_distance_reranker(
        distance_executor,
        ['center', 'disconnected'],
        'center',
    )
    mentions = await operations.episode_mentions_reranker(
        mentions_executor,
        ['disconnected'],
    )

    assert distance == (['center', 'disconnected'], [0.1, 0.0])
    assert mentions == (['disconnected'], [0.0])
