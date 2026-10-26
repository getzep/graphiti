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

from graphiti_core.driver.oracle.codecs import (
    community_node_to_binds,
    edge_to_binds,
    entity_node_to_binds,
    episode_node_to_binds,
)
from graphiti_core.driver.oracle.operations.search import OracleSearchOperations
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from tests.driver.oracle.operations.recording import RecordingExecutor

NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)


def entity() -> EntityNode:
    return EntityNode(
        uuid='entity-a',
        name='Alpha',
        group_id='group-a',
        labels=['Concept'],
        summary='neutral search fixture',
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_node_fulltext_uses_bound_oracle_text_and_stable_ranking() -> None:
    executor = RecordingExecutor([{**entity_node_to_binds(entity()), 'search_score': 12.0}])

    results = await OracleSearchOperations().node_fulltext_search(
        executor,
        'Alpha / beta!',
        SearchFilters(node_labels=['Concept']),
        group_ids=['group-a'],
        limit=7,
    )

    statement, binds = executor.calls[0]
    assert results[0].uuid == 'entity-a'
    assert 'CONTAINS(n.search_text, :text_query, 1) > 0' in statement
    assert 'SCORE(1) DESC, n.uuid' in statement
    assert 'Alpha' not in statement
    assert binds['text_query'] == '{Alpha} ACCUM {beta}'
    assert binds['result_limit'] == 7
    assert binds['group_ids_json'] == '["group-a"]'


@pytest.mark.asyncio
async def test_edge_episode_and_community_text_search_decode_records() -> None:
    edge = EntityEdge(
        uuid='edge-a',
        source_node_uuid='entity-a',
        target_node_uuid='entity-b',
        group_id='group-a',
        name='RELATES_TO',
        fact='Alpha relates to Beta',
        created_at=NOW,
    )
    episode = EpisodicNode(
        uuid='episode-a',
        name='Neutral event',
        group_id='group-a',
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic',
        content='{"event":"created"}',
        created_at=NOW,
        valid_at=NOW,
    )
    community = CommunityNode(
        uuid='community-a',
        name='Neutral cluster',
        group_id='group-a',
        labels=[],
        summary='Alpha and Beta',
        created_at=NOW,
    )
    operations = OracleSearchOperations()

    edge_executor = RecordingExecutor([edge_to_binds(edge, kind='RELATES_TO')])
    episode_executor = RecordingExecutor([episode_node_to_binds(episode)])
    community_executor = RecordingExecutor([community_node_to_binds(community)])

    edge_results = await operations.edge_fulltext_search(
        edge_executor, 'relates', SearchFilters(), group_ids=['group-a']
    )
    episode_results = await operations.episode_fulltext_search(
        episode_executor, 'created', SearchFilters(), group_ids=['group-a']
    )
    community_results = await operations.community_fulltext_search(
        community_executor, 'cluster', group_ids=['group-a']
    )

    assert [result.uuid for result in edge_results] == ['edge-a']
    assert [result.uuid for result in episode_results] == ['episode-a']
    assert [result.uuid for result in community_results] == ['community-a']
    assert 'CONTAINS(e.search_text' in edge_executor.calls[0][0]
    assert 'CONTAINS(n.search_text' in episode_executor.calls[0][0]
    assert 'CONTAINS(n.search_text' in community_executor.calls[0][0]


@pytest.mark.asyncio
async def test_empty_group_scope_short_circuits_before_executor() -> None:
    executor = RecordingExecutor()

    results = await OracleSearchOperations().node_fulltext_search(
        executor, 'alpha', SearchFilters(), group_ids=[]
    )

    assert results == []
    assert executor.calls == []


@pytest.mark.asyncio
async def test_scoped_search_without_explicit_groups_binds_request_group() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    await OracleSearchOperations().node_fulltext_search(
        executor,
        'alpha',
        SearchFilters(),
        group_ids=None,
    )

    assert executor.calls[0][1]['group_ids_json'] == '["group-b"]'


@pytest.mark.asyncio
async def test_scoped_search_intersects_explicit_group_list() -> None:
    executor = RecordingExecutor()
    executor._scope_group_id = 'group-b'

    result = await OracleSearchOperations().node_fulltext_search(
        executor,
        'alpha',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert result == []
    assert executor.calls == []


@pytest.mark.asyncio
async def test_edge_fulltext_pushes_down_directed_endpoint_filters() -> None:
    executor = RecordingExecutor()

    await OracleSearchOperations().edge_fulltext_search(
        executor,
        'relates',
        SearchFilters(
            edge_source_node_uuids=['entity-a'],
            edge_target_node_uuids=['entity-b'],
        ),
        group_ids=['group-a'],
    )

    statement, binds = executor.calls[0]
    assert 'src.uuid' in statement
    assert 'dst.uuid' in statement
    assert binds['edge_source_node_uuids_json'] == '["entity-a"]'
    assert binds['edge_target_node_uuids_json'] == '["entity-b"]'
    assert binds['group_ids_json'] == '["group-a"]'
