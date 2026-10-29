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

from array import array
from datetime import UTC, datetime

import pytest

from graphiti_core.driver.oracle.codecs import (
    community_node_to_binds,
    edge_to_binds,
    entity_node_to_binds,
)
from graphiti_core.driver.oracle.errors import InvalidEmbeddingDimension
from graphiti_core.driver.oracle.operations.search import OracleSearchOperations
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.search.search_filters import SearchFilters
from tests.driver.oracle.operations.recording import RecordingExecutor

NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)
VECTOR = [1.0] + [0.0] * 1023


def entity() -> EntityNode:
    return EntityNode(
        uuid='entity-a',
        name='Alpha',
        group_id='group-a',
        labels=['Concept'],
        summary='neutral vector fixture',
        name_embedding=VECTOR,
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_node_similarity_uses_cosine_threshold_and_uuid_tie_break() -> None:
    executor = RecordingExecutor([{**entity_node_to_binds(entity()), 'similarity': 1.0}])

    results = await OracleSearchOperations().node_similarity_search(
        executor,
        VECTOR,
        SearchFilters(),
        group_ids=['group-a'],
        limit=5,
        min_score=0.75,
    )

    statement, binds = executor.calls[0]
    assert [result.uuid for result in results] == ['entity-a']
    assert 'VECTOR_DISTANCE(n.name_embedding, :search_vector, COSINE)' in statement
    assert 'n.name_embedding IS NOT NULL' in statement
    assert 'ORDER BY VECTOR_DISTANCE' in statement
    assert 'n.uuid' in statement
    assert isinstance(binds['search_vector'], array)
    assert binds['search_vector'].typecode == 'f'
    assert list(binds['search_vector']) == VECTOR
    assert binds['min_score'] == 0.75
    assert binds['result_limit'] == 5


@pytest.mark.asyncio
async def test_invalid_vector_is_rejected_before_executor_access() -> None:
    executor = RecordingExecutor()

    with pytest.raises(InvalidEmbeddingDimension, match='expected 1024, received 3'):
        await OracleSearchOperations().node_similarity_search(
            executor, [1.0, 0.0, 0.0], SearchFilters()
        )

    assert executor.calls == []


@pytest.mark.asyncio
async def test_edge_similarity_always_applies_endpoint_filters() -> None:
    edge = EntityEdge(
        uuid='edge-a',
        source_node_uuid='entity-a',
        target_node_uuid='entity-b',
        group_id='group-a',
        name='RELATES_TO',
        fact='Alpha relates to Beta',
        fact_embedding=VECTOR,
        created_at=NOW,
    )
    executor = RecordingExecutor([{**edge_to_binds(edge, kind='RELATES_TO'), 'similarity': 1.0}])

    results = await OracleSearchOperations().edge_similarity_search(
        executor,
        VECTOR,
        source_node_uuid='entity-a',
        target_node_uuid='entity-b',
        search_filter=SearchFilters(),
        group_ids=None,
    )

    statement, binds = executor.calls[0]
    assert [result.uuid for result in results] == ['edge-a']
    assert 'e.source_uuid = :source_node_uuid' in statement
    assert 'e.target_uuid = :target_node_uuid' in statement
    assert binds['source_node_uuid'] == 'entity-a'
    assert binds['target_node_uuid'] == 'entity-b'


@pytest.mark.asyncio
async def test_community_similarity_decodes_non_null_vector_record() -> None:
    community = CommunityNode(
        uuid='community-a',
        name='Neutral cluster',
        group_id='group-a',
        labels=[],
        summary='Alpha and Beta',
        name_embedding=VECTOR,
        created_at=NOW,
    )
    executor = RecordingExecutor([{**community_node_to_binds(community), 'similarity': 0.9}])

    results = await OracleSearchOperations().community_similarity_search(
        executor, VECTOR, group_ids=['group-a'], min_score=0.8
    )

    assert [result.uuid for result in results] == ['community-a']
    assert "n.kind = 'COMMUNITY'" in executor.calls[0][0]


@pytest.mark.asyncio
async def test_edge_similarity_combines_scalar_and_list_endpoint_filters() -> None:
    executor = RecordingExecutor()

    await OracleSearchOperations().edge_similarity_search(
        executor,
        VECTOR,
        source_node_uuid='entity-a',
        target_node_uuid=None,
        search_filter=SearchFilters(
            edge_source_node_uuids=['entity-a', 'entity-c'],
            edge_target_node_uuids=['entity-b'],
        ),
        group_ids=['group-a'],
    )

    statement, binds = executor.calls[0]
    assert 'e.source_uuid = :source_node_uuid' in statement
    assert 'src.uuid' in statement
    assert 'dst.uuid' in statement
    assert binds['source_node_uuid'] == 'entity-a'
    assert binds['edge_source_node_uuids_json'] == '["entity-a","entity-c"]'
    assert binds['edge_target_node_uuids_json'] == '["entity-b"]'
