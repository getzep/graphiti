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

from graphiti_core.driver.oracle.codecs import edge_to_binds, entity_node_to_binds
from graphiti_core.driver.oracle.operations.search import OracleSearchOperations
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters
from tests.driver.oracle.operations.recording import RecordingExecutor

NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)


def entity(uuid: str) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=uuid,
        group_id='group-a',
        labels=['Concept'],
        summary='bounded traversal fixture',
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_node_bfs_is_bounded_deduplicated_and_group_scoped() -> None:
    executor = RecordingExecutor([entity_node_to_binds(entity('entity-b'))])

    results = await OracleSearchOperations().node_bfs_search(
        executor,
        ['entity-a', 'entity-a'],
        SearchFilters(),
        max_depth=3,
        group_ids=['group-a'],
        limit=4,
    )

    statement, binds = executor.calls[0]
    assert [result.uuid for result in results] == ['entity-b']
    assert 'GRAPH_TABLE' in statement
    assert '{1,3}' in statement
    assert 'SELECT DISTINCT' in statement
    assert "path_edge.kind IN ('RELATES_TO', 'MENTIONS')" in statement
    assert 'ORDER BY n.uuid' in statement
    assert binds['origin_uuids_json'] == '["entity-a"]'
    assert binds['group_ids_json'] == '["group-a"]'
    assert binds['result_limit'] == 4


@pytest.mark.asyncio
async def test_edge_bfs_emits_each_entity_edge_once() -> None:
    edge = EntityEdge(
        uuid='edge-a',
        source_node_uuid='entity-a',
        target_node_uuid='entity-b',
        group_id='group-a',
        name='RELATES_TO',
        fact='Alpha relates to Beta',
        created_at=NOW,
    )
    executor = RecordingExecutor([edge_to_binds(edge, kind='RELATES_TO')])

    results = await OracleSearchOperations().edge_bfs_search(
        executor,
        ['entity-a'],
        max_depth=2,
        search_filter=SearchFilters(),
        group_ids=['group-a'],
    )

    statement, _ = executor.calls[0]
    assert [result.uuid for result in results] == ['edge-a']
    assert 'ONE ROW PER STEP' in statement
    assert 'step_edge.uuid AS edge_uuid' in statement
    assert "reachable.edge_kind = 'RELATES_TO'" in statement


@pytest.mark.asyncio
async def test_empty_bfs_origins_short_circuit() -> None:
    executor = RecordingExecutor()

    results = await OracleSearchOperations().node_bfs_search(
        executor, [], SearchFilters(), max_depth=2
    )

    assert results == []
    assert executor.calls == []
