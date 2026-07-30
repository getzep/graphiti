"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.driver.drevo_search_interface import (
    DrevoSearchInterface,
    rank_uuids_by_cosine,
)
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters


class TestRankUuidsByCosine:
    """The library-side ranker drevo relies on until drevo#202 lands a native
    cosine scalar usable in ORDER BY."""

    def test_orders_by_similarity_descending(self):
        query = [1.0, 0.0, 0.0]
        rows = [
            {'uuid': 'far', 'embedding': [0.0, 1.0, 0.0]},  # orthogonal -> ~0.5 normalized
            {'uuid': 'near', 'embedding': [1.0, 0.0, 0.0]},  # identical -> 1.0
            {'uuid': 'mid', 'embedding': [1.0, 1.0, 0.0]},
        ]
        ranked = rank_uuids_by_cosine(rows, query, min_score=-1.0, limit=10)
        assert [uuid for uuid, _ in ranked] == ['near', 'mid', 'far']
        assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]

    def test_min_score_filters(self):
        query = [1.0, 0.0, 0.0]
        rows = [
            {'uuid': 'near', 'embedding': [1.0, 0.0, 0.0]},
            {'uuid': 'far', 'embedding': [-1.0, 0.0, 0.0]},
        ]
        ranked = rank_uuids_by_cosine(rows, query, min_score=0.9, limit=10)
        assert [uuid for uuid, _ in ranked] == ['near']

    def test_limit_truncates(self):
        query = [1.0, 0.0]
        rows = [{'uuid': str(i), 'embedding': [1.0, 0.0]} for i in range(5)]
        ranked = rank_uuids_by_cosine(rows, query, min_score=0.0, limit=2)
        assert len(ranked) == 2

    def test_skips_rows_without_embedding_or_uuid(self):
        query = [1.0, 0.0]
        rows = [
            {'uuid': 'ok', 'embedding': [1.0, 0.0]},
            {'uuid': 'no_emb', 'embedding': None},
            {'embedding': [1.0, 0.0]},  # missing uuid
        ]
        ranked = rank_uuids_by_cosine(rows, query, min_score=0.0, limit=10)
        assert [uuid for uuid, _ in ranked] == ['ok']

    def test_empty(self):
        assert rank_uuids_by_cosine([], [1.0, 0.0], min_score=0.0, limit=10) == []


def _drevo_driver_stub(execute_query: AsyncMock):
    """A minimal stand-in exposing just what the interface touches."""
    return SimpleNamespace(provider=GraphProvider.DREVO, execute_query=execute_query)


@pytest.mark.asyncio
class TestNodeSimilaritySearch:
    async def test_ranks_then_fetches_in_ranked_order(self):
        # First call: candidate uuids+embeddings. Second call: full node records.
        candidates = [
            {'uuid': 'a', 'embedding': [0.0, 1.0]},
            {'uuid': 'b', 'embedding': [1.0, 0.0]},
        ]
        node_records = [{'uuid': 'a'}, {'uuid': 'b'}]
        execute_query = AsyncMock(
            side_effect=[(candidates, None, None), (node_records, None, None)]
        )
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_entity_node_from_record',
            side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().node_similarity_search(
                driver,
                search_vector=[1.0, 0.0],
                search_filter=SearchFilters(),
                limit=10,
                min_score=-1.0,
            )

        # 'b' is identical to the query vector, so it must rank first.
        assert [n.uuid for n in result] == ['b', 'a']
        # The fetch step must request exactly the ranked uuids.
        assert execute_query.await_args_list[1].kwargs['uuids'] == ['b', 'a']

    async def test_returns_empty_without_second_query_when_nothing_ranks(self):
        candidates = [{'uuid': 'a', 'embedding': [-1.0, 0.0]}]
        execute_query = AsyncMock(side_effect=[(candidates, None, None)])
        driver = _drevo_driver_stub(execute_query)

        result = await DrevoSearchInterface().node_similarity_search(
            driver,
            search_vector=[1.0, 0.0],
            search_filter=SearchFilters(),
            limit=10,
            min_score=0.9,
        )
        assert result == []
        assert execute_query.await_count == 1  # no fetch query issued


@pytest.mark.asyncio
class TestEdgeSimilaritySearch:
    async def test_ranks_edges_library_side(self):
        candidates = [
            {'uuid': 'e1', 'embedding': [1.0, 0.0]},
            {'uuid': 'e2', 'embedding': [0.0, 1.0]},
        ]
        edge_records = [{'uuid': 'e1'}, {'uuid': 'e2'}]
        execute_query = AsyncMock(
            side_effect=[(candidates, None, None), (edge_records, None, None)]
        )
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_entity_edge_from_record',
            side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().edge_similarity_search(
                driver,
                search_vector=[1.0, 0.0],
                source_node_uuid=None,
                target_node_uuid=None,
                search_filter=SearchFilters(),
                limit=10,
                min_score=-1.0,
            )

        assert [e.uuid for e in result] == ['e1', 'e2']
