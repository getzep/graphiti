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

from graphiti_core.driver.drevo_search_interface import DrevoSearchInterface
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters


def _drevo_driver_stub(execute_query: AsyncMock):
    """A minimal stand-in exposing just what the interface touches."""
    return SimpleNamespace(provider=GraphProvider.DREVO, execute_query=execute_query)


@pytest.mark.asyncio
class TestNodeSimilaritySearch:
    async def test_uses_native_cosine_scalar_server_side(self):
        # Server-side ORDER BY means the DB returns rows already ranked; the
        # interface must preserve that order and pass the ranking params through.
        node_records = [{'uuid': 'b'}, {'uuid': 'a'}]
        execute_query = AsyncMock(return_value=(node_records, None, None))
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_entity_node_from_record',
            side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().node_similarity_search(
                driver,
                search_vector=[1.0, 0.0],
                search_filter=SearchFilters(),
                limit=5,
                min_score=0.5,
            )

        assert [n.uuid for n in result] == ['b', 'a']

        query = execute_query.await_args.args[0]
        assert 'cosine_similarity(n.name_embedding, $search_vector)' in query
        assert 'ORDER BY score DESC' in query
        assert 'n.name_embedding IS NOT NULL' in query

        kwargs = execute_query.await_args.kwargs
        assert kwargs['search_vector'] == [1.0, 0.0]
        assert kwargs['min_score'] == 0.5
        assert kwargs['limit'] == 5

    async def test_group_ids_filter_is_applied(self):
        execute_query = AsyncMock(return_value=([], None, None))
        driver = _drevo_driver_stub(execute_query)

        await DrevoSearchInterface().node_similarity_search(
            driver,
            search_vector=[1.0, 0.0],
            search_filter=SearchFilters(),
            group_ids=['g1'],
            limit=5,
            min_score=0.5,
        )

        query = execute_query.await_args.args[0]
        assert 'n.group_id IN $group_ids' in query
        assert execute_query.await_args.kwargs['group_ids'] == ['g1']


@pytest.mark.asyncio
class TestEdgeSimilaritySearch:
    async def test_uses_native_cosine_scalar_on_fact_embedding(self):
        edge_records = [{'uuid': 'e1'}, {'uuid': 'e2'}]
        execute_query = AsyncMock(return_value=(edge_records, None, None))
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
                min_score=0.3,
            )

        assert [e.uuid for e in result] == ['e1', 'e2']

        query = execute_query.await_args.args[0]
        assert 'cosine_similarity(e.fact_embedding, $search_vector)' in query
        assert 'ORDER BY score DESC' in query

    async def test_source_and_target_filters_applied(self):
        execute_query = AsyncMock(return_value=([], None, None))
        driver = _drevo_driver_stub(execute_query)

        await DrevoSearchInterface().edge_similarity_search(
            driver,
            search_vector=[1.0, 0.0],
            source_node_uuid='src',
            target_node_uuid='tgt',
            search_filter=SearchFilters(),
            limit=10,
            min_score=0.3,
        )

        query = execute_query.await_args.args[0]
        kwargs = execute_query.await_args.kwargs
        assert 'n.uuid = $source_uuid' in query
        assert 'm.uuid = $target_uuid' in query
        assert kwargs['source_uuid'] == 'src'
        assert kwargs['target_uuid'] == 'tgt'
