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

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.driver.drevo_search_interface import DrevoSearchInterface
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters


def _drevo_driver_stub(execute_query: AsyncMock):
    """A minimal stand-in exposing just what the interface touches."""
    return SimpleNamespace(provider=GraphProvider.DREVO, execute_query=execute_query)


_UNSUPPORTED_COSINE = Exception('unsupported Cypher feature `function call `cosine_similarity``')
_UNSUPPORTED_FTS = Exception('unsupported procedure `fts.search`')


@pytest.mark.asyncio
class TestNodeSimilaritySearch:
    async def test_uses_native_cosine_when_supported(self):
        # Native path: drevo ranks server-side, so the DB returns rows already
        # ordered; the interface preserves that order and threads the params.
        records = [{'uuid': 'b'}, {'uuid': 'a'}]
        execute_query = AsyncMock(return_value=(records, None, None))
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
        assert execute_query.await_count == 1
        query = execute_query.await_args.args[0]
        assert 'cosine_similarity(n.name_embedding, $search_vector)' in query
        assert 'ORDER BY score DESC' in query
        kwargs = execute_query.await_args.kwargs
        assert kwargs['search_vector'] == [1.0, 0.0]
        assert kwargs['min_score'] == 0.5
        assert kwargs['limit'] == 5

    async def test_falls_back_to_library_side_with_warning(self, caplog):
        # Old drevo rejects cosine_similarity -> one warning, then library-side.
        fallback_records = [
            {'uuid': 'a', '_search_embedding': [0.0, 1.0]},
            {'uuid': 'b', '_search_embedding': [1.0, 0.0]},
        ]
        execute_query = AsyncMock(side_effect=[_UNSUPPORTED_COSINE, (fallback_records, None, None)])
        driver = _drevo_driver_stub(execute_query)

        with (
            patch(
                'graphiti_core.driver.drevo_search_interface.get_entity_node_from_record',
                side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = await DrevoSearchInterface().node_similarity_search(
                driver,
                search_vector=[1.0, 0.0],
                search_filter=SearchFilters(),
                limit=5,
                min_score=-1.0,
            )

        assert [n.uuid for n in result] == ['b', 'a']  # ranked library-side
        assert execute_query.await_count == 2
        fallback_query = execute_query.await_args_list[1].args[0]
        assert 'n.name_embedding AS _search_embedding' in fallback_query
        assert 'cosine_similarity' not in fallback_query
        assert 'cosine_similarity' in caplog.text and 'library-side' in caplog.text

    async def test_warns_only_once_then_stays_on_fallback(self, caplog):
        fallback_records = [{'uuid': 'a', '_search_embedding': [1.0, 0.0]}]
        # 1st call: native fails then fallback; 2nd call: fallback only (native skipped).
        execute_query = AsyncMock(
            side_effect=[
                _UNSUPPORTED_COSINE,
                (fallback_records, None, None),
                (fallback_records, None, None),
            ]
        )
        driver = _drevo_driver_stub(execute_query)
        interface = DrevoSearchInterface()

        with (
            patch(
                'graphiti_core.driver.drevo_search_interface.get_entity_node_from_record',
                side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
            ),
            caplog.at_level(logging.WARNING),
        ):
            await interface.node_similarity_search(
                driver, search_vector=[1.0, 0.0], search_filter=SearchFilters(), min_score=-1.0
            )
            await interface.node_similarity_search(
                driver, search_vector=[1.0, 0.0], search_filter=SearchFilters(), min_score=-1.0
            )

        # 2 (fail+fallback) + 1 (fallback only, native skipped) = 3 calls; 1 warning.
        assert execute_query.await_count == 3
        assert caplog.text.count('falling back to slower') == 1

    async def test_non_cosine_error_propagates(self):
        execute_query = AsyncMock(side_effect=ValueError('some other db error'))
        driver = _drevo_driver_stub(execute_query)

        with pytest.raises(ValueError):
            await DrevoSearchInterface().node_similarity_search(
                driver, search_vector=[1.0, 0.0], search_filter=SearchFilters()
            )
        assert execute_query.await_count == 1  # no fallback attempted

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

        query = execute_query.await_args.args[0]  # native query
        assert 'n.group_id IN $group_ids' in query
        assert execute_query.await_args.kwargs['group_ids'] == ['g1']


@pytest.mark.asyncio
class TestEdgeSimilaritySearch:
    async def test_uses_native_cosine_on_fact_embedding(self):
        records = [{'uuid': 'e1'}, {'uuid': 'e2'}]
        execute_query = AsyncMock(return_value=(records, None, None))
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

    async def test_edge_falls_back_library_side(self):
        fallback_records = [{'uuid': 'e1', '_search_embedding': [1.0, 0.0]}]
        execute_query = AsyncMock(side_effect=[_UNSUPPORTED_COSINE, (fallback_records, None, None)])
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
                min_score=-1.0,
            )

        assert [e.uuid for e in result] == ['e1']
        fallback_query = execute_query.await_args_list[1].args[0]
        assert 'e.fact_embedding AS _search_embedding' in fallback_query

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

        query = execute_query.await_args.args[0]  # native query
        kwargs = execute_query.await_args.kwargs
        assert 'n.uuid = $source_uuid' in query
        assert 'm.uuid = $target_uuid' in query
        assert kwargs['source_uuid'] == 'src'
        assert kwargs['target_uuid'] == 'tgt'


@pytest.mark.asyncio
class TestNodeFulltextSearch:
    async def test_uses_native_fts_search(self):
        # Native BM25: drevo returns rows already ranked (server-side ORDER BY).
        records = [{'uuid': 'x'}, {'uuid': 'y'}]
        execute_query = AsyncMock(return_value=(records, None, None))
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_entity_node_from_record',
            side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().node_fulltext_search(
                driver,
                query='alpha beta',
                search_filter=SearchFilters(),
                group_ids=['g1'],
                limit=10,
            )

        assert [n.uuid for n in result] == ['x', 'y']
        cypher = execute_query.await_args.args[0]
        kwargs = execute_query.await_args.kwargs
        assert 'CALL fts.search($fts_query, $fts_k)' in cypher
        assert "'Entity' IN labels(n)" in cypher
        assert 'n.group_id IN $group_ids' in cypher
        assert 'ORDER BY score DESC' in cypher
        assert kwargs['fts_query'] == 'alpha beta'
        assert kwargs['group_ids'] == ['g1']
        assert 'CONTAINS' not in cypher

    async def test_falls_back_to_lexical_with_warning(self, caplog):
        lexical_records = [
            {'uuid': 'x', 'fulltext_name': 'alpha beta', 'fulltext_summary': ''},
            {'uuid': 'y', 'fulltext_name': 'alpha only', 'fulltext_summary': ''},
        ]
        execute_query = AsyncMock(side_effect=[_UNSUPPORTED_FTS, (lexical_records, None, None)])
        driver = _drevo_driver_stub(execute_query)

        with (
            patch(
                'graphiti_core.driver.drevo_search_interface.get_entity_node_from_record',
                side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = await DrevoSearchInterface().node_fulltext_search(
                driver, query='alpha beta', search_filter=SearchFilters(), limit=10
            )

        assert [n.uuid for n in result] == ['x', 'y']  # lexical term-overlap order
        assert execute_query.await_count == 2
        fallback_cypher = execute_query.await_args_list[1].args[0]
        assert 'toLower(n.name) CONTAINS $term_0' in fallback_cypher
        assert 'fts.search' in caplog.text and 'lexical' in caplog.text

    async def test_empty_query_short_circuits(self):
        execute_query = AsyncMock(return_value=([], None, None))
        driver = _drevo_driver_stub(execute_query)

        result = await DrevoSearchInterface().node_fulltext_search(
            driver, query='   ', search_filter=SearchFilters(), limit=10
        )
        assert result == []
        execute_query.assert_not_awaited()


@pytest.mark.asyncio
class TestEdgeFulltextSearch:
    async def test_matches_name_and_fact(self):
        records = [{'uuid': 'e1', 'fulltext_name': 'rel', 'fulltext_fact': 'alpha beta fact'}]
        execute_query = AsyncMock(return_value=(records, None, None))
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_entity_edge_from_record',
            side_effect=lambda record, provider: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().edge_fulltext_search(
                driver, query='alpha beta', search_filter=SearchFilters(), limit=10
            )

        assert [e.uuid for e in result] == ['e1']
        cypher = execute_query.await_args.args[0]
        assert 'toLower(e.name) CONTAINS $term_0' in cypher
        assert 'toLower(e.fact) CONTAINS $term_0' in cypher


@pytest.mark.asyncio
class TestEpisodeFulltextSearch:
    async def test_uses_native_fts_search(self):
        records = [{'uuid': 'ep1'}]
        execute_query = AsyncMock(return_value=(records, None, None))
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_episodic_node_from_record',
            side_effect=lambda record: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().episode_fulltext_search(
                driver,
                query='alpha beta',
                search_filter=SearchFilters(),
                group_ids=['g1'],
                limit=10,
            )

        assert [n.uuid for n in result] == ['ep1']
        cypher = execute_query.await_args.args[0]
        assert 'CALL fts.search($fts_query, $fts_k)' in cypher
        assert "'Episodic' IN labels(e)" in cypher
        assert 'e.group_id IN $group_ids' in cypher

    async def test_falls_back_to_lexical(self):
        lexical_records = [
            {
                'uuid': 'ep1',
                'fulltext_content': 'alpha beta',
                'fulltext_source': 'x',
                'fulltext_source_description': '',
            },
        ]
        execute_query = AsyncMock(side_effect=[_UNSUPPORTED_FTS, (lexical_records, None, None)])
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_episodic_node_from_record',
            side_effect=lambda record: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().episode_fulltext_search(
                driver, query='alpha beta', search_filter=SearchFilters(), limit=10
            )

        assert [n.uuid for n in result] == ['ep1']
        fallback_cypher = execute_query.await_args_list[1].args[0]
        assert 'toLower(e.content) CONTAINS $term_0' in fallback_cypher


@pytest.mark.asyncio
class TestCommunityFulltextSearch:
    async def test_uses_native_fts_search(self):
        records = [{'uuid': 'c1'}]
        execute_query = AsyncMock(return_value=(records, None, None))
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_community_node_from_record',
            side_effect=lambda record: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().community_fulltext_search(
                driver, query='alpha', group_ids=['g1'], limit=10
            )

        assert [c.uuid for c in result] == ['c1']
        cypher = execute_query.await_args.args[0]
        assert 'CALL fts.search($fts_query, $fts_k)' in cypher
        assert "'Community' IN labels(c)" in cypher
        assert 'c.group_id IN $group_ids' in cypher


@pytest.mark.asyncio
class TestCommunitySimilaritySearch:
    async def test_uses_native_cosine_on_name_embedding(self):
        records = [{'uuid': 'c1'}, {'uuid': 'c2'}]
        execute_query = AsyncMock(return_value=(records, None, None))
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_community_node_from_record',
            side_effect=lambda record: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().community_similarity_search(
                driver, search_vector=[1.0, 0.0], limit=10, min_score=0.4
            )

        assert [c.uuid for c in result] == ['c1', 'c2']
        cypher = execute_query.await_args.args[0]
        assert 'cosine_similarity(c.name_embedding, $search_vector)' in cypher
        assert 'ORDER BY score DESC' in cypher

    async def test_community_falls_back_library_side(self):
        fallback_records = [{'uuid': 'c1', 'name_embedding': [1.0, 0.0]}]
        execute_query = AsyncMock(side_effect=[_UNSUPPORTED_COSINE, (fallback_records, None, None)])
        driver = _drevo_driver_stub(execute_query)

        with patch(
            'graphiti_core.driver.drevo_search_interface.get_community_node_from_record',
            side_effect=lambda record: SimpleNamespace(uuid=record['uuid']),
        ):
            result = await DrevoSearchInterface().community_similarity_search(
                driver, search_vector=[1.0, 0.0], min_score=-1.0
            )

        assert [c.uuid for c in result] == ['c1']
        assert execute_query.await_count == 2
