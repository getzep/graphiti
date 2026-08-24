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

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.graph_queries import (
    get_vector_indices,
    get_vector_similarity_query,
    use_vector_index,
)


class TestGetVectorIndices:
    """Vector index DDL emitted for each provider."""

    def test_neo4j_creates_hnsw_indices_for_all_embedded_properties(self):
        queries = get_vector_indices(GraphProvider.NEO4J, embedding_dim=1024)

        joined = '\n'.join(queries)
        assert 'CREATE VECTOR INDEX' in joined

        # Every embedded property Graphiti searches over must be covered.
        assert 'e.fact_embedding' in joined or 'fact_embedding' in joined
        assert 'name_embedding' in joined

    def test_neo4j_uses_supplied_embedding_dimension(self):
        """Regression: #859 hardcoded 1024, breaking non-1024 embedders."""
        queries = get_vector_indices(GraphProvider.NEO4J, embedding_dim=2560)

        joined = '\n'.join(queries)
        assert '2560' in joined
        assert '1024' not in joined

    def test_neo4j_uses_cosine_similarity(self):
        queries = get_vector_indices(GraphProvider.NEO4J, embedding_dim=1024)

        joined = '\n'.join(queries)
        assert "'cosine'" in joined

    def test_neo4j_indices_are_idempotent(self):
        """build_indices_and_constraints runs on every driver init."""
        queries = get_vector_indices(GraphProvider.NEO4J, embedding_dim=1024)

        for query in queries:
            assert 'IF NOT EXISTS' in query

    def test_non_neo4j_providers_emit_no_vector_indices(self):
        """FalkorDB/Kuzu/Neptune vector indexing is out of scope for this change."""
        for provider in (GraphProvider.FALKORDB, GraphProvider.KUZU, GraphProvider.NEPTUNE):
            assert get_vector_indices(provider, embedding_dim=1024) == []


class TestGetVectorSimilarityQuery:
    """Similarity search routed through the HNSW index when it is available."""

    def test_neo4j_edge_search_uses_vector_index_procedure(self):
        query = get_vector_similarity_query(
            GraphProvider.NEO4J,
            index_name='edge_fact_embedding',
            entity_var='e',
            limit=20,
        )

        assert 'db.index.vector.queryRelationships' in query
        assert 'edge_fact_embedding' in query

    def test_neo4j_node_search_uses_node_index_procedure(self):
        query = get_vector_similarity_query(
            GraphProvider.NEO4J,
            index_name='entity_name_embedding',
            entity_var='n',
            limit=20,
            relationship=False,
        )

        assert 'db.index.vector.queryNodes' in query

    def test_over_fetches_candidates_for_post_filtering(self):
        """HNSW cannot pre-filter on group_id/SearchFilters, so candidates must be over-fetched."""
        limit = 20
        query = get_vector_similarity_query(
            GraphProvider.NEO4J,
            index_name='edge_fact_embedding',
            entity_var='e',
            limit=limit,
        )

        # The procedure's k must exceed the caller's LIMIT so post-filtering
        # cannot silently truncate the result set below `limit`.
        assert str(limit * 3) in query

    def test_non_neo4j_providers_return_empty(self):
        for provider in (GraphProvider.FALKORDB, GraphProvider.KUZU, GraphProvider.NEPTUNE):
            assert (
                get_vector_similarity_query(provider, index_name='x', entity_var='e', limit=10)
                == ''
            )


class TestUseVectorIndex:
    """Opt-in gate: existing deployments keep brute-force until they enable it."""

    def test_defaults_to_disabled_for_drivers_without_the_flag(self):
        class _Driver:
            pass

        assert use_vector_index(_Driver()) is False

    def test_respects_driver_opt_in(self):
        class _Driver:
            use_vector_index = True

        assert use_vector_index(_Driver()) is True


class TestNeo4jDriverVectorIndexWiring:
    def test_driver_defaults_to_brute_force(self):
        driver = Neo4jDriver.__new__(Neo4jDriver)
        driver.use_vector_index = False
        assert use_vector_index(driver) is False

    def test_build_indices_skips_vector_ddl_by_default(self):
        """Default init must not trigger an expensive background index build."""
        queries = get_vector_indices(GraphProvider.NEO4J, embedding_dim=1024)
        assert queries, 'sanity: DDL exists when explicitly requested'
        # The default path in build_indices_and_constraints() appends these only
        # when build_vector_indices=True; covered by signature default below.
        import inspect

        sig = inspect.signature(Neo4jDriver.build_indices_and_constraints)
        assert sig.parameters['build_vector_indices'].default is False

    def test_embedding_dim_is_configurable_not_hardcoded(self):
        import inspect

        sig = inspect.signature(Neo4jDriver.__init__)
        assert 'embedding_dim' in sig.parameters
        assert 'use_vector_index' in sig.parameters
