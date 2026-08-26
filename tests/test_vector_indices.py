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

from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.graph_queries import (
    get_vector_indices,
    get_vector_similarity_query,
    use_vector_index,
)


class TestGetVectorIndices:
    """Vector index DDL emitted for each provider."""

    def test_neo4j_vector_ddl_is_exact_and_stably_named(self):
        assert get_vector_indices(GraphProvider.NEO4J, embedding_dim=1024) == [
            'CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS FOR (n:Entity) '
            'ON (n.name_embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1024, '
            "`vector.similarity_function`: 'cosine'}}",
            'CREATE VECTOR INDEX community_name_embedding IF NOT EXISTS FOR (n:Community) '
            'ON (n.name_embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1024, '
            "`vector.similarity_function`: 'cosine'}}",
            'CREATE VECTOR INDEX edge_fact_embedding IF NOT EXISTS FOR ()-[e:RELATES_TO]-() '
            'ON (e.fact_embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1024, '
            "`vector.similarity_function`: 'cosine'}}",
        ]

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

    def test_uses_public_limit_without_claiming_post_filter_guarantees(self):
        """Filtered searches use exact cosine; this helper is only for unfiltered HNSW."""
        limit = 20
        query = get_vector_similarity_query(
            GraphProvider.NEO4J,
            index_name='edge_fact_embedding',
            entity_var='e',
            limit=limit,
        )

        assert f', {limit}, $search_vector' in query

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
            provider = GraphProvider.NEO4J
            use_vector_index = True

        assert use_vector_index(_Driver()) is True

    def test_non_neo4j_opt_in_does_not_leak(self):
        class _Driver:
            provider = GraphProvider.FALKORDB
            use_vector_index = True

        assert use_vector_index(_Driver()) is False


class TestNeo4jDriverVectorIndexWiring:
    def test_driver_defaults_to_brute_force(self):
        driver = Neo4jDriver.__new__(Neo4jDriver)
        driver.use_vector_index = False
        assert use_vector_index(driver) is False

    @pytest.mark.asyncio
    async def test_public_build_uses_driver_vector_opt_in(self):
        """Graphiti.build_indices_and_constraints() must honor the driver opt-in."""
        driver = Neo4jDriver.__new__(Neo4jDriver)
        driver.embedding_dim = 768
        driver.use_vector_index = True
        driver._execute_index_query = AsyncMock()

        await driver.build_indices_and_constraints()

        queries = [call.args[0] for call in driver._execute_index_query.await_args_list]
        assert get_vector_indices(GraphProvider.NEO4J, embedding_dim=768) == [
            query for query in queries if 'CREATE VECTOR INDEX' in query
        ]

    def test_embedding_dim_is_configurable_not_hardcoded(self):
        import inspect

        sig = inspect.signature(Neo4jDriver.__init__)
        assert 'embedding_dim' in sig.parameters
        assert 'use_vector_index' in sig.parameters
