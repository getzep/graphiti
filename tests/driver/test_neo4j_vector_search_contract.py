from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from neo4j.exceptions import ClientError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_similarity_search


def _record(uuid: str, source: str, target: str, group_id: str = 'group') -> dict:
    return {
        'uuid': uuid,
        'source_node_uuid': source,
        'target_node_uuid': target,
        'group_id': group_id,
        'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
        'name': 'RELATES_TO',
        'fact': uuid,
        'episodes': [],
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'reference_time': None,
        'attributes': {'uuid': uuid, 'custom': f'attribute-{uuid}'},
    }


def _executor(*responses):
    executor = AsyncMock()
    executor.provider = GraphProvider.NEO4J
    executor.use_vector_index = True
    executor.embedding_dim = 3
    executor.execute_query.side_effect = list(responses)
    return executor


def _client_error(code: str, message: str) -> ClientError:
    error = ClientError(message)
    error._neo4j_code = code
    return error


@pytest.mark.asyncio
async def test_unfiltered_opt_in_uses_hnsw_and_preserves_shape_and_order():
    records = [_record('higher', 'source-1', 'target-1'), _record('lower', 'source-2', 'target-2')]
    executor = _executor((records, None, None))

    edges = await Neo4jSearchOperations().edge_similarity_search(
        executor, [0.1, 0.2, 0.3], None, None, SearchFilters(), limit=2
    )

    query = executor.execute_query.await_args.args[0]
    assert 'db.index.vector.queryRelationships' in query
    assert 'startNode(e) AS n' in query
    assert 'endNode(e) AS m' in query
    assert 'matched.uuid = e.uuid' not in query
    assert 'ORDER BY score DESC' in query
    assert [(edge.uuid, edge.source_node_uuid, edge.target_node_uuid) for edge in edges] == [
        ('higher', 'source-1', 'target-1'),
        ('lower', 'source-2', 'target-2'),
    ]
    assert [edge.attributes for edge in edges] == [
        {'custom': 'attribute-higher'},
        {'custom': 'attribute-lower'},
    ]


@pytest.mark.asyncio
async def test_public_search_dispatch_reaches_neo4j_hnsw():
    records = [_record('edge', 'source', 'target')]
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        search_interface=None,
        search_ops=Neo4jSearchOperations(),
        use_vector_index=True,
        embedding_dim=3,
        execute_query=AsyncMock(return_value=(records, None, None)),
    )

    edges = await edge_similarity_search(
        driver,
        [0.1, 0.2, 0.3],
        None,
        None,
        SearchFilters(),
        limit=1,
        min_score=0.75,
    )

    query = driver.execute_query.await_args.args[0]
    kwargs = driver.execute_query.await_args.kwargs
    assert 'db.index.vector.queryRelationships' in query
    assert kwargs['limit'] == 1
    assert kwargs['min_score'] == 0.75
    assert [(edge.uuid, edge.source_node_uuid, edge.target_node_uuid) for edge in edges] == [
        ('edge', 'source', 'target')
    ]


@pytest.mark.asyncio
async def test_public_search_does_not_leak_vector_opt_in_to_other_providers():
    provider_search = AsyncMock(side_effect=AssertionError('provider search must not be called'))
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        search_interface=None,
        search_ops=SimpleNamespace(edge_similarity_search=provider_search),
        use_vector_index=True,
        execute_query=AsyncMock(return_value=([], None, None)),
    )

    await edge_similarity_search(driver, [0.1, 0.2, 0.3], None, None, SearchFilters(), limit=1)

    provider_search.assert_not_awaited()
    assert 'db.index.vector' not in driver.execute_query.await_args.args[0]


@pytest.mark.asyncio
async def test_node_and_community_similarity_paths_remain_exact_when_edge_hnsw_is_enabled():
    executor = _executor(([], None, None), ([], None, None))
    operations = Neo4jSearchOperations()

    await operations.node_similarity_search(executor, [0.1, 0.2, 0.3], SearchFilters(), limit=2)
    await operations.community_similarity_search(executor, [0.1, 0.2, 0.3], limit=2)

    node_query = executor.execute_query.await_args_list[0].args[0]
    community_query = executor.execute_query.await_args_list[1].args[0]
    assert 'vector.similarity.cosine(n.name_embedding' in node_query
    assert 'vector.similarity.cosine(c.name_embedding' in community_query
    assert 'db.index.vector' not in node_query + community_query


@pytest.mark.asyncio
async def test_filtered_search_uses_exact_prefilter_path_not_fixed_overfetch():
    limit = 2
    # Six globally closer results are outside the accepted group. The two valid
    # results begin at ranks 7 and 8, beyond a fixed 3x candidate window.
    globally_ranked_groups = ['other-group'] * (limit * 3) + ['sparse-group'] * limit
    acceptable_records = [
        _record(f'accepted-{i}', 'source', 'target', group_id='sparse-group') for i in range(limit)
    ]

    async def execute_query(query, **kwargs):
        if 'db.index.vector.queryRelationships' in query:
            candidate_groups = globally_ranked_groups[: limit * 3]
            assert 'sparse-group' not in candidate_groups
            return ([], None, None)
        return (acceptable_records, None, None)

    executor = _executor()
    executor.execute_query.side_effect = execute_query

    edges = await Neo4jSearchOperations().edge_similarity_search(
        executor,
        [0.1, 0.2, 0.3],
        None,
        None,
        SearchFilters(edge_types=['works_at']),
        group_ids=['sparse-group'],
        limit=limit,
    )

    query = executor.execute_query.await_args.args[0]
    assert 'db.index.vector.queryRelationships' not in query
    assert 'e.group_id IN $group_ids' in query
    assert 'e.name in $edge_types' in query
    assert 'LIMIT $limit' in query
    assert [edge.uuid for edge in edges] == ['accepted-0', 'accepted-1']


@pytest.mark.asyncio
async def test_endpoint_filters_apply_without_group_ids():
    executor = _executor(([], None, None))

    await Neo4jSearchOperations().edge_similarity_search(
        executor,
        [0.1, 0.2, 0.3],
        'source-id',
        'target-id',
        SearchFilters(),
        limit=2,
    )

    query = executor.execute_query.await_args.args[0]
    kwargs = executor.execute_query.await_args.kwargs
    assert 'n.uuid = $source_uuid' in query
    assert 'm.uuid = $target_uuid' in query
    assert kwargs['source_uuid'] == 'source-id'
    assert kwargs['target_uuid'] == 'target-id'


@pytest.mark.asyncio
async def test_known_vector_lifecycle_failure_falls_back_to_exact_search(monkeypatch):
    class VectorLifecycleError(Exception):
        code = 'Neo.ClientError.Schema.IndexNotFound'

    monkeypatch.setattr(
        'graphiti_core.driver.neo4j.operations.search_ops.ClientError',
        VectorLifecycleError,
        raising=False,
    )
    executor = _executor(VectorLifecycleError('index is not ONLINE'), ([], None, None))

    await Neo4jSearchOperations().edge_similarity_search(
        executor, [0.1, 0.2, 0.3], None, None, SearchFilters(), limit=2
    )

    assert executor.execute_query.await_count == 2
    first_query = executor.execute_query.await_args_list[0].args[0]
    fallback_query = executor.execute_query.await_args_list[1].args[0]
    assert 'db.index.vector.queryRelationships' in first_query
    assert 'db.index.vector.queryRelationships' not in fallback_query


@pytest.mark.parametrize(
    ('code', 'message'),
    [
        (
            'Neo.ClientError.Procedure.ProcedureNotFound',
            'There is no procedure with the name db.index.vector.queryRelationships',
        ),
        (
            'Neo.ClientError.Procedure.ProcedureCallFailed',
            "The index 'edge_fact_embedding' is not ONLINE.",
        ),
        (
            'Neo.ClientError.Procedure.ProcedureCallFailed',
            "The index 'edge_fact_embedding' is still POPULATING.",
        ),
        (
            'Neo.ClientError.Procedure.ProcedureCallFailed',
            "The index 'edge_fact_embedding' is OFFLINE.",
        ),
        (
            'Neo.ClientError.Procedure.ProcedureCallFailed',
            'Index query vector has 3 dimensions, but indexed vectors have 2.',
        ),
        (
            'Neo.ClientError.Procedure.ProcedureCallFailed',
            "The index 'edge_fact_embedding' is not a vector index.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bounded_fallback_for_known_vector_lifecycle_failures(code, message):
    executor = _executor(_client_error(code, message), ([], None, None))

    await Neo4jSearchOperations().edge_similarity_search(
        executor, [0.1, 0.2, 0.3], None, None, SearchFilters(), limit=2
    )

    assert executor.execute_query.await_count == 2
    assert 'db.index.vector.queryRelationships' in executor.execute_query.await_args_list[0].args[0]
    assert (
        'db.index.vector.queryRelationships'
        not in executor.execute_query.await_args_list[1].args[0]
    )


@pytest.mark.asyncio
async def test_programmer_error_is_not_hidden_by_fallback():
    executor = _executor(TypeError('bad query composition'))

    with pytest.raises(TypeError, match='bad query composition'):
        await Neo4jSearchOperations().edge_similarity_search(
            executor, [0.1, 0.2, 0.3], None, None, SearchFilters(), limit=2
        )

    assert executor.execute_query.await_count == 1


@pytest.mark.asyncio
async def test_unrecognized_client_error_is_not_hidden_by_fallback():
    error = _client_error(
        'Neo.ClientError.Statement.SyntaxError', 'Invalid input caused by a programming error'
    )
    executor = _executor(error)

    with pytest.raises(ClientError, match='programming error'):
        await Neo4jSearchOperations().edge_similarity_search(
            executor, [0.1, 0.2, 0.3], None, None, SearchFilters(), limit=2
        )

    assert executor.execute_query.await_count == 1


@pytest.mark.asyncio
async def test_vector_dimension_mismatch_is_clear_before_query_execution():
    executor = _executor(([], None, None))

    with pytest.raises(
        ValueError, match='query embedding dimension 2.*configured index dimension 3'
    ):
        await Neo4jSearchOperations().edge_similarity_search(
            executor, [0.1, 0.2], None, None, SearchFilters(), limit=2
        )

    executor.execute_query.assert_not_awaited()


def test_graphiti_default_neo4j_driver_uses_actual_embedder_dimension(monkeypatch):
    driver_factory = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr('graphiti_core.graphiti.Neo4jDriver', driver_factory)
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients', Mock(return_value=SimpleNamespace())
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=768))
    llm_client = SimpleNamespace(set_tracer=Mock())

    Graphiti(
        uri='bolt://unused',
        user='neo4j',
        password='unused',
        embedder=embedder,
        llm_client=llm_client,
        cross_encoder=SimpleNamespace(),
        use_vector_index=True,
    )

    driver_factory.assert_called_once_with(
        'bolt://unused',
        'neo4j',
        'unused',
        embedding_dim=768,
        use_vector_index=True,
    )


def test_graphiti_does_not_apply_neo4j_vector_args_to_supplied_non_neo4j_driver(monkeypatch):
    driver = SimpleNamespace(provider=GraphProvider.FALKORDB)
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients', Mock(return_value=SimpleNamespace())
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    llm_client = SimpleNamespace(set_tracer=Mock())

    graphiti = Graphiti(
        graph_driver=driver,
        embedder=SimpleNamespace(config=SimpleNamespace(embedding_dim=768)),
        llm_client=llm_client,
        cross_encoder=SimpleNamespace(),
        use_vector_index=True,
    )

    assert graphiti.driver is driver
    assert not hasattr(driver, 'use_vector_index')


def test_graphiti_environment_opt_in_reaches_default_neo4j_driver(monkeypatch):
    monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', 'true')
    monkeypatch.setattr('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', Mock())
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients', Mock(return_value=SimpleNamespace())
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    llm_client = SimpleNamespace(set_tracer=Mock())

    graphiti = Graphiti(
        uri='bolt://unused',
        user='neo4j',
        password='unused',
        embedder=SimpleNamespace(config=SimpleNamespace(embedding_dim=768)),
        llm_client=llm_client,
        cross_encoder=SimpleNamespace(),
    )

    assert graphiti.driver.provider == GraphProvider.NEO4J
    assert graphiti.driver.embedding_dim == 768
    assert graphiti.driver.use_vector_index is True
