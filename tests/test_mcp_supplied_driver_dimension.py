"""Exercise MCP's supplied-driver initialization without importing the server runtime."""

import __future__

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_filters import SearchFilters


@pytest.mark.asyncio
@pytest.mark.parametrize(('dimensions', 'vector_index'), [(1536, True), (768, True), (8192, False)])
async def test_mcp_supplied_driver_dimension_reaches_index_initialization_and_search(
    monkeypatch, dimensions, vector_index
):
    # Execute the actual service class, not a duplicate constructor or the default-driver path.
    # Exclude module-level MCP setup, dotenv loading and optional server dependencies.
    server_path = Path(__file__).parents[1] / 'mcp_server/src/graphiti_mcp_server.py'
    tree = ast.parse(server_path.read_text())
    service_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'GraphitiService'
    )
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=dimensions))
    config = SimpleNamespace(
        llm=SimpleNamespace(provider='mock', model='mock'),
        embedder=SimpleNamespace(provider='mock', dimensions=dimensions),
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            entity_types={}, edge_types={}, edge_type_map={}, group_id='test-group'
        ),
    )
    namespace = {
        'asyncio': asyncio,
        'Graphiti': Graphiti,
        'logger': Mock(),
        'LLMClientFactory': SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(set_tracer=Mock()))
        ),
        'EmbedderFactory': SimpleNamespace(create=Mock(return_value=embedder)),
        'CrossEncoderFactory': SimpleNamespace(create=Mock(return_value=SimpleNamespace())),
        'DatabaseDriverFactory': SimpleNamespace(
            create_config=Mock(
                return_value=dict(
                    uri='bolt://unused', user='unused', password='unused', database='test'
                )
            )
        ),
        'build_entity_types': Mock(return_value={}),
        'build_edge_types': Mock(return_value={}),
        'build_edge_type_map': Mock(return_value={}),
    }
    exec(
        compile(
            ast.Module(body=[service_class], type_ignores=[]),
            str(server_path),
            'exec',
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    monkeypatch.setenv('GRAPHITI_NEO4J_USE_VECTOR_INDEX', str(vector_index).lower())
    monkeypatch.setattr('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', Mock())
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients',
        Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    initialized_dimensions = []

    async def build_indices(driver, *args, **kwargs):
        initialized_dimensions.append((driver.embedding_dim, driver.use_vector_index))

    monkeypatch.setattr(Neo4jDriver, 'build_indices_and_constraints', build_indices)
    service = namespace['GraphitiService'](config)
    await service.initialize()
    graphiti = service.client
    driver = graphiti.driver
    await driver._init_task

    assert graphiti.embedder is embedder
    assert driver.embedding_dim == dimensions
    assert driver.use_vector_index is vector_index
    assert initialized_dimensions == [(dimensions, vector_index), (dimensions, vector_index)]

    execute_query = AsyncMock(return_value=([], None, None))
    monkeypatch.setattr(driver, 'execute_query', execute_query)
    assert (
        await driver.search_ops.node_similarity_search(
            driver, [0.1] * dimensions, SearchFilters(), limit=1
        )
        == []
    )
    assert (
        await driver.search_ops.edge_similarity_search(
            driver, [0.1] * dimensions, None, None, SearchFilters(), limit=1
        )
        == []
    )
    assert execute_query.await_count == 2
    for call, procedure in zip(
        execute_query.await_args_list,
        ['db.index.vector.queryNodes', 'db.index.vector.queryRelationships'],
        strict=True,
    ):
        assert (procedure in call.args[0]) is vector_index
