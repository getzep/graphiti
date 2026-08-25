"""Regression tests: entity-edge save must always persist source_node_uuid /
target_node_uuid, never source_uuid / target_uuid (no live database).

EntityEdge.save's single-edge write path built its MATCH-key payload as
`source_uuid`/`target_uuid`, while the bulk-edge write path used
`edge.model_dump()` -- whose real Pydantic field names are
`source_node_uuid`/`target_node_uuid`. Both queries did `SET e = <payload>`
(or the Neptune/FalkorDB equivalent), so the raw property name a relationship
ended up with depended on which write path created it. Downstream code (the
record parsers, the return-query aliases) has always expected
`source_node_uuid`/`target_node_uuid`, so the differently-named single-save
properties leaked into `EntityEdge.attributes` on read and produced two
incompatible property shapes for the same relationship type in the same
graph.

The fix standardizes every write path -- the shared query builder
(edge_db_queries.py), the generic EntityEdge.save fallback (edges.py), and
each provider's EntityEdgeOperations.save (the actual live path when
driver.graph_operations_interface is set) -- on source_node_uuid /
target_node_uuid, matching what bulk save and the return-query aliases
already use.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.entity_edge_ops import FalkorEntityEdgeOperations
from graphiti_core.driver.kuzu.operations.entity_edge_ops import KuzuEntityEdgeOperations
from graphiti_core.driver.neo4j.operations.entity_edge_ops import Neo4jEntityEdgeOperations
from graphiti_core.driver.neptune.operations.entity_edge_ops import NeptuneEntityEdgeOperations
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_query

PROVIDERS = [
    GraphProvider.NEO4J,
    GraphProvider.FALKORDB,
    GraphProvider.NEPTUNE,
    GraphProvider.KUZU,
]


def _assert_no_bare_source_target_uuid(text: str) -> None:
    # Substring checks alone would false-positive on source_node_uuid /
    # target_node_uuid, so require the bare names not be followed by "_uuid"
    # having already matched -- simplest is checking the exact param/property
    # tokens graphiti-core actually emits.
    assert 'source_uuid' not in text.replace('source_node_uuid', '')
    assert 'target_uuid' not in text.replace('target_node_uuid', '')


@pytest.mark.parametrize('provider', PROVIDERS)
def test_entity_edge_save_query_never_emits_bare_source_target_uuid(provider):
    query = get_entity_edge_save_query(provider)
    _assert_no_bare_source_target_uuid(query)
    assert 'source_node_uuid' in query
    assert 'target_node_uuid' in query


class RecordingDriver:
    """Captures the query and kwargs EntityEdge.save emits (no live database)."""

    search_interface = None
    graph_operations_interface = None
    fulltext_syntax = ''

    def __init__(self, provider: GraphProvider):
        self.provider = provider
        self.query = ''
        self.kwargs: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.query = cypher_query_
        self.kwargs = kwargs
        return [], None, None


def _make_edge() -> EntityEdge:
    return EntityEdge(
        source_node_uuid='src-uuid',
        target_node_uuid='tgt-uuid',
        name='RELATES_TO',
        fact='src relates to tgt',
        group_id='group-a',
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', PROVIDERS)
async def test_entity_edge_save_fallback_never_sends_bare_source_target_uuid(provider):
    driver = RecordingDriver(provider)
    edge = _make_edge()

    await edge.save(driver)  # type: ignore[arg-type]

    payload = driver.kwargs if provider == GraphProvider.KUZU else driver.kwargs['edge_data']

    assert 'source_uuid' not in payload
    assert 'target_uuid' not in payload
    assert payload['source_node_uuid'] == 'src-uuid'
    assert payload['target_node_uuid'] == 'tgt-uuid'


class RecordingExecutor:
    def __init__(self):
        self.query = ''
        self.kwargs: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.query = cypher_query_
        self.kwargs = kwargs
        return [], None, None


OPERATIONS_CLASSES = [
    Neo4jEntityEdgeOperations,
    FalkorEntityEdgeOperations,
    NeptuneEntityEdgeOperations,
    KuzuEntityEdgeOperations,
]


@pytest.mark.asyncio
@pytest.mark.parametrize('ops_cls', OPERATIONS_CLASSES)
async def test_entity_edge_operations_save_never_sends_bare_source_target_uuid(ops_cls):
    executor = RecordingExecutor()
    edge = _make_edge()

    await ops_cls().save(executor, edge)  # type: ignore[arg-type]

    payload = (
        executor.kwargs if ops_cls is KuzuEntityEdgeOperations else executor.kwargs['edge_data']
    )

    assert 'source_uuid' not in payload
    assert 'target_uuid' not in payload
    assert payload['source_node_uuid'] == 'src-uuid'
    assert payload['target_node_uuid'] == 'tgt-uuid'
