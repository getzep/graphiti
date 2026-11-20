"""Unit tests for the entity edge save operations of the graph drivers.

`graphiti.edges.entity.save()` and `save_bulk()` run these operations. The save
queries replace every property of the relationship (`SET e = $edge_data`), so a
field that the payload leaves out is erased from a stored edge.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.falkordb.operations.entity_edge_ops import FalkorEntityEdgeOperations
from graphiti_core.driver.neo4j.operations.entity_edge_ops import Neo4jEntityEdgeOperations
from graphiti_core.driver.neptune.operations.entity_edge_ops import NeptuneEntityEdgeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.edges import EntityEdge

REFERENCE_TIME = datetime(2024, 12, 9, tzinfo=timezone.utc)

ENTITY_EDGE_OPS = [
    Neo4jEntityEdgeOperations,
    FalkorEntityEdgeOperations,
    NeptuneEntityEdgeOperations,
]


class RecordingExecutor(QueryExecutor):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return [], [], None


def _edge() -> EntityEdge:
    return EntityEdge(
        source_node_uuid='source',
        target_node_uuid='target',
        name='WORKS_AT',
        fact='Alice works at Acme',
        group_id='group',
        created_at=REFERENCE_TIME,
        reference_time=REFERENCE_TIME,
    )


@pytest.mark.parametrize('ops_cls', ENTITY_EDGE_OPS)
async def test_save_persists_reference_time(ops_cls: type[EntityEdgeOperations]):
    executor = RecordingExecutor()

    await ops_cls().save(executor, _edge())

    assert executor.calls[0]['edge_data']['reference_time'] == REFERENCE_TIME


@pytest.mark.parametrize('ops_cls', ENTITY_EDGE_OPS)
async def test_save_bulk_persists_reference_time(ops_cls: type[EntityEdgeOperations]):
    executor = RecordingExecutor()

    await ops_cls().save_bulk(executor, [_edge()])

    assert executor.calls[0]['entity_edges'][0]['reference_time'] == REFERENCE_TIME
