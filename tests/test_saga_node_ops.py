"""Saga node operations must write and read back every SagaNode field (no live database).

These operations back the ``graphiti.nodes.saga`` namespace. The shared save query sets
the summary and episode-tracking properties, so each backend must pass all of them as
parameters, and reads must keep them instead of rebuilding a bare SagaNode.
"""

import re
from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.falkordb.operations.saga_node_ops import FalkorSagaNodeOperations
from graphiti_core.driver.kuzu.operations.saga_node_ops import KuzuSagaNodeOperations
from graphiti_core.driver.neo4j.operations.saga_node_ops import Neo4jSagaNodeOperations
from graphiti_core.driver.neptune.operations.saga_node_ops import NeptuneSagaNodeOperations
from graphiti_core.nodes import SagaNode

SAGA_OPERATIONS = [
    Neo4jSagaNodeOperations,
    FalkorSagaNodeOperations,
    KuzuSagaNodeOperations,
    NeptuneSagaNodeOperations,
]


class RecordingExecutor:
    """Captures each query and its parameters, returning the given rows."""

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.records = records or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.calls.append((cypher_query_, kwargs))
        return self.records, None, None


def _saga() -> SagaNode:
    return SagaNode(
        name='support-thread',
        group_id='group',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        summary='The customer asked for a refund.',
        first_episode_uuid='episode-1',
        last_episode_uuid='episode-3',
        last_summarized_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        last_summarized_episode_valid_at=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('operations_class', SAGA_OPERATIONS)
async def test_save_passes_every_parameter_of_the_save_query(operations_class):
    saga = _saga()
    executor = RecordingExecutor()

    await operations_class().save(executor, saga)

    [(query, params)] = executor.calls
    assert set(re.findall(r'\$(\w+)', query)) <= set(params)
    for field in (
        'summary',
        'first_episode_uuid',
        'last_episode_uuid',
        'last_summarized_at',
        'last_summarized_episode_valid_at',
    ):
        assert params[field] == getattr(saga, field)


@pytest.mark.asyncio
@pytest.mark.parametrize('operations_class', SAGA_OPERATIONS)
async def test_reads_keep_every_saga_field(operations_class):
    saga = _saga()
    record = saga.model_dump(exclude={'labels'})
    executor = RecordingExecutor([record])
    operations = operations_class()

    # Node equality only compares uuids, so compare the dumped fields.
    loaded = [
        await operations.get_by_uuid(executor, saga.uuid),
        *await operations.get_by_uuids(executor, [saga.uuid]),
        *await operations.get_by_group_ids(executor, [saga.group_id]),
    ]
    assert [node.model_dump() for node in loaded] == [saga.model_dump()] * 3
