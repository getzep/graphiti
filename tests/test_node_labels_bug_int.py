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

import os
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from graphiti_core.driver.driver import AsyncGraphDatabase
from graphiti_core.embedder import EmbedderClient
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")


@pytest.fixture
def sample_entity_node_empty_labels():
    return EntityNode(
        uuid=str(uuid4()),
        name="Test Entity Empty Labels",
        group_id="test_group_bug_empty",
        labels=[],
        name_embedding=[0.5] * 1024,
        summary="Entity Summary",
    )


@pytest.fixture
def sample_entity_node_none_labels():
    return EntityNode(
        uuid=str(uuid4()),
        name="Test Entity None Labels",
        group_id="test_group_bug_none",
        labels=None,
        name_embedding=[0.5] * 1024,
        summary="Entity Summary",
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bulk_save_with_problematic_labels(
    sample_entity_node_empty_labels,
    sample_entity_node_none_labels,
):
    """Tests that bulk saving nodes with empty or None labels completes successfully."""
    neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    nodes_to_save = [sample_entity_node_empty_labels, sample_entity_node_none_labels]

    # Mock the embedder client
    mock_embedder = AsyncMock(spec=EmbedderClient)
    mock_embedder.create_batch.return_value = [[0.1] * 10 for _ in nodes_to_save]

    # Use the high-level bulk save function
    await add_nodes_and_edges_bulk(
        driver=neo4j_driver,
        episodic_nodes=[],
        episodic_edges=[],
        entity_nodes=nodes_to_save,
        entity_edges=[],
        embedder=mock_embedder,
    )

    # Verify first node
    retrieved_empty = await EntityNode.get_by_uuid(
        neo4j_driver, sample_entity_node_empty_labels.uuid
    )
    assert retrieved_empty is not None
    assert retrieved_empty.uuid == sample_entity_node_empty_labels.uuid

    # Verify second node
    retrieved_none = await EntityNode.get_by_uuid(
        neo4j_driver, sample_entity_node_none_labels.uuid
    )
    assert retrieved_none is not None
    assert retrieved_none.uuid == sample_entity_node_none_labels.uuid

    # Cleanup
    await sample_entity_node_empty_labels.delete(neo4j_driver)
    await sample_entity_node_none_labels.delete(neo4j_driver)
    await neo4j_driver.close()
