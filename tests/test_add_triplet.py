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

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode
from tests.helpers_test import group_id

pytest_plugins = ('pytest_asyncio', 'tests.helpers_test')


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM"""
    mock_llm = Mock(spec=LLMClient)
    mock_llm.config = Mock()
    mock_llm.model = 'test-model'
    mock_llm.small_model = 'test-small-model'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None

    # Mock the public method that's actually called
    mock_llm.generate_response = AsyncMock()
    mock_llm.generate_response.return_value = {
        'duplicate_facts': [],
        'invalidate_facts': [],
    }

    return mock_llm


@pytest.fixture
def mock_cross_encoder_client():
    """Create a mock cross encoder"""
    mock_ce = Mock(spec=CrossEncoderClient)
    mock_ce.config = Mock()
    mock_ce.rerank = AsyncMock()
    mock_ce.rerank.return_value = []

    return mock_ce


@pytest.mark.asyncio
async def test_add_triplet_merges_attributes(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that attributes are merged (not replaced) when adding a triplet."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create an existing node with some attributes
    existing_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Existing summary',
        attributes={'age': 30, 'city': 'New York'},
    )
    await existing_source.generate_name_embedding(mock_embedder)
    await existing_source.save(graph_driver)

    # Create a user-provided node with additional attributes
    user_source = EntityNode(
        uuid=existing_source.uuid,  # Same UUID to trigger direct lookup
        name='Alice',
        group_id=group_id,
        labels=['Person', 'Employee'],
        created_at=now,
        summary='Updated summary',
        attributes={'age': 31, 'department': 'Engineering'},  # age updated, department added
    )

    # Create target node
    user_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25},
    )

    # Create edge
    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='WORKS_WITH',
        fact='Alice works with Bob',
        group_id=group_id,
        created_at=now,
    )

    # Mock the search functions to return empty results
    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        mock_resolve_edge.return_value = (edge, [], [])

        await graphiti.add_triplet(user_source, edge, user_target)

        # Verify attributes were merged (not replaced)
        # The resolved node should have both existing and new attributes
        retrieved_source = await EntityNode.get_by_uuid(graph_driver, existing_source.uuid)
        assert 'age' in retrieved_source.attributes
        assert retrieved_source.attributes['age'] == 31  # Updated value
        assert retrieved_source.attributes['city'] == 'New York'  # Preserved
        assert retrieved_source.attributes['department'] == 'Engineering'  # Added
        assert retrieved_source.summary == 'Updated summary'


@pytest.mark.asyncio
async def test_add_triplet_updates_summary(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that summary is updated when provided by user."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create an existing node with a summary
    existing_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Old summary',
        attributes={},
    )
    await existing_target.generate_name_embedding(mock_embedder)
    await existing_target.save(graph_driver)

    # Create user-provided nodes
    user_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={},
    )

    user_target = EntityNode(
        uuid=existing_target.uuid,
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='New summary for Bob',
        attributes={},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        mock_resolve_edge.return_value = (edge, [], [])

        await graphiti.add_triplet(user_source, edge, user_target)

        # Verify summary was updated
        retrieved_target = await EntityNode.get_by_uuid(graph_driver, existing_target.uuid)
        assert retrieved_target.summary == 'New summary for Bob'


@pytest.mark.asyncio
async def test_add_triplet_updates_labels(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that labels are updated when provided by user."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create an existing node with labels
    existing_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='',
        attributes={},
    )
    await existing_source.generate_name_embedding(mock_embedder)
    await existing_source.save(graph_driver)

    # Create user-provided node with different labels
    user_source = EntityNode(
        uuid=existing_source.uuid,
        name='Alice',
        group_id=group_id,
        labels=['Person', 'Employee', 'Manager'],
        created_at=now,
        summary='',
        attributes={},
    )

    user_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='',
        attributes={},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='MANAGES',
        fact='Alice manages Bob',
        group_id=group_id,
        created_at=now,
    )

    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        mock_resolve_edge.return_value = (edge, [], [])

        await graphiti.add_triplet(user_source, edge, user_target)

        # Verify labels were updated
        retrieved_source = await EntityNode.get_by_uuid(graph_driver, existing_source.uuid)
        # Labels should be set to user-provided labels (not merged)
        assert set(retrieved_source.labels) == {'Person', 'Employee', 'Manager'}


@pytest.mark.asyncio
async def test_add_triplet_with_new_nodes_no_uuid(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test add_triplet with nodes that don't have UUIDs (will be resolved)."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create user-provided nodes without UUIDs
    user_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30},
    )

    user_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    with patch('graphiti_core.graphiti.search') as mock_search:
        mock_search.return_value = Mock(edges=[])
        with patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge:
            mock_resolve_edge.return_value = (edge, [], [])

            result = await graphiti.add_triplet(user_source, edge, user_target)

            # Verify nodes were created with user-provided attributes
            assert len(result.nodes) >= 2
            # Find the nodes in the result
            source_in_result = next((n for n in result.nodes if n.name == 'Alice'), None)
            target_in_result = next((n for n in result.nodes if n.name == 'Bob'), None)

            if source_in_result:
                assert source_in_result.attributes.get('age') == 30
                assert source_in_result.summary == 'Alice summary'
            if target_in_result:
                assert target_in_result.attributes.get('age') == 25
                assert target_in_result.summary == 'Bob summary'


@pytest.mark.asyncio
async def test_add_triplet_preserves_existing_attributes(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that existing attributes are preserved when merging new ones."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create an existing node with multiple attributes
    existing_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Existing summary',
        attributes={
            'age': 30,
            'city': 'New York',
            'country': 'USA',
            'email': 'alice@example.com',
        },
    )
    await existing_source.generate_name_embedding(mock_embedder)
    await existing_source.save(graph_driver)

    # Create user-provided node with only some attributes
    user_source = EntityNode(
        uuid=existing_source.uuid,
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Updated summary',
        attributes={'age': 31, 'city': 'Boston'},  # Only updating age and city
    )

    user_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='',
        attributes={},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        mock_resolve_edge.return_value = (edge, [], [])

        await graphiti.add_triplet(user_source, edge, user_target)

        # Verify all attributes are preserved/updated correctly
        retrieved_source = await EntityNode.get_by_uuid(graph_driver, existing_source.uuid)
        assert retrieved_source.attributes['age'] == 31  # Updated
        assert retrieved_source.attributes['city'] == 'Boston'  # Updated
        assert retrieved_source.attributes['country'] == 'USA'  # Preserved
        assert retrieved_source.attributes['email'] == 'alice@example.com'  # Preserved
        assert retrieved_source.summary == 'Updated summary'


@pytest.mark.asyncio
async def test_add_triplet_empty_attributes_preserved(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that nodes with empty attributes don't overwrite existing attributes."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create an existing node with attributes
    existing_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Existing summary',
        attributes={'age': 30, 'city': 'New York'},
    )
    await existing_source.generate_name_embedding(mock_embedder)
    await existing_source.save(graph_driver)

    # Create user-provided node with empty attributes
    user_source = EntityNode(
        uuid=existing_source.uuid,
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='',  # Empty summary should not overwrite
        attributes={},  # Empty attributes should not overwrite
    )

    user_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='',
        attributes={},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        mock_resolve_edge.return_value = (edge, [], [])

        await graphiti.add_triplet(user_source, edge, user_target)

        # Verify existing attributes are preserved when user provides empty dict
        retrieved_source = await EntityNode.get_by_uuid(graph_driver, existing_source.uuid)
        # Empty attributes dict should not clear existing attributes
        assert 'age' in retrieved_source.attributes
        assert 'city' in retrieved_source.attributes
        # Empty summary should not overwrite existing summary
        assert retrieved_source.summary == 'Existing summary'


@pytest.mark.asyncio
async def test_add_triplet_invalid_source_uuid(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that ValueError is raised when source_node has a UUID that doesn't exist."""
    from uuid import uuid4

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create a node with a UUID that doesn't exist in the database
    invalid_uuid = str(uuid4())
    user_source = EntityNode(
        uuid=invalid_uuid,
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30},
    )

    user_target = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    # Should raise ValueError for invalid source UUID
    with pytest.raises(ValueError, match=f'Node with UUID {invalid_uuid} not found'):
        await graphiti.add_triplet(user_source, edge, user_target)


@pytest.mark.asyncio
async def test_add_triplet_invalid_target_uuid(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that ValueError is raised when target_node has a UUID that doesn't exist."""
    from uuid import uuid4

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create an existing source node
    existing_source = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30},
    )
    await existing_source.generate_name_embedding(mock_embedder)
    await existing_source.save(graph_driver)

    # Create a target node with a UUID that doesn't exist in the database
    invalid_uuid = str(uuid4())
    user_source = EntityNode(
        uuid=existing_source.uuid,
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30},
    )

    user_target = EntityNode(
        uuid=invalid_uuid,
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    # Should raise ValueError for invalid target UUID
    with pytest.raises(ValueError, match=f'Node with UUID {invalid_uuid} not found'):
        await graphiti.add_triplet(user_source, edge, user_target)


@pytest.mark.asyncio
async def test_add_triplet_invalid_both_uuids(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that ValueError is raised for source_node first when both UUIDs are invalid."""
    from uuid import uuid4

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create nodes with UUIDs that don't exist in the database
    invalid_source_uuid = str(uuid4())
    invalid_target_uuid = str(uuid4())

    user_source = EntityNode(
        uuid=invalid_source_uuid,
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30},
    )

    user_target = EntityNode(
        uuid=invalid_target_uuid,
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25},
    )

    edge = EntityEdge(
        source_node_uuid=user_source.uuid,
        target_node_uuid=user_target.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )

    # Should raise ValueError for source UUID first (source is checked before target)
    with pytest.raises(ValueError, match=f'Node with UUID {invalid_source_uuid} not found'):
        await graphiti.add_triplet(user_source, edge, user_target)


@pytest.mark.asyncio
async def test_add_triplet_edge_uuid_with_different_nodes_creates_new_edge(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that providing an edge UUID with different src/dst nodes creates a new edge."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create existing nodes: Alice and Bob
    alice = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={},
    )
    await alice.generate_name_embedding(mock_embedder)
    await alice.save(graph_driver)

    bob = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={},
    )
    await bob.generate_name_embedding(mock_embedder)
    await bob.save(graph_driver)

    # Create a third node: Charlie
    charlie = EntityNode(
        name='Charlie',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Charlie summary',
        attributes={},
    )
    await charlie.generate_name_embedding(mock_embedder)
    await charlie.save(graph_driver)

    # Create an existing edge between Alice and Bob
    existing_edge = EntityEdge(
        source_node_uuid=alice.uuid,
        target_node_uuid=bob.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )
    await existing_edge.generate_embedding(mock_embedder)
    await existing_edge.save(graph_driver)

    # Now try to add a triplet using the existing edge UUID but with different nodes (Alice -> Charlie)
    new_edge_with_same_uuid = EntityEdge(
        uuid=existing_edge.uuid,  # Reuse the existing edge's UUID
        source_node_uuid=alice.uuid,
        target_node_uuid=charlie.uuid,  # Different target!
        name='KNOWS',
        fact='Alice knows Charlie',
        group_id=group_id,
        created_at=now,
    )

    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        # Return the edge as-is (simulating no deduplication)
        mock_resolve_edge.return_value = (new_edge_with_same_uuid, [], [])

        result = await graphiti.add_triplet(alice, new_edge_with_same_uuid, charlie)

        # The original edge (Alice -> Bob) should still exist
        original_edge = await EntityEdge.get_by_uuid(graph_driver, existing_edge.uuid)
        assert original_edge.source_node_uuid == alice.uuid
        assert original_edge.target_node_uuid == bob.uuid
        assert original_edge.fact == 'Alice knows Bob'

        # The new edge should have a different UUID
        new_edge = result.edges[0]
        assert new_edge.uuid != existing_edge.uuid
        assert new_edge.source_node_uuid == alice.uuid
        assert new_edge.target_node_uuid == charlie.uuid


@pytest.mark.asyncio
async def test_add_triplet_edge_uuid_with_same_nodes_updates_edge(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    """Test that providing an edge UUID with same src/dst nodes allows updating the edge."""
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create existing nodes: Alice and Bob
    alice = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Alice summary',
        attributes={},
    )
    await alice.generate_name_embedding(mock_embedder)
    await alice.save(graph_driver)

    bob = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Person'],
        created_at=now,
        summary='Bob summary',
        attributes={},
    )
    await bob.generate_name_embedding(mock_embedder)
    await bob.save(graph_driver)

    # Create an existing edge between Alice and Bob
    existing_edge = EntityEdge(
        source_node_uuid=alice.uuid,
        target_node_uuid=bob.uuid,
        name='KNOWS',
        fact='Alice knows Bob',
        group_id=group_id,
        created_at=now,
    )
    await existing_edge.generate_embedding(mock_embedder)
    await existing_edge.save(graph_driver)

    # Now update the edge with the same source/target but different fact
    updated_edge = EntityEdge(
        uuid=existing_edge.uuid,  # Reuse the existing edge's UUID
        source_node_uuid=alice.uuid,
        target_node_uuid=bob.uuid,  # Same target
        name='WORKS_WITH',
        fact='Alice works with Bob',  # Updated fact
        group_id=group_id,
        created_at=now,
    )

    with (
        patch('graphiti_core.graphiti.search') as mock_search,
        patch('graphiti_core.graphiti.resolve_extracted_edge') as mock_resolve_edge,
    ):
        mock_search.return_value = Mock(edges=[])
        mock_resolve_edge.return_value = (updated_edge, [], [])

        result = await graphiti.add_triplet(alice, updated_edge, bob)

        # The edge should keep the same UUID (update allowed)
        result_edge = result.edges[0]
        assert result_edge.uuid == existing_edge.uuid
