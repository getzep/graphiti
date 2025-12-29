"""Integration test for Neo4j nested attributes serialization.

Tests that entities and edges with complex nested attributes (Maps of Lists, Lists of Maps)
are properly serialized to JSON strings for Neo4j storage.

This test addresses a bug where Neo4j would reject entity/edge attributes containing
nested structures with the error:
Neo.ClientError.Statement.TypeError - Property values can only be of primitive types
or arrays thereof.
"""

import pytest
from datetime import datetime, UTC

from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.driver.driver import GraphProvider


@pytest.mark.integration
async def test_nested_entity_attributes(graph_driver, embedder):
    """Test that entities with nested attributes are stored and retrieved correctly in Neo4j."""
    if graph_driver.provider != GraphProvider.NEO4J:
        pytest.skip("This test is specific to Neo4j nested attributes serialization")
    
    # Create entity with nested attributes (Maps of Lists, Lists of Maps)
    entity = EntityNode(
        uuid="test-entity-nested-attrs-001",
        name="Test Entity with Nested Attributes",
        group_id="test-group-nested",
        labels=["Entity", "TestType"],
        created_at=datetime.now(UTC),
        summary="Test entity for nested attributes",
        attributes={
            # Simple array of primitives - should work
            "discovered_resources": ["resource1", "resource2", "resource3"],
            # Nested map with list values - the problematic case
            "metadata": {
                "analysis": ["analysis_item1", "analysis_item2"],
                "nested_map": {"key1": "value1", "key2": "value2"}
            },
            # Map with complex nested structure
            "activity_log": {
                "initiated_actions": ["action1", "action2"],
                "completed_tasks": {
                    "task_list": ["task1", "task2"],
                    "priority": "high"
                }
            },
            # Simple primitive attributes
            "count": 42,
            "status": "active"
        }
    )
    
    await entity.generate_name_embedding(embedder)
    
    # Save entity - this would previously crash Neo4j with nested structures
    await entity.save(graph_driver)
    
    # Retrieve entity and verify attributes are preserved
    retrieved = await EntityNode.get_by_uuid(graph_driver, entity.uuid)
    
    assert retrieved is not None, "Entity should be retrievable"
    assert retrieved.uuid == entity.uuid
    assert retrieved.name == entity.name
    
    # Verify nested attributes are correctly preserved
    assert retrieved.attributes == entity.attributes, "Attributes should be preserved exactly"
    assert retrieved.attributes["discovered_resources"] == ["resource1", "resource2", "resource3"]
    assert retrieved.attributes["metadata"]["analysis"] == ["analysis_item1", "analysis_item2"]
    assert retrieved.attributes["metadata"]["nested_map"]["key1"] == "value1"
    assert retrieved.attributes["activity_log"]["completed_tasks"]["task_list"] == ["task1", "task2"]
    assert retrieved.attributes["count"] == 42
    assert retrieved.attributes["status"] == "active"


@pytest.mark.integration
async def test_nested_edge_attributes(graph_driver, embedder):
    """Test that edges with nested attributes are stored and retrieved correctly in Neo4j."""
    if graph_driver.provider != GraphProvider.NEO4J:
        pytest.skip("This test is specific to Neo4j nested attributes serialization")
    
    # First create two entity nodes to connect
    source_entity = EntityNode(
        uuid="test-source-entity-001",
        name="Source Entity",
        group_id="test-group-nested",
        labels=["Entity", "TestType"],
        created_at=datetime.now(UTC),
        summary="Source entity for edge test",
        attributes={}
    )
    
    target_entity = EntityNode(
        uuid="test-target-entity-001",
        name="Target Entity",
        group_id="test-group-nested",
        labels=["Entity", "TestType"],
        created_at=datetime.now(UTC),
        summary="Target entity for edge test",
        attributes={}
    )
    
    await source_entity.generate_name_embedding(embedder)
    await target_entity.generate_name_embedding(embedder)
    await source_entity.save(graph_driver)
    await target_entity.save(graph_driver)
    
    # Create edge with nested attributes
    edge = EntityEdge(
        uuid="test-edge-nested-attrs-001",
        source_node_uuid=source_entity.uuid,
        target_node_uuid=target_entity.uuid,
        name="RELATES_TO",
        fact="Source entity relates to target entity with complex metadata",
        group_id="test-group-nested",
        episodes=["episode1", "episode2"],
        created_at=datetime.now(UTC),
        valid_at=datetime.now(UTC),
        attributes={
            # Nested map with list values
            "relationship_metadata": {
                "interaction_types": ["collaboration", "communication"],
                "details": {
                    "frequency": "daily",
                    "confidence": 0.95
                }
            },
            # Map with complex structure
            "historical_data": {
                "events": ["event1", "event2", "event3"],
                "analysis": {
                    "trends": ["increasing", "positive"],
                    "factors": {"external": True, "internal": False}
                }
            },
            # Simple attributes
            "weight": 0.85,
            "verified": True
        }
    )
    
    await edge.generate_embedding(embedder)
    
    # Save edge - this would previously crash Neo4j with nested structures
    await edge.save(graph_driver)
    
    # Retrieve edge and verify attributes are preserved
    retrieved = await EntityEdge.get_by_uuid(graph_driver, edge.uuid)
    
    assert retrieved is not None, "Edge should be retrievable"
    assert retrieved.uuid == edge.uuid
    assert retrieved.fact == edge.fact
    
    # Verify nested attributes are correctly preserved
    assert retrieved.attributes == edge.attributes, "Edge attributes should be preserved exactly"
    assert retrieved.attributes["relationship_metadata"]["interaction_types"] == ["collaboration", "communication"]
    assert retrieved.attributes["relationship_metadata"]["details"]["frequency"] == "daily"
    assert retrieved.attributes["historical_data"]["events"] == ["event1", "event2", "event3"]
    assert retrieved.attributes["historical_data"]["analysis"]["factors"]["external"] is True
    assert retrieved.attributes["weight"] == 0.85
    assert retrieved.attributes["verified"] is True


@pytest.mark.integration
async def test_empty_and_none_attributes(graph_driver, embedder):
    """Test that empty and None attributes are handled correctly."""
    if graph_driver.provider != GraphProvider.NEO4J:
        pytest.skip("This test is specific to Neo4j nested attributes serialization")
    
    # Entity with empty attributes
    entity_empty = EntityNode(
        uuid="test-entity-empty-attrs-001",
        name="Entity with Empty Attributes",
        group_id="test-group-nested",
        labels=["Entity", "TestType"],
        created_at=datetime.now(UTC),
        summary="Test entity with empty attributes",
        attributes={}
    )
    
    await entity_empty.generate_name_embedding(embedder)
    await entity_empty.save(graph_driver)
    
    retrieved_empty = await EntityNode.get_by_uuid(graph_driver, entity_empty.uuid)
    assert retrieved_empty is not None
    assert retrieved_empty.attributes == {}
    
    # Entity with None-valued attributes
    entity_none = EntityNode(
        uuid="test-entity-none-attrs-001",
        name="Entity with None Attributes",
        group_id="test-group-nested",
        labels=["Entity", "TestType"],
        created_at=datetime.now(UTC),
        summary="Test entity with None attributes",
        attributes={"key1": None, "key2": "value2"}
    )
    
    await entity_none.generate_name_embedding(embedder)
    await entity_none.save(graph_driver)
    
    retrieved_none = await EntityNode.get_by_uuid(graph_driver, entity_none.uuid)
    assert retrieved_none is not None
    assert retrieved_none.attributes["key1"] is None
    assert retrieved_none.attributes["key2"] == "value2"

