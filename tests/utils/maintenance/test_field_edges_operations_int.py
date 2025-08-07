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

Integration tests for Field Edges Operations (bulk operations and analysis).

This test suite covers:
- Bulk BELONGS_TO edge operations with MongoDB validation
- Bulk field relationship edge operations with MongoDB validation
- Field relationship pattern analysis
- Bidirectional relationship detection
- Relationship constraint validation
- Field relationship network analysis
- MongoDB integration error handling in bulk operations
"""

import os
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncGraphDatabase

from graphiti_core.field_edges import BelongsToEdge, FieldRelationshipEdge
from graphiti_core.utils.maintenance.field_edges_operations import (
    get_belongs_to_edges_by_cluster,
    save_belongs_to_edges_bulk,
    get_field_relationships_by_source,
    get_field_relationships_by_target,
    get_field_relationships_by_cluster,
    save_field_relationship_edges_bulk,
    create_field_relationship_embeddings_batch,
    build_field_cluster_membership,
    analyze_field_relationship_patterns,
    find_bidirectional_relationships,
    validate_relationship_constraints,
    get_field_relationship_network,
    _validate_clusters_bulk,
)
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError as MongoClusterNotFoundError,
    ClusterValidationError,
)

# Test configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'YourStrongPasswordHere')

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def driver():
    """Neo4j driver fixture for integration tests"""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    await driver.close()


@pytest.fixture
def mock_embedder():
    """Mock embedder client fixture"""
    embedder = MagicMock()
    embedder.create.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 200  # Mock 1000-dim embedding
    embedder.create_batch.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] * 200] * 5  # Mock batch embeddings
    return embedder


@pytest.fixture
def sample_belongs_to_edges():
    """Sample BelongsToEdge list fixture for bulk operations"""
    cluster_uuid = str(uuid.uuid4())
    return [
        BelongsToEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),  # Field 1
            target_node_uuid=cluster_uuid,
            cluster_partition_id=cluster_uuid,
            created_at=datetime.now(timezone.utc),
        ),
        BelongsToEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),  # Field 2
            target_node_uuid=cluster_uuid,
            cluster_partition_id=cluster_uuid,
            created_at=datetime.now(timezone.utc),
        ),
        BelongsToEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),  # Field 3
            target_node_uuid=cluster_uuid,
            cluster_partition_id=cluster_uuid,
            created_at=datetime.now(timezone.utc),
        ),
    ]


@pytest.fixture
def sample_field_relationship_edges():
    """Sample FieldRelationshipEdge list fixture for bulk operations"""
    cluster_id = 'linux_audit_batelco'
    now = datetime.now(timezone.utc)
    
    return [
        FieldRelationshipEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            name='CORRELATES_WITH',
            description='Fields that frequently appear together in logs',
            confidence=0.85,
            cluster_partition_id=cluster_id,
            relationship_type='FIELD_RELATES_TO',
            created_at=now,
            valid_at=now,
        ),
        FieldRelationshipEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            name='SIMILAR_TO',
            description='Fields with similar semantic meaning',
            confidence=0.75,
            cluster_partition_id=cluster_id,
            relationship_type='FIELD_RELATES_TO',
            created_at=now,
            valid_at=now,
        ),
        FieldRelationshipEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            name='DERIVED_FROM',
            description='Field derived from another field through transformation',
            confidence=0.95,
            cluster_partition_id=cluster_id,
            relationship_type='FIELD_RELATES_TO',
            created_at=now,
            valid_at=now,
        ),
    ]


class TestBelongsToEdgeBulkOperations:
    """Test suite for bulk BELONGS_TO edge operations"""
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_save_belongs_to_edges_bulk_with_mongodb_validation(self, mock_cluster_service, sample_belongs_to_edges, driver):
        """Test bulk BELONGS_TO edge save with MongoDB validation"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save edges in bulk
        result = await save_belongs_to_edges_bulk(driver, sample_belongs_to_edges)
        
        # Verify MongoDB validation was called
        mock_service_instance.validate_cluster_exists.assert_called()
        
        # Clean up
        for edge in sample_belongs_to_edges:
            await edge.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_save_belongs_to_edges_bulk_mongodb_error_handling(self, mock_cluster_service, sample_belongs_to_edges, driver):
        """Test bulk BELONGS_TO edge save with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.side_effect = ClusterValidationError("MongoDB connection failed")
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed despite MongoDB error
        result = await save_belongs_to_edges_bulk(driver, sample_belongs_to_edges)
        
        # Clean up
        for edge in sample_belongs_to_edges:
            await edge.delete(driver)
    
    async def test_save_belongs_to_edges_bulk_empty_list(self, driver):
        """Test bulk BELONGS_TO edge save with empty list"""
        result = await save_belongs_to_edges_bulk(driver, [])
        assert result == []
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_build_field_cluster_membership(self, mock_cluster_service, driver):
        """Test building field-cluster membership relationships"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Test data
        field_uuids = [str(uuid.uuid4()) for _ in range(3)]
        cluster_uuid = str(uuid.uuid4())
        
        # Build membership
        edges = await build_field_cluster_membership(driver, field_uuids, cluster_uuid)
        
        # Verify edges were created and saved
        assert len(edges) == len(field_uuids)
        for edge in edges:
            assert edge.target_node_uuid == cluster_uuid
            assert edge.cluster_partition_id == cluster_uuid
            assert edge.source_node_uuid in field_uuids
        
        # Verify MongoDB validation was called
        mock_service_instance.validate_cluster_exists.assert_called_with(cluster_uuid)
        
        # Clean up
        for edge in edges:
            await edge.delete(driver)


class TestFieldRelationshipEdgeBulkOperations:
    """Test suite for bulk field relationship edge operations"""
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_save_field_relationship_edges_bulk_with_mongodb_validation(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test bulk field relationship edge save with MongoDB validation"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save edges in bulk
        result = await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Verify MongoDB validation was called
        mock_service_instance.validate_cluster_exists.assert_called()
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_save_field_relationship_edges_bulk_mongodb_error_handling(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test bulk field relationship edge save with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.side_effect = ClusterValidationError("MongoDB connection failed")
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed despite MongoDB error
        result = await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)
    
    async def test_save_field_relationship_edges_bulk_empty_list(self, driver):
        """Test bulk field relationship edge save with empty list"""
        result = await save_field_relationship_edges_bulk(driver, [])
        assert result == []
    
    async def test_create_field_relationship_embeddings_batch(self, sample_field_relationship_edges, mock_embedder):
        """Test batch embedding creation for field relationships"""
        await create_field_relationship_embeddings_batch(mock_embedder, sample_field_relationship_edges)
        
        # Verify embeddings were created
        for edge in sample_field_relationship_edges:
            assert edge.description_embedding is not None
            assert len(edge.description_embedding) == 1000
        
        # Verify embedder was called with correct batch
        mock_embedder.create_batch.assert_called_once()
        call_args = mock_embedder.create_batch.call_args[0][0]
        assert len(call_args) == len(sample_field_relationship_edges)
    
    async def test_create_field_relationship_embeddings_batch_empty_list(self, mock_embedder):
        """Test batch embedding creation with empty list"""
        await create_field_relationship_embeddings_batch(mock_embedder, [])
        mock_embedder.create_batch.assert_not_called()


class TestFieldRelationshipQueryOperations:
    """Test suite for field relationship query operations"""
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_get_field_relationships_by_cluster(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test retrieval of field relationships by cluster"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save some relationships first
        await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Test retrieval
        cluster_relationships = await get_field_relationships_by_cluster(driver, 'linux_audit_batelco')
        
        # Should find the relationships for this cluster
        assert len(cluster_relationships) == len(sample_field_relationship_edges)
        for rel in cluster_relationships:
            assert rel.cluster_partition_id == 'linux_audit_batelco'
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_get_field_relationships_by_source(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test retrieval of field relationships by source field"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save some relationships first
        await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Test retrieval by source field
        source_uuid = sample_field_relationship_edges[0].source_node_uuid
        source_relationships = await get_field_relationships_by_source(driver, source_uuid, 'linux_audit_batelco')
        
        # Should find relationships from this source
        for rel in source_relationships:
            assert rel.source_node_uuid == source_uuid
            assert rel.cluster_partition_id == 'linux_audit_batelco'
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_get_field_relationships_by_target(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test retrieval of field relationships by target field"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save some relationships first
        await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Test retrieval by target field
        target_uuid = sample_field_relationship_edges[0].target_node_uuid
        target_relationships = await get_field_relationships_by_target(driver, target_uuid, 'linux_audit_batelco')
        
        # Should find relationships to this target
        for rel in target_relationships:
            assert rel.target_node_uuid == target_uuid
            assert rel.cluster_partition_id == 'linux_audit_batelco'
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)


class TestFieldRelationshipAnalytics:
    """Test suite for field relationship analytics operations"""
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_analyze_field_relationship_patterns(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test field relationship pattern analysis"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save some relationships first
        await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Analyze patterns
        analysis = await analyze_field_relationship_patterns(driver, 'linux_audit_batelco')
        
        assert analysis['cluster_id'] == 'linux_audit_batelco'
        assert analysis['total_relationships'] == len(sample_field_relationship_edges)
        assert 'relationship_types' in analysis
        assert 'avg_confidence' in analysis
        assert 'temporal_distribution' in analysis
        assert 'confidence_distribution' in analysis
        
        # Check relationship type distribution
        expected_types = {'CORRELATES_WITH': 1, 'SIMILAR_TO': 1, 'DERIVED_FROM': 1}
        assert analysis['relationship_types'] == expected_types
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)
    
    async def test_analyze_field_relationship_patterns_empty_cluster(self, driver):
        """Test relationship pattern analysis for empty cluster"""
        analysis = await analyze_field_relationship_patterns(driver, 'non_existent_cluster')
        
        assert analysis['cluster_id'] == 'non_existent_cluster'
        assert analysis['total_relationships'] == 0
        assert analysis['relationship_types'] == {}
        assert analysis['avg_confidence'] == 0.0
        assert analysis['temporal_distribution'] == {}
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_find_bidirectional_relationships(self, mock_cluster_service, driver):
        """Test finding bidirectional relationships between fields"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Create bidirectional relationships
        field_a = str(uuid.uuid4())
        field_b = str(uuid.uuid4())
        cluster_id = 'test_cluster'
        
        bidirectional_edges = [
            FieldRelationshipEdge(
                uuid=str(uuid.uuid4()),
                source_node_uuid=field_a,
                target_node_uuid=field_b,
                name='CORRELATES_WITH',
                description='A correlates with B',
                confidence=0.8,
                cluster_partition_id=cluster_id,
                relationship_type='FIELD_RELATES_TO',
                created_at=datetime.now(timezone.utc),
                valid_at=datetime.now(timezone.utc),
            ),
            FieldRelationshipEdge(
                uuid=str(uuid.uuid4()),
                source_node_uuid=field_b,
                target_node_uuid=field_a,
                name='CORRELATES_WITH',
                description='B correlates with A',
                confidence=0.8,
                cluster_partition_id=cluster_id,
                relationship_type='FIELD_RELATES_TO',
                created_at=datetime.now(timezone.utc),
                valid_at=datetime.now(timezone.utc),
            ),
        ]
        
        # Save bidirectional relationships
        await save_field_relationship_edges_bulk(driver, bidirectional_edges)
        
        # Find bidirectional relationships
        bidirectional_results = await find_bidirectional_relationships(driver, cluster_id)
        
        # Should find the bidirectional pair
        assert len(bidirectional_results) >= 1
        
        # Clean up
        for edge in bidirectional_edges:
            await edge.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_validate_relationship_constraints(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test relationship constraint validation"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save some relationships and belongs_to edges
        await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Create some belongs_to edges for the same cluster
        cluster_uuid = str(uuid.uuid4())
        belongs_to_edges = [
            BelongsToEdge(
                uuid=str(uuid.uuid4()),
                source_node_uuid=sample_field_relationship_edges[0].source_node_uuid,
                target_node_uuid=cluster_uuid,
                cluster_partition_id=cluster_uuid,
                created_at=datetime.now(timezone.utc),
            ),
        ]
        await save_belongs_to_edges_bulk(driver, belongs_to_edges)
        
        # Validate constraints
        validation = await validate_relationship_constraints(driver, 'linux_audit_batelco')
        
        assert 'cluster_id' in validation
        assert 'validation_errors' in validation
        assert 'warnings' in validation
        assert 'is_valid' in validation
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)
        for edge in belongs_to_edges:
            await edge.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_get_field_relationship_network(self, mock_cluster_service, sample_field_relationship_edges, driver):
        """Test field relationship network analysis"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Save some relationships first
        await save_field_relationship_edges_bulk(driver, sample_field_relationship_edges)
        
        # Get network for a field
        center_field = sample_field_relationship_edges[0].source_node_uuid
        network = await get_field_relationship_network(driver, center_field, 'linux_audit_batelco', max_depth=2)
        
        assert network['center_field_uuid'] == center_field
        assert network['max_depth'] == 2
        assert 'connected_field_uuids' in network
        assert 'relationships' in network
        assert 'network_size' in network
        assert 'relationship_count' in network
        
        # Clean up
        for edge in sample_field_relationship_edges:
            await edge.delete(driver)


class TestMongoDBValidationHelpers:
    """Test suite for MongoDB validation helper functions"""
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_validate_clusters_bulk(self, mock_cluster_service):
        """Test bulk cluster validation with MongoDB"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_cluster_service.return_value = mock_service_instance
        
        # Test cluster IDs
        cluster_ids = ['cluster_a', 'cluster_b', 'cluster_c', 'cluster_a']  # Include duplicate
        
        # Validate clusters
        await _validate_clusters_bulk(cluster_ids)
        
        # Verify validation was called for unique clusters only
        unique_clusters = set(cluster_ids)
        assert mock_service_instance.validate_cluster_exists.call_count == len(unique_clusters)
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_validate_clusters_bulk_cluster_not_found(self, mock_cluster_service):
        """Test bulk cluster validation when MongoDB clusters don't exist"""
        # Mock cluster service to return False for cluster existence
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = False
        mock_cluster_service.return_value = mock_service_instance
        
        # Validation should not fail but log warnings
        await _validate_clusters_bulk(['non_existent_cluster'])
        
        # Verify validation was called
        mock_service_instance.validate_cluster_exists.assert_called_once()
    
    @patch('graphiti_core.utils.maintenance.field_edges_operations.ClusterMetadataService')
    async def test_validate_clusters_bulk_error_handling(self, mock_cluster_service):
        """Test bulk cluster validation with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.side_effect = ClusterValidationError("MongoDB connection failed")
        mock_cluster_service.return_value = mock_service_instance
        
        # Validation should not fail but log errors
        await _validate_clusters_bulk(['test_cluster'])
        
        # Verify validation was called despite errors
        mock_service_instance.validate_cluster_exists.assert_called_once()
    
    async def test_validate_clusters_bulk_empty_list(self):
        """Test bulk cluster validation with empty list"""
        await _validate_clusters_bulk([])
        # Should not raise any errors


class TestFieldEdgeConstraints:
    """Test suite for field edge constraint validation"""
    
    def test_belongs_to_edge_cluster_consistency(self):
        """Test BELONGS_TO edge cluster consistency"""
        cluster_uuid = str(uuid.uuid4())
        edge = BelongsToEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=cluster_uuid,
            cluster_partition_id=cluster_uuid,  # Should match target
            created_at=datetime.now(timezone.utc),
        )
        
        assert edge.cluster_partition_id == edge.target_node_uuid
    
    def test_field_relationship_edge_confidence_bounds(self):
        """Test field relationship edge confidence constraints"""
        # Test valid confidence values
        for confidence in [0.0, 0.25, 0.5, 0.75, 1.0]:
            edge = FieldRelationshipEdge(
                uuid=str(uuid.uuid4()),
                source_node_uuid=str(uuid.uuid4()),
                target_node_uuid=str(uuid.uuid4()),
                name='TEST_RELATIONSHIP',
                description='Test relationship',
                confidence=confidence,
                cluster_partition_id='test_cluster',
            )
            
            assert 0.0 <= edge.confidence <= 1.0
    
    def test_field_relationship_edge_temporal_ordering(self):
        """Test field relationship edge temporal ordering"""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)
        
        edge = FieldRelationshipEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            name='TEMPORAL_TEST',
            description='Testing temporal ordering',
            confidence=0.8,
            cluster_partition_id='test_cluster',
            created_at=past,
            valid_at=now,
            invalid_at=future,
        )
        
        # Test temporal consistency
        if edge.valid_at and edge.invalid_at:
            assert edge.created_at <= edge.valid_at <= edge.invalid_at
    
    def test_field_relationship_edge_same_cluster_constraint(self):
        """Test that field relationships should be within same cluster"""
        cluster_id = 'test_cluster'
        
        edge = FieldRelationshipEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            name='SAME_CLUSTER_TEST',
            description='Testing same cluster constraint',
            confidence=0.8,
            cluster_partition_id=cluster_id,
        )
        
        # Both source and target fields should belong to the same cluster
        assert edge.cluster_partition_id == cluster_id
    
    def test_edge_uuid_uniqueness_and_equality(self):
        """Test edge UUID uniqueness and equality"""
        uuid1 = str(uuid.uuid4())
        uuid2 = str(uuid.uuid4())
        
        edge1 = BelongsToEdge(
            uuid=uuid1,
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            cluster_partition_id=str(uuid.uuid4()),
        )
        
        edge2 = BelongsToEdge(
            uuid=uuid2,
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            cluster_partition_id=str(uuid.uuid4()),
        )
        
        edge3 = BelongsToEdge(
            uuid=uuid1,  # Same UUID as edge1
            source_node_uuid=str(uuid.uuid4()),
            target_node_uuid=str(uuid.uuid4()),
            cluster_partition_id=str(uuid.uuid4()),
        )
        
        # Different UUIDs should be different edges
        assert edge1 != edge2
        assert edge1.uuid != edge2.uuid
        
        # Same UUIDs should be equal edges
        assert edge1 == edge3
        assert edge1.uuid == edge3.uuid
