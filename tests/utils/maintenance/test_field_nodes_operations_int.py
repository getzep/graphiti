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

Integration tests for Field Nodes Operations (bulk operations and analysis).

This test suite covers:
- Bulk field node operations with MongoDB synchronization
- Bulk cluster node operations with MongoDB synchronization
- Field distribution analysis
- Field similarity detection
- Field-cluster consistency validation
- Complex field operations and analytics
- MongoDB integration error handling in bulk operations
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncGraphDatabase

from graphiti_core.field_nodes import FieldNode, ClusterNode
from graphiti_core.utils.maintenance.field_nodes_operations import (
    get_fields_by_cluster_id,
    search_fields_by_name,
    save_fields_bulk,
    create_field_embeddings_batch,
    get_clusters_by_organization,
    save_clusters_bulk,
    analyze_field_distribution_by_cluster,
    find_similar_fields_across_clusters,
    validate_field_cluster_consistency,
    _sync_field_counts_bulk,
    _sync_clusters_bulk,
)
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError as MongoClusterNotFoundError,
    ClusterValidationError,
    DuplicateClusterError,
    InvalidOrganizationError,
)

# Test configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')

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
def sample_field_nodes():
    """Sample FieldNode list fixture for bulk operations"""
    return [
        FieldNode(
            uuid=str(uuid.uuid4()),
            name='source_ip',
            description='Source IP address for network connections',
            examples=['192.168.1.10', '10.0.0.5'],
            data_type='string',
            count=1500,
            distinct_count=150,
            primary_cluster_id='linux_audit_batelco',
        ),
        FieldNode(
            uuid=str(uuid.uuid4()),
            name='dest_port',
            description='Destination port number for network traffic',
            examples=['80', '443', '22'],
            data_type='integer',
            count=2000,
            distinct_count=25,
            primary_cluster_id='linux_audit_batelco',
        ),
        FieldNode(
            uuid=str(uuid.uuid4()),
            name='user_name',
            description='Username for authentication events',
            examples=['admin', 'user1', 'service_account'],
            data_type='string',
            count=800,
            distinct_count=45,
            primary_cluster_id='windows_security_sico',
        ),
    ]


@pytest.fixture
def sample_cluster_nodes():
    """Sample ClusterNode list fixture for bulk operations"""
    return [
        ClusterNode(
            uuid=str(uuid.uuid4()),
            name='linux_audit_batelco',
            organization='batelco',
            macro_name='linux_audit',
        ),
        ClusterNode(
            uuid=str(uuid.uuid4()),
            name='windows_security_sico',
            organization='sico',
            macro_name='windows_security',
        ),
        ClusterNode(
            uuid=str(uuid.uuid4()),
            name='network_traffic_batelco',
            organization='batelco',
            macro_name='network_traffic',
        ),
    ]


class TestFieldNodesBulkOperations:
    """Test suite for bulk field node operations"""
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_save_fields_bulk_with_mongodb_sync(self, mock_cluster_service, sample_field_nodes, driver):
        """Test bulk field save with MongoDB synchronization"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_service_instance.increment_field_count.return_value = {'total_fields': 15}
        mock_cluster_service.return_value = mock_service_instance
        
        # Save fields in bulk
        result = await save_fields_bulk(driver, sample_field_nodes)
        
        # Verify MongoDB sync was called for each cluster
        expected_calls = 2  # linux_audit_batelco (2 fields) + windows_security_sico (1 field)
        assert mock_service_instance.validate_cluster_exists.call_count == expected_calls
        
        # Clean up
        for field in sample_field_nodes:
            await field.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_save_fields_bulk_mongodb_error_handling(self, mock_cluster_service, sample_field_nodes, driver):
        """Test bulk field save with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.side_effect = ClusterValidationError("MongoDB connection failed")
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed despite MongoDB error
        result = await save_fields_bulk(driver, sample_field_nodes)
        
        # Clean up
        for field in sample_field_nodes:
            await field.delete(driver)
    
    async def test_save_fields_bulk_empty_list(self, driver):
        """Test bulk field save with empty list"""
        result = await save_fields_bulk(driver, [])
        assert result == []
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_save_clusters_bulk_with_mongodb_sync(self, mock_cluster_service, sample_cluster_nodes, driver):
        """Test bulk cluster save with MongoDB synchronization"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.return_value = None  # Clusters don't exist
        mock_service_instance.create_cluster.return_value = {'_id': 'test_cluster'}
        mock_cluster_service.return_value = mock_service_instance
        
        # Save clusters in bulk
        result = await save_clusters_bulk(driver, sample_cluster_nodes)
        
        # Verify MongoDB sync was called for each cluster
        assert mock_service_instance.get_cluster.call_count == len(sample_cluster_nodes)
        assert mock_service_instance.create_cluster.call_count == len(sample_cluster_nodes)
        
        # Clean up
        for cluster in sample_cluster_nodes:
            await cluster.delete(driver)
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_save_clusters_bulk_existing_mongodb_clusters(self, mock_cluster_service, sample_cluster_nodes, driver):
        """Test bulk cluster save when MongoDB clusters already exist"""
        # Mock cluster service to return existing clusters
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.return_value = {'_id': 'existing_cluster', 'status': 'active'}
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed
        result = await save_clusters_bulk(driver, sample_cluster_nodes)
        
        # Verify only get_cluster was called, not create_cluster
        assert mock_service_instance.get_cluster.call_count == len(sample_cluster_nodes)
        mock_service_instance.create_cluster.assert_not_called()
        
        # Clean up
        for cluster in sample_cluster_nodes:
            await cluster.delete(driver)
    
    async def test_create_field_embeddings_batch(self, sample_field_nodes, mock_embedder):
        """Test batch embedding creation for fields"""
        await create_field_embeddings_batch(mock_embedder, sample_field_nodes)
        
        # Verify embeddings were created
        for field in sample_field_nodes:
            assert field.embedding is not None
            assert len(field.embedding) == 1000
        
        # Verify embedder was called with correct batch
        mock_embedder.create_batch.assert_called_once()
        call_args = mock_embedder.create_batch.call_args[0][0]
        assert len(call_args) == len(sample_field_nodes)
    
    async def test_create_field_embeddings_batch_empty_list(self, mock_embedder):
        """Test batch embedding creation with empty list"""
        await create_field_embeddings_batch(mock_embedder, [])
        mock_embedder.create_batch.assert_not_called()


class TestFieldNodesQueryOperations:
    """Test suite for field node query operations"""
    
    async def test_get_fields_by_cluster_id(self, sample_field_nodes, driver):
        """Test retrieval of fields by cluster ID"""
        # Save some fields first
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_fields_bulk(driver, sample_field_nodes)
        
        # Test retrieval
        cluster_fields = await get_fields_by_cluster_id(driver, 'linux_audit_batelco')
        
        # Should find 2 fields for this cluster
        assert len(cluster_fields) == 2
        field_names = [field.name for field in cluster_fields]
        assert 'source_ip' in field_names
        assert 'dest_port' in field_names
        
        # Clean up
        for field in sample_field_nodes:
            await field.delete(driver)
    
    async def test_search_fields_by_name(self, sample_field_nodes, driver):
        """Test field search by name pattern"""
        # Save some fields first
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_fields_bulk(driver, sample_field_nodes)
        
        # Test search by name pattern
        results = await search_fields_by_name(driver, 'source', limit=10)
        
        # Should find fields containing 'source'
        assert len(results) >= 1
        assert any('source' in field.name for field in results)
        
        # Test search with cluster filter
        results_filtered = await search_fields_by_name(driver, 'dest', cluster_id='linux_audit_batelco', limit=10)
        
        # Should find dest_port in the specified cluster
        assert len(results_filtered) >= 1
        assert any('dest' in field.name for field in results_filtered)
        
        # Clean up
        for field in sample_field_nodes:
            await field.delete(driver)
    
    async def test_get_clusters_by_organization(self, sample_cluster_nodes, driver):
        """Test retrieval of clusters by organization"""
        # Save some clusters first
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_cluster.return_value = None
            mock_service_instance.create_cluster.return_value = {'_id': 'test_cluster'}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_clusters_bulk(driver, sample_cluster_nodes)
        
        # Test retrieval by organization
        batelco_clusters = await get_clusters_by_organization(driver, 'batelco')
        
        # Should find 2 clusters for batelco
        assert len(batelco_clusters) == 2
        for cluster in batelco_clusters:
            assert cluster.organization == 'batelco'
        
        # Clean up
        for cluster in sample_cluster_nodes:
            await cluster.delete(driver)


class TestFieldAnalyticsOperations:
    """Test suite for field analytics and analysis operations"""
    
    async def test_analyze_field_distribution_by_cluster(self, sample_field_nodes, driver):
        """Test field distribution analysis for a cluster"""
        # Save some fields first
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_fields_bulk(driver, sample_field_nodes)
        
        # Analyze distribution
        analysis = await analyze_field_distribution_by_cluster(driver, 'linux_audit_batelco')
        
        assert analysis['cluster_id'] == 'linux_audit_batelco'
        assert analysis['total_fields'] == 2
        assert analysis['total_events'] == 3500  # 1500 + 2000
        assert 'data_type_distribution' in analysis
        assert analysis['data_type_distribution']['string'] == 1
        assert analysis['data_type_distribution']['integer'] == 1
        
        # Clean up
        for field in sample_field_nodes:
            await field.delete(driver)
    
    async def test_analyze_field_distribution_empty_cluster(self, driver):
        """Test field distribution analysis for empty cluster"""
        analysis = await analyze_field_distribution_by_cluster(driver, 'non_existent_cluster')
        
        assert analysis['cluster_id'] == 'non_existent_cluster'
        assert analysis['total_fields'] == 0
        assert analysis['total_events'] == 0
        assert analysis['data_type_distribution'] == {}
        assert analysis['avg_distinct_ratio'] == 0.0
    
    async def test_find_similar_fields_across_clusters(self, driver):
        """Test finding similar fields across different clusters"""
        # Create fields with similar names in different clusters
        similar_fields = [
            FieldNode(
                uuid=str(uuid.uuid4()),
                name='src_ip',
                description='Source IP address',
                data_type='string',
                count=100,
                distinct_count=50,
                primary_cluster_id='cluster_a',
            ),
            FieldNode(
                uuid=str(uuid.uuid4()),
                name='source_ip_addr',
                description='Source IP address field',
                data_type='string',
                count=150,
                distinct_count=75,
                primary_cluster_id='cluster_b',
            ),
        ]
        
        # Save fields
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_fields_bulk(driver, similar_fields)
        
        # Find similar fields
        similar_results = await find_similar_fields_across_clusters(driver, 'ip', min_similarity_threshold=0.8)
        
        # Should find fields with 'ip' in the name across different clusters
        assert len(similar_results) >= 1
        
        # Clean up
        for field in similar_fields:
            await field.delete(driver)
    
    async def test_validate_field_cluster_consistency(self, sample_field_nodes, driver):
        """Test field-cluster consistency validation"""
        # Save some fields first
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_fields_bulk(driver, sample_field_nodes)
        
        # Validate consistency
        validation = await validate_field_cluster_consistency(driver, 'linux_audit_batelco')
        
        assert validation['cluster_id'] == 'linux_audit_batelco'
        assert validation['total_fields'] == 2
        assert 'validation_errors' in validation
        assert 'warnings' in validation
        assert 'is_valid' in validation
        
        # Clean up
        for field in sample_field_nodes:
            await field.delete(driver)
    
    async def test_validate_field_cluster_consistency_with_errors(self, driver):
        """Test field-cluster consistency validation with problematic data"""
        # Create field with invalid data
        invalid_field = FieldNode(
            uuid=str(uuid.uuid4()),
            name='',  # Empty name should trigger error
            description='',  # Empty description should trigger error
            data_type='',  # Empty data type should trigger error
            count=100,
            distinct_count=150,  # Invalid: distinct > count
            primary_cluster_id='test_cluster',
        )
        
        # Save field
        with patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await save_fields_bulk(driver, [invalid_field])
        
        # Validate consistency
        validation = await validate_field_cluster_consistency(driver, 'test_cluster')
        
        assert validation['is_valid'] is False
        assert len(validation['validation_errors']) > 0
        
        # Clean up
        await invalid_field.delete(driver)


class TestMongoDBSyncHelpers:
    """Test suite for MongoDB synchronization helper functions"""
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_sync_field_counts_bulk(self, mock_cluster_service, sample_field_nodes):
        """Test bulk field count synchronization with MongoDB"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_service_instance.increment_field_count.return_value = {'total_fields': 15}
        mock_cluster_service.return_value = mock_service_instance
        
        # Sync field counts
        await _sync_field_counts_bulk(sample_field_nodes)
        
        # Verify cluster validation was called for unique clusters
        expected_clusters = {'linux_audit_batelco', 'windows_security_sico'}
        assert mock_service_instance.validate_cluster_exists.call_count == len(expected_clusters)
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_sync_field_counts_bulk_cluster_not_found(self, mock_cluster_service, sample_field_nodes):
        """Test bulk field count sync when MongoDB cluster doesn't exist"""
        # Mock cluster service to return False for cluster existence
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = False
        mock_cluster_service.return_value = mock_service_instance
        
        # Sync should not fail but log warnings
        await _sync_field_counts_bulk(sample_field_nodes)
        
        # Verify validation was called but increment wasn't
        assert mock_service_instance.validate_cluster_exists.call_count >= 1
        mock_service_instance.increment_field_count.assert_not_called()
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_sync_clusters_bulk(self, mock_cluster_service, sample_cluster_nodes):
        """Test bulk cluster synchronization with MongoDB"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.return_value = None  # Clusters don't exist
        mock_service_instance.create_cluster.return_value = {'_id': 'test_cluster'}
        mock_cluster_service.return_value = mock_service_instance
        
        # Sync clusters
        await _sync_clusters_bulk(sample_cluster_nodes)
        
        # Verify MongoDB operations were called for each cluster
        assert mock_service_instance.get_cluster.call_count == len(sample_cluster_nodes)
        assert mock_service_instance.create_cluster.call_count == len(sample_cluster_nodes)
    
    @patch('graphiti_core.utils.maintenance.field_nodes_operations.ClusterMetadataService')
    async def test_sync_clusters_bulk_error_handling(self, mock_cluster_service, sample_cluster_nodes):
        """Test bulk cluster sync with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.side_effect = InvalidOrganizationError("Invalid organization")
        mock_cluster_service.return_value = mock_service_instance
        
        # Sync should not fail but log errors
        await _sync_clusters_bulk(sample_cluster_nodes)
        
        # Verify get_cluster was called despite errors
        assert mock_service_instance.get_cluster.call_count == len(sample_cluster_nodes)
    
    async def test_sync_field_counts_bulk_empty_list(self):
        """Test bulk field count sync with empty list"""
        await _sync_field_counts_bulk([])
        # Should not raise any errors
    
    async def test_sync_clusters_bulk_empty_list(self):
        """Test bulk cluster sync with empty list"""
        await _sync_clusters_bulk([])
        # Should not raise any errors


class TestFieldNodesConstraints:
    """Test suite for field nodes constraint validation"""
    
    def test_field_count_consistency_detection(self):
        """Test detection of field count consistency issues"""
        field = FieldNode(
            uuid=str(uuid.uuid4()),
            name='test_field',
            description='Test field with count issues',
            data_type='string',
            count=100,
            distinct_count=150,  # Invalid: greater than count
            primary_cluster_id='test_cluster',
        )
        
        # This should be caught by validation functions
        assert field.distinct_count > field.count
    
    def test_cluster_organization_validation(self):
        """Test cluster organization format validation"""
        cluster = ClusterNode(
            uuid=str(uuid.uuid4()),
            name='test_cluster',
            organization='batelco',
            macro_name='linux_audit',
        )
        
        assert cluster.organization == 'batelco'
        assert cluster.macro_name == 'linux_audit'
        assert cluster.name == 'test_cluster'
    
    def test_field_data_type_constraints(self):
        """Test field data type constraints"""
        valid_types = ['string', 'integer', 'float', 'boolean', 'timestamp']
        
        for data_type in valid_types:
            field = FieldNode(
                uuid=str(uuid.uuid4()),
                name=f'test_field_{data_type}',
                description=f'Test field for {data_type} type',
                data_type=data_type,
                count=100,
                distinct_count=50,
                primary_cluster_id='test_cluster',
            )
            
            assert field.data_type == data_type
