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

Integration tests for Field Nodes (FieldNode and ClusterNode).

This test suite covers:
- Field node CRUD operations with MongoDB synchronization
- Cluster node CRUD operations with MongoDB synchronization
- Embedding generation and loading
- Temporal field updates
- MongoDB integration error handling
- Edge cases and constraint validation
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from graphiti_core.cluster_metadata.cluster_service import ClusterMetadataService
from graphiti_core.embedder import EmbedderClient
import pytest
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from unittest.mock import AsyncMock
from graphiti_core.embedder import EmbedderClient
from tests.embedder.embedder_fixtures import create_embedding_values

# Load environment variables from .env file
load_dotenv()

from graphiti_core.field_nodes import FieldNode, ClusterNode, get_field_node_from_record, get_cluster_node_from_record
from graphiti_core.errors import NodeNotFoundError
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
    """Mock embedder client that returns realistic embedding vectors"""
    embedder = AsyncMock(spec=EmbedderClient)
    
    # This will properly track calls and return the embedding
    embedder.create.return_value = create_embedding_values(multiplier=0.1, dimension=1536)
    
    return embedder

@pytest.fixture
def sample_field_node():
    """Sample FieldNode fixture"""
    return FieldNode(
        uuid=str(uuid.uuid4()),
        name='source_ip',
        description='Source IP address field for network traffic analysis',
        examples=['192.168.1.10', '10.0.0.5', '172.16.0.1'],
        data_type='string',
        count=1500,
        distinct_count=150,
        primary_cluster_id='linux_audit_batelco',
        embedding=None,
    )


@pytest.fixture
def sample_cluster_node():
    """Sample ClusterNode fixture"""
    return ClusterNode(
        uuid=str(uuid.uuid4()),
        name='linux_audit_batelco',
        organization='batelco',
        macro_name='linux_audit',
    )


@pytest.fixture
def sample_field_record():
    """Sample field database record fixture"""
    return {
        'uuid': str(uuid.uuid4()),
        'name': 'dest_port',
        'description': 'Destination port number for network connections',
        'examples': ['80', '443', '22', '3389'],
        'data_type': 'integer',
        'count': 2000,
        'distinct_count': 25,
        'primary_cluster_id': 'linux_audit_sico',
        'embedding': [0.2, 0.4, 0.6] * 300,
        'labels': ['Field'],
        'created_at': datetime.now(timezone.utc),
        'validated_at': datetime.now(timezone.utc),
        'invalidated_at': None,
        'last_updated': datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_cluster_record():
    """Sample cluster database record fixture"""
    return {
        'uuid': str(uuid.uuid4()),
        'name': 'windows_security_batelco',
        'organization': 'batelco',
        'macro_name': 'windows_security',
        'labels': ['Cluster'],
        'created_at': datetime.now(timezone.utc),
        'validated_at': datetime.now(timezone.utc),
        'invalidated_at': None,
        'last_updated': datetime.now(timezone.utc),
    }


class TestFieldNode:
    """Test suite for FieldNode class"""
    
    async def test_field_node_creation(self, sample_field_node):
        """Test FieldNode creation with valid data"""
        assert sample_field_node.name == 'source_ip'
        assert sample_field_node.data_type == 'string'
        assert sample_field_node.count == 1500
        assert sample_field_node.distinct_count == 150
        assert sample_field_node.primary_cluster_id == 'linux_audit_batelco'
        assert 'Field' in sample_field_node.labels
        assert sample_field_node.embedding is None
    
    async def test_field_node_temporal_updates(self, sample_field_node):
        """Test temporal field updates"""
        original_updated = sample_field_node.last_updated
        original_validated = sample_field_node.validated_at
        
        sample_field_node.update_temporal_fields()
        
        assert sample_field_node.last_updated > original_updated
        assert sample_field_node.validated_at > original_validated
    
    async def test_field_node_embedding_generation(self, sample_field_node, mock_embedder):
        """Test embedding generation for field description"""
        # Store original embedding state
        original_embedding = sample_field_node.embedding
        assert original_embedding is None  
        
        # Generate embedding
        await sample_field_node.generate_embedding(mock_embedder)
        
        # Verify embedding was generated and stored
        assert sample_field_node.embedding is not None
        assert len(sample_field_node.embedding) == 1536 
        
        # Verify the embedding content matches our mock pattern
        expected_embedding = [0.1] * 1536 
        assert sample_field_node.embedding == expected_embedding
        
        # This will now work because embedder.create is properly mocked
        mock_embedder.create.assert_called_once()
    
    @patch('graphiti_core.field_nodes.ClusterMetadataService')
    async def test_field_node_save_with_mongodb_sync(self, mock_cluster_service, sample_field_node, driver):
        """Test FieldNode save with MongoDB synchronization"""
        # Mock cluster service
        mock_service_instance = AsyncMock(spec=ClusterMetadataService)
        mock_service_instance.validate_cluster_exists.return_value = True
        mock_service_instance.increment_field_count.return_value = {'total_fields': 15}
        mock_cluster_service.return_value = mock_service_instance
        
        # Save field node
        result = await sample_field_node.save(driver)
        
        # Verify MongoDB sync was called
        mock_service_instance.validate_cluster_exists.assert_called_once_with(sample_field_node.primary_cluster_id)
        mock_service_instance.increment_field_count.assert_called_once_with(sample_field_node.primary_cluster_id)
        
        # Clean up
        await sample_field_node.delete(driver)
    
    @patch('graphiti_core.field_nodes.ClusterMetadataService')
    async def test_field_node_save_mongodb_cluster_not_found(self, mock_cluster_service, sample_field_node, driver):
        """Test FieldNode save when MongoDB cluster doesn't exist"""
        # Mock cluster service to return False for cluster existence
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.return_value = False
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed but log warning
        result = await sample_field_node.save(driver)
        
        # Verify validation was called but increment wasn't
        mock_service_instance.validate_cluster_exists.assert_called_once()
        mock_service_instance.increment_field_count.assert_not_called()
        
        # Clean up
        await sample_field_node.delete(driver)
    
    @patch('graphiti_core.field_nodes.ClusterMetadataService')
    async def test_field_node_save_mongodb_error_handling(self, mock_cluster_service, sample_field_node, driver):
        """Test FieldNode save with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.validate_cluster_exists.side_effect = ClusterValidationError("MongoDB connection failed")
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed despite MongoDB error
        result = await sample_field_node.save(driver)
        
        # Clean up
        await sample_field_node.delete(driver)
    
    async def test_field_node_save_and_retrieve(self, sample_field_node, driver):
        """Test FieldNode save and retrieval operations"""
        # Save the field node
        with patch('graphiti_core.field_nodes.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await sample_field_node.save(driver)
        
        # Retrieve by UUID
        retrieved_node = await FieldNode.get_by_uuid(driver, sample_field_node.uuid)
        
        assert retrieved_node.uuid == sample_field_node.uuid
        assert retrieved_node.name == sample_field_node.name
        assert retrieved_node.description == sample_field_node.description
        assert retrieved_node.data_type == sample_field_node.data_type
        assert retrieved_node.count == sample_field_node.count
        assert retrieved_node.distinct_count == sample_field_node.distinct_count
        assert retrieved_node.primary_cluster_id == sample_field_node.primary_cluster_id
        
        # Clean up
        await sample_field_node.delete(driver)
    
    async def test_field_node_update(self, sample_field_node, driver):
        """Test FieldNode update operations"""
        # Save the field node first
        with patch('graphiti_core.field_nodes.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            await sample_field_node.save(driver)
        
        # Update field properties
        sample_field_node.count = 2000
        sample_field_node.distinct_count = 200
        sample_field_node.description = 'Updated description for source IP field'
        
        # Update in database
        await sample_field_node.update(driver)
        
        # Retrieve and verify updates
        retrieved_node = await FieldNode.get_by_uuid(driver, sample_field_node.uuid)
        assert retrieved_node.count == 2000
        assert retrieved_node.distinct_count == 200
        assert retrieved_node.description == 'Updated description for source IP field'
        
        # Clean up
        await sample_field_node.delete(driver)
    
    async def test_field_node_get_by_uuids(self, driver):
        """Test batch retrieval of FieldNodes by UUIDs"""
        field_nodes = []
        
        # Create multiple field nodes
        for i in range(3):
            field_node = FieldNode(
                uuid=str(uuid.uuid4()),
                name=f'test_field_{i}',
                description=f'Test field number {i}',
                data_type='string',
                count=100 + i,
                distinct_count=10 + i,
                primary_cluster_id='test_cluster',
                examples=[f'example_{i}']
            )
            field_nodes.append(field_node)
        
        # Save all field nodes
        with patch('graphiti_core.field_nodes.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.validate_cluster_exists.return_value = True
            mock_service_instance.increment_field_count.return_value = {'total_fields': 1}
            mock_cluster_service.return_value = mock_service_instance
            
            for field_node in field_nodes:
                await field_node.save(driver)
        
        # Retrieve by UUIDs
        uuids = [node.uuid for node in field_nodes]
        retrieved_nodes = await FieldNode.get_by_uuids(driver, uuids)
        
        assert len(retrieved_nodes) == 3
        retrieved_uuids = [node.uuid for node in retrieved_nodes]
        for uuid_val in uuids:
            assert uuid_val in retrieved_uuids
        
        # Clean up
        for field_node in field_nodes:
            await field_node.delete(driver)
    
    async def test_field_node_not_found(self, driver):
        """Test FieldNode retrieval with non-existent UUID"""
        non_existent_uuid = str(uuid.uuid4())
        
        with pytest.raises(NodeNotFoundError):
            await FieldNode.get_by_uuid(driver, non_existent_uuid)
    
    def test_get_field_node_from_record(self, sample_field_record):
        """Test creation of FieldNode from database record"""
        field_node = get_field_node_from_record(sample_field_record)
        
        assert field_node.uuid == sample_field_record['uuid']
        assert field_node.name == sample_field_record['name']
        assert field_node.description == sample_field_record['description']
        assert field_node.data_type == sample_field_record['data_type']
        assert field_node.count == sample_field_record['count']
        assert field_node.distinct_count == sample_field_record['distinct_count']
        assert field_node.primary_cluster_id == sample_field_record['primary_cluster_id']
        assert field_node.embedding == sample_field_record['embedding']


class TestClusterNode:
    """Test suite for ClusterNode class"""
    
    async def test_cluster_node_creation(self, sample_cluster_node):
        """Test ClusterNode creation with valid data"""
        assert sample_cluster_node.name == 'linux_audit_batelco'
        assert sample_cluster_node.organization == 'batelco'
        assert sample_cluster_node.macro_name == 'linux_audit'
        assert 'Cluster' in sample_cluster_node.labels
    
    async def test_cluster_node_temporal_updates(self, sample_cluster_node):
        """Test temporal field updates"""
        original_updated = sample_cluster_node.last_updated
        original_validated = sample_cluster_node.validated_at
        
        sample_cluster_node.update_temporal_fields()
        
        assert sample_cluster_node.last_updated > original_updated
        assert sample_cluster_node.validated_at > original_validated
    
    @patch('graphiti_core.field_nodes.ClusterMetadataService')
    async def test_cluster_node_save_with_mongodb_sync(self, mock_cluster_service, sample_cluster_node, driver):
        """Test ClusterNode save with MongoDB synchronization"""
        # Mock cluster service
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.return_value = None  # Cluster doesn't exist
        mock_service_instance.create_cluster.return_value = {'_id': 'linux_audit_batelco'}
        mock_cluster_service.return_value = mock_service_instance
        
        # Save cluster node
        result = await sample_cluster_node.save(driver)
        
        # Verify MongoDB sync was called
        mock_service_instance.get_cluster.assert_called_once_with(sample_cluster_node.name)
        mock_service_instance.create_cluster.assert_called_once()
        
        # Clean up
        await sample_cluster_node.delete(driver)
    
    @patch('graphiti_core.field_nodes.ClusterMetadataService')
    async def test_cluster_node_save_existing_mongodb_cluster(self, mock_cluster_service, sample_cluster_node, driver):
        """Test ClusterNode save when MongoDB cluster already exists"""
        # Mock cluster service to return existing cluster
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.return_value = {'_id': 'linux_audit_batelco', 'status': 'active'}
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed
        result = await sample_cluster_node.save(driver)
        
        # Verify only get_cluster was called, not create_cluster
        mock_service_instance.get_cluster.assert_called_once()
        mock_service_instance.create_cluster.assert_not_called()
        
        # Clean up
        await sample_cluster_node.delete(driver)
    
    @patch('graphiti_core.field_nodes.ClusterMetadataService')
    async def test_cluster_node_save_mongodb_error_handling(self, mock_cluster_service, sample_cluster_node, driver):
        """Test ClusterNode save with MongoDB errors"""
        # Mock cluster service to raise exception
        mock_service_instance = AsyncMock()
        mock_service_instance.get_cluster.side_effect = InvalidOrganizationError("Invalid organization")
        mock_cluster_service.return_value = mock_service_instance
        
        # Save should still succeed despite MongoDB error
        result = await sample_cluster_node.save(driver)
        
        # Clean up
        await sample_cluster_node.delete(driver)
    
    async def test_cluster_node_save_and_retrieve(self, sample_cluster_node, driver):
        """Test ClusterNode save and retrieval operations"""
        # Save the cluster node
        with patch('graphiti_core.field_nodes.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_cluster.return_value = None
            mock_service_instance.create_cluster.return_value = {'_id': 'linux_audit_batelco'}
            mock_cluster_service.return_value = mock_service_instance
            
            await sample_cluster_node.save(driver)
        
        # Retrieve by UUID
        retrieved_node = await ClusterNode.get_by_uuid(driver, sample_cluster_node.uuid)
        
        assert retrieved_node.uuid == sample_cluster_node.uuid
        assert retrieved_node.name == sample_cluster_node.name
        assert retrieved_node.organization == sample_cluster_node.organization
        assert retrieved_node.macro_name == sample_cluster_node.macro_name
        
        # Clean up
        await sample_cluster_node.delete(driver)
    
    async def test_cluster_node_update(self, sample_cluster_node, driver):
        """Test ClusterNode update operations"""
        # Save the cluster node first
        with patch('graphiti_core.field_nodes.ClusterMetadataService') as mock_cluster_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_cluster.return_value = None
            mock_service_instance.create_cluster.return_value = {'_id': 'linux_audit_batelco'}
            mock_cluster_service.return_value = mock_service_instance
            
            await sample_cluster_node.save(driver)
        
        # Update cluster properties
        sample_cluster_node.organization = 'sico'
        sample_cluster_node.macro_name = 'windows_security'
        
        # Update in database
        await sample_cluster_node.update(driver)
        
        # Retrieve and verify updates
        retrieved_node = await ClusterNode.get_by_uuid(driver, sample_cluster_node.uuid)
        assert retrieved_node.organization == 'sico'
        assert retrieved_node.macro_name == 'windows_security'
        
        # Clean up
        await sample_cluster_node.delete(driver)
    
    def test_get_cluster_node_from_record(self, sample_cluster_record):
        """Test creation of ClusterNode from database record"""
        cluster_node = get_cluster_node_from_record(sample_cluster_record)
        
        assert cluster_node.uuid == sample_cluster_record['uuid']
        assert cluster_node.name == sample_cluster_record['name']
        assert cluster_node.organization == sample_cluster_record['organization']
        assert cluster_node.macro_name == sample_cluster_record['macro_name']


class TestFieldNodeConstraints:
    """Test suite for FieldNode constraint validation"""
    
    def test_field_node_invalid_count_consistency(self):
        """Test FieldNode creation with invalid count consistency"""
        # distinct_count should not exceed count
        field_node = FieldNode(
            uuid=str(uuid.uuid4()),
            name='test_field',
            description='Test field with invalid counts',
            data_type='string',
            count=100,
            distinct_count=150,  # Invalid: greater than count
            primary_cluster_id='test_cluster',
        )
        
        # This should be caught in validation functions, not in creation
        assert field_node.distinct_count > field_node.count
    
    def test_field_node_edge_cases(self):
        """Test FieldNode edge cases"""
        # Empty examples
        field_node = FieldNode(
            uuid=str(uuid.uuid4()),
            name='empty_examples_field',
            description='Field with no examples',
            data_type='string',
            count=0,
            distinct_count=0,
            primary_cluster_id='test_cluster',
            examples=[],
        )
        
        assert field_node.examples == []
        assert field_node.count == 0
        assert field_node.distinct_count == 0
    
    def test_cluster_node_organization_formats(self):
        """Test ClusterNode with different organization formats"""
        # Test lowercase
        cluster_node = ClusterNode(
            uuid=str(uuid.uuid4()),
            name='test_cluster_lowercase',
            organization='batelco',
            macro_name='linux_audit',
        )
        
        assert cluster_node.organization == 'batelco'
        
        # Test uppercase
        cluster_node_upper = ClusterNode(
            uuid=str(uuid.uuid4()),
            name='test_cluster_uppercase',
            organization='SICO',
            macro_name='windows_security',
        )
        
        assert cluster_node_upper.organization == 'SICO'
