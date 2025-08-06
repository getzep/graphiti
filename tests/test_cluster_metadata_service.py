"""
Comprehensive test suite for ClusterMetadataService.

This test suite covers all aspects of the cluster metadata service including:
- CRUD operations (create, read, update, delete)
- Search functionality with various criteria
- Organization and macro validation
- Error handling and edge cases
- Field count management
- Statistics generation
- Cache functionality
"""

import pytest
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv

from graphiti_core.cluster_metadata.cluster_service import ClusterMetadataService
from graphiti_core.cluster_metadata.models import (
    ClusterCreateRequest,
    ClusterUpdateRequest,
    ClusterSearchRequest,
    ClusterCriteriaRequest,
    ClusterStats
)
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError,
    DuplicateClusterError,
    ClusterValidationError,
    InvalidOrganizationError,
    ClusterUpdateError
)
from graphiti_core.utils.datetime_utils import utc_now

# Load environment variables
load_dotenv()

# Test configuration
TEST_ORG_BATELCO = "Batelco"
TEST_ORG_SICO = "SICO"
TEST_MACRO_LINUX_AUDIT = "linux_audit"  # Back to original with underscore
TEST_CLUSTER_ID_BATELCO = "linux_audit_batelco"
TEST_CLUSTER_ID_SICO = "linux_audit_sico"

pytestmark = pytest.mark.asyncio


def setup_logging():
    """Setup logging for tests"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


@pytest.fixture
def logger():
    """Logger fixture for tests"""
    return setup_logging()


@pytest.fixture
def cluster_service():
    """Fixture providing ClusterMetadataService instance"""
    return ClusterMetadataService()


@pytest.fixture
def mock_collection():
    """Fixture providing mock MongoDB collection"""
    return AsyncMock()


@pytest.fixture
def sample_cluster_batelco():
    """Sample cluster data for Batelco organization"""
    now = utc_now()
    return {
        "_id": TEST_CLUSTER_ID_BATELCO,
        "cluster_id": TEST_CLUSTER_ID_BATELCO,
        "cluster_uuid": str(uuid.uuid4()),
        "macro_name": TEST_MACRO_LINUX_AUDIT,
        "organization": TEST_ORG_BATELCO,  # Stored in proper case as per updated model validation
        "description": "Linux audit cluster for Batelco organization",
        "status": "active",
        "total_fields": 25,
        "created_at": now,
        "last_updated": now,
        "created_by": "test_user"
    }


@pytest.fixture
def sample_cluster_sico():
    """Sample cluster data for SICO organization"""
    now = utc_now()
    return {
        "_id": TEST_CLUSTER_ID_SICO,
        "cluster_id": TEST_CLUSTER_ID_SICO,
        "cluster_uuid": str(uuid.uuid4()),
        "macro_name": TEST_MACRO_LINUX_AUDIT,
        "organization": TEST_ORG_SICO,  # Stored in proper case as per updated model validation
        "description": "Linux audit cluster for SICO organization",
        "status": "active",
        "total_fields": 30,
        "created_at": now,
        "last_updated": now,
        "created_by": "test_user"
    }


@pytest.fixture
def create_request_batelco():
    """ClusterCreateRequest for Batelco"""
    return ClusterCreateRequest(
        macro_name=TEST_MACRO_LINUX_AUDIT,
        organization=TEST_ORG_BATELCO,
        cluster_uuid=str(uuid.uuid4()),
        description="Test Linux audit cluster for Batelco",
        cluster_id=TEST_CLUSTER_ID_BATELCO,
        status="active",
        total_fields=0,
        created_by="test_user"
    )


@pytest.fixture
def create_request_sico():
    """ClusterCreateRequest for SICO"""
    return ClusterCreateRequest(
        macro_name=TEST_MACRO_LINUX_AUDIT,
        organization=TEST_ORG_SICO,
        cluster_uuid=str(uuid.uuid4()),
        description="Test Linux audit cluster for SICO",
        cluster_id=TEST_CLUSTER_ID_SICO,
        status="active",
        total_fields=0,
        created_by="test_user"
    )


class TestClusterMetadataServiceCRUD:
    """Test suite for basic CRUD operations"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_success(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test successful cluster retrieval"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.get_cluster(TEST_CLUSTER_ID_BATELCO)
        
        # Verify
        assert result is not None
        assert result["cluster_id"] == TEST_CLUSTER_ID_BATELCO
        assert result["organization"] == TEST_ORG_BATELCO
        assert result["macro_name"] == TEST_MACRO_LINUX_AUDIT
        mock_collection.find_one.assert_called_once_with({"_id": TEST_CLUSTER_ID_BATELCO})
        logger.info(f"✓ Successfully retrieved cluster: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_not_found(self, mock_get_collection, cluster_service, logger):
        """Test cluster retrieval when cluster doesn't exist"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.get_cluster("nonexistent_cluster")
        
        # Verify
        assert result is None
        mock_collection.find_one.assert_called_once_with({"_id": "nonexistent_cluster"})
        logger.info("✓ Correctly handled non-existent cluster")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_database_error(self, mock_get_collection, cluster_service, logger):
        """Test cluster retrieval with database error"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.side_effect = Exception("Database connection error")
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        with pytest.raises(ClusterValidationError, match="Failed to retrieve cluster"):
            await cluster_service.get_cluster(TEST_CLUSTER_ID_BATELCO)
        logger.info("✓ Correctly handled database error during retrieval")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_create_cluster_success_batelco(self, mock_get_collection, cluster_service, create_request_batelco, logger):
        """Test successful cluster creation for Batelco"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None  # No existing cluster
        mock_collection.insert_one.return_value = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock validator
        cluster_service.validator.validate_organization = AsyncMock(return_value=TEST_ORG_BATELCO)
        cluster_service.validator.validate_macro_name = AsyncMock(return_value=True)
        
        # Execute
        result = await cluster_service.create_cluster(create_request_batelco)
        
        # Verify
        assert result["cluster_id"] == TEST_CLUSTER_ID_BATELCO
        assert result["organization"] == TEST_ORG_BATELCO  # Model preserves original case
        assert result["macro_name"] == TEST_MACRO_LINUX_AUDIT
        assert result["status"] == "active"
        assert result["total_fields"] == 0
        mock_collection.insert_one.assert_called_once()
        logger.info(f"✓ Successfully created cluster for Batelco: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_create_cluster_success_sico(self, mock_get_collection, cluster_service, create_request_sico, logger):
        """Test successful cluster creation for SICO"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None  # No existing cluster
        mock_collection.insert_one.return_value = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock validator
        cluster_service.validator.validate_organization = AsyncMock(return_value=TEST_ORG_SICO)
        cluster_service.validator.validate_macro_name = AsyncMock(return_value=True)
        
        # Execute
        result = await cluster_service.create_cluster(create_request_sico)
        
        # Verify
        assert result["cluster_id"] == TEST_CLUSTER_ID_SICO
        assert result["organization"] == TEST_ORG_SICO  # Model preserves original case
        assert result["macro_name"] == TEST_MACRO_LINUX_AUDIT
        assert result["status"] == "active"
        assert result["total_fields"] == 0
        mock_collection.insert_one.assert_called_once()
        logger.info(f"✓ Successfully created cluster for SICO: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_create_cluster_duplicate_error(self, mock_get_collection, cluster_service, create_request_batelco, sample_cluster_batelco, logger):
        """Test cluster creation when cluster already exists"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco  # Existing cluster
        mock_get_collection.return_value = mock_collection
        
        # Mock validator
        cluster_service.validator.validate_organization = AsyncMock(return_value=TEST_ORG_BATELCO)
        cluster_service.validator.validate_macro_name = AsyncMock(return_value=True)
        
        # Execute & Verify
        with pytest.raises(DuplicateClusterError):
            await cluster_service.create_cluster(create_request_batelco)
        logger.info("✓ Correctly prevented duplicate cluster creation")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_create_cluster_invalid_organization(self, mock_get_collection, cluster_service, create_request_batelco, logger):
        """Test cluster creation with invalid organization"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Mock validator to raise InvalidOrganizationError
        cluster_service.validator.validate_organization = AsyncMock(
            side_effect=InvalidOrganizationError("invalid_org")
        )
        
        # Execute & Verify
        with pytest.raises(ClusterValidationError, match="Organization validation failed"):
            await cluster_service.create_cluster(create_request_batelco)
        logger.info("✓ Correctly handled invalid organization")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_create_cluster_invalid_macro(self, mock_get_collection, cluster_service, create_request_batelco, logger):
        """Test cluster creation with invalid macro name"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Mock validator
        cluster_service.validator.validate_organization = AsyncMock(return_value=TEST_ORG_BATELCO)
        cluster_service.validator.validate_macro_name = AsyncMock(return_value=False)
        
        # Execute & Verify
        with pytest.raises(ClusterValidationError, match="Macro .* is not valid"):
            await cluster_service.create_cluster(create_request_batelco)
        logger.info("✓ Correctly handled invalid macro name")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_update_cluster_success(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test successful cluster update"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.side_effect = [
            sample_cluster_batelco,  # Existing cluster check
            {**sample_cluster_batelco, "description": "Updated description", "total_fields": 50}  # Updated cluster
        ]
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        mock_get_collection.return_value = mock_collection
        
        # Execute
        update_request = ClusterUpdateRequest(
            cluster_uuid=sample_cluster_batelco["cluster_uuid"],
            description="Updated description",
            total_fields=50,
            status="active"
        )
        result = await cluster_service.update_cluster(TEST_CLUSTER_ID_BATELCO, update_request)
        
        # Verify
        assert result["description"] == "Updated description"
        assert result["total_fields"] == 50
        mock_collection.update_one.assert_called_once()
        logger.info(f"✓ Successfully updated cluster: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_update_cluster_not_found(self, mock_get_collection, cluster_service, logger):
        """Test cluster update when cluster doesn't exist"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        update_request = ClusterUpdateRequest(
            cluster_uuid=str(uuid.uuid4()),
            description="Updated description",
            status="active"
        )
        
        with pytest.raises(ClusterNotFoundError):
            await cluster_service.update_cluster("nonexistent_cluster", update_request)
        logger.info("✓ Correctly handled update of non-existent cluster")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_update_cluster_empty_description(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test cluster update with empty description"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        update_request = ClusterUpdateRequest(
            cluster_uuid=sample_cluster_batelco["cluster_uuid"],
            description="   ",  # Empty string with whitespace
            status="active"
        )
        
        with pytest.raises(ClusterValidationError, match="Description cannot be empty"):
            await cluster_service.update_cluster(TEST_CLUSTER_ID_BATELCO, update_request)
        logger.info("✓ Correctly validated empty description")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_delete_cluster_success(self, mock_get_collection, cluster_service, logger):
        """Test successful cluster deletion"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.delete_cluster(TEST_CLUSTER_ID_BATELCO)
        
        # Verify
        assert result is True
        mock_collection.delete_one.assert_called_once_with({"_id": TEST_CLUSTER_ID_BATELCO})
        logger.info(f"✓ Successfully deleted cluster: {TEST_CLUSTER_ID_BATELCO}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_delete_cluster_not_found(self, mock_get_collection, cluster_service, logger):
        """Test cluster deletion when cluster doesn't exist"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.delete_one.return_value = MagicMock(deleted_count=0)
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.delete_cluster("nonexistent_cluster")
        
        # Verify
        assert result is False
        mock_collection.delete_one.assert_called_once_with({"_id": "nonexistent_cluster"})
        logger.info("✓ Correctly handled deletion of non-existent cluster")


class TestClusterMetadataServiceSearch:
    """Test suite for search functionality"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_cluster_by_cluster_id(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test search cluster by cluster ID"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute
        search_request = ClusterSearchRequest(
            document_id=None,
            organization=None,
            macro_name=None,
            cluster_id=TEST_CLUSTER_ID_BATELCO,
            cluster_uuid=None
        )
        result = await cluster_service.search_cluster(search_request)
        
        # Verify
        assert result is not None
        assert result["cluster_id"] == TEST_CLUSTER_ID_BATELCO
        mock_collection.find_one.assert_called_once_with({"_id": TEST_CLUSTER_ID_BATELCO})
        logger.info(f"✓ Successfully searched cluster by ID: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_cluster_by_organization_and_macro(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test search cluster by organization and macro name combination"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute
        search_request = ClusterSearchRequest(
            document_id=None,
            organization=TEST_ORG_BATELCO,
            macro_name=TEST_MACRO_LINUX_AUDIT,
            cluster_id=None,
            cluster_uuid=None
        )
        result = await cluster_service.search_cluster(search_request)
        
        # Verify
        assert result is not None
        assert result["cluster_id"] == TEST_CLUSTER_ID_BATELCO
        mock_collection.find_one.assert_called_once_with({"_id": TEST_CLUSTER_ID_BATELCO})
        logger.info(f"✓ Successfully searched cluster by org+macro: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_cluster_by_uuid(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test search cluster by cluster UUID"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute
        search_request = ClusterSearchRequest(
            document_id=None,
            organization=None,
            macro_name=None,
            cluster_id=None,
            cluster_uuid=sample_cluster_batelco["cluster_uuid"]
        )
        result = await cluster_service.search_cluster(search_request)
        
        # Verify
        assert result is not None
        assert result["cluster_uuid"] == sample_cluster_batelco["cluster_uuid"]
        mock_collection.find_one.assert_called_once_with({"cluster_uuid": sample_cluster_batelco["cluster_uuid"]})
        logger.info(f"✓ Successfully searched cluster by UUID: {result['cluster_uuid']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_clusters_by_criteria_organization(self, mock_get_collection, cluster_service, sample_cluster_batelco, sample_cluster_sico, logger):
        """Test search clusters by organization criteria"""
        # Setup
        mock_collection = MagicMock()
        
        # Create a mock cursor that supports the chaining pattern
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor  # sort() returns the cursor for chaining
        mock_cursor.to_list = AsyncMock(return_value=[sample_cluster_batelco])
        
        # Mock the find method to return the mock cursor
        mock_collection.find.return_value = mock_cursor
        
        # Mock get_collection to return our mock collection
        mock_get_collection.return_value = mock_collection
        
        # Execute
        criteria = ClusterCriteriaRequest(
            organization=TEST_ORG_BATELCO,
            macro_name=None,
            status=None,
            created_by=None
        )
        results = await cluster_service.search_clusters_by_criteria(criteria)
        
        # Verify
        assert len(results) == 1
        assert results[0]["organization"] == TEST_ORG_BATELCO
        mock_collection.find.assert_called_once_with({"organization": TEST_ORG_BATELCO})
        logger.info(f"✓ Successfully searched clusters by organization: {len(results)} found")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_clusters_by_criteria_organization_and_macro(self, mock_get_collection, cluster_service, sample_cluster_batelco, sample_cluster_sico, logger):
        """Test search clusters by organization and macro criteria"""
        # Setup
        mock_collection = MagicMock()
        
        # Create a mock cursor that supports the chaining pattern
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor  # sort() returns the cursor for chaining
        mock_cursor.to_list = AsyncMock(return_value=[sample_cluster_batelco, sample_cluster_sico])
        
        # Mock the find method to return the mock cursor
        mock_collection.find.return_value = mock_cursor
        mock_get_collection.return_value = mock_collection
        
        # Execute
        criteria = ClusterCriteriaRequest(
            organization=TEST_ORG_BATELCO,
            macro_name=TEST_MACRO_LINUX_AUDIT,
            status=None,
            created_by=None
        )
        results = await cluster_service.search_clusters_by_criteria(criteria)
        
        # Verify
        assert len(results) == 2
        expected_query = {"organization": TEST_ORG_BATELCO, "macro_name": TEST_MACRO_LINUX_AUDIT}
        mock_collection.find.assert_called_once_with(expected_query)
        logger.info(f"✓ Successfully searched clusters by org+macro: {len(results)} found")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_clusters_by_criteria_status(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test search clusters by status criteria"""
        # Setup
        mock_collection = MagicMock()
        
        # Create a mock cursor that supports the chaining pattern
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor  # sort() returns the cursor for chaining
        mock_cursor.to_list = AsyncMock(return_value=[sample_cluster_batelco])
        
        # Mock the find method to return the mock cursor
        mock_collection.find.return_value = mock_cursor
        mock_get_collection.return_value = mock_collection
        
        # Execute
        criteria = ClusterCriteriaRequest(
            organization=None,
            macro_name=None,
            status="active",
            created_by=None
        )
        results = await cluster_service.search_clusters_by_criteria(criteria)
        
        # Verify
        assert len(results) == 1
        assert results[0]["status"] == "active"
        mock_collection.find.assert_called_once_with({"status": "active"})
        logger.info(f"✓ Successfully searched clusters by status: {len(results)} found")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_clusters_by_criteria_created_by(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test search clusters by created_by criteria"""
        # Setup
        mock_collection = MagicMock()
        
        # Create a mock cursor that supports the chaining pattern
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor  # sort() returns the cursor for chaining
        mock_cursor.to_list = AsyncMock(return_value=[sample_cluster_batelco])
        
        # Mock the find method to return the mock cursor
        mock_collection.find.return_value = mock_cursor
        mock_get_collection.return_value = mock_collection
        
        # Execute
        criteria = ClusterCriteriaRequest(
            organization=None,
            macro_name=None,
            status=None,
            created_by="test_user"
        )
        results = await cluster_service.search_clusters_by_criteria(criteria)
        
        # Verify
        assert len(results) == 1
        assert results[0]["created_by"] == "test_user"
        mock_collection.find.assert_called_once_with({"created_by": "test_user"})
        logger.info(f"✓ Successfully searched clusters by creator: {len(results)} found")


class TestClusterMetadataServiceFieldManagement:
    """Test suite for field count management"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_increment_field_count_success(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test successful field count increment"""
        # Setup
        updated_cluster = {**sample_cluster_batelco, "total_fields": 26}
        mock_collection = AsyncMock()
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        mock_collection.find_one.return_value = updated_cluster
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.increment_field_count(TEST_CLUSTER_ID_BATELCO)
        
        # Verify
        assert result["total_fields"] == 26
        mock_collection.update_one.assert_called_once()
        logger.info(f"✓ Successfully incremented field count for: {result['cluster_id']}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_increment_field_count_cluster_not_found(self, mock_get_collection, cluster_service, logger):
        """Test field count increment when cluster doesn't exist"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.update_one.return_value = MagicMock(matched_count=0)
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        with pytest.raises(ClusterNotFoundError):
            await cluster_service.increment_field_count("nonexistent_cluster")
        logger.info("✓ Correctly handled field count increment for non-existent cluster")


class TestClusterMetadataServiceStatistics:
    """Test suite for statistics and analytics"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_statistics_success(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test successful cluster statistics retrieval"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute
        stats = await cluster_service.get_cluster_statistics(TEST_CLUSTER_ID_BATELCO)
        
        # Verify
        assert isinstance(stats, ClusterStats)
        assert stats.cluster_id == TEST_CLUSTER_ID_BATELCO
        assert stats.organization == TEST_ORG_BATELCO
        assert stats.macro_name == TEST_MACRO_LINUX_AUDIT
        assert stats.total_fields == 25
        assert stats.status == "active"
        logger.info(f"✓ Successfully retrieved statistics for: {stats.cluster_id}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_statistics_not_found(self, mock_get_collection, cluster_service, logger):
        """Test cluster statistics when cluster doesn't exist"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        with pytest.raises(ClusterNotFoundError):
            await cluster_service.get_cluster_statistics("nonexistent_cluster")
        logger.info("✓ Correctly handled statistics request for non-existent cluster")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_validate_cluster_exists_success(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test successful cluster existence validation"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.validate_cluster_exists(TEST_CLUSTER_ID_BATELCO)
        
        # Verify
        assert result is True
        logger.info(f"✓ Successfully validated cluster exists: {TEST_CLUSTER_ID_BATELCO}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_validate_cluster_exists_inactive(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test cluster existence validation for inactive cluster"""
        # Setup
        inactive_cluster = {**sample_cluster_batelco, "status": "inactive"}
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = inactive_cluster
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.validate_cluster_exists(TEST_CLUSTER_ID_BATELCO)
        
        # Verify
        assert result is False
        logger.info(f"✓ Correctly identified inactive cluster: {TEST_CLUSTER_ID_BATELCO}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_validate_cluster_exists_not_found(self, mock_get_collection, cluster_service, logger):
        """Test cluster existence validation when cluster doesn't exist"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.validate_cluster_exists("nonexistent_cluster")
        
        # Verify
        assert result is False
        logger.info("✓ Correctly identified non-existent cluster")


class TestClusterMetadataServiceOrganizationMacroListing:
    """Test suite for organization and macro listing functionality"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_list_organizations_success(self, mock_get_collection, cluster_service, logger):
        """Test successful organization listing"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.distinct.return_value = [TEST_ORG_BATELCO, TEST_ORG_SICO, ""]
        mock_get_collection.return_value = mock_collection
        
        # Execute
        organizations = await cluster_service.list_organizations()
        
        # Verify
        assert TEST_ORG_BATELCO in organizations
        assert TEST_ORG_SICO in organizations
        assert "" not in organizations  # Empty strings should be filtered out
        assert len(organizations) == 2
        mock_collection.distinct.assert_called_once_with("organization")
        logger.info(f"✓ Successfully listed organizations: {organizations}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_list_macros_all(self, mock_get_collection, cluster_service, logger):
        """Test listing all macro names"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.distinct.return_value = [TEST_MACRO_LINUX_AUDIT, "windows_audit", ""]
        mock_get_collection.return_value = mock_collection
        
        # Execute
        macros = await cluster_service.list_macros()
        
        # Verify
        assert TEST_MACRO_LINUX_AUDIT in macros
        assert "windows_audit" in macros
        assert "" not in macros  # Empty strings should be filtered out
        assert len(macros) == 2
        mock_collection.distinct.assert_called_once_with("macro_name", {})
        logger.info(f"✓ Successfully listed all macros: {macros}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_list_macros_by_organization(self, mock_get_collection, cluster_service, logger):
        """Test listing macro names filtered by organization"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.distinct.return_value = [TEST_MACRO_LINUX_AUDIT]
        mock_get_collection.return_value = mock_collection
        
        # Mock validator
        cluster_service.validator.validate_organization = AsyncMock(return_value=TEST_ORG_BATELCO)
        
        # Execute
        macros = await cluster_service.list_macros(organization=TEST_ORG_BATELCO)
        
        # Verify
        assert TEST_MACRO_LINUX_AUDIT in macros
        assert len(macros) == 1
        mock_collection.distinct.assert_called_once_with("macro_name", {"organization": TEST_ORG_BATELCO.lower()})
        cluster_service.validator.validate_organization.assert_called_once_with(TEST_ORG_BATELCO)
        logger.info(f"✓ Successfully listed macros for {TEST_ORG_BATELCO}: {macros}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_list_macros_invalid_organization(self, mock_get_collection, cluster_service, logger):
        """Test listing macros with invalid organization"""
        # Setup
        mock_collection = AsyncMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock validator to raise error
        cluster_service.validator.validate_organization = AsyncMock(
            side_effect=InvalidOrganizationError("invalid_org")
        )
        
        # Execute & Verify
        with pytest.raises(ClusterValidationError, match="Failed to list macros"):
            await cluster_service.list_macros(organization="invalid_org")
        logger.info("✓ Correctly handled invalid organization for macro listing")


class TestClusterMetadataServiceErrorHandling:
    """Test suite for error handling and edge cases"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_search_cluster_no_criteria(self, mock_get_collection, cluster_service, logger):
        """Test search cluster with no valid criteria"""
        # Setup
        mock_collection = AsyncMock()
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        with pytest.raises(ClusterValidationError, match="No valid search criteria provided"):
            search_request = ClusterSearchRequest(
                document_id=None,
                organization=None,
                macro_name=None,
                cluster_id=None,
                cluster_uuid=None
            )  # No criteria provided
            await cluster_service.search_cluster(search_request)
        logger.info("✓ Correctly handled search with no criteria")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_update_cluster_no_match(self, mock_get_collection, cluster_service, sample_cluster_batelco, logger):
        """Test cluster update when MongoDB update operation matches no documents"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = sample_cluster_batelco  # Cluster exists for initial check
        mock_collection.update_one.return_value = MagicMock(matched_count=0)  # But no match during update
        mock_get_collection.return_value = mock_collection
        
        # Execute & Verify
        update_request = ClusterUpdateRequest(
            cluster_uuid=sample_cluster_batelco["cluster_uuid"],
            description="Updated description",
            status="active"
        )
        
        with pytest.raises(ClusterUpdateError, match="No matching cluster found for update"):
            await cluster_service.update_cluster(TEST_CLUSTER_ID_BATELCO, update_request)
        logger.info("✓ Correctly handled update operation with no match")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_by_object_id_invalid(self, mock_get_collection, cluster_service, logger):
        """Test getting cluster by invalid ObjectId"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.side_effect = Exception("Invalid ObjectId")
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.get_cluster_by_object_id("invalid_object_id")
        
        # Verify
        assert result is None
        logger.info("✓ Correctly handled invalid ObjectId")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_get_cluster_by_uuid_not_found(self, mock_get_collection, cluster_service, logger):
        """Test getting cluster by UUID when not found"""
        # Setup
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_get_collection.return_value = mock_collection
        
        # Execute
        result = await cluster_service.get_cluster_by_uuid(str(uuid.uuid4()))
        
        # Verify
        assert result is None
        logger.info("✓ Correctly handled UUID lookup with no match")


class TestClusterMetadataServiceIntegration:
    """Integration-style tests for complex scenarios"""

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_full_cluster_lifecycle_batelco(self, mock_get_collection, cluster_service, create_request_batelco, logger):
        """Test complete cluster lifecycle for Batelco organization"""
        # Setup mock collection
        mock_collection = AsyncMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock validator
        cluster_service.validator.validate_organization = AsyncMock(return_value=TEST_ORG_BATELCO)
        cluster_service.validator.validate_macro_name = AsyncMock(return_value=True)
        
        # Phase 1: Create cluster
        mock_collection.find_one.return_value = None  # No existing cluster
        mock_collection.insert_one.return_value = MagicMock()
        
        created_cluster = await cluster_service.create_cluster(create_request_batelco)
        assert created_cluster["cluster_id"] == TEST_CLUSTER_ID_BATELCO
        logger.info(f"✓ Phase 1: Created cluster {created_cluster['cluster_id']}")
        
        # Phase 2: Search cluster
        mock_collection.find_one.return_value = created_cluster
        search_request = ClusterSearchRequest(
            document_id=None,
            organization=None,
            macro_name=None,
            cluster_id=TEST_CLUSTER_ID_BATELCO,
            cluster_uuid=None
        )
        found_cluster = await cluster_service.search_cluster(search_request)
        assert found_cluster["cluster_id"] == TEST_CLUSTER_ID_BATELCO
        logger.info(f"✓ Phase 2: Found cluster {found_cluster['cluster_id']}")
        
        # Phase 3: Update cluster
        updated_cluster = {**created_cluster, "description": "Updated description", "total_fields": 10}
        mock_collection.find_one.side_effect = [created_cluster, updated_cluster]
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        
        update_request = ClusterUpdateRequest(
            cluster_uuid=created_cluster["cluster_uuid"],
            description="Updated description",
            total_fields=10,
            status="active"
        )
        result_cluster = await cluster_service.update_cluster(TEST_CLUSTER_ID_BATELCO, update_request)
        assert result_cluster["description"] == "Updated description"
        logger.info(f"✓ Phase 3: Updated cluster {result_cluster['cluster_id']}")
        
        # Phase 4: Increment field count
        incremented_cluster = {**updated_cluster, "total_fields": 11}
        mock_collection.find_one.return_value = incremented_cluster
        mock_collection.update_one.return_value = MagicMock(matched_count=1)
        
        final_cluster = await cluster_service.increment_field_count(TEST_CLUSTER_ID_BATELCO)
        assert final_cluster["total_fields"] == 11
        logger.info(f"✓ Phase 4: Incremented field count for {final_cluster['cluster_id']}")
        
        # Phase 5: Get statistics
        mock_collection.find_one.return_value = final_cluster
        stats = await cluster_service.get_cluster_statistics(TEST_CLUSTER_ID_BATELCO)
        assert stats.total_fields == 11
        assert stats.organization == TEST_ORG_BATELCO
        logger.info(f"✓ Phase 5: Retrieved statistics for {stats.cluster_id}")
        
        # Phase 6: Delete cluster
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        deleted = await cluster_service.delete_cluster(TEST_CLUSTER_ID_BATELCO)
        assert deleted is True
        logger.info(f"✓ Phase 6: Deleted cluster {TEST_CLUSTER_ID_BATELCO}")

    @patch('graphiti_core.cluster_metadata.cluster_service.get_collection')
    async def test_multi_organization_scenario(self, mock_get_collection, cluster_service, create_request_batelco, create_request_sico, sample_cluster_batelco, sample_cluster_sico, logger):
        """Test scenario with multiple organizations"""
        # Setup
        mock_collection = MagicMock()  # Use MagicMock for collections that use find()
        mock_get_collection.return_value = mock_collection
        
        # Mock validator for both organizations
        def mock_validate_org(org):
            # Handle case-insensitive lookup
            org_lower = org.lower()
            return {"batelco": TEST_ORG_BATELCO, "sico": TEST_ORG_SICO}[org_lower]
        
        cluster_service.validator.validate_organization = AsyncMock(side_effect=mock_validate_org)
        cluster_service.validator.validate_macro_name = AsyncMock(return_value=True)
        
        # Create clusters for both organizations
        mock_collection.find_one.return_value = None
        mock_collection.insert_one.return_value = MagicMock()
        
        batelco_cluster = await cluster_service.create_cluster(create_request_batelco)
        sico_cluster = await cluster_service.create_cluster(create_request_sico)
        
        assert batelco_cluster["organization"] == TEST_ORG_BATELCO
        assert sico_cluster["organization"] == TEST_ORG_SICO
        logger.info(f"✓ Created clusters for both organizations")
        
        # Search clusters by organization
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = [sample_cluster_batelco]
        
        # Mock the find method to return a mock cursor object
        mock_collection.find.return_value = mock_cursor
        mock_cursor.sort.return_value = mock_cursor  # sort() returns the cursor for chaining
        
        criteria = ClusterCriteriaRequest(
            organization=TEST_ORG_BATELCO,
            macro_name=None,
            status=None,
            created_by=None
        )
        batelco_clusters = await cluster_service.search_clusters_by_criteria(criteria)
        
        assert len(batelco_clusters) == 1
        assert batelco_clusters[0]["organization"] == TEST_ORG_BATELCO
        logger.info(f"✓ Successfully searched clusters by organization filter")
        
        # List organizations
        mock_collection.distinct.return_value = [TEST_ORG_BATELCO, TEST_ORG_SICO]
        organizations = await cluster_service.list_organizations()
        
        assert TEST_ORG_BATELCO in organizations
        assert TEST_ORG_SICO in organizations
        logger.info(f"✓ Successfully listed all organizations: {organizations}")


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
