"""
MongoDB Cluster Metadata Service for Hybrid Architecture

This service manages cluster metadata in MongoDB while Neo4j handles field relationships.
Following the hybrid architecture pattern where MongoDB provides fast cluster lookups
and Neo4j manages the graph data with field relationships.

Key Features:
- Fast cluster discovery and validation
- Organization and macro validation
- Statistics tracking for field counts
- Integration with Neo4j field operations
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from bson import ObjectId

from graphiti_core.utils.datetime_utils import utc_now
from .models import (
    ClusterCreateRequest, 
    ClusterUpdateRequest, 
    ClusterStats,
    ClusterSearchRequest,
    ClusterCriteriaRequest
)
from .mongo_service import (
    get_collection
)
from .exceptions import (
    ClusterNotFoundError,
    DuplicateClusterError,
    ClusterValidationError,
    InvalidOrganizationError,
    ClusterUpdateError
)
from .validator import EnhancedClusterValidator

logger = logging.getLogger(__name__)


class ClusterMetadataService:
    """
    Service for managing cluster metadata in MongoDB.
    
    This service provides CRUD operations for the cluster_metadata collection,
    enabling fast cluster discovery and validation for the hybrid architecture.
    """
    
    def __init__(self, collection_name: str = "cluster_metadata"):
        self.collection_name = collection_name
        self.validator = EnhancedClusterValidator()
        
    
    async def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cluster metadata by ID (legacy method - use search_cluster for advanced search).
        
        Args:
            cluster_id: The cluster identifier (e.g., 'linux_audit_batelco')
            
        Returns:
            Cluster document or None if not found
        """
        try:
            collection = await get_collection(self.collection_name)
            cluster = await collection.find_one({"_id": cluster_id})
            
            if cluster:
                logger.debug(f"Retrieved cluster: {cluster_id}")
                return cluster
            else:
                logger.debug(f"Cluster not found: {cluster_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cluster {cluster_id}: {e}")
            raise ClusterValidationError(f"Failed to retrieve cluster: {e}")

    async def search_cluster(self, search_request: ClusterSearchRequest) -> Optional[Dict[str, Any]]:
        """
        Search for cluster using flexible criteria with Pydantic validation.
        
        Args:
            search_request: ClusterSearchRequest with validation
            
        Returns:
            Cluster document or None if not found
            
        Raises:
            ClusterValidationError: If search fails
        """
        try:
            collection = await get_collection(self.collection_name)
            
            # The ClusterSearchRequest model validation ensures only one search approach is used
            query = {}
            if search_request.document_id:
                # Search by MongoDB document ID
                try:
                    # Try as ObjectId first
                    query = {"_id": ObjectId(search_request.document_id)}
                except:
                    # If not a valid ObjectId, try as string (cluster_id)
                    query = {"_id": search_request.document_id}
            
            elif search_request.organization and search_request.macro_name:
                # Search by organization + macro_name combination
                cluster_id = f"{search_request.macro_name}_{search_request.organization.lower()}"
                query = {"_id": cluster_id}
            
            elif search_request.cluster_id:
                # Search by cluster_id (macro_organization format)
                query = {"_id": search_request.cluster_id}
            
            elif search_request.cluster_uuid:
                # Search by cluster UUID
                query = {"cluster_uuid": search_request.cluster_uuid}
            
            else:
                raise ClusterValidationError("No valid search criteria provided")
            
            cluster = await collection.find_one(query)
            
            if cluster:
                logger.debug(f"Retrieved cluster with query: {query}")
                return cluster
            else:
                logger.debug(f"Cluster not found with query: {query}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching cluster with request {search_request}: {e}")
            raise ClusterValidationError(f"Failed to search cluster: {e}")
    
    async def search_clusters_by_criteria(self, criteria: ClusterCriteriaRequest) -> List[Dict[str, Any]]:
        """
        Search for multiple clusters using flexible criteria with validation.
        
        Args:
            criteria: ClusterCriteriaRequest with validated search criteria
            
        Returns:
            List of matching cluster documents
            
        Example:
            # Search by organization
            criteria = ClusterCriteriaRequest(organization="batelco")
            clusters = await service.search_clusters_by_criteria(criteria)
            
            # Search by organization and macro
            criteria = ClusterCriteriaRequest(
                organization="batelco", 
                macro_name="linux_audit"
            )
            clusters = await service.search_clusters_by_criteria(criteria)
            
            # Search by status
            criteria = ClusterCriteriaRequest(status="active")
            clusters = await service.search_clusters_by_criteria(criteria)
        """
        try:
            collection = await get_collection(self.collection_name)
            
            # Build query from validated criteria
            query = {}
            
            if criteria.organization:
                query["organization"] = criteria.organization
            if criteria.macro_name:
                query["macro_name"] = criteria.macro_name
            if criteria.status:
                query["status"] = criteria.status
            if criteria.created_by:
                query["created_by"] = criteria.created_by
            
            # Execute query
            cursor = collection.find(query).sort("cluster_id", 1)
            clusters = await cursor.to_list(length=None)
            
            logger.debug(f"Found {len(clusters)} clusters matching criteria: {criteria.model_dump(exclude_none=True)}")
            return clusters
            
        except Exception as e:
            logger.error(f"Error searching clusters with criteria {criteria.model_dump(exclude_none=True)}: {e}")
            raise ClusterValidationError(f"Failed to search clusters: {e}")
    
    async def create_cluster(self, cluster_data: ClusterCreateRequest) -> Dict[str, Any]:
        """
        Create a new cluster in MongoDB.
        
        Args:
            cluster_data: Cluster creation request data
            
        Returns:
            Created cluster document
            
        Raises:
            DuplicateClusterError: If cluster already exists
            ClusterValidationError: If cluster data is invalid
        """
        try:
            # Validate organization and macro separately
            try:
                await self.validator.validate_organization(cluster_data.organization)
            except InvalidOrganizationError as e:
                raise ClusterValidationError(f"Organization validation failed: {e}")
            
            macro_valid = await self.validator.validate_macro_name(cluster_data.macro_name, cluster_data.organization)
            if not macro_valid:
                raise ClusterValidationError(f"Macro '{cluster_data.macro_name}' is not valid for organization '{cluster_data.organization}'")
            
            # Check if cluster already exists
            existing = await self.get_cluster(cluster_data.cluster_id)
            if existing:
                raise DuplicateClusterError(cluster_data.cluster_id)
            
            # Create cluster document
            now = utc_now()
            cluster_doc = {
                "_id": cluster_data.cluster_id,
                "cluster_id": cluster_data.cluster_id,
                "cluster_uuid": cluster_data.cluster_uuid,
                "macro_name": cluster_data.macro_name,
                "organization": cluster_data.organization,
                "description": cluster_data.description,
                "status": cluster_data.status or "active",
                "total_fields": cluster_data.total_fields or 0,
                "created_at": now,
                "last_updated": now,
                "created_by": cluster_data.created_by or "system"
            }
            
            # Insert into MongoDB
            collection = await get_collection(self.collection_name)
            result = await collection.insert_one(cluster_doc)
            
            logger.info(f"Created cluster: {cluster_data.cluster_id}")
            return cluster_doc
            
        except (DuplicateClusterError, ClusterValidationError):
            raise
        except Exception as e:
            logger.error(f"Error creating cluster {cluster_data.cluster_id}: {e}")
            raise ClusterValidationError(f"Failed to create cluster: {e}")
    
    async def update_cluster(self, cluster_id: str, updates: ClusterUpdateRequest) -> Dict[str, Any]:
        """
        Update cluster metadata.
        
        Args:
            cluster_id: The cluster identifier
            updates: Update data
            
        Returns:
            Updated cluster document
            
        Raises:
            ClusterNotFoundError: If cluster doesn't exist
        """
        try:
            # Check if cluster exists
            existing = await self.get_cluster(cluster_id)
            if not existing:
                raise ClusterNotFoundError(cluster_id)
            
            # Basic validation for updates
            if updates.description is not None and not updates.description.strip():
                raise ClusterValidationError("Description cannot be empty")
            
            # Build update document
            update_doc: Dict[str, Any] = {"last_updated": utc_now()}
            
            if updates.description:
                update_doc["description"] = updates.description
            if updates.status:
                update_doc["status"] = updates.status
            if updates.total_fields is not None:
                update_doc["total_fields"] = updates.total_fields
            
            # Update in MongoDB
            collection = await get_collection(self.collection_name)
            result = await collection.update_one(
                {"_id": cluster_id},
                {"$set": update_doc}
            )
            
            # Check if update was successful
            if result.matched_count == 0:
                raise ClusterUpdateError(cluster_id, "No matching cluster found for update")
            
            # Return updated document
            updated_cluster = await self.get_cluster(cluster_id)
            if not updated_cluster:
                raise ClusterValidationError(f"Failed to retrieve updated cluster: {cluster_id}")
            logger.info(f"Updated cluster: {cluster_id}")
            return updated_cluster
            
        except ClusterNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating cluster {cluster_id}: {e}")
            raise ClusterValidationError(f"Failed to update cluster: {e}")
    
    async def delete_cluster(self, cluster_id: str) -> bool:
        """
        Delete cluster from MongoDB.
        
        Note: This only deletes metadata. Neo4j field cleanup must be handled separately.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            collection = await get_collection(self.collection_name)
            result = await collection.delete_one({"_id": cluster_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted cluster: {cluster_id}")
                return True
            else:
                logger.debug(f"Cluster not found for deletion: {cluster_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cluster {cluster_id}: {e}")
            raise ClusterValidationError(f"Failed to delete cluster: {e}")
    
    async def increment_field_count(self, cluster_id: str) -> Dict[str, Any]:
        """
        Increment field count for cluster.
        
        This is called when a new field is added to Neo4j to keep MongoDB stats in sync.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            Updated cluster document
        """
        try:
            collection = await get_collection(self.collection_name)
            now = utc_now()
            
            # Build update operations
            update_ops = {
                "$inc": {"total_fields": 1},
                "$set": {"last_updated": now}
            }
            
            # Update cluster
            result = await collection.update_one(
                {"_id": cluster_id},
                update_ops
            )
            
            # Check if update was successful
            if result.matched_count == 0:
                raise ClusterNotFoundError(cluster_id)
            
            # Return updated cluster
            updated_cluster = await self.get_cluster(cluster_id)
            if not updated_cluster:
                raise ClusterValidationError(f"Failed to retrieve updated cluster after field count increment: {cluster_id}")
            logger.debug(f"Incremented field count for cluster: {cluster_id}")
            return updated_cluster
            
        except Exception as e:
            logger.error(f"Error incrementing field count for {cluster_id}: {e}")
            raise ClusterValidationError(f"Failed to increment field count: {e}")
    
    async def get_cluster_by_object_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get cluster by MongoDB ObjectId."""
        try:
            collection = await get_collection(self.collection_name)
            cluster = await collection.find_one({"_id": ObjectId(object_id)})
            return cluster
        except Exception:
            return None
    
    async def get_cluster_by_uuid(self, cluster_uuid: str) -> Optional[Dict[str, Any]]:
        """Get cluster by cluster UUID."""
        try:
            collection = await get_collection(self.collection_name)
            cluster = await collection.find_one({"cluster_uuid": cluster_uuid})
            return cluster
        except Exception:
            return None

    async def get_cluster_statistics(self, cluster_id: str) -> ClusterStats:
        """
        Get comprehensive cluster statistics.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            Cluster statistics including field counts
        """
        cluster = await self.get_cluster(cluster_id)
        if not cluster:
            raise ClusterNotFoundError(cluster_id)
        
        stats = ClusterStats(
            cluster_uuid=cluster.get("cluster_uuid", ""),
            cluster_id=cluster_id,
            total_fields=cluster.get("total_fields", 0),
            status=cluster.get("status", "unknown"),
            created_at=cluster.get("created_at") or utc_now(),
            last_updated=cluster.get("last_updated") or utc_now(),
            organization=cluster.get("organization", ""),
            created_by=cluster.get("created_by", "system"),
            description=cluster.get("description", ""),
            macro_name=cluster.get("macro_name", "")
        )
        
        return stats
    
    async def validate_cluster_exists(self, cluster_id: str) -> bool:
        """
        Fast validation that cluster exists.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            True if cluster exists and is active
        """
        cluster = await self.get_cluster(cluster_id)
        return cluster is not None and cluster.get("status") == "active"
    
    async def list_organizations(self) -> List[str]:
        """
        Get list of all organizations from clusters.
        
        Returns:
            List of unique organization names
        """
        try:
            collection = await get_collection(self.collection_name)
            organizations = await collection.distinct("organization")
            return sorted([org for org in organizations if org])
            
        except Exception as e:
            logger.error(f"Error listing organizations: {e}")
            raise ClusterValidationError(f"Failed to list organizations: {e}")
    
    async def list_macros(self, organization: Optional[str] = None) -> List[str]:
        """
        Get list of all macro names from clusters, optionally filtered by organization.
        
        Args:
            organization: Optional organization filter (e.g., 'batelco', 'sico')
            
        Returns:
            List of unique macro names
        """
        try:
            collection = await get_collection(self.collection_name)
            
            # Build query filter
            filter_query = {}
            if organization:
                # Validate organization before querying
                await self.validator.validate_organization(organization)
                filter_query = {"organization": organization}
            
            # Get distinct macro names with optional organization filter
            macros = await collection.distinct("macro_name", filter_query)
            
            # Filter out empty/null values and sort
            filtered_macros = sorted([macro for macro in macros if macro])
            
            logger.debug(f"Retrieved {len(filtered_macros)} macros" + 
                        (f" for organization: {organization}" if organization else ""))
            return filtered_macros
            
        except Exception as e:
            logger.error(f"Error listing macros{' for organization ' + organization if organization else ''}: {e}")
            raise ClusterValidationError(f"Failed to list macros: {e}")
        
        