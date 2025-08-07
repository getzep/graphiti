"""
MongoDB Cluster Fields Service for Hybrid Architecture

This service manages field metadata within clusters in MongoDB while maintaining
synchronization with Neo4j Field nodes. It provides a bridge between the MongoDB
cluster metadata and Neo4j graph structure for optimal performance.

Key Features:
- Field-to-cluster association management
- Bulk field operations with transaction support
- Constraint validation based on Neo4j graph constraints
- Integration with ClusterMetadataService for cluster validation
- Field existence checking and metadata tracking
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from bson import ObjectId
from pymongo import UpdateOne, InsertOne, DeleteOne
from pymongo.errors import BulkWriteError, DuplicateKeyError
from .exceptions import (
    ClusterNotFoundError,
    FieldAlreadyExistsError,
    FieldNotFoundError,
    FieldValidationError
)
from graphiti_core.utils.datetime_utils import utc_now
from .models import (
    FieldCreateRequest, 
    FieldUpdateRequest,
    FieldSearchByUuidRequest,
    FieldSearchByNameAndClusterRequest,
    FieldSearchByClusterUuidRequest,
    FieldSearchByCreatorRequest,
    FieldSearchByFieldNameRequest
)
from .mongo_service import get_collection
from .cluster_service import ClusterMetadataService

logger = logging.getLogger(__name__)




class ClusterFieldsService:
    """
    Service for managing field metadata within clusters in MongoDB.
    
    This service maintains a separate 'cluster_fields' collection that stores
    field metadata associated with specific clusters, enabling fast field lookups
    and cluster-based field operations while keeping synchronization with Neo4j.
    """
    
    def __init__(self, collection_name: str = "cluster_fields"):
        self.collection_name = collection_name
        self.cluster_service = ClusterMetadataService()
        
    # ==================== FIELD VALIDATION BASED ON NEO4J CONSTRAINTS ====================
    
    def _validate_field_constraints(self, field_data: FieldCreateRequest) -> List[str]:
        """
        Validate field data based on Neo4j constraints from field_graph_queries.py
        
        Based on Neo4j constraints:
        - field_uuid must be unique (handled by MongoDB unique index)
        - field_name must not be null/empty
        - primary_cluster_id (cluster_uuid) must not be null/empty  
        - description must not be null/empty
        - data_type must not be null/empty
        - count must be >= 0
        - distinct_count must be >= 0 and <= count
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Field name validation (field_name_not_null constraint)
        if not field_data.field_name or not field_data.field_name.strip():
            errors.append("field_name must be a non-empty string")
        elif len(field_data.field_name.strip()) > 100:
            errors.append("field_name too long (max 100 characters)")
            
        # Cluster UUID validation (field_primary_cluster_not_null constraint)
        if not field_data.cluster_uuid or not field_data.cluster_uuid.strip():
            errors.append("cluster_uuid must be a non-empty string")
            
        # Description validation (field_description_not_null constraint)
        if not field_data.description or not field_data.description.strip():
            errors.append("description must be a non-empty string")
            
        # Data type validation (field_data_type_not_null constraint)
        if not field_data.data_type or not field_data.data_type.strip():
            errors.append("data_type must be a non-empty string")
            
        # Count validation (field_count_non_negative constraint)
        if field_data.count < 0:
            errors.append("count must be >= 0")
            
        # Distinct count validation (field_distinct_count constraints)
        if field_data.distinct_count < 0:
            errors.append("distinct_count must be >= 0")
        elif field_data.distinct_count > field_data.count:
            errors.append("distinct_count must be <= count")
            
        # Temporal validation (field_temporal_consistency constraint)
        if field_data.created_at > field_data.last_updated:
            errors.append("created_at must be <= last_updated")
            
        return errors
    
    def _validate_bulk_field_constraints(self, fields_data: List[FieldCreateRequest]) -> Dict[str, List[str]]:
        """
        Validate multiple fields and return validation errors per field.
        
        Returns:
            Dictionary mapping field_uuid to list of validation errors
        """
        validation_results = {}
        field_uuids = set()
        
        for field_data in fields_data:
            # Individual field validation
            errors = self._validate_field_constraints(field_data)
            
            # Check for duplicate UUIDs within the batch
            if field_data.field_uuid in field_uuids:
                errors.append(f"Duplicate field_uuid in batch: {field_data.field_uuid}")
            else:
                field_uuids.add(field_data.field_uuid)
            
            if errors:
                validation_results[field_data.field_uuid] = errors
                
        return validation_results
    
    # ==================== CORE FIELD OPERATIONS ====================
    async def add_field_to_cluster(self, field_data: FieldCreateRequest) -> Dict[str, Any]:
        """
        Add a single field to a cluster with comprehensive validation.
        
        This method:
        1. Validates that the cluster exists in MongoDB
        2. Validates field data against Neo4j constraints
        3. Checks for field uniqueness within cluster
        4. Creates field document in cluster_fields collection
        5. Updates cluster metadata (increment field count)
        
        Args:
            field_data: FieldCreateRequest with validated field information
            
        Returns:
            Created field document
            
        Raises:
            ClusterNotFoundError: If cluster doesn't exist or is inactive
            FieldValidationError: If field data violates constraints
            FieldAlreadyExistsError: If field already exists in cluster
        """
        field_inserted = False
        field_doc = None
        collection = None
        
        try:
            # Step 1: Validate cluster exists and is active
            cluster_exists = await self.cluster_service.validate_cluster_exists(field_data.cluster_uuid)
            if not cluster_exists:
                raise ClusterNotFoundError(field_data.cluster_uuid)
            
            # Step 2: Validate field data against Neo4j constraints
            validation_errors = self._validate_field_constraints(field_data)
            if validation_errors:
                raise FieldValidationError("Field validation failed", validation_errors)
            
            # Step 3: Check if field already exists in cluster
            existing_field = await self.check_field_exists_in_cluster(
                field_data.field_uuid, 
                field_data.cluster_uuid
            )
            if existing_field:
                raise FieldAlreadyExistsError(field_data.field_uuid, field_data.cluster_uuid)

            # Step 4: Prepare field document
            field_doc = {
                "field_uuid": field_data.field_uuid,
                "field_name": field_data.field_name.strip(),
                "cluster_parent_id": field_data.cluster_parent_id,
                "cluster_uuid": field_data.cluster_uuid,
                "description": field_data.description.strip(),
                "examples": field_data.examples,
                "data_type": field_data.data_type.strip(),
                "count": field_data.count,
                "distinct_count": field_data.distinct_count,
                "labels": field_data.labels,
                "embedding": field_data.embedding,
                "mongodb_cluster_id": field_data.mongodb_cluster_id,
                "created_at": field_data.created_at,
                "validated_at": field_data.validated_at,
                "invalidated_at": field_data.invalidated_at,
                "last_updated": field_data.last_updated,
                "created_by": field_data.created_by or "system"
            }
            
            # Step 5: Insert field document
            collection = await get_collection(self.collection_name)
            await collection.insert_one(field_doc)
            field_inserted = True
            
            # Step 6: Update cluster field count (with rollback on failure)
            try:
                await self.cluster_service.increment_field_count(field_data.cluster_uuid)
            except Exception as cluster_error:
                # Rollback: Remove the field we just inserted
                await collection.delete_one({"field_uuid": field_data.field_uuid})
                field_inserted = False
                logger.error(f"Rolled back field insertion due to cluster update failure: {cluster_error}")
                raise FieldValidationError(f"Failed to update cluster metadata: {cluster_error}")
            
            logger.info(f"Successfully added field '{field_data.field_uuid}' to cluster '{field_data.cluster_uuid}'")
            return field_doc
            
        except (ClusterNotFoundError, FieldValidationError, FieldAlreadyExistsError):
            # Re-raise known exceptions without modification
            raise
            
        except DuplicateKeyError:
            # Handle MongoDB duplicate key errors
            raise FieldAlreadyExistsError(field_data.field_uuid, field_data.cluster_uuid)
            
        except Exception as e:
            # Handle any other unexpected errors with rollback
            if field_inserted and collection:
                try:
                    await collection.delete_one({"field_uuid": field_data.field_uuid})
                    logger.info(f"Rolled back field insertion for {field_data.field_uuid} due to unexpected error")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback field insertion: {rollback_error}")
            
            logger.error(f"Error adding field {field_data.field_uuid} to cluster {field_data.cluster_uuid}: {e}")
            raise FieldValidationError(f"Failed to add field to cluster: {e}")
        
        
    async def remove_field_from_cluster(self, field_uuid: str, cluster_uuid: str) -> bool:
        """
        Remove a field from a cluster and update cluster metadata.
        
        Args:
            field_uuid: UUID of the field to remove
            cluster_uuid: UUID of the cluster containing the field
            
        Returns:
            True if field was removed, False if field was not found
            
        Raises:
            ClusterNotFoundError: If cluster doesn't exist
            FieldValidationError: If operation fails
        """
        field_removed = False
        removed_field_doc = None
        collection = None
        
        try:
            # Step 1: Validate cluster exists
            cluster_exists = await self.cluster_service.validate_cluster_exists(cluster_uuid)
            if not cluster_exists:
                raise ClusterNotFoundError(cluster_uuid)
            
            # Step 2: Get the field document before removal (for potential rollback)
            collection = await get_collection(self.collection_name)
            removed_field_doc = await collection.find_one({
                "field_uuid": field_uuid,
                "cluster_uuid": cluster_uuid
            })
            
            if not removed_field_doc:
                logger.debug(f"Field '{field_uuid}' not found in cluster '{cluster_uuid}'")
                return False
            
            # Step 3: Remove field from collection
            result = await collection.delete_one({
                "field_uuid": field_uuid,
                "cluster_uuid": cluster_uuid
            })
            
            if result.deleted_count > 0:
                field_removed = True
                
                # Step 4: Decrement cluster field count (with rollback on failure)
                try:
                    await self._decrement_cluster_field_count(cluster_uuid)
                except Exception as cluster_error:
                    # Rollback: Re-insert the field we just removed
                    await collection.insert_one(removed_field_doc)
                    field_removed = False
                    logger.error(f"Rolled back field removal due to cluster update failure: {cluster_error}")
                    raise FieldValidationError(f"Failed to update cluster metadata: {cluster_error}")
                
                logger.info(f"Removed field '{field_uuid}' from cluster '{cluster_uuid}'")
                return True
            else:
                logger.debug(f"Field '{field_uuid}' not found in cluster '{cluster_uuid}'")
                return False
                
        except ClusterNotFoundError:
            # Re-raise known exceptions without modification
            raise
            
        except Exception as e:
            # Handle any other unexpected errors with rollback
            if field_removed and removed_field_doc and collection:
                try:
                    await collection.insert_one(removed_field_doc)
                    logger.info(f"Rolled back field removal for {field_uuid} due to unexpected error")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback field removal: {rollback_error}")
            
            logger.error(f"Error removing field {field_uuid} from cluster {cluster_uuid}: {e}")
            raise FieldValidationError(f"Failed to remove field from cluster: {e}")
            

    async def check_field_exists_in_cluster(self, field_uuid: str, cluster_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Check if a field exists in a specific cluster.
        
        Args:
            field_uuid: UUID of the field to check
            cluster_uuid: UUID of the cluster to check in
            
        Returns:
            Field document if exists, None otherwise
        """
        try:
            collection = await get_collection(self.collection_name)
            field_doc = await collection.find_one({
                "field_uuid": field_uuid,
                "cluster_uuid": cluster_uuid
            })
            
            return field_doc
            
        except Exception as e:
            logger.error(f"Error checking field existence {field_uuid} in cluster {cluster_uuid}: {e}")
            return None

    async def add_bulk_fields_to_cluster(self, fields_data: List[FieldCreateRequest]) -> Dict[str, Any]:
        """
        Add multiple fields to clusters with transaction-like behavior.
        
        This method:
        1. Validates all fields and clusters
        2. Checks for duplicates within batch and existing fields
        3. Performs bulk insert operation
        4. Updates cluster field counts for affected clusters
        
        Args:
            fields_data: List of FieldCreateRequest objects
            
        Returns:
            Dictionary with success/failure statistics and details
            
        Raises:
            FieldValidationError: If validation fails for any field
        """
        if not fields_data:
            return {"success_count": 0, "failure_count": 0, "errors": []}
        
        try:
            # Step 1: Validate all fields
            validation_results = self._validate_bulk_field_constraints(fields_data)
            if validation_results:
                error_details = []
                for field_uuid, errors in validation_results.items():
                    error_details.extend([f"Field {field_uuid}: {error}" for error in errors])
                raise FieldValidationError("Bulk field validation failed", error_details)
            
            # Step 2: Validate all clusters exist
            cluster_uuids = {field.cluster_uuid for field in fields_data}
            for cluster_uuid in cluster_uuids:
                cluster_exists = await self.cluster_service.validate_cluster_exists(cluster_uuid)
                if not cluster_exists:
                    raise ClusterNotFoundError(cluster_uuid)
            
            # Step 3: Check for existing fields in clusters
            existing_checks = []
            for field_data in fields_data:
                existing_field = await self.check_field_exists_in_cluster(
                    field_data.field_uuid, 
                    field_data.cluster_uuid
                )
                if existing_field:
                    existing_checks.append(f"Field {field_data.field_uuid} already exists in cluster {field_data.cluster_uuid}")
            
            if existing_checks:
                raise FieldValidationError("Duplicate fields found", existing_checks)
            
            # Step 4: Prepare bulk operations
            now = utc_now()
            bulk_operations = []
            cluster_field_counts = {}
            
            for field_data in fields_data:
                field_doc = {
                    "field_uuid": field_data.field_uuid,
                    "field_name": field_data.field_name.strip(),
                    "cluster_parent_id": field_data.cluster_parent_id,
                    "cluster_uuid": field_data.cluster_uuid,
                    "description": field_data.description.strip(),
                    "examples": field_data.examples,
                    "data_type": field_data.data_type.strip(),
                    "count": field_data.count,
                    "distinct_count": field_data.distinct_count,
                    "labels": field_data.labels,
                    "embedding": field_data.embedding,
                    "mongodb_cluster_id": field_data.mongodb_cluster_id,
                    "created_at": field_data.created_at,
                    "validated_at": field_data.validated_at,
                    "invalidated_at": field_data.invalidated_at,
                    "last_updated": field_data.last_updated,
                    "created_by": field_data.created_by or "system"
                }
                
                bulk_operations.append(InsertOne(field_doc))
                
                # Count fields per cluster for metadata updates
                if field_data.cluster_uuid in cluster_field_counts:
                    cluster_field_counts[field_data.cluster_uuid] += 1
                else:
                    cluster_field_counts[field_data.cluster_uuid] = 1
            
            # Step 5: Execute bulk insert
            collection = await get_collection(self.collection_name)
            bulk_result = await collection.bulk_write(bulk_operations, ordered=False)
            fields_inserted = bulk_result.inserted_count > 0
            inserted_field_uuids = [field_data.field_uuid for field_data in fields_data[:bulk_result.inserted_count]]
            
            # Step 6: Update cluster field counts (with rollback on failure)
            cluster_update_errors = []
            successfully_updated_clusters = []
            
            try:
                for cluster_uuid, field_count in cluster_field_counts.items():
                    try:
                        await self._increment_cluster_field_count_by(cluster_uuid, field_count)
                        successfully_updated_clusters.append(cluster_uuid)
                    except Exception as cluster_error:
                        cluster_update_errors.append(f"Cluster {cluster_uuid}: {cluster_error}")
                
                # If any cluster updates failed, rollback all inserted fields
                if cluster_update_errors:
                    if fields_inserted and inserted_field_uuids:
                        try:
                            # Rollback: Remove all successfully inserted fields
                            rollback_result = await collection.delete_many({
                                "field_uuid": {"$in": inserted_field_uuids}
                            })
                            
                            # Also rollback successfully updated cluster counts
                            for cluster_uuid in successfully_updated_clusters:
                                try:
                                    await self._decrement_cluster_field_count_by(
                                        cluster_uuid, 
                                        cluster_field_counts[cluster_uuid]
                                    )
                                except Exception as rollback_cluster_error:
                                    logger.error(f"Failed to rollback cluster count for {cluster_uuid}: {rollback_cluster_error}")
                            
                            logger.error(f"Rolled back {rollback_result.deleted_count} fields due to cluster update failures")
                        except Exception as rollback_error:
                            logger.error(f"CRITICAL: Failed to rollback bulk field insertion: {rollback_error}")
                            raise FieldValidationError(
                                f"Bulk operation failed with incomplete rollback - manual intervention required",
                                cluster_update_errors + [f"Rollback error: {rollback_error}"]
                            )
                    
                    # Raise the original cluster update errors
                    raise FieldValidationError("Bulk cluster metadata updates failed", cluster_update_errors)
                
            except FieldValidationError:
                # Re-raise validation errors (these include our rollback scenarios)
                raise
            except Exception as unexpected_error:
                # Handle any other unexpected errors during cluster updates
                if fields_inserted and inserted_field_uuids:
                    try:
                        rollback_result = await collection.delete_many({
                            "field_uuid": {"$in": inserted_field_uuids}
                        })
                        logger.error(f"Rolled back {rollback_result.deleted_count} fields due to unexpected error: {unexpected_error}")
                    except Exception as rollback_error:
                        logger.error(f"Failed to rollback bulk field insertion: {rollback_error}")
                
                raise FieldValidationError(f"Unexpected error during bulk cluster updates: {unexpected_error}")
            
            logger.info(f"Bulk added {bulk_result.inserted_count} fields to {len(cluster_field_counts)} clusters")
            
            return {
                "success_count": bulk_result.inserted_count,
                "failure_count": len(fields_data) - bulk_result.inserted_count,
                "errors": [],
                "clusters_updated": list(cluster_field_counts.keys())
            }
            
        except (ClusterNotFoundError, FieldValidationError):
            raise
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")
            return {
                "success_count": bwe.details.get("nInserted", 0),
                "failure_count": len(fields_data) - bwe.details.get("nInserted", 0),
                "errors": [str(error) for error in bwe.details.get("writeErrors", [])],
                "clusters_updated": []
            }
        except Exception as e:
            logger.error(f"Error in bulk field addition: {e}")
            raise FieldValidationError(f"Failed to add fields in bulk: {e}")

    # ==================== FIELD SEARCH OPERATIONS ====================

    async def search_field_by_uuid(self, search_request: FieldSearchByUuidRequest) -> Optional[Dict[str, Any]]:
        """Search for a field by UUID across all clusters."""
        try:
            collection = await get_collection(self.collection_name)
            field_doc = await collection.find_one({"field_uuid": search_request.field_uuid})
            return field_doc
        except Exception as e:
            logger.error(f"Error searching field by UUID {search_request.field_uuid}: {e}")
            return None

    async def search_field_by_name_and_cluster(self, search_request: FieldSearchByNameAndClusterRequest) -> List[Dict[str, Any]]:
        """Search for fields by name within a specific cluster."""
        try:
            collection = await get_collection(self.collection_name)
            cursor = collection.find({
                "field_name": search_request.field_name,
                "cluster_parent_id": search_request.cluster_parent_id
            }).sort("created_at", -1)
            
            fields = await cursor.to_list(length=None)
            return fields
        except Exception as e:
            logger.error(f"Error searching fields by name and cluster: {e}")
            return []

    async def search_fields_by_cluster_uuid(self, search_request: FieldSearchByClusterUuidRequest) -> List[Dict[str, Any]]:
        """Search for all fields in a cluster, optionally filtered by field name."""
        try:
            collection = await get_collection(self.collection_name)
            
            query = {"cluster_uuid": search_request.cluster_uuid}
            if search_request.field_name:
                query["field_name"] = search_request.field_name
            
            cursor = collection.find(query).sort("field_name", 1)
            fields = await cursor.to_list(length=None)
            return fields
        except Exception as e:
            logger.error(f"Error searching fields by cluster UUID: {e}")
            return []

    async def search_fields_by_creator(self, search_request: FieldSearchByCreatorRequest) -> List[Dict[str, Any]]:
        """Search for fields by creator, optionally filtered by cluster."""
        try:
            collection = await get_collection(self.collection_name)
            
            query = {"created_by": search_request.created_by}
            if search_request.cluster_parent_id:
                query["cluster_parent_id"] = search_request.cluster_parent_id
            
            cursor = collection.find(query).sort("created_at", -1)
            fields = await cursor.to_list(length=None)
            return fields
        except Exception as e:
            logger.error(f"Error searching fields by creator: {e}")
            return []

    async def search_fields_by_name(self, search_request: FieldSearchByFieldNameRequest) -> List[Dict[str, Any]]:
        """Search for fields by name across all clusters."""
        try:
            collection = await get_collection(self.collection_name)
            cursor = collection.find({
                "field_name": search_request.field_name
            }).sort([("cluster_uuid", 1), ("created_at", -1)])
            
            fields = await cursor.to_list(length=None)
            return fields
        except Exception as e:
            logger.error(f"Error searching fields by name: {e}")
            return []

    # ==================== FIELD UPDATE OPERATIONS ====================

    async def update_field(self, field_uuid: str, updates: FieldUpdateRequest) -> Optional[Dict[str, Any]]:
        """
        Update field metadata with validation.
        
        Args:
            field_uuid: UUID of the field to update
            updates: FieldUpdateRequest with update data
            
        Returns:
            Updated field document or None if not found
        """
        try:
            # Check if field exists
            existing_field = await self.search_field_by_uuid(FieldSearchByUuidRequest(field_uuid=field_uuid))
            if not existing_field:
                raise FieldNotFoundError(field_uuid)
            
            # Build update document
            update_doc: Dict[str, Any] = {"last_updated": utc_now()}
            
            if updates.field_name is not None:
                if not updates.field_name.strip():
                    raise FieldValidationError("field_name cannot be empty")
                update_doc["field_name"] = updates.field_name.strip()
            
            if updates.description is not None:
                if not updates.description.strip():
                    raise FieldValidationError("description cannot be empty")
                update_doc["description"] = updates.description.strip()
            
            if updates.examples is not None:
                update_doc["examples"] = updates.examples
            
            if updates.data_type is not None:
                if not updates.data_type.strip():
                    raise FieldValidationError("data_type cannot be empty")
                update_doc["data_type"] = updates.data_type.strip()
            
            if updates.count is not None:
                if updates.count < 0:
                    raise FieldValidationError("count must be >= 0")
                update_doc["count"] = updates.count
            
            if updates.distinct_count is not None:
                if updates.distinct_count < 0:
                    raise FieldValidationError("distinct_count must be >= 0")
                # Check against current or updated count
                current_count = updates.count if updates.count is not None else existing_field["count"]
                if updates.distinct_count > current_count:
                    raise FieldValidationError("distinct_count must be <= count")
                update_doc["distinct_count"] = updates.distinct_count
            
            if updates.embedding is not None:
                update_doc["embedding"] = updates.embedding
            
            if updates.cluster_parent_id is not None:
                update_doc["cluster_parent_id"] = updates.cluster_parent_id
            
            # Perform update
            collection = await get_collection(self.collection_name)
            result = await collection.update_one(
                {"field_uuid": field_uuid},
                {"$set": update_doc}
            )
            
            if result.matched_count > 0:
                # Return updated field
                updated_field = await self.search_field_by_uuid(FieldSearchByUuidRequest(field_uuid=field_uuid))
                logger.info(f"Updated field: {field_uuid}")
                return updated_field
            else:
                return None
                
        except (FieldNotFoundError, FieldValidationError):
            raise
        except Exception as e:
            logger.error(f"Error updating field {field_uuid}: {e}")
            raise FieldValidationError(f"Failed to update field: {e}")

    # ==================== CLUSTER STATISTICS AND UTILITIES ====================

    async def get_cluster_field_statistics(self, cluster_uuid: str) -> Dict[str, Any]:
        """
        Get comprehensive field statistics for a cluster.
        
        Args:
            cluster_uuid: UUID of the cluster
            
        Returns:
            Dictionary with field statistics
        """
        try:
            collection = await get_collection(self.collection_name)
            
            # Aggregation pipeline for statistics
            pipeline = [
                {"$match": {"cluster_uuid": cluster_uuid}},
                {"$group": {
                    "_id": "$cluster_uuid",
                    "total_fields": {"$sum": 1},
                    "total_count": {"$sum": "$count"},
                    "total_distinct_count": {"$sum": "$distinct_count"},
                    "data_types": {"$addToSet": "$data_type"},
                    "avg_count": {"$avg": "$count"},
                    "max_count": {"$max": "$count"},
                    "min_count": {"$min": "$count"},
                    "latest_created": {"$max": "$created_at"},
                    "earliest_created": {"$min": "$created_at"}
                }}
            ]
            
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                stats = results[0]
                stats["cluster_uuid"] = cluster_uuid
                return stats
            else:
                return {
                    "cluster_uuid": cluster_uuid,
                    "total_fields": 0,
                    "total_count": 0,
                    "total_distinct_count": 0,
                    "data_types": [],
                    "avg_count": 0,
                    "max_count": 0,
                    "min_count": 0,
                    "latest_created": None,
                    "earliest_created": None
                }
                
        except Exception as e:
            logger.error(f"Error getting cluster field statistics for {cluster_uuid}: {e}")
            raise FieldValidationError(f"Failed to get cluster statistics: {e}")

    async def list_fields_in_cluster(self, cluster_uuid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all fields in a cluster with optional limit.
        
        Args:
            cluster_uuid: UUID of the cluster
            limit: Optional maximum number of fields to return
            
        Returns:
            List of field documents
        """
        try:
            collection = await get_collection(self.collection_name)
            cursor = collection.find({"cluster_uuid": cluster_uuid}).sort("field_name", 1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            fields = await cursor.to_list(length=None)
            return fields
            
        except Exception as e:
            logger.error(f"Error listing fields in cluster {cluster_uuid}: {e}")
            return []

    async def get_field_count_by_cluster(self, cluster_uuid: str) -> int:
        """Get the count of fields in a specific cluster."""
        try:
            collection = await get_collection(self.collection_name)
            count = await collection.count_documents({"cluster_uuid": cluster_uuid})
            return count
        except Exception as e:
            logger.error(f"Error getting field count for cluster {cluster_uuid}: {e}")
            return 0

    # ==================== PRIVATE HELPER METHODS ====================

    async def _decrement_cluster_field_count(self, cluster_uuid: str) -> Dict[str, Any]:
        """Decrement field count for cluster."""
        try:
            cluster_service = ClusterMetadataService()
            cluster = await cluster_service.get_cluster(cluster_uuid)
            
            if cluster:
                # Update with decremented count
                current_count = cluster.get("total_fields", 0)
                new_count = max(0, current_count - 1)  # Ensure count doesn't go negative
                
                collection = await get_collection(cluster_service.collection_name)
                await collection.update_one(
                    {"_id": cluster_uuid},
                    {
                        "$set": {
                            "total_fields": new_count,
                            "last_updated": utc_now()
                        }
                    }
                )
                
            return cluster or {}
        except Exception as e:
            logger.error(f"Error decrementing field count for cluster {cluster_uuid}: {e}")
            return {}

    async def _increment_cluster_field_count_by(self, cluster_uuid: str, count: int) -> Dict[str, Any]:
        """Increment field count for cluster by specified amount."""
        try:
            cluster_service = ClusterMetadataService()
            collection = await get_collection(cluster_service.collection_name)
            
            await collection.update_one(
                {"_id": cluster_uuid},
                {
                    "$inc": {"total_fields": count},
                    "$set": {"last_updated": utc_now()}
                }
            )
            
            return await cluster_service.get_cluster(cluster_uuid) or {}
        except Exception as e:
            logger.error(f"Error incrementing field count by {count} for cluster {cluster_uuid}: {e}")
            return {}

    async def _decrement_cluster_field_count_by(self, cluster_uuid: str, count: int) -> Dict[str, Any]:
        """Decrement field count for cluster by specified amount."""
        try:
            cluster_service = ClusterMetadataService()
            collection = await get_collection(cluster_service.collection_name)
            
            await collection.update_one(
                {"_id": cluster_uuid},
                {
                    "$inc": {"total_fields": -count},  # Negative increment for decrement
                    "$set": {"last_updated": utc_now()}
                }
            )
            
            return await cluster_service.get_cluster(cluster_uuid) or {}
        except Exception as e:
            logger.error(f"Error decrementing field count by {count} for cluster {cluster_uuid}: {e}")
            return {}

    # ==================== INDEX MANAGEMENT ====================

    async def create_indexes(self):
        """
        Create indexes for the cluster_fields collection.
        Based on Neo4j constraints and common query patterns.
        """
        try:
            collection = await get_collection(self.collection_name)
            
            # Create indexes based on Neo4j constraints and search patterns
            indexes = [
                # Unique field UUID (primary constraint)
                ("field_uuid", 1),
                
                # Cluster-based queries
                ("cluster_uuid", 1),
                ("cluster_parent_id", 1),
                
                # Field name queries
                ("field_name", 1),
                ("field_name", 1, "cluster_uuid", 1),  # Compound for name + cluster searches
                
                # Creator and temporal queries
                ("created_by", 1),
                ("created_at", -1),
                ("last_updated", -1),
                
                # Data type analysis
                ("data_type", 1),
                
                # Count-based queries
                ("count", -1),
                ("distinct_count", -1)
            ]
            
            for index_spec in indexes:
                if len(index_spec) == 2:
                    await collection.create_index([(index_spec[0], index_spec[1])], unique=(index_spec[0] == "field_uuid"))
                else:
                    # Compound index
                    index_pairs = [(index_spec[i], index_spec[i+1]) for i in range(0, len(index_spec), 2)]
                    await collection.create_index(index_pairs)
            
            logger.info(f"Created indexes for {self.collection_name} collection")
            
        except Exception as e:
            logger.error(f"Error creating indexes for {self.collection_name}: {e}")
