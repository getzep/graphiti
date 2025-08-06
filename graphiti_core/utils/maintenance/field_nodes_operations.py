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

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.field_nodes import FieldNode, ClusterNode, get_field_node_from_record, get_cluster_node_from_record
from graphiti_core.models.nodes.field_db_queries import (
    FIELD_NODE_GET_BY_CLUSTER_ID,
    FIELD_NODE_SEARCH_BY_NAME,
    FIELD_NODE_SAVE_BULK,
    CLUSTER_NODE_GET_BY_ORGANIZATION,
    CLUSTER_NODE_SAVE_BULK,
)
from graphiti_core.cluster_metadata.cluster_service import ClusterMetadataService
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError as MongoClusterNotFoundError,
    ClusterValidationError,
    DuplicateClusterError,
    InvalidOrganizationError,
)
from graphiti_core.cluster_metadata.models import ClusterCreateRequest

logger = logging.getLogger(__name__)


# ==================== FIELD NODE OPERATIONS ====================

async def get_fields_by_cluster_id(driver: GraphDriver, cluster_id: str) -> list[FieldNode]:
    """Get all Field nodes belonging to a specific cluster"""
    records, _, _ = await driver.execute_query(
        FIELD_NODE_GET_BY_CLUSTER_ID,
        cluster_id=cluster_id,
        routing_='r',
    )

    return [get_field_node_from_record(record) for record in records]


async def search_fields_by_name(
    driver: GraphDriver, 
    name: str, 
    cluster_id: str | None = None, 
    limit: int = 50
) -> list[FieldNode]:
    """Search Field nodes by name with optional cluster filtering"""
    records, _, _ = await driver.execute_query(
        FIELD_NODE_SEARCH_BY_NAME,
        name=name,
        cluster_id=cluster_id,
        limit=limit,
        routing_='r',
    )

    return [get_field_node_from_record(record) for record in records]


async def save_fields_bulk(driver: GraphDriver, fields: list[FieldNode]) -> Any:
    """Save multiple Field nodes in bulk operation with MongoDB synchronization"""
    if not fields:
        return []

    # Update temporal fields for all nodes
    for field in fields:
        field.update_temporal_fields()

    field_data = [
        {
            'uuid': field.uuid,
            'name': field.name,
            'description': field.description,
            'examples': field.examples,
            'data_type': field.data_type,
            'count': field.count,
            'distinct_count': field.distinct_count,
            'primary_cluster_id': field.primary_cluster_id,
            'embedding': field.embedding,
            'validated_at': field.validated_at,
            'invalidated_at': field.invalidated_at,
            'last_updated': field.last_updated,
            'created_at': field.created_at,
        }
        for field in fields
    ]

    result = await driver.execute_query(
        FIELD_NODE_SAVE_BULK,
        fields=field_data,
    )

    # Sync field counts with MongoDB cluster metadata in batch
    await _sync_field_counts_bulk(fields)

    logger.debug(f'Saved {len(fields)} Field Nodes to Graph in bulk')
    return result


async def create_field_embeddings_batch(embedder: EmbedderClient, fields: list[FieldNode]) -> None:
    """Create embeddings for multiple field nodes in batch"""
    if not fields:  # Handle empty list case
        return
    
    descriptions = [field.description for field in fields]
    embeddings = await embedder.create_batch(descriptions)
    
    for field, embedding in zip(fields, embeddings, strict=True):
        field.embedding = embedding


# ==================== CLUSTER NODE OPERATIONS ====================

async def get_clusters_by_organization(driver: GraphDriver, organization: str) -> list[ClusterNode]:
    """Get all Cluster nodes for a specific organization"""
    records, _, _ = await driver.execute_query(
        CLUSTER_NODE_GET_BY_ORGANIZATION,
        organization=organization,
        routing_='r',
    )

    return [get_cluster_node_from_record(record) for record in records]


async def save_clusters_bulk(driver: GraphDriver, clusters: list[ClusterNode]) -> Any:
    """Save multiple Cluster nodes in bulk operation with MongoDB synchronization"""
    if not clusters:
        return []

    # Update temporal fields for all nodes
    for cluster in clusters:
        cluster.update_temporal_fields()

    cluster_data = [
        {
            'uuid': cluster.uuid,
            'name': cluster.name,
            'organization': cluster.organization,
            'macro_name': cluster.macro_name,
            'validated_at': cluster.validated_at,
            'invalidated_at': cluster.invalidated_at,
            'last_updated': cluster.last_updated,
            'created_at': cluster.created_at,
        }
        for cluster in clusters
    ]

    result = await driver.execute_query(
        CLUSTER_NODE_SAVE_BULK,
        clusters=cluster_data,
    )

    # Sync clusters with MongoDB cluster metadata in batch
    await _sync_clusters_bulk(clusters)

    logger.debug(f'Saved {len(clusters)} Cluster Nodes to Graph in bulk')
    return result


# ==================== COMPLEX FIELD OPERATIONS ====================

async def analyze_field_distribution_by_cluster(
    driver: GraphDriver, 
    cluster_id: str
) -> dict[str, Any]:
    """Analyze field distribution and statistics for a cluster"""
    fields = await get_fields_by_cluster_id(driver, cluster_id)
    
    if not fields:
        return {
            'cluster_id': cluster_id,
            'total_fields': 0,
            'data_type_distribution': {},
            'total_events': 0,
            'avg_distinct_ratio': 0.0,
        }
    
    data_type_distribution = {}
    total_events = 0
    distinct_ratios = []
    
    for field in fields:
        # Data type distribution
        data_type_distribution[field.data_type] = data_type_distribution.get(field.data_type, 0) + 1
        
        # Total events
        total_events += field.count
        
        # Distinct ratio calculation
        if field.count > 0:
            distinct_ratios.append(field.distinct_count / field.count)
    
    avg_distinct_ratio = sum(distinct_ratios) / len(distinct_ratios) if distinct_ratios else 0.0
    
    return {
        'cluster_id': cluster_id,
        'total_fields': len(fields),
        'data_type_distribution': data_type_distribution,
        'total_events': total_events,
        'avg_distinct_ratio': avg_distinct_ratio,
        'fields_with_examples': len([f for f in fields if f.examples]),
    }


async def find_similar_fields_across_clusters(
    driver: GraphDriver,
    field_name_pattern: str,
    min_similarity_threshold: float = 0.8
) -> list[dict[str, Any]]:
    """Find fields with similar names across different clusters"""
    # This would typically use embedding similarity or fuzzy matching
    # For now, we'll use a simple name-based search
    all_matches = await search_fields_by_name(driver, field_name_pattern, limit=100)
    
    # Group by cluster
    cluster_groups = {}
    for field in all_matches:
        cluster_id = field.primary_cluster_id
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(field)
    
    # Return fields that appear in multiple clusters
    similar_fields = []
    for cluster_id, fields in cluster_groups.items():
        for field in fields:
            similar_in_other_clusters = []
            for other_cluster_id, other_fields in cluster_groups.items():
                if other_cluster_id != cluster_id:
                    # Simple name similarity check
                    for other_field in other_fields:
                        if field.name.lower() in other_field.name.lower() or other_field.name.lower() in field.name.lower():
                            similar_in_other_clusters.append({
                                'field': other_field,
                                'cluster_id': other_cluster_id,
                            })
            
            if similar_in_other_clusters:
                similar_fields.append({
                    'source_field': field,
                    'source_cluster_id': cluster_id,
                    'similar_fields': similar_in_other_clusters,
                })
    
    return similar_fields


async def validate_field_cluster_consistency(
    driver: GraphDriver,
    cluster_id: str
) -> dict[str, Any]:
    """Validate that all fields in a cluster have consistent metadata"""
    fields = await get_fields_by_cluster_id(driver, cluster_id)
    
    validation_results = {
        'cluster_id': cluster_id,
        'total_fields': len(fields),
        'validation_errors': [],
        'warnings': [],
        'is_valid': True,
    }
    
    for field in fields:
        # Check required fields
        if not field.description.strip():
            validation_results['validation_errors'].append(f"Field {field.name} has empty description")
            validation_results['is_valid'] = False
        
        if not field.data_type.strip():
            validation_results['validation_errors'].append(f"Field {field.name} has empty data_type")
            validation_results['is_valid'] = False
        
        if field.primary_cluster_id != cluster_id:
            validation_results['validation_errors'].append(
                f"Field {field.name} has mismatched cluster_id: {field.primary_cluster_id} != {cluster_id}"
            )
            validation_results['is_valid'] = False
        
        # Check data consistency
        if field.distinct_count > field.count:
            validation_results['validation_errors'].append(
                f"Field {field.name} has distinct_count ({field.distinct_count}) > count ({field.count})"
            )
            validation_results['is_valid'] = False
        
        # Warnings for potential issues
        if field.count == 0:
            validation_results['warnings'].append(f"Field {field.name} has zero occurrences")
        
        if not field.examples:
            validation_results['warnings'].append(f"Field {field.name} has no examples")
        
        if field.embedding is None:
            validation_results['warnings'].append(f"Field {field.name} has no embedding")
    
    return validation_results


# ==================== MONGODB SYNCHRONIZATION HELPERS ====================

async def _sync_field_counts_bulk(fields: list[FieldNode]) -> None:
    """Sync field counts with MongoDB cluster metadata in batch"""
    if not fields:
        return
    
    # Group fields by cluster to optimize MongoDB operations
    cluster_field_counts = {}
    for field in fields:
        cluster_id = field.primary_cluster_id
        cluster_field_counts[cluster_id] = cluster_field_counts.get(cluster_id, 0) + 1
    
    # Initialize cluster metadata service
    cluster_service = ClusterMetadataService()
    
    # Update field counts for each cluster
    for cluster_id, field_count in cluster_field_counts.items():
        try:
            # Validate cluster exists before updating
            cluster_exists = await cluster_service.validate_cluster_exists(cluster_id)
            if cluster_exists:
                # For bulk operations, we increment by the number of fields being added
                for _ in range(field_count):
                    await cluster_service.increment_field_count(cluster_id)
                logger.debug(f'Synchronized {field_count} fields with MongoDB cluster {cluster_id}')
            else:
                logger.warning(f'Cluster {cluster_id} not found in MongoDB for {field_count} fields')
                
        except (MongoClusterNotFoundError, ClusterValidationError) as e:
            # Log the error but don't fail the bulk operation
            logger.warning(f'MongoDB sync failed for cluster {cluster_id}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the bulk operation
            logger.error(f'Unexpected error during MongoDB sync for cluster {cluster_id}: {e}')


async def _sync_clusters_bulk(clusters: list[ClusterNode]) -> None:
    """Sync cluster data with MongoDB cluster metadata in batch"""
    if not clusters:
        return
    
    # Initialize cluster metadata service
    cluster_service = ClusterMetadataService()
    
    for cluster in clusters:
        try:
            # Check if cluster already exists in MongoDB
            existing_cluster = await cluster_service.get_cluster(cluster.name)
            
            if not existing_cluster:
                # Create new cluster in MongoDB
                cluster_request = ClusterCreateRequest(
                    cluster_id=cluster.name,
                    cluster_uuid=cluster.uuid,
                    macro_name=cluster.macro_name,
                    organization=cluster.organization,
                    description=f"Cluster for {cluster.macro_name} in {cluster.organization}",
                    status="active",
                    total_fields=0,
                    created_by="neo4j_bulk_sync"
                )
                
                await cluster_service.create_cluster(cluster_request)
                logger.debug(f'Created cluster in MongoDB: {cluster.name}')
            else:
                logger.debug(f'Cluster {cluster.name} already exists in MongoDB')
                
        except (DuplicateClusterError, InvalidOrganizationError, ClusterValidationError) as e:
            # Log the error but don't fail the bulk operation
            logger.warning(f'MongoDB sync failed for cluster {cluster.uuid}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the bulk operation
            logger.error(f'Unexpected error during MongoDB sync for cluster {cluster.uuid}: {e}')
