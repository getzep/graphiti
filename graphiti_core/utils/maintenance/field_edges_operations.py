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
from graphiti_core.field_edges import BelongsToEdge, FieldRelationshipEdge, get_belongs_to_edge_from_record, get_field_relationship_edge_from_record
from graphiti_core.models.edges.field_edges_db_queries import (
    BELONGS_TO_EDGES_GET_BY_CLUSTER,
    BELONGS_TO_EDGE_SAVE_BULK,
    FIELD_RELATIONSHIP_EDGES_GET_BY_SOURCE,
    FIELD_RELATIONSHIP_EDGES_GET_BY_TARGET,
    FIELD_RELATIONSHIP_EDGES_GET_BY_CLUSTER,
    FIELD_RELATIONSHIP_EDGE_SAVE_BULK,
)
from graphiti_core.cluster_metadata.cluster_service import ClusterMetadataService
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError as MongoClusterNotFoundError,
    ClusterValidationError,
)

logger = logging.getLogger(__name__)


# ==================== BELONGS TO EDGE OPERATIONS ====================

async def get_belongs_to_edges_by_cluster(driver: GraphDriver, cluster_uuid: str) -> list[BelongsToEdge]:
    """Get all BELONGS_TO edges for a specific cluster"""
    records, _, _ = await driver.execute_query(
        BELONGS_TO_EDGES_GET_BY_CLUSTER,
        cluster_uuid=cluster_uuid,
        routing_='r',
    )

    return [get_belongs_to_edge_from_record(record) for record in records]


async def save_belongs_to_edges_bulk(driver: GraphDriver, edges: list[BelongsToEdge]) -> Any:
    """Save multiple BELONGS_TO edges in bulk operation with MongoDB validation"""
    if not edges:
        return []

    # Validate clusters exist in MongoDB before creating relationships
    await _validate_clusters_bulk([edge.target_node_uuid for edge in edges])

    edge_data = [
        {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'cluster_partition_id': edge.cluster_partition_id,
            'created_at': edge.created_at,
        }
        for edge in edges
    ]

    result = await driver.execute_query(
        BELONGS_TO_EDGE_SAVE_BULK,
        belongs_to_edges=edge_data,
    )

    logger.debug(f'Saved {len(edges)} BELONGS_TO edges to Graph in bulk')
    return result


# ==================== FIELD RELATIONSHIP EDGE OPERATIONS ====================

async def get_field_relationships_by_source(
    driver: GraphDriver, 
    source_field_uuid: str, 
    cluster_partition_id: str
) -> list[FieldRelationshipEdge]:
    """Get all outgoing Field relationships from a source field"""
    records, _, _ = await driver.execute_query(
        FIELD_RELATIONSHIP_EDGES_GET_BY_SOURCE,
        source_field_uuid=source_field_uuid,
        cluster_partition_id=cluster_partition_id,
        routing_='r',
    )

    return [get_field_relationship_edge_from_record(record) for record in records]


async def get_field_relationships_by_target(
    driver: GraphDriver, 
    target_field_uuid: str, 
    cluster_partition_id: str
) -> list[FieldRelationshipEdge]:
    """Get all incoming Field relationships to a target field"""
    records, _, _ = await driver.execute_query(
        FIELD_RELATIONSHIP_EDGES_GET_BY_TARGET,
        target_field_uuid=target_field_uuid,
        cluster_partition_id=cluster_partition_id,
        routing_='r',
    )

    return [get_field_relationship_edge_from_record(record) for record in records]


async def get_field_relationships_by_cluster(
    driver: GraphDriver, 
    cluster_partition_id: str
) -> list[FieldRelationshipEdge]:
    """Get all Field relationships within a specific cluster"""
    records, _, _ = await driver.execute_query(
        FIELD_RELATIONSHIP_EDGES_GET_BY_CLUSTER,
        cluster_partition_id=cluster_partition_id,
        routing_='r',
    )

    return [get_field_relationship_edge_from_record(record) for record in records]


async def save_field_relationship_edges_bulk(
    driver: GraphDriver, 
    edges: list[FieldRelationshipEdge]
) -> Any:
    """Save multiple Field relationship edges in bulk operation with MongoDB validation"""
    if not edges:
        return []

    # Validate clusters exist in MongoDB before creating relationships
    await _validate_clusters_bulk([edge.cluster_partition_id for edge in edges])

    edge_data = [
        {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge.name,
            'description': edge.description,
            'confidence': edge.confidence,
            'cluster_partition_id': edge.cluster_partition_id,
            'relationship_type': edge.relationship_type,
            'description_embedding': edge.description_embedding,
            'created_at': edge.created_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
        }
        for edge in edges
    ]

    result = await driver.execute_query(
        FIELD_RELATIONSHIP_EDGE_SAVE_BULK,
        field_relationship_edges=edge_data,
    )

    logger.debug(f'Saved {len(edges)} FIELD_RELATES_TO edges to Graph in bulk')
    return result


async def create_field_relationship_embeddings_batch(
    embedder: EmbedderClient, 
    edges: list[FieldRelationshipEdge]
) -> None:
    """Create embeddings for multiple field relationship edges in batch"""
    if not edges:  # Handle empty list case
        return
    
    descriptions = [edge.description for edge in edges]
    embeddings = await embedder.create_batch(descriptions)
    
    for edge, embedding in zip(edges, embeddings, strict=True):
        edge.description_embedding = embedding


# ==================== COMPLEX RELATIONSHIP OPERATIONS ====================

async def build_field_cluster_membership(
    driver: GraphDriver,
    field_uuids: list[str],
    cluster_uuid: str
) -> list[BelongsToEdge]:
    """Build and save BELONGS_TO relationships for fields joining a cluster with MongoDB validation"""
    from graphiti_core.utils.datetime_utils import utc_now
    
    # Validate cluster exists in MongoDB before creating relationships
    await _validate_clusters_bulk([cluster_uuid])
    
    edges = []
    current_time = utc_now()
    
    # Create BelongsToEdge objects
    for field_uuid in field_uuids:
        edge = BelongsToEdge(
            source_node_uuid=field_uuid,
            target_node_uuid=cluster_uuid,
            cluster_partition_id=cluster_uuid,
            created_at=current_time,
        )
        edges.append(edge)
    
    # Save all edges to database using bulk operation
    if edges:
        await save_belongs_to_edges_bulk(driver, edges)
        logger.debug(f'Built and saved {len(edges)} BELONGS_TO relationships for cluster {cluster_uuid}')
    
    return edges


async def analyze_field_relationship_patterns(
    driver: GraphDriver,
    cluster_partition_id: str
) -> dict[str, Any]:
    """Analyze relationship patterns within a cluster"""
    relationships = await get_field_relationships_by_cluster(driver, cluster_partition_id)
    
    if not relationships:
        return {
            'cluster_id': cluster_partition_id,
            'total_relationships': 0,
            'relationship_types': {},
            'avg_confidence': 0.0,
            'temporal_distribution': {},
        }
    
    # Analyze relationship types
    relationship_types = {}
    confidences = []
    temporal_distribution = {}
    
    for rel in relationships:
        # Count relationship types
        relationship_types[rel.name] = relationship_types.get(rel.name, 0) + 1
        
        # Collect confidences
        confidences.append(rel.confidence)
        
        # Temporal distribution by month
        month_key = rel.created_at.strftime('%Y-%m')
        temporal_distribution[month_key] = temporal_distribution.get(month_key, 0) + 1
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        'cluster_id': cluster_partition_id,
        'total_relationships': len(relationships),
        'relationship_types': relationship_types,
        'avg_confidence': avg_confidence,
        'temporal_distribution': temporal_distribution,
        'confidence_distribution': {
            'min': min(confidences) if confidences else 0.0,
            'max': max(confidences) if confidences else 0.0,
            'avg': avg_confidence,
        },
    }


async def find_bidirectional_relationships(
    driver: GraphDriver,
    cluster_partition_id: str
) -> list[dict[str, Any]]:
    """Find bidirectional relationships between fields in a cluster"""
    relationships = await get_field_relationships_by_cluster(driver, cluster_partition_id)
    
    # Create a mapping of relationships
    relationship_map = {}
    for rel in relationships:
        key = (rel.source_node_uuid, rel.target_node_uuid)
        relationship_map[key] = rel
    
    bidirectional_pairs = []
    processed_pairs = set()
    
    for rel in relationships:
        source_uuid = rel.source_node_uuid
        target_uuid = rel.target_node_uuid
        reverse_key = (target_uuid, source_uuid)
        
        # Check if we've already processed this pair
        pair_key = tuple(sorted([source_uuid, target_uuid]))
        if pair_key in processed_pairs:
            continue
        
        # Check if reverse relationship exists
        if reverse_key in relationship_map:
            reverse_rel = relationship_map[reverse_key]
            bidirectional_pairs.append({
                'field_1_uuid': source_uuid,
                'field_2_uuid': target_uuid,
                'forward_relationship': {
                    'name': rel.name,
                    'confidence': rel.confidence,
                    'description': rel.description,
                },
                'reverse_relationship': {
                    'name': reverse_rel.name,
                    'confidence': reverse_rel.confidence,
                    'description': reverse_rel.description,
                },
                'avg_confidence': (rel.confidence + reverse_rel.confidence) / 2,
            })
            processed_pairs.add(pair_key)
    
    return bidirectional_pairs


async def validate_relationship_constraints(
    driver: GraphDriver,
    cluster_partition_id: str
) -> dict[str, Any]:
    """Validate that relationships follow field system constraints"""
    relationships = await get_field_relationships_by_cluster(driver, cluster_partition_id)
    belongs_to_edges = await get_belongs_to_edges_by_cluster(driver, cluster_partition_id)
    
    validation_results = {
        'cluster_id': cluster_partition_id,
        'total_relationships': len(relationships),
        'total_belongs_to': len(belongs_to_edges),
        'validation_errors': [],
        'warnings': [],
        'is_valid': True,
    }
    
    # Get all field UUIDs that belong to this cluster
    cluster_field_uuids = {edge.source_node_uuid for edge in belongs_to_edges}
    
    for rel in relationships:
        # Check cluster isolation
        if rel.source_node_uuid not in cluster_field_uuids:
            validation_results['validation_errors'].append(
                f"Relationship {rel.uuid}: source field {rel.source_node_uuid} not in cluster"
            )
            validation_results['is_valid'] = False
        
        if rel.target_node_uuid not in cluster_field_uuids:
            validation_results['validation_errors'].append(
                f"Relationship {rel.uuid}: target field {rel.target_node_uuid} not in cluster"
            )
            validation_results['is_valid'] = False
        
        # Check confidence range
        if not (0.0 <= rel.confidence <= 1.0):
            validation_results['validation_errors'].append(
                f"Relationship {rel.uuid}: confidence {rel.confidence} out of range [0.0, 1.0]"
            )
            validation_results['is_valid'] = False
        
        # Check required fields
        if not rel.name.strip():
            validation_results['validation_errors'].append(
                f"Relationship {rel.uuid}: empty name"
            )
            validation_results['is_valid'] = False
        
        if not rel.description.strip():
            validation_results['validation_errors'].append(
                f"Relationship {rel.uuid}: empty description"
            )
            validation_results['is_valid'] = False
        
        # Warnings for potential issues
        if rel.confidence < 0.5:
            validation_results['warnings'].append(
                f"Relationship {rel.uuid}: low confidence {rel.confidence}"
            )
        
        if rel.description_embedding is None:
            validation_results['warnings'].append(
                f"Relationship {rel.uuid}: no description embedding"
            )
        
        # Check temporal validity
        if rel.valid_at and rel.invalid_at and rel.valid_at >= rel.invalid_at:
            validation_results['validation_errors'].append(
                f"Relationship {rel.uuid}: valid_at >= invalid_at"
            )
            validation_results['is_valid'] = False
    
    return validation_results


async def get_field_relationship_network(
    driver: GraphDriver,
    field_uuid: str,
    cluster_partition_id: str,
    max_depth: int = 2
) -> dict[str, Any]:
    """Get the relationship network around a specific field"""
    if max_depth < 1:
        return {'nodes': [], 'relationships': []}
    
    visited_fields = set()
    all_relationships = []
    fields_to_process = [(field_uuid, 0)]  # (field_uuid, depth)
    
    while fields_to_process:
        current_field_uuid, depth = fields_to_process.pop(0)
        
        if current_field_uuid in visited_fields or depth >= max_depth:
            continue
        
        visited_fields.add(current_field_uuid)
        
        # Get outgoing relationships
        outgoing = await get_field_relationships_by_source(driver, current_field_uuid, cluster_partition_id)
        all_relationships.extend(outgoing)
        
        # Get incoming relationships
        incoming = await get_field_relationships_by_target(driver, current_field_uuid, cluster_partition_id)
        all_relationships.extend(incoming)
        
        # Add connected fields to processing queue
        if depth + 1 < max_depth:
            for rel in outgoing:
                if rel.target_node_uuid not in visited_fields:
                    fields_to_process.append((rel.target_node_uuid, depth + 1))
            
            for rel in incoming:
                if rel.source_node_uuid not in visited_fields:
                    fields_to_process.append((rel.source_node_uuid, depth + 1))
    
    # Remove duplicates from relationships
    unique_relationships = []
    seen_rel_uuids = set()
    for rel in all_relationships:
        if rel.uuid not in seen_rel_uuids:
            unique_relationships.append(rel)
            seen_rel_uuids.add(rel.uuid)
    
    return {
        'center_field_uuid': field_uuid,
        'max_depth': max_depth,
        'connected_field_uuids': list(visited_fields),
        'relationships': unique_relationships,
        'network_size': len(visited_fields),
        'relationship_count': len(unique_relationships),
    }


# ==================== MONGODB SYNCHRONIZATION HELPERS ====================

async def _validate_clusters_bulk(cluster_ids: list[str]) -> None:
    """Validate that clusters exist in MongoDB for bulk edge operations"""
    if not cluster_ids:
        return
    
    # Remove duplicates
    unique_cluster_ids = list(set(cluster_ids))
    
    # Initialize cluster metadata service
    cluster_service = ClusterMetadataService()
    
    # Validate each cluster
    for cluster_id in unique_cluster_ids:
        try:
            cluster_exists = await cluster_service.validate_cluster_exists(cluster_id)
            if not cluster_exists:
                logger.warning(f'Cluster {cluster_id} not found or inactive in MongoDB for bulk edge operation')
                
        except (MongoClusterNotFoundError, ClusterValidationError) as e:
            # Log the warning but don't fail the bulk operation
            logger.warning(f'MongoDB cluster validation failed for cluster {cluster_id}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the bulk operation
            logger.error(f'Unexpected error during MongoDB cluster validation for cluster {cluster_id}: {e}')
