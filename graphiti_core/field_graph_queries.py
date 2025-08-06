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

"""
Database query utilities for the Security Field System.

This module provides Neo4j index creation and query generation for the security field 
architecture including Field nodes, Cluster nodes, and their relationships.
"""

from typing import Any
from typing_extensions import LiteralString

from graphiti_core.models.edges.field_edges_db_queries import (
    BELONGS_TO_EDGE_SAVE_BULK,
    FIELD_RELATIONSHIP_EDGE_SAVE_BULK,
)
from graphiti_core.models.nodes.field_db_queries import (
    FIELD_NODE_SAVE_BULK,
    CLUSTER_NODE_SAVE_BULK,
)


def get_field_range_indices() -> list[LiteralString]:
    """
    Get all range (performance) indexes for the Security Field System.
    
    Returns:
        List of Cypher queries to create range indexes for Field and Cluster nodes
        and their relationships (BELONGS_TO, FIELD_RELATES_TO).
    """
    return [
        # Field Node Indexes
        'CREATE INDEX field_uuid_index IF NOT EXISTS FOR (f:Field) ON (f.uuid)',
        'CREATE INDEX field_name_index IF NOT EXISTS FOR (f:Field) ON (f.name)',
        'CREATE INDEX field_cluster_index IF NOT EXISTS FOR (f:Field) ON (f.primary_cluster_id)',
        'CREATE INDEX field_data_type_index IF NOT EXISTS FOR (f:Field) ON (f.data_type)',
        'CREATE INDEX field_count_index IF NOT EXISTS FOR (f:Field) ON (f.count)',
        'CREATE INDEX field_created_at_index IF NOT EXISTS FOR (f:Field) ON (f.created_at)',
        'CREATE INDEX field_validated_at_index IF NOT EXISTS FOR (f:Field) ON (f.validated_at)',
        'CREATE INDEX field_last_updated_index IF NOT EXISTS FOR (f:Field) ON (f.last_updated)',
        
        # Cluster Node Indexes
        'CREATE INDEX cluster_uuid_index IF NOT EXISTS FOR (c:Cluster) ON (c.uuid)',
        'CREATE INDEX cluster_name_index IF NOT EXISTS FOR (c:Cluster) ON (c.name)',
        'CREATE INDEX cluster_org_index IF NOT EXISTS FOR (c:Cluster) ON (c.organization)',
        'CREATE INDEX cluster_macro_index IF NOT EXISTS FOR (c:Cluster) ON (c.macro_name)',
        'CREATE INDEX cluster_created_at_index IF NOT EXISTS FOR (c:Cluster) ON (c.created_at)',
        'CREATE INDEX cluster_validated_at_index IF NOT EXISTS FOR (c:Cluster) ON (c.validated_at)',
        
        # BELONGS_TO Relationship Indexes
        'CREATE INDEX belongs_to_uuid_index IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.uuid)',
        'CREATE INDEX belongs_to_source_index IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.source_node_uuid)',
        'CREATE INDEX belongs_to_target_index IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.target_node_uuid)',
        'CREATE INDEX belongs_to_cluster_index IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.cluster_partition_id)',
        'CREATE INDEX belongs_to_created_at_index IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.created_at)',
        
        # FIELD_RELATES_TO Relationship Indexes
        'CREATE INDEX field_relationship_uuid_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.uuid)',
        'CREATE INDEX field_relationship_source_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.source_node_uuid)',
        'CREATE INDEX field_relationship_target_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.target_node_uuid)',
        'CREATE INDEX field_relationship_name_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.name)',
        'CREATE INDEX field_relationship_confidence_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.confidence)',
        'CREATE INDEX field_relationship_cluster_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.cluster_partition_id)',
        'CREATE INDEX field_relationship_created_at_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.created_at)',
        'CREATE INDEX field_relationship_valid_at_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.valid_at)',
        'CREATE INDEX field_relationship_invalid_at_index IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() ON (r.invalid_at)',
    ]


def get_field_fulltext_indices() -> list[LiteralString]:
    """
    Get all fulltext search indexes for the Security Field System.
    
    Returns:
        List of Cypher queries to create fulltext indexes for searchable text fields
        in Field and Cluster nodes and their relationships.
    """
    return [
        # Field Node Fulltext Indexes
        """CREATE FULLTEXT INDEX field_fulltext IF NOT EXISTS 
        FOR (f:Field) ON EACH [f.name, f.description]""",
        
        # Cluster Node Fulltext Indexes  
        """CREATE FULLTEXT INDEX cluster_fulltext IF NOT EXISTS 
        FOR (c:Cluster) ON EACH [c.name, c.organization, c.macro_name]""",
        
        # Field Relationship Fulltext Indexes
        """CREATE FULLTEXT INDEX field_relationship_fulltext IF NOT EXISTS 
        FOR ()-[r:FIELD_RELATES_TO]-() ON EACH [r.name, r.description]""",
    ]


def get_field_vector_indices() -> list[LiteralString]:
    """
    Get all vector indexes for the Security Field System.
    
    Returns:
        List of Cypher queries to create vector indexes for embedding-based
        semantic search on Field descriptions and relationship descriptions.
    """
    return [
        # Field Node Vector Indexes
        """CREATE VECTOR INDEX field_embedding_index IF NOT EXISTS 
        FOR (f:Field) ON (f.embedding) 
        OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}""",
        
        # Field Relationship Vector Indexes
        """CREATE VECTOR INDEX field_relationship_embedding_index IF NOT EXISTS 
        FOR ()-[r:FIELD_RELATES_TO]-() ON (r.description_embedding) 
        OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}""",
    ]


def get_field_node_constraints() -> list[LiteralString]:
    """
    Get uniqueness and existence constraints for Field nodes.
    
    Returns:
        List of Cypher queries to create Field node constraints ensuring
        data integrity and required field validation.
    """
    return [
        # Field Node Uniqueness and Existence Constraints
        'CREATE CONSTRAINT field_uuid_unique IF NOT EXISTS FOR (f:Field) REQUIRE f.uuid IS UNIQUE',
        'CREATE CONSTRAINT field_name_not_null IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS NOT NULL',
        'CREATE CONSTRAINT field_primary_cluster_not_null IF NOT EXISTS FOR (f:Field) REQUIRE f.primary_cluster_id IS NOT NULL',
        'CREATE CONSTRAINT field_description_not_null IF NOT EXISTS FOR (f:Field) REQUIRE f.description IS NOT NULL',
        'CREATE CONSTRAINT field_data_type_not_null IF NOT EXISTS FOR (f:Field) REQUIRE f.data_type IS NOT NULL',
    ]


def get_cluster_node_constraints() -> list[LiteralString]:
    """
    Get uniqueness and existence constraints for Cluster nodes.
    
    Returns:
        List of Cypher queries to create Cluster node constraints ensuring
        organizational data integrity and required field validation.
    """
    return [
        # Cluster Node Uniqueness and Existence Constraints
        'CREATE CONSTRAINT cluster_uuid_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE c.uuid IS UNIQUE',
        'CREATE CONSTRAINT cluster_name_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE c.name IS UNIQUE',
        'CREATE CONSTRAINT cluster_name_not_null IF NOT EXISTS FOR (c:Cluster) REQUIRE c.name IS NOT NULL',
        'CREATE CONSTRAINT cluster_organization_not_null IF NOT EXISTS FOR (c:Cluster) REQUIRE c.organization IS NOT NULL',
        'CREATE CONSTRAINT cluster_macro_name_not_null IF NOT EXISTS FOR (c:Cluster) REQUIRE c.macro_name IS NOT NULL',
        'CREATE CONSTRAINT cluster_macro_organization_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE (c.macro_name, c.organization) IS UNIQUE',
    ]


def get_belongs_to_relationship_constraints() -> list[LiteralString]:
    """
    Get constraints for BELONGS_TO relationships between Field and Cluster nodes.
    
    Returns:
        List of Cypher queries to create BELONGS_TO relationship constraints
        ensuring proper Field-Cluster membership and uniqueness.
    """
    return [
        # BELONGS_TO Relationship Constraints
        'CREATE CONSTRAINT belongs_to_uuid_unique IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() REQUIRE r.uuid IS UNIQUE',
        'CREATE CONSTRAINT belongs_to_source_target_unique IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() REQUIRE (r.source_node_uuid, r.target_node_uuid) IS UNIQUE',
        'CREATE CONSTRAINT belongs_to_cluster_partition_not_null IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() REQUIRE r.cluster_partition_id IS NOT NULL',
        'CREATE CONSTRAINT belongs_to_no_self_relationship IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() REQUIRE r.source_node_uuid <> r.target_node_uuid',
        '''CREATE CONSTRAINT belongs_to_field_to_cluster_only IF NOT EXISTS 
        FOR (f:Field)-[r:BELONGS_TO]->(c:Cluster) REQUIRE 
        EXISTS { MATCH (f:Field)-[r:BELONGS_TO]->(c:Cluster) }''',
    ]


def get_field_relationship_constraints() -> list[LiteralString]:
    """
    Get constraints for FIELD_RELATES_TO relationships between Field nodes.
    
    Returns:
        List of Cypher queries to create FIELD_RELATES_TO relationship constraints
        ensuring proper semantic relationships and confidence validation.
    """
    return [
        # FIELD_RELATES_TO Relationship Constraints  
        'CREATE CONSTRAINT field_relationship_uuid_unique IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE r.uuid IS UNIQUE',
        'CREATE CONSTRAINT field_relationship_name_not_null IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE r.name IS NOT NULL',
        'CREATE CONSTRAINT field_relationship_cluster_partition_not_null IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE r.cluster_partition_id IS NOT NULL',
        'CREATE CONSTRAINT field_relationship_confidence_valid IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE r.confidence >= 0.0 AND r.confidence <= 1.0',
        'CREATE CONSTRAINT field_relationship_no_self_relationship IF NOT EXISTS FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE r.source_node_uuid <> r.target_node_uuid',
    ]


def get_cluster_isolation_constraints() -> list[LiteralString]:
    """
    Get constraints that enforce cluster isolation and organizational boundaries.
    
    Returns:
        List of Cypher queries to create constraints ensuring Fields can only
        belong to their designated clusters and relationships respect isolation.
    """
    return [
        # Constraint: Fields can only have BELONGS_TO relationships with their primary cluster
        '''CREATE CONSTRAINT field_cluster_consistency IF NOT EXISTS 
        FOR (f:Field) REQUIRE 
        EXISTS { 
            MATCH (f)-[:BELONGS_TO]->(c:Cluster) 
            WHERE f.primary_cluster_id = c.uuid 
        }''',
        
        # Constraint: Ensure FIELD_RELATES_TO relationships respect cluster isolation
        '''CREATE CONSTRAINT field_relationship_cluster_isolation IF NOT EXISTS
        FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE
        EXISTS {
            MATCH (f1:Field)-[r:FIELD_RELATES_TO]->(f2:Field)
            MATCH (f1)-[:BELONGS_TO]->(c:Cluster {uuid: r.cluster_partition_id})
            MATCH (f2)-[:BELONGS_TO]->(c:Cluster {uuid: r.cluster_partition_id})
        }''',
    ]


def get_temporal_validation_constraints() -> list[LiteralString]:
    """
    Get constraints that ensure temporal field consistency across nodes and relationships.
    
    Returns:
        List of Cypher queries to create temporal validation constraints ensuring
        logical date/time relationships and lifecycle management.
    """
    return [
        # Node Temporal Consistency
        'CREATE CONSTRAINT field_temporal_consistency IF NOT EXISTS FOR (f:Field) REQUIRE f.created_at <= f.last_updated',
        'CREATE CONSTRAINT cluster_temporal_consistency IF NOT EXISTS FOR (c:Cluster) REQUIRE c.created_at <= c.last_updated',
        
        # Relationship Temporal Consistency
        '''CREATE CONSTRAINT field_relationship_temporal_consistency IF NOT EXISTS 
        FOR ()-[r:FIELD_RELATES_TO]-() REQUIRE 
        (r.invalid_at IS NULL) OR (r.valid_at IS NULL) OR (r.valid_at <= r.invalid_at)''',
    ]


def get_data_quality_constraints() -> list[LiteralString]:
    """
    Get constraints that ensure data quality and logical value ranges.
    
    Returns:
        List of Cypher queries to create data quality constraints ensuring
        field counts, confidence scores, and other metrics are logically valid.
    """
    return [
        # Field Count Validation
        'CREATE CONSTRAINT field_count_non_negative IF NOT EXISTS FOR (f:Field) REQUIRE f.count >= 0',
        'CREATE CONSTRAINT field_distinct_count_non_negative IF NOT EXISTS FOR (f:Field) REQUIRE f.distinct_count >= 0',
        'CREATE CONSTRAINT field_distinct_count_logical IF NOT EXISTS FOR (f:Field) REQUIRE f.distinct_count <= f.count',
    ]


def get_field_constraints() -> list[LiteralString]:
    """
    Get all constraints for the Security Field System.
    
    Returns:
        Combined list of all constraint categories for complete data integrity
        enforcement across the security field graph system.
    """
    return (
        get_field_node_constraints() +
        get_cluster_node_constraints() +
        get_belongs_to_relationship_constraints() +
        get_field_relationship_constraints() +
        get_cluster_isolation_constraints() +
        get_temporal_validation_constraints() +
        get_data_quality_constraints()
    )


def get_field_nodes_query(name: str, query: str | None = None) -> str:
    """
    Generate fulltext search query for Field or Cluster nodes.
    
    Args:
        name: Index name ('field_fulltext' or 'cluster_fulltext')
        query: Search query string
        
    Returns:
        Formatted Cypher query for fulltext node search
    """
    return f'CALL db.index.fulltext.queryNodes("{name}", {query}, {{limit: $limit}})'


def get_field_relationships_query(name: str, query: str | None = None) -> str:
    """
    Generate fulltext search query for Field relationships.
    
    Args:
        name: Index name ('field_relationship_fulltext')
        query: Search query string
        
    Returns:
        Formatted Cypher query for fulltext relationship search
    """
    return f'CALL db.index.fulltext.queryRelationships("{name}", {query}, {{limit: $limit}})'


def get_field_vector_cosine_func_query(vec1: str, vec2: str) -> str:
    """
    Generate vector similarity function query for Field embeddings.
    
    Args:
        vec1: First vector reference (e.g., 'f.embedding')
        vec2: Second vector reference (e.g., '$query_embedding')
        
    Returns:
        Formatted Cypher expression for cosine similarity calculation
    """
    return f'vector.similarity.cosine({vec1}, {vec2})'


def get_field_node_save_bulk_query() -> str:
    """Get bulk save query for Field nodes."""
    return FIELD_NODE_SAVE_BULK


def get_cluster_node_save_bulk_query() -> str:
    """Get bulk save query for Cluster nodes."""
    return CLUSTER_NODE_SAVE_BULK


def get_belongs_to_edge_save_bulk_query() -> str:
    """Get bulk save query for BELONGS_TO relationships."""
    return BELONGS_TO_EDGE_SAVE_BULK


def get_field_relationship_edge_save_bulk_query() -> str:
    """Get bulk save query for FIELD_RELATES_TO relationships."""
    return FIELD_RELATIONSHIP_EDGE_SAVE_BULK


def build_all_field_indices_and_constraints() -> list[LiteralString]:
    """
    Get all indexes and constraints for the Security Field System.
    
    Returns:
        Combined list of all Cypher queries needed to set up the complete
        index and constraint structure for optimal performance and data integrity.
    """
    return (
        get_field_range_indices() +
        get_field_fulltext_indices() +
        get_field_vector_indices() +
        get_field_constraints()
    )


# Mapping for field system components
FIELD_SYSTEM_MAPPING = {
    'field_fulltext': 'Field',
    'cluster_fulltext': 'Cluster', 
    'field_relationship_fulltext': 'FIELD_RELATES_TO',
}