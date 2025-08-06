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

# BELONGS_TO Edge Queries - Links Field nodes to their Cluster for organizational hierarchy
BELONGS_TO_EDGE_SAVE = """
    MATCH (field:Field {uuid: $source_node_uuid})
    MATCH (cluster:Cluster {uuid: $target_node_uuid})
    MERGE (field)-[r:BELONGS_TO {uuid: $uuid}]->(cluster)
    SET r = {uuid: $uuid, source_node_uuid: $source_node_uuid, target_node_uuid: $target_node_uuid, 
             cluster_partition_id: $cluster_partition_id, created_at: $created_at}
    RETURN r.uuid AS uuid"""

BELONGS_TO_EDGE_SAVE_BULK = """
    UNWIND $belongs_to_edges AS edge
    MATCH (field:Field {uuid: edge.source_node_uuid})
    MATCH (cluster:Cluster {uuid: edge.target_node_uuid})
    MERGE (field)-[r:BELONGS_TO {uuid: edge.uuid}]->(cluster)
    SET r = {uuid: edge.uuid, source_node_uuid: edge.source_node_uuid, target_node_uuid: edge.target_node_uuid,
             cluster_partition_id: edge.cluster_partition_id, created_at: edge.created_at}
    RETURN r.uuid AS uuid"""

BELONGS_TO_EDGE_GET = """
    MATCH (field:Field {uuid: $field_uuid})-[r:BELONGS_TO]->(cluster:Cluster)
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.cluster_partition_id AS cluster_partition_id, r.created_at AS created_at,
           field.uuid AS field_uuid, cluster.uuid AS cluster_uuid"""

BELONGS_TO_EDGES_GET_BY_CLUSTER = """
    MATCH (field:Field)-[r:BELONGS_TO]->(cluster:Cluster {uuid: $cluster_uuid})
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.cluster_partition_id AS cluster_partition_id, r.created_at AS created_at,
           field.uuid AS field_uuid, cluster.uuid AS cluster_uuid"""

BELONGS_TO_EDGE_DELETE = """
    MATCH (field:Field {uuid: $field_uuid})-[r:BELONGS_TO]->(cluster:Cluster {uuid: $cluster_uuid})
    DELETE r
    RETURN COUNT(r) AS deleted_count"""

# FIELD_RELATES_TO Edge Queries - Links Field nodes with contextual relationships
FIELD_RELATIONSHIP_EDGE_SAVE = """
    MATCH (source:Field {uuid: $source_node_uuid})
    MATCH (target:Field {uuid: $target_node_uuid})
    MATCH (source)-[:BELONGS_TO]->(c:Cluster {uuid: $cluster_partition_id})
    MATCH (target)-[:BELONGS_TO]->(c)
    WHERE source.count >= target.count
    AND ($invalid_at IS NULL OR $valid_at IS NULL OR $valid_at <= $invalid_at)
    AND $confidence >= 0.0 AND $confidence <= 1.0
    MERGE (source)-[r:FIELD_RELATES_TO {uuid: $uuid}]->(target)
    SET r = {uuid: $uuid, source_node_uuid: $source_node_uuid, target_node_uuid: $target_node_uuid,
             name: $name, description: $description, confidence: $confidence, 
             cluster_partition_id: $cluster_partition_id, relationship_type: $relationship_type,
             description_embedding: $description_embedding, created_at: $created_at, 
             valid_at: $valid_at, invalid_at: $invalid_at}
    RETURN r.uuid AS uuid"""

FIELD_RELATIONSHIP_EDGE_SAVE_BULK = """
    UNWIND $field_relationship_edges AS edge
    MATCH (source:Field {uuid: edge.source_node_uuid})
    MATCH (target:Field {uuid: edge.target_node_uuid})
    MATCH (source)-[:BELONGS_TO]->(c:Cluster {uuid: edge.cluster_partition_id})
    MATCH (target)-[:BELONGS_TO]->(c)
    WHERE source.count >= target.count
    AND (edge.invalid_at IS NULL OR edge.valid_at IS NULL OR edge.valid_at <= edge.invalid_at)
    AND edge.confidence >= 0.0 AND edge.confidence <= 1.0
    MERGE (source)-[r:FIELD_RELATES_TO {uuid: edge.uuid}]->(target)
    SET r = {uuid: edge.uuid, source_node_uuid: edge.source_node_uuid, target_node_uuid: edge.target_node_uuid,
             name: edge.name, description: edge.description, confidence: edge.confidence,
             cluster_partition_id: edge.cluster_partition_id, relationship_type: edge.relationship_type,
             description_embedding: edge.description_embedding, created_at: edge.created_at, 
             valid_at: edge.valid_at, invalid_at: edge.invalid_at}
    RETURN r.uuid AS uuid"""

FIELD_RELATIONSHIP_EDGE_GET = """
    MATCH (source:Field {uuid: $source_field_uuid})-[r:FIELD_RELATES_TO {uuid: $edge_uuid}]->(target:Field)
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.name AS name, r.description AS description, r.confidence AS confidence,
           r.cluster_partition_id AS cluster_partition_id, r.relationship_type AS relationship_type,
           r.description_embedding AS description_embedding, r.created_at AS created_at,
           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, target.uuid AS target_field_uuid"""

# Source -> Targets
FIELD_RELATIONSHIP_EDGES_GET_BY_SOURCE = """
    MATCH (source:Field {uuid: $source_field_uuid})-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.name AS name, r.description AS description, r.confidence AS confidence,
           r.cluster_partition_id AS cluster_partition_id, r.relationship_type AS relationship_type,
           r.description_embedding AS description_embedding, r.created_at AS created_at,
           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, target.uuid AS target_field_uuid
    ORDER BY r.created_at DESC"""

# Target -> Sources
FIELD_RELATIONSHIP_EDGES_GET_BY_TARGET = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field {uuid: $target_field_uuid})
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.name AS name, r.description AS description, r.confidence AS confidence,
           r.cluster_partition_id AS cluster_partition_id, r.relationship_type AS relationship_type,
           r.description_embedding AS description_embedding, r.created_at AS created_at,
           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, target.uuid AS target_field_uuid
    ORDER BY r.created_at DESC"""

# Returns all relationships in a cluster
FIELD_RELATIONSHIP_EDGES_GET_BY_CLUSTER = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.name AS name, r.description AS description, r.confidence AS confidence,
           r.cluster_partition_id AS cluster_partition_id, r.relationship_type AS relationship_type,
           r.description_embedding AS description_embedding, r.created_at AS created_at,
           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, target.uuid AS target_field_uuid
    ORDER BY r.created_at DESC"""

FIELD_RELATIONSHIP_EDGE_UPDATE = """
    MATCH (source:Field {uuid: $source_field_uuid})-[r:FIELD_RELATES_TO {uuid: $edge_uuid}]->(target:Field)
    WHERE $confidence >= 0.0 AND $confidence <= 1.0
    AND (r.invalid_at IS NULL OR $valid_at IS NULL OR $valid_at <= r.invalid_at)
    SET r.name = $name, r.description = $description, r.confidence = $confidence, r.valid_at = $valid_at
    RETURN r.uuid AS uuid"""

FIELD_RELATIONSHIP_EDGE_EXPIRE = """
    MATCH (source:Field {uuid: $source_field_uuid})-[r:FIELD_RELATES_TO {uuid: $edge_uuid}]->(target:Field)
    SET r.invalid_at = datetime()
    RETURN r.uuid AS uuid, r.invalid_at AS invalid_at"""

FIELD_RELATIONSHIP_EDGE_DELETE = """
    MATCH (source:Field {uuid: $source_field_uuid})-[r:FIELD_RELATES_TO {uuid: $edge_uuid}]->(target:Field)
    DELETE r
    RETURN COUNT(r) AS deleted_count"""

# Search and Discovery Queries for Field Relationships
FIELD_RELATIONSHIPS_SEARCH_BY_NAME = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
    AND toLower(r.name) CONTAINS toLower($search_term)
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.name AS name, r.description AS description, r.confidence AS confidence,
           r.cluster_partition_id AS cluster_partition_id, r.relationship_type AS relationship_type,
           r.description_embedding AS description_embedding, r.created_at AS created_at,
           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, source.name AS source_field_name,
           target.uuid AS target_field_uuid, target.name AS target_field_name
    ORDER BY r.created_at DESC
    LIMIT $limit"""

FIELD_RELATIONSHIPS_SEARCH_BY_DESCRIPTION = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
    AND toLower(r.description) CONTAINS toLower($search_term)
    RETURN r.uuid AS uuid, r.source_node_uuid AS source_node_uuid, r.target_node_uuid AS target_node_uuid,
           r.name AS name, r.description AS description, r.confidence AS confidence,
           r.cluster_partition_id AS cluster_partition_id, r.relationship_type AS relationship_type,
           r.description_embedding AS description_embedding, r.created_at AS created_at,
           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, source.name AS source_field_name,
           target.uuid AS target_field_uuid, target.name AS target_field_name
    ORDER BY r.created_at DESC
    LIMIT $limit"""

# Field relationship statistics and analytics
FIELD_RELATIONSHIP_COUNT_BY_CLUSTER = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
    RETURN COUNT(r) AS relationship_count"""

FIELD_RELATIONSHIP_COUNT_BY_FIELD = """
    MATCH (field:Field {uuid: $field_uuid})
    OPTIONAL MATCH (field)-[out_rel:FIELD_RELATES_TO]->(target:Field)
    WHERE out_rel.cluster_partition_id = $cluster_partition_id 
    AND (out_rel.invalid_at IS NULL OR out_rel.invalid_at > datetime())
    OPTIONAL MATCH (source:Field)-[in_rel:FIELD_RELATES_TO]->(field)
    WHERE in_rel.cluster_partition_id = $cluster_partition_id 
    AND (in_rel.invalid_at IS NULL OR in_rel.invalid_at > datetime())
    RETURN COUNT(DISTINCT out_rel) AS outgoing_count, COUNT(DISTINCT in_rel) AS incoming_count"""

# Field relationship validation and integrity checks
FIELD_RELATIONSHIPS_GET_EXPIRED = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND r.invalid_at IS NOT NULL AND r.invalid_at <= datetime()
    RETURN r.uuid AS uuid, r.name AS name, r.invalid_at AS invalid_at,
           source.uuid AS source_field_uuid, target.uuid AS target_field_uuid
    ORDER BY r.invalid_at DESC"""

FIELD_RELATIONSHIPS_CLEANUP_EXPIRED = """
    MATCH (source:Field)-[r:FIELD_RELATES_TO]->(target:Field)
    WHERE r.cluster_partition_id = $cluster_partition_id
    AND r.invalid_at IS NOT NULL AND r.invalid_at <= datetime() - duration('P30D')
    DELETE r
    RETURN COUNT(r) AS deleted_count"""
