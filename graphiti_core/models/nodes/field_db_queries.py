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

# =============================================================================
# SECURITY FIELD SYSTEM NODE QUERIES
# =============================================================================

# -----------------------------------------------------------------------------
# Field Node Queries - Core security field entities
# -----------------------------------------------------------------------------

FIELD_NODE_SAVE = """
        MATCH (c:Cluster {uuid: $primary_cluster_id})
        WHERE $count >= 0 AND $distinct_count >= 0 AND $distinct_count <= $count 
        AND $created_at <= $last_updated
        MERGE (f:Field {uuid: $uuid})
        SET f = {uuid: $uuid, name: $name, description: $description, 
                examples: $examples, data_type: $data_type, count: $count, distinct_count: $distinct_count,
                primary_cluster_id: $primary_cluster_id, validated_at: $validated_at, 
                invalidated_at: $invalidated_at, last_updated: $last_updated, created_at: $created_at}
        WITH f 
        CALL {
            WITH f
            WHERE $embedding IS NOT NULL
            CALL db.create.setNodeVectorProperty(f, "embedding", $embedding)
            RETURN f
        UNION
            WITH f
            WHERE $embedding IS NULL
            RETURN f
        }
        RETURN f.uuid AS uuid"""

FIELD_NODE_SAVE_BULK = """
    UNWIND $fields AS field
    MATCH (c:Cluster {uuid: field.primary_cluster_id})
    WHERE field.count >= 0 AND field.distinct_count >= 0 AND field.distinct_count <= field.count
    AND field.created_at <= field.last_updated
    MERGE (f:Field {uuid: field.uuid})
    SET f = {uuid: field.uuid, name: field.name, description: field.description, 
            examples: field.examples, data_type: field.data_type, count: field.count, 
            distinct_count: field.distinct_count, primary_cluster_id: field.primary_cluster_id, 
            validated_at: field.validated_at, invalidated_at: field.invalidated_at, 
            last_updated: field.last_updated, created_at: field.created_at}
    WITH f, field
    CALL {
        WITH f, field
        WHERE field.embedding IS NOT NULL
        CALL db.create.setNodeVectorProperty(f, "embedding", field.embedding)
        RETURN f
    UNION
        WITH f, field
        WHERE field.embedding IS NULL
        RETURN f
    }
    RETURN f.uuid AS uuid"""

FIELD_NODE_GET_BY_UUID = """
    MATCH (f:Field {uuid: $uuid})
    RETURN f.uuid as uuid, f.name as name, f.description as description, 
           f.examples as examples, f.data_type as data_type, f.count as count, 
           f.distinct_count as distinct_count, f.primary_cluster_id as primary_cluster_id, 
           f.embedding as embedding, f.validated_at as validated_at, 
           f.invalidated_at as invalidated_at, f.last_updated as last_updated, 
           f.created_at as created_at, labels(f) as labels"""

FIELD_NODE_GET_BY_UUIDS = """
    MATCH (f:Field)
    WHERE f.uuid IN $uuids
    RETURN f.uuid as uuid, f.name as name, f.description as description, 
           f.examples as examples, f.data_type as data_type, f.count as count, 
           f.distinct_count as distinct_count, f.primary_cluster_id as primary_cluster_id, 
           f.embedding as embedding, f.validated_at as validated_at, 
           f.invalidated_at as invalidated_at, f.last_updated as last_updated, 
           f.created_at as created_at, labels(f) as labels"""

# Return fields in a cluster
FIELD_NODE_GET_BY_CLUSTER_ID = """
    MATCH (f:Field)-[:BELONGS_TO]->(c:Cluster {uuid: $cluster_id})
    RETURN f.uuid as uuid, f.name as name, f.description as description, 
           f.examples as examples, f.data_type as data_type, f.count as count, 
           f.distinct_count as distinct_count, f.primary_cluster_id as primary_cluster_id, 
           f.embedding as embedding, f.validated_at as validated_at, 
           f.invalidated_at as invalidated_at, f.last_updated as last_updated, 
           f.created_at as created_at, labels(f) as labels"""

FIELD_NODE_SEARCH_BY_NAME = """
    MATCH (f:Field)
    WHERE f.name CONTAINS $name
    AND ($cluster_id IS NULL OR f.primary_cluster_id = $cluster_id)
    OPTIONAL MATCH (c:Cluster {uuid: f.primary_cluster_id})
    WHERE $cluster_id IS NULL OR c.uuid IS NOT NULL
    RETURN f.uuid as uuid, f.name as name, f.description as description, 
           f.examples as examples, f.data_type as data_type, f.count as count, 
           f.distinct_count as distinct_count, f.primary_cluster_id as primary_cluster_id, 
           f.embedding as embedding, f.validated_at as validated_at, 
           f.invalidated_at as invalidated_at, f.last_updated as last_updated, 
           f.created_at as created_at
    ORDER BY f.count DESC
    LIMIT $limit"""

FIELD_NODE_UPDATE = """
    MATCH (f:Field {uuid: $uuid})
    WHERE $count >= 0 AND $distinct_count >= 0 AND $distinct_count <= $count
    AND f.created_at <= $last_updated
    SET f.name = $name, f.description = $description, f.examples = $examples,
        f.data_type = $data_type, f.count = $count, f.distinct_count = $distinct_count,
        f.validated_at = $validated_at, f.last_updated = $last_updated
    WITH f
    CALL {
        WITH f
        WHERE $embedding IS NOT NULL
        CALL db.create.setNodeVectorProperty(f, "embedding", $embedding)
        RETURN f
    UNION
        WITH f
        WHERE $embedding IS NULL
        RETURN f
    }
    RETURN f.uuid AS uuid"""

FIELD_NODE_DELETE = """
    MATCH (f:Field {uuid: $uuid})
    DETACH DELETE f
    RETURN COUNT(f) AS deleted_count"""

# -----------------------------------------------------------------------------
# Cluster Node Queries - Organizational containers for field isolation
# -----------------------------------------------------------------------------

CLUSTER_NODE_SAVE = """
        WHERE $created_at <= $last_updated
        MERGE (c:Cluster {uuid: $uuid})
        SET c = {uuid: $uuid, name: $name, organization: $organization, macro_name: $macro_name,
                validated_at: $validated_at, invalidated_at: $invalidated_at, 
                last_updated: $last_updated, created_at: $created_at}
        RETURN c.uuid AS uuid"""

CLUSTER_NODE_SAVE_BULK = """
    UNWIND $clusters AS cluster
    WHERE cluster.created_at <= cluster.last_updated
    MERGE (c:Cluster {uuid: cluster.uuid})
    SET c = {uuid: cluster.uuid, name: cluster.name, organization: cluster.organization, 
            macro_name: cluster.macro_name, validated_at: cluster.validated_at, 
            invalidated_at: cluster.invalidated_at, last_updated: cluster.last_updated, 
            created_at: cluster.created_at}
    RETURN c.uuid AS uuid"""

CLUSTER_NODE_GET_BY_UUID = """
    MATCH (c:Cluster {uuid: $uuid})
    RETURN c.uuid as uuid, c.name as name, c.organization as organization,
           c.macro_name as macro_name, c.validated_at as validated_at, 
           c.invalidated_at as invalidated_at, c.last_updated as last_updated,
           c.created_at as created_at"""

CLUSTER_NODE_GET_BY_UUIDS = """
    MATCH (c:Cluster)
    WHERE c.uuid IN $uuids
    RETURN c.uuid as uuid, c.name as name, c.organization as organization,
           c.macro_name as macro_name, c.validated_at as validated_at, 
           c.invalidated_at as invalidated_at, c.last_updated as last_updated,
           c.created_at as created_at"""

CLUSTER_NODE_GET_BY_ORGANIZATION = """
    MATCH (c:Cluster {organization: $organization})
    RETURN c.uuid as uuid, c.name as name, c.organization as organization,
           c.macro_name as macro_name, c.validated_at as validated_at, 
           c.invalidated_at as invalidated_at, c.last_updated as last_updated,
           c.created_at as created_at
    ORDER BY c.created_at DESC"""

CLUSTER_NODE_GET_FIELD_COUNT = """
    MATCH (c:Cluster {uuid: $uuid})<-[:BELONGS_TO]-(f:Field)
    RETURN count(f) as field_count"""

CLUSTER_NODE_UPDATE = """
    MATCH (c:Cluster {uuid: $uuid})
    WHERE c.created_at <= $last_updated
    SET c.name = $name, c.organization = $organization, c.macro_name = $macro_name,
        c.validated_at = $validated_at, c.last_updated = $last_updated
    RETURN c.uuid AS uuid"""

CLUSTER_NODE_DELETE = """
    MATCH (c:Cluster {uuid: $uuid})
    OPTIONAL MATCH (c)<-[:BELONGS_TO]-(f:Field)
    DETACH DELETE c, f
    RETURN count(c) + count(f) as deleted_count"""

# -----------------------------------------------------------------------------
# Security Graph Analysis Queries - Node-focused analytics
# -----------------------------------------------------------------------------

CLUSTER_FIELD_SUMMARY = """
    MATCH (c:Cluster {uuid: $cluster_id})<-[:BELONGS_TO]-(f:Field)
    RETURN c.name as cluster_name, c.organization as organization,
           count(f) as total_fields,
           collect(f.name)[0..10] as sample_fields,
           sum(f.count) as total_field_events,
           avg(f.count) as avg_field_events"""

ORGANIZATION_CLUSTER_OVERVIEW = """
    MATCH (c:Cluster {organization: $organization})
    OPTIONAL MATCH (c)<-[:BELONGS_TO]-(f:Field)
    RETURN c.uuid as cluster_id, c.name as cluster_name, c.macro_name as macro_name,
           count(f) as field_count,
    ORDER BY field_count DESC"""

FIELD_CORRELATION_DISCOVERY = """
    MATCH (f1:Field)-[:BELONGS_TO]->(c:Cluster {uuid: $cluster_id})
    MATCH (f2:Field)-[:BELONGS_TO]->(c)
    WHERE f1.uuid <> f2.uuid
    AND NOT (f1)-[:FIELD_RELATES_TO]-(f2)
    AND NOT (f2)-[:FIELD_RELATES_TO]-(f1)
    WITH f1, f2, 
         gds.similarity.cosine(f1.embedding, f2.embedding) as similarity
    WHERE similarity > $similarity_threshold
    RETURN f1.name as field1, f2.name as field2, 
           similarity,
           f1.description as field1_desc, f2.description as field2_desc
    ORDER BY similarity DESC
    LIMIT $limit"""

# -----------------------------------------------------------------------------
# Maintenance and Cleanup Queries - Node-focused operations
# -----------------------------------------------------------------------------

DELETE_ORPHANED_FIELDS = """
    MATCH (f:Field)
    WHERE NOT (f)-[:BELONGS_TO]->(:Cluster)
    DELETE f
    RETURN count(f) as deleted_count"""

VALIDATE_CLUSTER_ISOLATION = """
    MATCH (c1:Cluster)--(c2:Cluster)
    WHERE c1.uuid <> c2.uuid
    RETURN c1.name as cluster1, c2.name as cluster2, 
           type(r) as unexpected_relationship"""

# -----------------------------------------------------------------------------
# Search and Discovery Queries - Node-focused search
# -----------------------------------------------------------------------------

FIELD_FULLTEXT_SEARCH = """
    CALL db.index.fulltext.queryNodes("field_fulltext", $query) YIELD node, score
    WHERE ($cluster_id IS NULL OR node.primary_cluster_id = $cluster_id)
    RETURN node.uuid as uuid, node.name as name, node.description as description,
           score
    ORDER BY score DESC
    LIMIT $limit"""

FIELD_VECTOR_SEARCH = """
    MATCH (f:Field)
    WHERE ($cluster_id IS NULL OR f.primary_cluster_id = $cluster_id)
    AND f.embedding IS NOT NULL
    WITH f, gds.similarity.cosine(f.embedding, $query_embedding) as similarity
    WHERE similarity > $similarity_threshold
    RETURN f.uuid as uuid, f.name as name, f.description as description,
           similarity as score
    ORDER BY similarity DESC
    LIMIT $limit"""

FIELD_HYBRID_SEARCH = """
    CALL {
        // Fulltext search branch
        CALL db.index.fulltext.queryNodes("field_fulltext", $query) YIELD node as f, score
        WHERE ($cluster_id IS NULL OR f.primary_cluster_id = $cluster_id)
        RETURN f, score * $fulltext_weight as weighted_score, 'fulltext' as source
        
        UNION ALL
        
        // Vector search branch  
        MATCH (f:Field)
        WHERE ($cluster_id IS NULL OR f.primary_cluster_id = $cluster_id)
        AND f.embedding IS NOT NULL
        WITH f, gds.similarity.cosine(f.embedding, $query_embedding) as similarity
        WHERE similarity > $similarity_threshold
        RETURN f, similarity * $vector_weight as weighted_score, 'vector' as source
    }
    WITH f, max(weighted_score) as final_score, collect(DISTINCT source) as sources
    RETURN f.uuid as uuid, f.name as name, f.description as description,
           final_score as score, sources
    ORDER BY final_score DESC
    LIMIT $limit"""

# -----------------------------------------------------------------------------
# Migration Helper Queries
# -----------------------------------------------------------------------------

MIGRATE_ENTITY_TO_FIELD = """
    MATCH (e:Entity)
    WHERE e.group_id IS NOT NULL
    CREATE (f:Field {
        uuid: e.uuid,
        name: e.name,
        description: COALESCE(e.summary, 'Migrated from Entity node'),
        data_type: 'string',
        count: 1,
        distinct_count: 1,
        primary_cluster_id: e.group_id,
        embedding: e.name_embedding,
        created_at: e.created_at,
        validated_at: e.created_at,
        last_updated: e.created_at
    })
    RETURN count(f) as migrated_fields"""

# -----------------------------------------------------------------------------
# Performance and Index Queries for Nodes
# -----------------------------------------------------------------------------

CREATE_FIELD_INDEXES = """
    CREATE INDEX field_uuid_index IF NOT EXISTS FOR (f:Field) ON (f.uuid);
    CREATE INDEX field_name_index IF NOT EXISTS FOR (f:Field) ON (f.name);
    CREATE INDEX field_cluster_index IF NOT EXISTS FOR (f:Field) ON (f.primary_cluster_id);
    CREATE TEXT INDEX field_description_index IF NOT EXISTS FOR (f:Field) ON (f.description);"""

CREATE_CLUSTER_INDEXES = """
    CREATE INDEX cluster_uuid_index IF NOT EXISTS FOR (c:Cluster) ON (c.uuid);
    CREATE INDEX cluster_org_index IF NOT EXISTS FOR (c:Cluster) ON (c.organization);
    CREATE INDEX cluster_macro_index IF NOT EXISTS FOR (c:Cluster) ON (c.macro_name);"""