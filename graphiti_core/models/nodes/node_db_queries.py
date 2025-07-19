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

EPISODIC_NODE_SAVE = """
    MERGE (n:Episodic {uuid: $uuid})
    SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content, 
    entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
    RETURN n.uuid AS uuid
"""

EPISODIC_NODE_SAVE_BULK = """
    UNWIND $episodes AS episode
    MERGE (n:Episodic {uuid: episode.uuid})
    SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, 
        source: episode.source, content: episode.content, 
    entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
    RETURN n.uuid AS uuid
"""

EPISODIC_NODE_RETURN = """
    e.content AS content,
    e.created_at AS created_at,
    e.valid_at AS valid_at,
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.source_description AS source_description,
    e.source AS source,
    e.entity_edges AS entity_edges
"""

ENTITY_NODE_SAVE = """
    MERGE (n:Entity {uuid: $entity_data.uuid})
    SET n:$($labels)
    SET n = $entity_data
    WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)
    RETURN n.uuid AS uuid
"""

ENTITY_NODE_SAVE_BULK = """
    UNWIND $nodes AS node
    MERGE (n:Entity {uuid: node.uuid})
    SET n:$(node.labels)
    SET n = node
    WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
    RETURN n.uuid AS uuid
"""

ENTITY_NODE_RETURN = """
    n.uuid As uuid,
    n.name AS name,
    n.group_id AS group_id,
    n.created_at AS created_at, 
    n.summary AS summary,
    labels(n) AS labels,
    properties(n) AS attributes
"""

COMMUNITY_NODE_SAVE = """
    MERGE (n:Community {uuid: $uuid})
    SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
    WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $name_embedding)
    RETURN n.uuid AS uuid
"""

COMMUNITY_NODE_RETURN = """
    n.uuid As uuid,
    n.name AS name,
    n.name_embedding AS name_embedding,
    n.group_id AS group_id,
    n.summary AS summary,
    n.created_at AS created_at
"""
