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

EPISODIC_EDGE_SAVE = """
        MATCH (episode:Episodic {uuid: $episode_uuid}) 
        MATCH (node:Entity {uuid: $entity_uuid}) 
        MERGE (episode)-[r:MENTIONS {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN r.uuid AS uuid"""

EPISODIC_EDGE_SAVE_BULK = """
    UNWIND $episodic_edges AS edge
    MATCH (episode:Episodic {uuid: edge.source_node_uuid}) 
    MATCH (node:Entity {uuid: edge.target_node_uuid}) 
    MERGE (episode)-[r:MENTIONS {uuid: edge.uuid}]->(node)
    SET r = {uuid: edge.uuid, group_id: edge.group_id, created_at: edge.created_at}
    RETURN r.uuid AS uuid
"""

ENTITY_EDGE_SAVE = """
        MATCH (source:Entity {uuid: $source_uuid}) 
        MATCH (target:Entity {uuid: $target_uuid}) 
        MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
        SET r = {uuid: $uuid, name: $name, group_id: $group_id, fact: $fact, episodes: $episodes, 
        created_at: $created_at, expired_at: $expired_at, valid_at: $valid_at, invalid_at: $invalid_at}
        WITH r CALL db.create.setRelationshipVectorProperty(r, "fact_embedding", $fact_embedding)
        RETURN r.uuid AS uuid"""

ENTITY_EDGE_SAVE_BULK = """
    UNWIND $entity_edges AS edge
    MATCH (source:Entity {uuid: edge.source_node_uuid}) 
    MATCH (target:Entity {uuid: edge.target_node_uuid}) 
    MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
    SET r = {uuid: edge.uuid, name: edge.name, group_id: edge.group_id, fact: edge.fact, episodes: edge.episodes, 
    created_at: edge.created_at, expired_at: edge.expired_at, valid_at: edge.valid_at, invalid_at: edge.invalid_at}
    WITH r, edge CALL db.create.setRelationshipVectorProperty(r, "fact_embedding", edge.fact_embedding)
    RETURN r.uuid AS uuid
"""

COMMUNITY_EDGE_SAVE = """
        MATCH (community:Community {uuid: $community_uuid}) 
        MATCH (node:Entity | Community {uuid: $entity_uuid}) 
        MERGE (community)-[r:HAS_MEMBER {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN r.uuid AS uuid"""
