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

from graphiti_core.driver.driver import GraphProvider

EPISODIC_EDGE_SAVE = """
    MATCH (episode:Episodic {uuid: $episode_uuid})
    MATCH (node:Entity {uuid: $entity_uuid})
    MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
    SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
    RETURN e.uuid AS uuid
"""

EPISODIC_EDGE_SAVE_BULK = """
    UNWIND $episodic_edges AS edge
    MATCH (episode:Episodic {uuid: edge.source_node_uuid})
    MATCH (node:Entity {uuid: edge.target_node_uuid})
    MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
    SET e = {uuid: edge.uuid, group_id: edge.group_id, created_at: edge.created_at}
    RETURN e.uuid AS uuid
"""

EPISODIC_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


def get_entity_edge_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        return """
            MATCH (source:Entity {uuid: $edge_data.source_uuid})
            MATCH (target:Entity {uuid: $edge_data.target_uuid})
            MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
            SET e = $edge_data
            RETURN e.uuid AS uuid
        """

    return """
        MATCH (source:Entity {uuid: $edge_data.source_uuid})
        MATCH (target:Entity {uuid: $edge_data.target_uuid})
        MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
        SET e = $edge_data
        WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)
        RETURN e.uuid AS uuid
    """


def get_entity_edge_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        return """
            UNWIND $entity_edges AS edge
            MATCH (source:Entity {uuid: edge.source_node_uuid})
            MATCH (target:Entity {uuid: edge.target_node_uuid})
            MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
            SET r = {uuid: edge.uuid, name: edge.name, group_id: edge.group_id, fact: edge.fact, episodes: edge.episodes,
            created_at: edge.created_at, expired_at: edge.expired_at, valid_at: edge.valid_at, invalid_at: edge.invalid_at, fact_embedding: vecf32(edge.fact_embedding)}
            WITH r, edge
            RETURN edge.uuid AS uuid
        """

    return """
        UNWIND $entity_edges AS edge
        MATCH (source:Entity {uuid: edge.source_node_uuid})
        MATCH (target:Entity {uuid: edge.target_node_uuid})
        MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
        SET e = edge
        WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)
        RETURN edge.uuid AS uuid
    """


ENTITY_EDGE_RETURN = """
    e.uuid AS uuid,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.group_id AS group_id,
    e.name AS name,
    e.fact AS fact,
    e.episodes AS episodes,
    e.created_at AS created_at,
    e.expired_at AS expired_at,
    e.valid_at AS valid_at,
    e.invalid_at AS invalid_at,
    properties(e) AS attributes
"""


def get_community_edge_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        return """
            MATCH (community:Community {uuid: $community_uuid})
            MATCH (node {uuid: $entity_uuid})
            MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
            SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
            RETURN e.uuid AS uuid
        """

    return """
        MATCH (community:Community {uuid: $community_uuid})
        MATCH (node:Entity | Community {uuid: $entity_uuid})
        MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
        SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN e.uuid AS uuid
    """


COMMUNITY_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""
