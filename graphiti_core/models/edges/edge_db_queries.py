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
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""


def get_episodic_edge_save_bulk_query(provider: GraphProvider) -> str:
    return """
        UNWIND $episodic_edges AS edge
        MATCH (episode:Episodic {uuid: edge.source_node_uuid})
        MATCH (node:Entity {uuid: edge.target_node_uuid})
        MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
        SET
            e.group_id = edge.group_id,
            e.created_at = edge.created_at
        RETURN e.uuid AS uuid
    """


EPISODIC_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


def get_entity_edge_save_query(provider: GraphProvider, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = $edge_data
                SET e.fact_embedding = vecf32($edge_data.fact_embedding)
                RETURN e.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = removeKeyFromMap(removeKeyFromMap($edge_data, "fact_embedding"), "episodes")
                SET e.fact_embedding = join([x IN coalesce($edge_data.fact_embedding, []) | toString(x) ], ",")
                SET e.episodes = join($edge_data.episodes, ",")
                RETURN $edge_data.uuid AS uuid
            """
        case GraphProvider.RELATE:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = $edge_data
                RETURN e.uuid AS uuid
            """
        case _:  # Neo4j
            save_embedding_query = (
                """WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)"""
                if not has_aoss
                else ''
            )
            return (
                (
                    """
                        MATCH (source:Entity {uuid: $edge_data.source_uuid})
                        MATCH (target:Entity {uuid: $edge_data.target_uuid})
                        MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                        SET e = $edge_data
                        """
                    + save_embedding_query
                )
                + """
                RETURN e.uuid AS uuid
                """
            )


def get_entity_edge_save_bulk_query(provider: GraphProvider, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = edge
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = removeKeyFromMap(removeKeyFromMap(edge, "fact_embedding"), "episodes")
                SET r.fact_embedding = join([x IN coalesce(edge.fact_embedding, []) | toString(x) ], ",")
                SET r.episodes = join(edge.episodes, ",")
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.RELATE:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
                SET e = edge
                RETURN edge.uuid AS uuid
            """
        case _:
            save_embedding_query = (
                'WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)'
                if not has_aoss
                else ''
            )
            return (
                """
                    UNWIND $entity_edges AS edge
                    MATCH (source:Entity {uuid: edge.source_node_uuid})
                    MATCH (target:Entity {uuid: edge.target_node_uuid})
                    MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
                    SET e = edge
                    """
                + save_embedding_query
                + """
                RETURN edge.uuid AS uuid
            """
            )


def get_entity_edge_update_bulk_query(provider: GraphProvider, has_aoss: bool = False) -> str:
    """Return the bulk save query of the provider, with `MATCH` in the place of `MERGE`.

    The query changes an edge that exists and creates nothing. Concurrent ingestion
    uses it to update a stored edge, so that an edge that a different writer deleted
    is not created again.
    """
    query = get_entity_edge_save_bulk_query(provider, has_aoss)
    if query.count('MERGE (source)-[') != 1:
        raise ValueError(f'no update query for the entity edge save query of {provider}')
    return query.replace('MERGE (source)-[', 'MATCH (source)-[')


def get_entity_edge_return_query(provider: GraphProvider) -> str:
    # `fact_embedding` is not returned by default and must be manually loaded using `load_fact_embedding()`.
    # Direction semantics: NEO4J/FALKORDB/NEPTUNE read source/target from the relationship itself
    # via Cypher's built-in `startNode(e)`/`endNode(e)`, so the returned direction matches the
    # stored relationship regardless of whether the surrounding MATCH is directed or undirected.
    if provider == GraphProvider.NEPTUNE:
        return """
        e.uuid AS uuid,
        startNode(e).uuid AS source_node_uuid,
        endNode(e).uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.name AS name,
        e.fact AS fact,
        split(e.episodes, ',') AS episodes,
        e.created_at AS created_at,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        e.reference_time AS reference_time,
        properties(e) AS attributes
    """

    return """
        e.uuid AS uuid,
        startNode(e).uuid AS source_node_uuid,
        endNode(e).uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        e.reference_time AS reference_time,
        properties(e) AS attributes
    """


def get_community_edge_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
                RETURN e.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                WHERE node:Entity OR node:Community
                MERGE (community)-[r:HAS_MEMBER {uuid: $uuid}]->(node)
                SET r.uuid= $uuid
                SET r.group_id= $group_id
                SET r.created_at= $created_at
                RETURN r.uuid AS uuid
            """
        case _:  # Neo4j
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


HAS_EPISODE_EDGE_SAVE = """
    MATCH (saga:Saga {uuid: $saga_uuid, group_id: $group_id})
    MATCH (episode:Episodic {uuid: $episode_uuid, group_id: $group_id})
    MERGE (saga)-[e:HAS_EPISODE {uuid: $uuid}]->(episode)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""

HAS_EPISODE_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


NEXT_EPISODE_EDGE_SAVE = """
    MATCH (source_episode:Episodic {uuid: $source_episode_uuid, group_id: $group_id})
    MATCH (target_episode:Episodic {uuid: $target_episode_uuid, group_id: $group_id})
    MERGE (source_episode)-[e:NEXT_EPISODE {uuid: $uuid}]->(target_episode)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""

NEXT_EPISODE_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""
