"""
Database query utilities for different graph database backends.

This module provides database-agnostic query generation for Neo4j and FalkorDB,
supporting index creation, fulltext search, and bulk operations.
"""

from typing import Any

from typing_extensions import LiteralString

from graphiti_core.models.edges.edge_db_queries import (
    ENTITY_EDGE_SAVE_BULK,
)
from graphiti_core.models.nodes.node_db_queries import (
    ENTITY_NODE_SAVE_BULK,
)

# Mapping from Neo4j fulltext index names to FalkorDB node labels
NEO4J_TO_FALKORDB_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}


def get_range_indices(db_type: str = 'neo4j') -> list[LiteralString]:
    if db_type == 'falkordb':
        return [
            # Entity node
            'CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)',
            # Episodic node
            'CREATE INDEX FOR (n:Episodic) ON (n.uuid, n.group_id, n.created_at, n.valid_at)',
            # Community node
            'CREATE INDEX FOR (n:Community) ON (n.uuid)',
            # RELATES_TO edge
            'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)',
            # MENTIONS edge
            'CREATE INDEX FOR ()-[e:MENTIONS]-() ON (e.uuid, e.group_id)',
            # HAS_MEMBER edge
            'CREATE INDEX FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        ]
    else:
        return [
            'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
            'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
            'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
            'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
            'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
            'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
            'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
            'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
            'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
            'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
            'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
            'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
            'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
            'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
            'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
            'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
            'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
            'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
            'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
        ]


def get_fulltext_indices(db_type: str = 'neo4j') -> list[LiteralString]:
    if db_type == 'falkordb':
        return [
            """CREATE FULLTEXT INDEX FOR (e:Episodic) ON (e.content, e.source, e.source_description, e.group_id)""",
            """CREATE FULLTEXT INDEX FOR (n:Entity) ON (n.name, n.summary, n.group_id)""",
            """CREATE FULLTEXT INDEX FOR (n:Community) ON (n.name, n.group_id)""",
            """CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)""",
        ]
    else:
        return [
            """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS 
            FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
            """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS 
            FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
            """CREATE FULLTEXT INDEX community_name IF NOT EXISTS 
            FOR (n:Community) ON EACH [n.name, n.group_id]""",
            """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS 
            FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
        ]


def get_nodes_query(db_type: str = 'neo4j', name: str = '', query: str | None = None) -> str:
    if db_type == 'falkordb':
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryNodes('{label}', {query})"
    else:
        return f'CALL db.index.fulltext.queryNodes("{name}", {query}, {{limit: $limit}})'


def get_vector_cosine_func_query(vec1, vec2, db_type: str = 'neo4j') -> str:
    if db_type == 'falkordb':
        # FalkorDB uses a different syntax for regular cosine similarity and Neo4j uses normalized cosine similarity
        return f'(2 - vec.cosineDistance({vec1}, vecf32({vec2})))/2'
    else:
        return f'vector.similarity.cosine({vec1}, {vec2})'


def get_relationships_query(name: str, db_type: str = 'neo4j') -> str:
    if db_type == 'falkordb':
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryRelationships('{label}', $query)"
    else:
        return f'CALL db.index.fulltext.queryRelationships("{name}", $query, {{limit: $limit}})'


def get_entity_node_save_bulk_query(nodes, db_type: str = 'neo4j') -> str | Any:
    if db_type == 'falkordb':
        queries = []
        for node in nodes:
            for label in node['labels']:
                queries.append(
                    (
                        f"""
                    UNWIND $nodes AS node
                    MERGE (n:Entity {{uuid: node.uuid}})
                    SET n:{label}
                    SET n = node
                    WITH n, node
                    SET n.name_embedding = vecf32(node.name_embedding)
                    RETURN n.uuid AS uuid
                """,
                        {'nodes': [node]},
                    )
                )
        return queries
    else:
        return ENTITY_NODE_SAVE_BULK


def get_entity_edge_save_bulk_query(db_type: str = 'neo4j') -> str:
    if db_type == 'falkordb':
        return """
        UNWIND $entity_edges AS edge
        MATCH (source:Entity {uuid: edge.source_node_uuid}) 
        MATCH (target:Entity {uuid: edge.target_node_uuid}) 
        MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
        SET r = {uuid: edge.uuid, name: edge.name, group_id: edge.group_id, fact: edge.fact, episodes: edge.episodes, 
        created_at: edge.created_at, expired_at: edge.expired_at, valid_at: edge.valid_at, invalid_at: edge.invalid_at, fact_embedding: vecf32(edge.fact_embedding)}
        WITH r, edge
        RETURN edge.uuid AS uuid"""
    else:
        return ENTITY_EDGE_SAVE_BULK
