"""
Database query utilities for different graph database backends.

This module provides database-agnostic query generation for Neo4j and FalkorDB,
supporting index creation, fulltext search, and bulk operations.
"""

from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphProvider

# Mapping from Neo4j fulltext index names to FalkorDB node labels
NEO4J_TO_FALKORDB_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}
# Mapping from fulltext index names to Kuzu node labels
INDEX_TO_LABEL_KUZU_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RelatesToNode_',
}

# ArcadeDB addresses a full-text index by "Type[prop1,prop2]" rather than by a logical
# name, so the index name is derived from the type and the exact property list it was
# created on. These MUST stay in step with ARCADEDB_FULLTEXT_DDL below.
ARCADEDB_FULLTEXT_INDEX_PROPERTIES: dict[str, tuple[str, tuple[str, ...]]] = {
    'node_name_and_summary': ('Entity', ('name', 'summary', 'group_id')),
    'community_name': ('Community', ('name', 'group_id')),
    'episode_content': ('Episodic', ('content', 'source', 'source_description', 'group_id')),
    'edge_name_and_fact': ('RELATES_TO', ('name', 'fact', 'group_id')),
}


def get_arcadedb_fulltext_index_name(name: str) -> str:
    """Return the ArcadeDB "Type[prop,...]" identifier for a graphiti index name."""
    type_name, properties = ARCADEDB_FULLTEXT_INDEX_PROPERTIES[name]
    return f'{type_name}[{",".join(properties)}]'


def get_range_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.ARCADEDB:
        # These are emitted over the Bolt/Cypher channel, so they must be Cypher --
        # ArcadeDB's own SQL DDL (CREATE VERTEX TYPE ..., CREATE INDEX ... NOTUNIQUE)
        # is rejected there with "mismatched input 'TYPE'/'ON'". Vertex and edge types
        # do not need to be declared up front: ArcadeDB creates them implicitly on the
        # first MERGE/CREATE. Standard, RANGE and TEXT indexes are all creatable via
        # Cypher; only FULLTEXT is rejected (see get_fulltext_indices), so these reuse
        # the same Cypher syntax as the default Neo4j case below.
        return [
            'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
            'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
            'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
            'CREATE INDEX saga_uuid IF NOT EXISTS FOR (n:Saga) ON (n.uuid)',
            'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
            'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
            'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
            'CREATE INDEX has_episode_uuid IF NOT EXISTS FOR ()-[e:HAS_EPISODE]-() ON (e.uuid)',
            'CREATE INDEX next_episode_uuid IF NOT EXISTS FOR ()-[e:NEXT_EPISODE]-() ON (e.uuid)',
            'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
            'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
            'CREATE INDEX community_group_id IF NOT EXISTS FOR (n:Community) ON (n.group_id)',
            'CREATE INDEX saga_group_id IF NOT EXISTS FOR (n:Saga) ON (n.group_id)',
            'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
            'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
            'CREATE INDEX has_episode_group_id IF NOT EXISTS FOR ()-[e:HAS_EPISODE]-() ON (e.group_id)',
            'CREATE INDEX next_episode_group_id IF NOT EXISTS FOR ()-[e:NEXT_EPISODE]-() ON (e.group_id)',
            'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
            'CREATE INDEX saga_name IF NOT EXISTS FOR (n:Saga) ON (n.name)',
            'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
            'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
            'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
            'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
            'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
            'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
            'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
            'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
        ]

    if provider == GraphProvider.FALKORDB:
        return [
            # Entity node
            'CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)',
            # Episodic node
            'CREATE INDEX FOR (n:Episodic) ON (n.uuid, n.group_id, n.created_at, n.valid_at)',
            # Community node
            'CREATE INDEX FOR (n:Community) ON (n.uuid)',
            # Saga node
            'CREATE INDEX FOR (n:Saga) ON (n.uuid, n.group_id, n.name)',
            # RELATES_TO edge
            'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)',
            # MENTIONS edge
            'CREATE INDEX FOR ()-[e:MENTIONS]-() ON (e.uuid, e.group_id)',
            # HAS_MEMBER edge
            'CREATE INDEX FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
            # HAS_EPISODE edge
            'CREATE INDEX FOR ()-[e:HAS_EPISODE]-() ON (e.uuid, e.group_id)',
            # NEXT_EPISODE edge
            'CREATE INDEX FOR ()-[e:NEXT_EPISODE]-() ON (e.uuid, e.group_id)',
        ]

    if provider == GraphProvider.KUZU:
        return []

    return [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX saga_uuid IF NOT EXISTS FOR (n:Saga) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX has_episode_uuid IF NOT EXISTS FOR ()-[e:HAS_EPISODE]-() ON (e.uuid)',
        'CREATE INDEX next_episode_uuid IF NOT EXISTS FOR ()-[e:NEXT_EPISODE]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX community_group_id IF NOT EXISTS FOR (n:Community) ON (n.group_id)',
        'CREATE INDEX saga_group_id IF NOT EXISTS FOR (n:Saga) ON (n.group_id)',
        'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
        'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
        'CREATE INDEX has_episode_group_id IF NOT EXISTS FOR ()-[e:HAS_EPISODE]-() ON (e.group_id)',
        'CREATE INDEX next_episode_group_id IF NOT EXISTS FOR ()-[e:NEXT_EPISODE]-() ON (e.group_id)',
        'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
        'CREATE INDEX saga_name IF NOT EXISTS FOR (n:Saga) ON (n.name)',
        'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
        'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
        'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
        'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
        'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
        'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
        'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
        'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
    ]


def get_fulltext_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.ARCADEDB:
        # NOTE: unlike every other provider, these are ArcadeDB *SQL* statements, not
        # Cypher. The Bolt/Cypher channel rejects full-text DDL outright ("Only
        # standard, RANGE and TEXT index types are supported"), and `CREATE TEXT INDEX`
        # builds a non-BM25 index that db.index.fulltext.queryNodes() will not accept.
        # ArcadeDBDriver therefore runs these over the HTTP/SQL endpoint; see
        # ArcadeDBDriver._create_fulltext_indices().
        #
        # Property lists must match ARCADEDB_FULLTEXT_INDEX_PROPERTIES above: ArcadeDB
        # names the resulting index after the type and properties it covers.
        return [
            'CREATE VERTEX TYPE Entity IF NOT EXISTS',
            'CREATE PROPERTY Entity.name IF NOT EXISTS STRING',
            'CREATE PROPERTY Entity.summary IF NOT EXISTS STRING',
            'CREATE PROPERTY Entity.group_id IF NOT EXISTS STRING',
            'CREATE INDEX IF NOT EXISTS ON Entity (name, summary, group_id) FULL_TEXT ENGINE LUCENE',
            'CREATE VERTEX TYPE Community IF NOT EXISTS',
            'CREATE PROPERTY Community.name IF NOT EXISTS STRING',
            'CREATE PROPERTY Community.group_id IF NOT EXISTS STRING',
            'CREATE INDEX IF NOT EXISTS ON Community (name, group_id) FULL_TEXT ENGINE LUCENE',
            'CREATE VERTEX TYPE Episodic IF NOT EXISTS',
            'CREATE PROPERTY Episodic.content IF NOT EXISTS STRING',
            'CREATE PROPERTY Episodic.source IF NOT EXISTS STRING',
            'CREATE PROPERTY Episodic.source_description IF NOT EXISTS STRING',
            'CREATE PROPERTY Episodic.group_id IF NOT EXISTS STRING',
            'CREATE INDEX IF NOT EXISTS ON Episodic (content, source, source_description, group_id) FULL_TEXT ENGINE LUCENE',
            'CREATE EDGE TYPE RELATES_TO IF NOT EXISTS',
            'CREATE PROPERTY RELATES_TO.name IF NOT EXISTS STRING',
            'CREATE PROPERTY RELATES_TO.fact IF NOT EXISTS STRING',
            'CREATE PROPERTY RELATES_TO.group_id IF NOT EXISTS STRING',
            'CREATE INDEX IF NOT EXISTS ON RELATES_TO (name, fact, group_id) FULL_TEXT ENGINE LUCENE',
        ]

    if provider == GraphProvider.FALKORDB:
        from typing import cast

        from graphiti_core.driver.falkordb import STOPWORDS

        # Convert to string representation for embedding in queries
        stopwords_str = str(STOPWORDS)

        # Use type: ignore to satisfy LiteralString requirement while maintaining single source of truth
        return cast(
            list[LiteralString],
            [
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Episodic',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'content', 'source', 'source_description', 'group_id'
                                                )""",
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Entity',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'name', 'summary', 'group_id'
                                                )""",
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Community',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'name', 'group_id'
                                                )""",
                """CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)""",
            ],
        )

    if provider == GraphProvider.KUZU:
        return [
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]

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


def get_nodes_query(name: str, query: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryNodes('{label}', {query})"

    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', {query}, TOP := $limit)"

    if provider == GraphProvider.ARCADEDB:
        # ArcadeDB's procedure takes exactly 2 arguments; Neo4j's {limit: $limit}
        # config map is a hard error. Callers already append ORDER BY score DESC
        # LIMIT $limit, so dropping it is safe.
        index_name = get_arcadedb_fulltext_index_name(name)
        return f"CALL db.index.fulltext.queryNodes('{index_name}', {query})"

    return f'CALL db.index.fulltext.queryNodes("{name}", {query}, {{limit: $limit}})'


def get_vector_cosine_func_query(vec1, vec2, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        # FalkorDB uses a different syntax for regular cosine similarity and Neo4j uses normalized cosine similarity
        return f'(2 - vec.cosineDistance({vec1}, vecf32({vec2})))/2'

    if provider == GraphProvider.KUZU:
        return f'array_cosine_similarity({vec1}, {vec2})'

    return f'vector.similarity.cosine({vec1}, {vec2})'


def get_relationships_query(name: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryRelationships('{label}', $query)"

    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', cast($query AS STRING), TOP := $limit)"

    if provider == GraphProvider.ARCADEDB:
        # See the note in get_nodes_query: 2 arguments, no config map.
        index_name = get_arcadedb_fulltext_index_name(name)
        return f"CALL db.index.fulltext.queryRelationships('{index_name}', $query)"

    return f'CALL db.index.fulltext.queryRelationships("{name}", $query, {{limit: $limit}})'
