"""
Database query utilities for different graph database backends.

This module provides database-agnostic query generation for Neo4j, FalkorDB, Kuzu, and Neptune,
supporting index creation, fulltext search, bulk operations, and Gremlin queries.
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


def get_range_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
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

    if provider == GraphProvider.KUZU:
        return []

    return [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX community_group_id IF NOT EXISTS FOR (n:Community) ON (n.group_id)',
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


def get_fulltext_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        from typing import cast

        from graphiti_core.driver.falkordb_driver import STOPWORDS

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

    return f'CALL db.index.fulltext.queryRelationships("{name}", $query, {{limit: $limit}})'


# Gremlin Query Generation Functions


def gremlin_match_node_by_property(
    label: str, property_name: str, property_value_param: str
) -> str:
    """
    Generate a Gremlin query to match a node by label and property.

    Args:
        label: Node label (e.g., 'Entity', 'Episodic')
        property_name: Property name to match on
        property_value_param: Parameter name for the property value

    Returns:
        Gremlin traversal string
    """
    return f"g.V().hasLabel('{label}').has('{property_name}', {property_value_param})"


def gremlin_match_nodes_by_uuids(label: str, uuids_param: str = 'uuids') -> str:
    """
    Generate a Gremlin query to match multiple nodes by UUIDs.

    Args:
        label: Node label (e.g., 'Entity', 'Episodic')
        uuids_param: Parameter name containing list of UUIDs

    Returns:
        Gremlin traversal string
    """
    return f"g.V().hasLabel('{label}').has('uuid', within({uuids_param}))"


def gremlin_match_edge_by_property(
    edge_label: str, property_name: str, property_value_param: str
) -> str:
    """
    Generate a Gremlin query to match an edge by label and property.

    Args:
        edge_label: Edge label (e.g., 'RELATES_TO', 'MENTIONS')
        property_name: Property name to match on
        property_value_param: Parameter name for the property value

    Returns:
        Gremlin traversal string
    """
    return f"g.E().hasLabel('{edge_label}').has('{property_name}', {property_value_param})"


def gremlin_get_outgoing_edges(
    source_label: str,
    edge_label: str,
    target_label: str,
    source_uuid_param: str = 'source_uuid',
) -> str:
    """
    Generate a Gremlin query to get outgoing edges from a node.

    Args:
        source_label: Source node label
        edge_label: Edge label
        target_label: Target node label
        source_uuid_param: Parameter name for source UUID

    Returns:
        Gremlin traversal string
    """
    return (
        f"g.V().hasLabel('{source_label}').has('uuid', {source_uuid_param})"
        f".outE('{edge_label}').as('e')"
        f".inV().hasLabel('{target_label}').as('target')"
        f".select('e', 'target')"
    )


def gremlin_bfs_traversal(
    start_label: str,
    edge_labels: list[str],
    max_depth: int,
    start_uuids_param: str = 'start_uuids',
) -> str:
    """
    Generate a Gremlin query for breadth-first search traversal.

    Args:
        start_label: Starting node label
        edge_labels: List of edge labels to traverse
        max_depth: Maximum traversal depth
        start_uuids_param: Parameter name for starting UUIDs

    Returns:
        Gremlin traversal string
    """
    edge_labels_str = "', '".join(edge_labels)
    return (
        f"g.V().hasLabel('{start_label}').has('uuid', within({start_uuids_param}))"
        f".repeat(bothE('{edge_labels_str}').otherV()).times({max_depth})"
        f'.dedup()'
    )


def gremlin_delete_all_nodes() -> str:
    """
    Generate a Gremlin query to delete all nodes and edges.

    Returns:
        Gremlin traversal string
    """
    return 'g.V().drop()'


def gremlin_delete_nodes_by_group_id(label: str, group_ids_param: str = 'group_ids') -> str:
    """
    Generate a Gremlin query to delete nodes by group_id.

    Args:
        label: Node label
        group_ids_param: Parameter name for group IDs list

    Returns:
        Gremlin traversal string
    """
    return f"g.V().hasLabel('{label}').has('group_id', within({group_ids_param})).drop()"


def gremlin_cosine_similarity_filter(
    embedding_property: str, search_vector_param: str, min_score: float
) -> str:
    """
    Generate a Gremlin query fragment for cosine similarity filtering.
    Note: This is a placeholder as Neptune Gremlin doesn't have built-in vector similarity.
    Vector similarity should be handled via OpenSearch integration.

    Args:
        embedding_property: Property name containing the embedding
        search_vector_param: Parameter name for search vector
        min_score: Minimum similarity score

    Returns:
        Gremlin query fragment (warning comment)
    """
    # Neptune Gremlin doesn't support vector similarity natively
    # This should be handled via OpenSearch AOSS integration
    return f"// Vector similarity for '{embedding_property}' must be handled via OpenSearch"


def gremlin_retrieve_episodes(
    reference_time_param: str = 'reference_time',
    group_ids_param: str = 'group_ids',
    limit_param: str = 'num_episodes',
    source_param: str | None = None,
) -> str:
    """
    Generate a Gremlin query to retrieve episodes filtered by time and optionally by group_id and source.

    Args:
        reference_time_param: Parameter name for reference timestamp
        group_ids_param: Parameter name for group IDs list
        limit_param: Parameter name for result limit
        source_param: Optional parameter name for source filter

    Returns:
        Gremlin traversal string
    """
    query = f"g.V().hasLabel('Episodic').has('valid_at', lte({reference_time_param}))"

    # Add group_id filter if specified
    query += f".has('group_id', within({group_ids_param}))"

    # Add source filter if specified
    if source_param:
        query += f".has('source', {source_param})"

    # Order by valid_at descending and limit
    query += f".order().by('valid_at', desc).limit({limit_param}).valueMap(true)"

    return query
