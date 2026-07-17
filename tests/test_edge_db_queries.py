"""Unit tests for the entity edge return-query builder.

Locks in the direction-correctness fix in `get_entity_edge_return_query`:
NEO4J/FALKORDB/NEPTUNE must read source/target from the relationship itself
(``startNode(e)``/``endNode(e)``) so the returned direction reflects the stored
relationship regardless of whether the surrounding MATCH is directed or undirected.
KUZU keeps ``n.uuid``/``m.uuid`` because its `e` is an intermediate node, not a
relationship.
"""

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query


def test_neo4j_uses_start_end_node():
    query = get_entity_edge_return_query(GraphProvider.NEO4J)
    assert 'startNode(e).uuid AS source_node_uuid' in query
    assert 'endNode(e).uuid AS target_node_uuid' in query
    assert 'n.uuid AS source_node_uuid' not in query
    assert 'm.uuid AS target_node_uuid' not in query


def test_falkordb_uses_start_end_node():
    query = get_entity_edge_return_query(GraphProvider.FALKORDB)
    assert 'startNode(e).uuid AS source_node_uuid' in query
    assert 'endNode(e).uuid AS target_node_uuid' in query


def test_neptune_uses_start_end_node_with_split_episodes():
    query = get_entity_edge_return_query(GraphProvider.NEPTUNE)
    assert 'startNode(e).uuid AS source_node_uuid' in query
    assert 'endNode(e).uuid AS target_node_uuid' in query
    assert "split(e.episodes, ',') AS episodes" in query


def test_kuzu_keeps_match_variables():
    # KUZU models relationships as an intermediate RelatesToNode_ node, so `e` is a node
    # and startNode/endNode are not applicable. Its MATCH is always directed, so n/m
    # already reflect the true source/target.
    query = get_entity_edge_return_query(GraphProvider.KUZU)
    assert 'n.uuid AS source_node_uuid' in query
    assert 'm.uuid AS target_node_uuid' in query
    assert 'startNode' not in query
    assert 'endNode' not in query
    assert 'e.attributes AS attributes' in query


def test_default_branch_returns_neo4j_or_falkordb_attributes():
    # Both NEO4J and FALKORDB use `properties(e)`, unlike KUZU's `e.attributes`.
    for provider in (GraphProvider.NEO4J, GraphProvider.FALKORDB):
        query = get_entity_edge_return_query(provider)
        assert 'properties(e) AS attributes' in query
