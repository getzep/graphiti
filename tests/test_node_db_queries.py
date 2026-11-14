"""Unit tests for the entity node return-query builder."""

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query


def test_neo4j_returns_properties_as_attributes():
    query = get_entity_node_return_query(GraphProvider.NEO4J)
    assert 'properties(n) AS attributes' in query


def test_falkordb_attributes_leave_out_the_name_embedding():
    # The record parser discards `name_embedding`, so FalkorDB does not send it.
    query = get_entity_node_return_query(GraphProvider.FALKORDB)
    assert 'n {.*, name_embedding: NULL} AS attributes' in query
    assert 'properties(n)' not in query
