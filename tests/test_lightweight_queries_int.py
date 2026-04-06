"""
Integration tests for lightweight query mode.

Validates that lightweight queries exclude embedding vectors while preserving
all other properties (including custom attributes like `labels`, `migrated_from`).

Tests cover both the new driver operations architecture (record_parsers, ops layer)
and the query generation layer (node_db_queries, edge_db_queries).

Run with:
    DISABLE_NEO4J=1 DISABLE_KUZU=1 FALKORDB_PORT=6380 \
      pytest tests/test_lightweight_queries_int.py -v
"""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.record_parsers import entity_edge_from_record, entity_node_from_record
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode
from tests.helpers_test import embedding_dim, group_id

# ──────────────────────────────────────────────
# Unit Tests: Query string validation
# ──────────────────────────────────────────────


class TestLightweightNodeQuery:
    """Unit tests for get_entity_node_return_query with lightweight=True."""

    def test_lightweight_excludes_properties_call(self):
        query = get_entity_node_return_query(GraphProvider.FALKORDB, lightweight=True)
        assert 'properties(n)' not in query

    def test_lightweight_excludes_embedding_keys(self):
        query = get_entity_node_return_query(GraphProvider.FALKORDB, lightweight=True)
        assert 'name_embedding' in query  # in the exclusion list
        assert 'summary_embedding' in query  # in the exclusion list

    def test_lightweight_includes_explicit_fields(self):
        query = get_entity_node_return_query(GraphProvider.FALKORDB, lightweight=True)
        assert 'n.uuid AS uuid' in query
        assert 'n.name AS name' in query
        assert 'n.group_id AS group_id' in query
        assert 'n.summary AS summary' in query
        assert 'labels(n) AS labels' in query

    def test_non_lightweight_uses_properties(self):
        query = get_entity_node_return_query(GraphProvider.FALKORDB, lightweight=False)
        assert 'properties(n) AS attributes' in query

    def test_kuzu_ignores_lightweight(self):
        query_lw = get_entity_node_return_query(GraphProvider.KUZU, lightweight=True)
        query_normal = get_entity_node_return_query(GraphProvider.KUZU, lightweight=False)
        assert query_lw == query_normal  # KUZU doesn't have the problem


class TestLightweightEdgeQuery:
    """Unit tests for get_entity_edge_return_query with lightweight=True."""

    def test_lightweight_excludes_properties_call(self):
        query = get_entity_edge_return_query(GraphProvider.FALKORDB, lightweight=True)
        assert 'properties(e)' not in query

    def test_lightweight_excludes_fact_embedding(self):
        query = get_entity_edge_return_query(GraphProvider.FALKORDB, lightweight=True)
        assert 'fact_embedding' in query  # in the exclusion list

    def test_lightweight_includes_explicit_fields(self):
        query = get_entity_edge_return_query(GraphProvider.FALKORDB, lightweight=True)
        assert 'e.uuid AS uuid' in query
        assert 'e.fact AS fact' in query
        assert 'e.name AS name' in query

    def test_non_lightweight_uses_properties(self):
        query = get_entity_edge_return_query(GraphProvider.FALKORDB, lightweight=False)
        assert 'properties(e) AS attributes' in query

    def test_neptune_lightweight(self):
        query = get_entity_edge_return_query(GraphProvider.NEPTUNE, lightweight=True)
        assert 'properties(e)' not in query
        assert '[] AS attributes' in query


# ──────────────────────────────────────────────
# Unit Tests: Record parser (new architecture)
# ──────────────────────────────────────────────


class TestNodeRecordParser:
    """Unit tests for entity_node_from_record (driver.record_parsers) with lightweight mode."""

    def test_lightweight_parses_list_of_pairs(self):
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Node',
            'group_id': 'main',
            'created_at': '2026-01-01T00:00:00+00:00',
            'summary': 'A test node',
            'labels': ['Entity', 'Concept'],
            'attributes': [
                ['labels', ['Entity', 'Concept']],
                ['migrated_from', 'old-group'],
            ],
        }
        node = entity_node_from_record(record, lightweight=True)
        assert node.attributes == {
            'labels': ['Entity', 'Concept'],
            'migrated_from': 'old-group',
        }

    def test_lightweight_empty_attributes(self):
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Node',
            'group_id': 'main',
            'created_at': '2026-01-01T00:00:00+00:00',
            'summary': '',
            'labels': ['Entity'],
            'attributes': [],
        }
        node = entity_node_from_record(record, lightweight=True)
        assert node.attributes == {}

    def test_lightweight_no_embedding_in_node(self):
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Node',
            'group_id': 'main',
            'created_at': '2026-01-01T00:00:00+00:00',
            'summary': 'A test',
            'labels': ['Entity'],
            'attributes': [],
        }
        node = entity_node_from_record(record, lightweight=True)
        assert node.name_embedding is None


class TestEdgeRecordParser:
    """Unit tests for entity_edge_from_record (driver.record_parsers) with lightweight mode."""

    def test_lightweight_parses_list_of_pairs(self):
        record = {
            'uuid': 'test-uuid',
            'source_node_uuid': 'src-uuid',
            'target_node_uuid': 'tgt-uuid',
            'group_id': 'main',
            'created_at': '2026-01-01T00:00:00+00:00',
            'name': 'RELATES_TO',
            'fact': 'A relates to B',
            'episodes': ['ep1'],
            'expired_at': None,
            'valid_at': None,
            'invalid_at': None,
            'attributes': [['custom_key', 'custom_val']],
        }
        edge = entity_edge_from_record(record, lightweight=True)
        assert edge.attributes == {'custom_key': 'custom_val'}
        assert edge.fact_embedding is None

    def test_lightweight_empty_attributes(self):
        record = {
            'uuid': 'test-uuid',
            'source_node_uuid': 'src-uuid',
            'target_node_uuid': 'tgt-uuid',
            'group_id': 'main',
            'created_at': '2026-01-01T00:00:00+00:00',
            'name': 'RELATES_TO',
            'fact': 'A relates to B',
            'episodes': [],
            'expired_at': None,
            'valid_at': None,
            'invalid_at': None,
            'attributes': [],
        }
        edge = entity_edge_from_record(record, lightweight=True)
        assert edge.attributes == {}


# ──────────────────────────────────────────────
# Integration Tests: Round-trip via operations layer
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_lightweight_roundtrip(graph_driver):
    """Save a node with embeddings, retrieve lightweight via ops — embeddings should be None."""
    embedding = np.random.uniform(0.0, 0.9, embedding_dim).tolist()
    node = EntityNode(
        uuid=str(uuid4()),
        name='Lightweight Test Node',
        group_id=group_id,
        labels=['Entity', 'Concept'],
        created_at=datetime.now(timezone.utc),
        name_embedding=embedding,
        summary='A node for testing lightweight queries',
        summary_embedding=embedding,
        attributes={'custom_attr': 'test_value'},
    )
    await node.save(graph_driver)

    # Retrieve through new operations layer with lightweight=True
    ops = graph_driver.entity_node_ops
    assert ops is not None, 'Driver does not implement entity_node_ops'
    nodes = await ops.get_by_group_ids(graph_driver, [group_id], lightweight=True)
    assert len(nodes) >= 1
    retrieved = next(n for n in nodes if n.uuid == node.uuid)

    # Core fields preserved
    assert retrieved.name == 'Lightweight Test Node'
    assert retrieved.summary == 'A node for testing lightweight queries'
    assert retrieved.group_id == group_id

    # Embeddings NOT loaded
    assert retrieved.name_embedding is None
    assert retrieved.summary_embedding is None


@pytest.mark.asyncio
async def test_node_lightweight_preserves_custom_attributes(graph_driver):
    """Lightweight mode must preserve custom attributes like labels, migrated_from."""
    embedding = np.random.uniform(0.0, 0.9, embedding_dim).tolist()
    node = EntityNode(
        uuid=str(uuid4()),
        name='Custom Attrs Node',
        group_id=group_id,
        labels=['Entity', 'Concept'],
        created_at=datetime.now(timezone.utc),
        name_embedding=embedding,
        summary='Test custom attrs',
        attributes={'migrated_from': 'boerse-trading', 'custom_flag': True},
    )
    await node.save(graph_driver)

    ops = graph_driver.entity_node_ops
    assert ops is not None
    nodes = await ops.get_by_group_ids(graph_driver, [group_id], lightweight=True)
    retrieved = next(n for n in nodes if n.uuid == node.uuid)

    assert retrieved.attributes.get('migrated_from') == 'boerse-trading'
    assert retrieved.attributes.get('custom_flag') is True


@pytest.mark.asyncio
async def test_node_non_lightweight_still_works(graph_driver):
    """Non-lightweight mode (default) should still work as before."""
    embedding = np.random.uniform(0.0, 0.9, embedding_dim).tolist()
    node = EntityNode(
        uuid=str(uuid4()),
        name='Full Node',
        group_id=group_id,
        labels=['Entity'],
        created_at=datetime.now(timezone.utc),
        name_embedding=embedding,
        summary='Full properties node',
        attributes={'keep_this': 'yes'},
    )
    await node.save(graph_driver)

    ops = graph_driver.entity_node_ops
    assert ops is not None
    nodes = await ops.get_by_group_ids(graph_driver, [group_id], lightweight=False)
    retrieved = next(n for n in nodes if n.uuid == node.uuid)

    assert retrieved.name == 'Full Node'
    assert retrieved.attributes.get('keep_this') == 'yes'


@pytest.mark.asyncio
async def test_edge_lightweight_roundtrip(graph_driver):
    """Save an edge with fact_embedding, retrieve lightweight via ops — embedding should be None."""
    embedding = np.random.uniform(0.0, 0.9, embedding_dim).tolist()

    # Create source and target nodes first
    src = EntityNode(
        uuid=str(uuid4()),
        name='Edge Source',
        group_id=group_id,
        labels=['Entity'],
        created_at=datetime.now(timezone.utc),
        name_embedding=embedding,
        summary='Source node',
    )
    tgt = EntityNode(
        uuid=str(uuid4()),
        name='Edge Target',
        group_id=group_id,
        labels=['Entity'],
        created_at=datetime.now(timezone.utc),
        name_embedding=embedding,
        summary='Target node',
    )
    await src.save(graph_driver)
    await tgt.save(graph_driver)

    edge = EntityEdge(
        uuid=str(uuid4()),
        source_node_uuid=src.uuid,
        target_node_uuid=tgt.uuid,
        name='TEST_RELATION',
        fact='Source relates to target for testing',
        fact_embedding=embedding,
        group_id=group_id,
        episodes=['ep-test'],
        created_at=datetime.now(timezone.utc),
    )
    await edge.save(graph_driver)

    # Retrieve through new operations layer with lightweight=True
    ops = graph_driver.entity_edge_ops
    assert ops is not None, 'Driver does not implement entity_edge_ops'
    edges = await ops.get_by_group_ids(graph_driver, [group_id], lightweight=True)
    assert len(edges) >= 1
    retrieved = next(e for e in edges if e.uuid == edge.uuid)

    # Core fields preserved
    assert retrieved.name == 'TEST_RELATION'
    assert retrieved.fact == 'Source relates to target for testing'
    assert retrieved.source_node_uuid == src.uuid
    assert retrieved.target_node_uuid == tgt.uuid

    # Embedding NOT loaded
    assert retrieved.fact_embedding is None
