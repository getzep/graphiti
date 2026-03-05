"""Unit tests for graphiti_core.vector_store.milvus_utils."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from graphiti_core.vector_store.milvus_utils import (
    COLLECTION_COMMUNITY_NODES,
    COLLECTION_ENTITY_EDGES,
    COLLECTION_ENTITY_NODES,
    COLLECTION_EPISODIC_NODES,
    build_group_ids_filter,
    build_milvus_filter_from_search_filters,
    community_node_to_milvus_dict,
    datetime_to_epoch_ms,
    entity_edge_to_milvus_dict,
    entity_node_to_milvus_dict,
    episodic_node_to_milvus_dict,
    epoch_ms_to_datetime,
    milvus_dict_to_community_node,
    milvus_dict_to_entity_edge,
    milvus_dict_to_entity_node,
    milvus_dict_to_episodic_node,
)

# ---- Collection Constants ----


class TestCollectionConstants:
    def test_collection_names(self):
        assert COLLECTION_ENTITY_NODES == 'entity_nodes'
        assert COLLECTION_ENTITY_EDGES == 'entity_edges'
        assert COLLECTION_EPISODIC_NODES == 'episodic_nodes'
        assert COLLECTION_COMMUNITY_NODES == 'community_nodes'


# ---- Datetime Helpers ----


class TestDatetimeHelpers:
    def test_datetime_to_epoch_ms_with_datetime(self):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = datetime_to_epoch_ms(dt)
        assert result == 1705320000000

    def test_datetime_to_epoch_ms_with_none(self):
        assert datetime_to_epoch_ms(None) == 0

    def test_epoch_ms_to_datetime_with_value(self):
        result = epoch_ms_to_datetime(1705320000000)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_epoch_ms_to_datetime_with_zero(self):
        assert epoch_ms_to_datetime(0) is None

    def test_roundtrip(self):
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        ms = datetime_to_epoch_ms(dt)
        result = epoch_ms_to_datetime(ms)
        assert result is not None
        assert result.replace(microsecond=0) == dt.replace(microsecond=0)


# ---- Filter Builders ----


class TestBuildGroupIdsFilter:
    def test_none(self):
        assert build_group_ids_filter(None) == ''

    def test_empty_list(self):
        assert build_group_ids_filter([]) == ''

    def test_single(self):
        result = build_group_ids_filter(['g1'])
        assert result == 'group_id in ["g1"]'

    def test_multiple(self):
        result = build_group_ids_filter(['g1', 'g2', 'g3'])
        assert result == 'group_id in ["g1", "g2", "g3"]'


class TestBuildMilvusFilterFromSearchFilters:
    def _make_filter(self, **kwargs):
        """Create a mock SearchFilters with given attributes."""
        sf = MagicMock()
        sf.node_labels = kwargs.get('node_labels', [])
        sf.edge_types = kwargs.get('edge_types', [])
        sf.edge_uuids = kwargs.get('edge_uuids', [])
        sf.valid_at = kwargs.get('valid_at', [])
        sf.invalid_at = kwargs.get('invalid_at', [])
        sf.created_at = kwargs.get('created_at', [])
        sf.expired_at = kwargs.get('expired_at', [])
        sf.property_filters = kwargs.get('property_filters', [])
        return sf

    def test_empty_filters(self):
        sf = self._make_filter()
        result = build_milvus_filter_from_search_filters(sf, 'node')
        assert result == ''

    def test_node_labels_filter_for_node_type(self):
        sf = self._make_filter(node_labels=['Person', 'Organization'])
        result = build_milvus_filter_from_search_filters(sf, 'node')
        assert 'json_contains(labels, "Person", "Organization")' in result

    def test_node_labels_not_applied_for_edge_type(self):
        sf = self._make_filter(node_labels=['Person'])
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert result == ''

    def test_edge_types_filter(self):
        sf = self._make_filter(edge_types=['WORKS_AT', 'LIVES_IN'])
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert 'name in ["WORKS_AT", "LIVES_IN"]' in result

    def test_edge_uuids_filter(self):
        sf = self._make_filter(edge_uuids=['uuid1', 'uuid2'])
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert 'uuid in ["uuid1", "uuid2"]' in result

    def test_date_filter_comparison(self):
        date_filter = MagicMock()
        date_filter.comparison_operator.value = '>='
        date_filter.date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        sf = self._make_filter(valid_at=[[date_filter]])
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert 'valid_at >=' in result

    def test_date_filter_is_null(self):
        date_filter = MagicMock()
        date_filter.comparison_operator.value = 'IS NULL'

        sf = self._make_filter(expired_at=[[date_filter]])
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert 'expired_at == 0' in result

    def test_date_filter_is_not_null(self):
        date_filter = MagicMock()
        date_filter.comparison_operator.value = 'IS NOT NULL'

        sf = self._make_filter(created_at=[[date_filter]])
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert 'created_at != 0' in result

    def test_combined_edge_filters(self):
        sf = self._make_filter(
            edge_types=['WORKS_AT'],
            edge_uuids=['uuid1'],
        )
        result = build_milvus_filter_from_search_filters(sf, 'edge')
        assert 'name in ["WORKS_AT"]' in result
        assert 'uuid in ["uuid1"]' in result
        assert ' and ' in result


# ---- Collection Schema Builders ----


class TestCollectionSchemas:
    @patch('pymilvus.MilvusClient')
    def test_entity_node_schema_returns_tuple(self, mock_client):
        from graphiti_core.vector_store.milvus_utils import get_entity_node_collection_schema

        schema, index_params = get_entity_node_collection_schema(dim=128)
        assert schema is not None
        assert index_params is not None

    @patch('pymilvus.MilvusClient')
    def test_entity_edge_schema_returns_tuple(self, mock_client):
        from graphiti_core.vector_store.milvus_utils import get_entity_edge_collection_schema

        schema, index_params = get_entity_edge_collection_schema(dim=128)
        assert schema is not None
        assert index_params is not None

    @patch('pymilvus.MilvusClient')
    def test_episodic_node_schema_returns_tuple(self, mock_client):
        from graphiti_core.vector_store.milvus_utils import get_episodic_node_collection_schema

        schema, index_params = get_episodic_node_collection_schema()
        assert schema is not None
        assert index_params is not None

    @patch('pymilvus.MilvusClient')
    def test_community_node_schema_returns_tuple(self, mock_client):
        from graphiti_core.vector_store.milvus_utils import get_community_node_collection_schema

        schema, index_params = get_community_node_collection_schema(dim=128)
        assert schema is not None
        assert index_params is not None


# ---- Serialization/Deserialization ----


class TestEntityNodeSerialization:
    def _make_node(self):
        node = MagicMock()
        node.uuid = 'node-uuid-1'
        node.group_id = 'g1'
        node.name = 'Alice'
        node.summary = 'A person named Alice'
        node.labels = ['Person']
        node.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
        node.attributes = {'role': 'engineer'}
        node.name_embedding = [0.1, 0.2, 0.3]
        return node

    def test_to_milvus_dict(self):
        node = self._make_node()
        result = entity_node_to_milvus_dict(node)
        assert result['uuid'] == 'node-uuid-1'
        assert result['group_id'] == 'g1'
        assert result['name'] == 'Alice'
        assert result['labels'] == ['Person']
        assert result['created_at'] == datetime_to_epoch_ms(node.created_at)
        assert result['name_embedding'] == [0.1, 0.2, 0.3]

    def test_roundtrip(self):
        node = self._make_node()
        data = entity_node_to_milvus_dict(node)
        restored = milvus_dict_to_entity_node(data)
        assert restored.uuid == 'node-uuid-1'
        assert restored.name == 'Alice'
        assert restored.group_id == 'g1'
        assert restored.labels == ['Person']

    def test_none_embedding_uses_default(self):
        node = self._make_node()
        node.name_embedding = None
        result = entity_node_to_milvus_dict(node)
        assert len(result['name_embedding']) == 1024
        assert all(v == 0.0 for v in result['name_embedding'])

    def test_deserialization_with_string_labels(self):
        data = {
            'uuid': 'u1',
            'name': 'Test',
            'group_id': 'g1',
            'labels': json.dumps(['Person']),
            'created_at': 1705320000000,
            'attributes': '{}',
        }
        node = milvus_dict_to_entity_node(data)
        assert node.labels == ['Person']


class TestEntityEdgeSerialization:
    def _make_edge(self):
        edge = MagicMock()
        edge.uuid = 'edge-uuid-1'
        edge.group_id = 'g1'
        edge.source_node_uuid = 'src-uuid'
        edge.target_node_uuid = 'tgt-uuid'
        edge.name = 'WORKS_AT'
        edge.fact = 'Alice works at Acme'
        edge.episodes = ['ep1', 'ep2']
        edge.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
        edge.expired_at = None
        edge.valid_at = datetime(2024, 1, 10, tzinfo=timezone.utc)
        edge.invalid_at = None
        edge.attributes = {}
        edge.fact_embedding = [0.4, 0.5, 0.6]
        return edge

    def test_to_milvus_dict(self):
        edge = self._make_edge()
        result = entity_edge_to_milvus_dict(edge)
        assert result['uuid'] == 'edge-uuid-1'
        assert result['source_node_uuid'] == 'src-uuid'
        assert result['target_node_uuid'] == 'tgt-uuid'
        assert result['fact'] == 'Alice works at Acme'
        assert result['expired_at'] == 0  # None -> sentinel

    def test_roundtrip(self):
        edge = self._make_edge()
        data = entity_edge_to_milvus_dict(edge)
        restored = milvus_dict_to_entity_edge(data)
        assert restored.uuid == 'edge-uuid-1'
        assert restored.name == 'WORKS_AT'
        assert restored.expired_at is None  # 0 -> None
        assert restored.valid_at is not None

    def test_deserialization_with_string_episodes(self):
        data = {
            'uuid': 'e1',
            'group_id': 'g1',
            'source_node_uuid': 'src',
            'target_node_uuid': 'tgt',
            'name': 'TEST',
            'fact': 'test fact',
            'episodes': json.dumps(['ep1']),
            'created_at': 1705320000000,
            'expired_at': 0,
            'valid_at': 0,
            'invalid_at': 0,
            'attributes': '{}',
        }
        edge = milvus_dict_to_entity_edge(data)
        assert edge.episodes == ['ep1']


class TestEpisodicNodeSerialization:
    def _make_episodic_node(self):
        node = MagicMock()
        node.uuid = 'ep-uuid-1'
        node.group_id = 'g1'
        node.name = 'episode1'
        node.content = 'Alice had a meeting with Bob'
        node.source.value = 'text'
        node.source_description = 'A conversation'
        node.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
        node.valid_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
        node.entity_edges = ['edge1', 'edge2']
        return node

    def test_to_milvus_dict(self):
        node = self._make_episodic_node()
        result = episodic_node_to_milvus_dict(node)
        assert result['uuid'] == 'ep-uuid-1'
        assert result['content'] == 'Alice had a meeting with Bob'
        assert result['source'] == 'text'

    def test_roundtrip(self):
        node = self._make_episodic_node()
        data = episodic_node_to_milvus_dict(node)
        restored = milvus_dict_to_episodic_node(data)
        assert restored.uuid == 'ep-uuid-1'
        assert restored.content == 'Alice had a meeting with Bob'


class TestCommunityNodeSerialization:
    def _make_community_node(self):
        node = MagicMock()
        node.uuid = 'comm-uuid-1'
        node.group_id = 'g1'
        node.name = 'Tech Community'
        node.summary = 'A group of tech companies'
        node.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
        node.name_embedding = [0.7, 0.8, 0.9]
        return node

    def test_to_milvus_dict(self):
        node = self._make_community_node()
        result = community_node_to_milvus_dict(node)
        assert result['uuid'] == 'comm-uuid-1'
        assert result['name'] == 'Tech Community'

    def test_roundtrip(self):
        node = self._make_community_node()
        data = community_node_to_milvus_dict(node)
        restored = milvus_dict_to_community_node(data)
        assert restored.uuid == 'comm-uuid-1'
        assert restored.name == 'Tech Community'
        assert restored.summary == 'A group of tech companies'
