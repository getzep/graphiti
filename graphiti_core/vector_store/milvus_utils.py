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

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Default collection names (can be prefixed via collection_prefix)
COLLECTION_ENTITY_NODES = 'entity_nodes'
COLLECTION_ENTITY_EDGES = 'entity_edges'
COLLECTION_EPISODIC_NODES = 'episodic_nodes'
COLLECTION_COMMUNITY_NODES = 'community_nodes'

DEFAULT_EMBEDDING_DIM = 1024


# ---- Datetime Helpers ----


def datetime_to_epoch_ms(dt: datetime | None) -> int:
    """Convert datetime to epoch milliseconds. Returns 0 for None (null sentinel)."""
    if dt is None:
        return 0
    return int(dt.timestamp() * 1000)


def epoch_ms_to_datetime(ms: int) -> datetime | None:
    """Convert epoch milliseconds to datetime. Returns None for 0 (null sentinel)."""
    if ms == 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ---- Filter Expression Builders ----


def build_group_ids_filter(group_ids: list[str] | None) -> str:
    """Build Milvus filter expression for group_ids."""
    if not group_ids:
        return ''
    ids_str = ', '.join(f'"{g}"' for g in group_ids)
    return f'group_id in [{ids_str}]'


def _translate_date_filters(field_name: str, date_filter_groups: list[list[Any]]) -> str:
    """Translate Graphiti date filter groups to Milvus boolean expression.

    Date filters use nested AND/OR logic:
    - Outer list: OR groups
    - Inner list: AND conditions within each group
    """
    or_parts = []
    for or_group in date_filter_groups:
        and_parts = []
        for df in or_group:
            op = df.comparison_operator
            if op.value == 'IS NULL':
                and_parts.append(f'{field_name} == 0')
            elif op.value == 'IS NOT NULL':
                and_parts.append(f'{field_name} != 0')
            else:
                epoch_ms = datetime_to_epoch_ms(df.date)
                and_parts.append(f'{field_name} {op.value} {epoch_ms}')
        if and_parts:
            or_parts.append('(' + ' and '.join(and_parts) + ')')
    if not or_parts:
        return ''
    if len(or_parts) == 1:
        return or_parts[0]
    return '(' + ' or '.join(or_parts) + ')'


def build_milvus_filter_from_search_filters(search_filter: Any, entity_type: str) -> str:
    """Translate Graphiti SearchFilters to a Milvus boolean expression string.

    Args:
        search_filter: SearchFilters instance
        entity_type: 'node' or 'edge'

    Returns:
        Milvus boolean expression string (empty string if no filters)
    """
    parts: list[str] = []

    if entity_type == 'edge':
        if search_filter.edge_types:
            types_str = ', '.join(f'"{t}"' for t in search_filter.edge_types)
            parts.append(f'name in [{types_str}]')
        if search_filter.edge_uuids:
            uuids_str = ', '.join(f'"{u}"' for u in search_filter.edge_uuids)
            parts.append(f'uuid in [{uuids_str}]')
        if search_filter.valid_at:
            expr = _translate_date_filters('valid_at', search_filter.valid_at)
            if expr:
                parts.append(expr)
        if search_filter.invalid_at:
            expr = _translate_date_filters('invalid_at', search_filter.invalid_at)
            if expr:
                parts.append(expr)
        if search_filter.created_at:
            expr = _translate_date_filters('created_at', search_filter.created_at)
            if expr:
                parts.append(expr)
        if search_filter.expired_at:
            expr = _translate_date_filters('expired_at', search_filter.expired_at)
            if expr:
                parts.append(expr)

    if search_filter.node_labels:
        labels_str = ', '.join(f'"{label}"' for label in search_filter.node_labels)
        if entity_type == 'node':
            parts.append(f'json_contains(labels, {labels_str})')

    return ' and '.join(parts)


# ---- Collection Schema Builders ----


def get_entity_node_collection_schema(dim: int = DEFAULT_EMBEDDING_DIM) -> tuple[Any, Any]:
    """Return (schema, index_params) for entity_nodes collection."""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema()
    schema.add_field('uuid', DataType.VARCHAR, max_length=36, is_primary=True)
    schema.add_field('group_id', DataType.VARCHAR, max_length=128, is_partition_key=True)
    schema.add_field(
        'name', DataType.VARCHAR, max_length=512, enable_analyzer=True, enable_match=True
    )
    schema.add_field(
        'summary', DataType.VARCHAR, max_length=8192, enable_analyzer=True, enable_match=True
    )
    schema.add_field('labels', DataType.JSON)
    schema.add_field('created_at', DataType.INT64)
    schema.add_field('attributes', DataType.JSON)
    schema.add_field('name_embedding', DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field('name_sparse', DataType.SPARSE_FLOAT_VECTOR)

    bm25_fn = Function(
        name='name_bm25',
        function_type=FunctionType.BM25,
        input_field_names=['name'],
        output_field_names=['name_sparse'],
    )
    schema.add_function(bm25_fn)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name='name_embedding',
        index_type='HNSW',
        metric_type='COSINE',
        params={'M': 16, 'efConstruction': 200},
    )
    index_params.add_index(
        field_name='name_sparse',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25',
    )

    return schema, index_params


def get_entity_edge_collection_schema(dim: int = DEFAULT_EMBEDDING_DIM) -> tuple[Any, Any]:
    """Return (schema, index_params) for entity_edges collection."""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema()
    schema.add_field('uuid', DataType.VARCHAR, max_length=36, is_primary=True)
    schema.add_field('group_id', DataType.VARCHAR, max_length=128, is_partition_key=True)
    schema.add_field('source_node_uuid', DataType.VARCHAR, max_length=36)
    schema.add_field('target_node_uuid', DataType.VARCHAR, max_length=36)
    schema.add_field(
        'name', DataType.VARCHAR, max_length=512, enable_analyzer=True, enable_match=True
    )
    schema.add_field(
        'fact', DataType.VARCHAR, max_length=8192, enable_analyzer=True, enable_match=True
    )
    schema.add_field('episodes', DataType.JSON)
    schema.add_field('created_at', DataType.INT64)
    schema.add_field('expired_at', DataType.INT64)
    schema.add_field('valid_at', DataType.INT64)
    schema.add_field('invalid_at', DataType.INT64)
    schema.add_field('attributes', DataType.JSON)
    schema.add_field('fact_embedding', DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field('fact_sparse', DataType.SPARSE_FLOAT_VECTOR)

    bm25_fn = Function(
        name='fact_bm25',
        function_type=FunctionType.BM25,
        input_field_names=['fact'],
        output_field_names=['fact_sparse'],
    )
    schema.add_function(bm25_fn)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name='fact_embedding',
        index_type='HNSW',
        metric_type='COSINE',
        params={'M': 16, 'efConstruction': 200},
    )
    index_params.add_index(
        field_name='fact_sparse',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25',
    )

    return schema, index_params


def get_episodic_node_collection_schema() -> tuple[Any, Any]:
    """Return (schema, index_params) for episodic_nodes collection.

    Note: Episodic nodes have no dense vector field -- only BM25 fulltext search.
    """
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema()
    schema.add_field('uuid', DataType.VARCHAR, max_length=36, is_primary=True)
    schema.add_field('group_id', DataType.VARCHAR, max_length=128, is_partition_key=True)
    schema.add_field('name', DataType.VARCHAR, max_length=512)
    schema.add_field(
        'content', DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True
    )
    schema.add_field('source', DataType.VARCHAR, max_length=64)
    schema.add_field('source_description', DataType.VARCHAR, max_length=512)
    schema.add_field('created_at', DataType.INT64)
    schema.add_field('valid_at', DataType.INT64)
    schema.add_field('entity_edges', DataType.JSON)
    schema.add_field('content_sparse', DataType.SPARSE_FLOAT_VECTOR)

    bm25_fn = Function(
        name='content_bm25',
        function_type=FunctionType.BM25,
        input_field_names=['content'],
        output_field_names=['content_sparse'],
    )
    schema.add_function(bm25_fn)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name='content_sparse',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25',
    )

    return schema, index_params


def get_community_node_collection_schema(dim: int = DEFAULT_EMBEDDING_DIM) -> tuple[Any, Any]:
    """Return (schema, index_params) for community_nodes collection."""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema()
    schema.add_field('uuid', DataType.VARCHAR, max_length=36, is_primary=True)
    schema.add_field('group_id', DataType.VARCHAR, max_length=128, is_partition_key=True)
    schema.add_field(
        'name', DataType.VARCHAR, max_length=512, enable_analyzer=True, enable_match=True
    )
    schema.add_field('summary', DataType.VARCHAR, max_length=8192)
    schema.add_field('created_at', DataType.INT64)
    schema.add_field('name_embedding', DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field('name_sparse', DataType.SPARSE_FLOAT_VECTOR)

    bm25_fn = Function(
        name='name_bm25',
        function_type=FunctionType.BM25,
        input_field_names=['name'],
        output_field_names=['name_sparse'],
    )
    schema.add_function(bm25_fn)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name='name_embedding',
        index_type='HNSW',
        metric_type='COSINE',
        params={'M': 16, 'efConstruction': 200},
    )
    index_params.add_index(
        field_name='name_sparse',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25',
    )

    return schema, index_params


# ---- Serialization: Graphiti models <-> Milvus dicts ----


def entity_node_to_milvus_dict(node: Any) -> dict[str, Any]:
    """Serialize an EntityNode to a Milvus-compatible dict for upsert."""
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': (node.name or '')[:512],
        'summary': (node.summary or '')[:8192],
        'labels': node.labels or [],
        'created_at': datetime_to_epoch_ms(node.created_at),
        'attributes': node.attributes or {},
        'name_embedding': node.name_embedding or [0.0] * DEFAULT_EMBEDDING_DIM,
    }


def milvus_dict_to_entity_node(data: dict[str, Any]) -> Any:
    """Deserialize a Milvus result dict to an EntityNode."""
    from graphiti_core.nodes import EntityNode

    labels = data.get('labels', [])
    if isinstance(labels, str):
        labels = json.loads(labels)

    attributes = data.get('attributes', {})
    if isinstance(attributes, str):
        attributes = json.loads(attributes)

    return EntityNode(
        uuid=data['uuid'],
        name=data.get('name', ''),
        group_id=data.get('group_id', ''),
        labels=labels,
        created_at=epoch_ms_to_datetime(data.get('created_at', 0)) or datetime.now(tz=timezone.utc),
        name_embedding=data.get('name_embedding'),
        summary=data.get('summary', ''),
        attributes=attributes,
    )


def entity_edge_to_milvus_dict(edge: Any) -> dict[str, Any]:
    """Serialize an EntityEdge to a Milvus-compatible dict for upsert."""
    return {
        'uuid': edge.uuid,
        'group_id': edge.group_id,
        'source_node_uuid': edge.source_node_uuid,
        'target_node_uuid': edge.target_node_uuid,
        'name': (edge.name or '')[:512],
        'fact': (edge.fact or '')[:8192],
        'episodes': edge.episodes or [],
        'created_at': datetime_to_epoch_ms(edge.created_at),
        'expired_at': datetime_to_epoch_ms(edge.expired_at),
        'valid_at': datetime_to_epoch_ms(edge.valid_at),
        'invalid_at': datetime_to_epoch_ms(edge.invalid_at),
        'attributes': edge.attributes or {},
        'fact_embedding': edge.fact_embedding or [0.0] * DEFAULT_EMBEDDING_DIM,
    }


def milvus_dict_to_entity_edge(data: dict[str, Any]) -> Any:
    """Deserialize a Milvus result dict to an EntityEdge."""
    from graphiti_core.edges import EntityEdge

    episodes = data.get('episodes', [])
    if isinstance(episodes, str):
        episodes = json.loads(episodes)

    attributes = data.get('attributes', {})
    if isinstance(attributes, str):
        attributes = json.loads(attributes)

    return EntityEdge(
        uuid=data['uuid'],
        group_id=data.get('group_id', ''),
        source_node_uuid=data.get('source_node_uuid', ''),
        target_node_uuid=data.get('target_node_uuid', ''),
        name=data.get('name', ''),
        fact=data.get('fact', ''),
        fact_embedding=data.get('fact_embedding'),
        episodes=episodes,
        created_at=epoch_ms_to_datetime(data.get('created_at', 0)) or datetime.now(tz=timezone.utc),
        expired_at=epoch_ms_to_datetime(data.get('expired_at', 0)),
        valid_at=epoch_ms_to_datetime(data.get('valid_at', 0)),
        invalid_at=epoch_ms_to_datetime(data.get('invalid_at', 0)),
        attributes=attributes,
    )


def episodic_node_to_milvus_dict(node: Any) -> dict[str, Any]:
    """Serialize an EpisodicNode to a Milvus-compatible dict for upsert."""
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': (node.name or '')[:512],
        'content': (node.content or '')[:65535],
        'source': node.source.value if hasattr(node.source, 'value') else str(node.source),
        'source_description': (node.source_description or '')[:512],
        'created_at': datetime_to_epoch_ms(node.created_at),
        'valid_at': datetime_to_epoch_ms(node.valid_at),
        'entity_edges': node.entity_edges or [],
    }


def milvus_dict_to_episodic_node(data: dict[str, Any]) -> Any:
    """Deserialize a Milvus result dict to an EpisodicNode."""
    from graphiti_core.nodes import EpisodeType, EpisodicNode

    entity_edges = data.get('entity_edges', [])
    if isinstance(entity_edges, str):
        entity_edges = json.loads(entity_edges)

    return EpisodicNode(
        uuid=data['uuid'],
        name=data.get('name', ''),
        group_id=data.get('group_id', ''),
        content=data.get('content', ''),
        source=EpisodeType.from_str(data.get('source', 'text')),
        source_description=data.get('source_description', ''),
        created_at=epoch_ms_to_datetime(data.get('created_at', 0)) or datetime.now(tz=timezone.utc),
        valid_at=epoch_ms_to_datetime(data.get('valid_at', 0)) or datetime.now(tz=timezone.utc),
        entity_edges=entity_edges,
    )


def community_node_to_milvus_dict(node: Any) -> dict[str, Any]:
    """Serialize a CommunityNode to a Milvus-compatible dict for upsert."""
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': (node.name or '')[:512],
        'summary': (node.summary or '')[:8192],
        'created_at': datetime_to_epoch_ms(node.created_at),
        'name_embedding': node.name_embedding or [0.0] * DEFAULT_EMBEDDING_DIM,
    }


def milvus_dict_to_community_node(data: dict[str, Any]) -> Any:
    """Deserialize a Milvus result dict to a CommunityNode."""
    from graphiti_core.nodes import CommunityNode

    return CommunityNode(
        uuid=data['uuid'],
        name=data.get('name', ''),
        group_id=data.get('group_id', ''),
        summary=data.get('summary', ''),
        created_at=epoch_ms_to_datetime(data.get('created_at', 0)) or datetime.now(tz=timezone.utc),
        name_embedding=data.get('name_embedding'),
    )
