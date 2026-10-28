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

# Encoders and decoders between Graphiti models and Spanner table rows.
#
# ``*_to_row`` returns a dict keyed by TABLE column name with codec-converted values, ready
# for ``mutations.upsert(table, rows, <KIND>_COLUMNS)``. For edges that means
# ``uuid`` = source node uuid and ``edge_uuid`` = Graphiti edge uuid.
#
# ``*_from_row`` takes a dict keyed by GRAPHITI field name, which is what SQL using
# ``schema.edge_select_columns`` and GQL over the property graph both return
# (``uuid`` = edge uuid, ``source_node_uuid`` = source node uuid). A raw edge-table row
# (it has ``edge_uuid`` but no ``source_node_uuid``) is renamed first, so both shapes work.

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from graphiti_core.driver.spanner.codecs import (
    coerce_embedding,
    from_spanner_value,
    to_jsonable,
    to_spanner_value,
)
from graphiti_core.driver.spanner.schema import EDGE_TABLES, NODE_TABLES
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from graphiti_core.utils.datetime_utils import ensure_utc

ENTITY_NODE_COLUMNS: tuple[str, ...] = NODE_TABLES['Entity'].writable_columns
EPISODIC_NODE_COLUMNS: tuple[str, ...] = NODE_TABLES['Episodic'].writable_columns
COMMUNITY_NODE_COLUMNS: tuple[str, ...] = NODE_TABLES['Community'].writable_columns
SAGA_NODE_COLUMNS: tuple[str, ...] = NODE_TABLES['Saga'].writable_columns
ENTITY_EDGE_COLUMNS: tuple[str, ...] = EDGE_TABLES['RelatesTo'].writable_columns
EPISODIC_EDGE_COLUMNS: tuple[str, ...] = EDGE_TABLES['Mentions'].writable_columns
COMMUNITY_EDGE_COLUMNS: tuple[str, ...] = EDGE_TABLES['HasMember'].writable_columns
HAS_EPISODE_EDGE_COLUMNS: tuple[str, ...] = EDGE_TABLES['HasEpisode'].writable_columns
NEXT_EPISODE_EDGE_COLUMNS: tuple[str, ...] = EDGE_TABLES['NextEpisode'].writable_columns

# Attribute keys that collide with core fields. Core fields win on write (as in
# EntityNode.save) and the keys are dropped on read (as in driver/record_parsers.py).
ENTITY_NODE_RESERVED_ATTRIBUTES: frozenset[str] = frozenset(
    {'uuid', 'name', 'group_id', 'name_embedding', 'summary', 'created_at', 'labels'}
)
ENTITY_EDGE_RESERVED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        'uuid',
        'source_node_uuid',
        'target_node_uuid',
        'fact',
        'fact_embedding',
        'name',
        'group_id',
        'episodes',
        'created_at',
        'expired_at',
        'valid_at',
        'invalid_at',
        'reference_time',
    }
)

ENTITY_LABEL = 'Entity'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp(value: datetime | None) -> datetime | None:
    return to_spanner_value(value)


def _attributes(attributes: Mapping[str, Any] | None, reserved: Iterable[str]) -> Any:
    reserved_keys = set(reserved)
    kept = {key: value for key, value in (attributes or {}).items() if key not in reserved_keys}
    return to_spanner_value(to_jsonable(kept))


def _read_attributes(value: Any, reserved: Iterable[str]) -> dict[str, Any]:
    decoded = from_spanner_value(value)
    if not isinstance(decoded, dict):
        return {}
    reserved_keys = set(reserved)
    return {key: item for key, item in decoded.items() if key not in reserved_keys}


def _read_datetime(value: Any) -> datetime | None:
    decoded = from_spanner_value(value)
    if decoded is None:
        return None
    if isinstance(decoded, str):
        decoded = datetime.fromisoformat(decoded)
    if not isinstance(decoded, datetime):
        raise TypeError(f'Expected a TIMESTAMP value, got {type(decoded).__name__}')
    return ensure_utc(decoded)


def _required_datetime(row: Mapping[str, Any], key: str) -> datetime:
    value = _read_datetime(row.get(key))
    if value is None:
        raise ValueError(f'{key} cannot be None for row {row.get("uuid", "unknown")}')
    return value


def _read_list(value: Any) -> list[Any]:
    decoded = from_spanner_value(value)
    return list(decoded) if decoded else []


def _read_embedding(value: Any) -> list[float] | None:
    decoded = from_spanner_value(value)
    return coerce_embedding(decoded) if decoded is not None else None


def _edge_key(edge_uuid: str, source_node_uuid: str) -> dict[str, Any]:
    return {'uuid': source_node_uuid, 'edge_uuid': edge_uuid}


def edge_record_from_table_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Rename a raw edge-table row to Graphiti names.

    ``uuid`` (source node) -> ``source_node_uuid`` and ``edge_uuid`` -> ``uuid``. Rows that are
    already in Graphiti shape (no ``edge_uuid`` key, or a ``source_node_uuid`` key) are
    returned as a copy unchanged.
    """
    record = dict(row)
    if 'edge_uuid' in record and 'source_node_uuid' not in record:
        record['source_node_uuid'] = record.pop('uuid', None)
        record['uuid'] = record.pop('edge_uuid')
    return record


def entity_labels(labels: Iterable[str]) -> list[str]:
    """Stored labels: the node's labels plus ``Entity``, de-duplicated, order preserved."""
    return list(dict.fromkeys([*labels, ENTITY_LABEL]))


# ---------------------------------------------------------------------------
# Node encoders
# ---------------------------------------------------------------------------


def entity_node_to_row(node: EntityNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': node.name,
        'labels': entity_labels(node.labels),
        'summary': node.summary,
        'attributes': _attributes(node.attributes, ENTITY_NODE_RESERVED_ATTRIBUTES),
        'name_embedding': coerce_embedding(node.name_embedding),
        'created_at': _timestamp(node.created_at),
    }


def episodic_node_to_row(node: EpisodicNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': node.name,
        'source': node.source.value,
        'source_description': node.source_description,
        'content': node.content,
        'valid_at': _timestamp(node.valid_at),
        'created_at': _timestamp(node.created_at),
        'entity_edges': list(node.entity_edges),
        'episode_metadata': to_spanner_value(node.episode_metadata),
    }


def community_node_to_row(node: CommunityNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': node.name,
        'summary': node.summary,
        'name_embedding': coerce_embedding(node.name_embedding),
        'created_at': _timestamp(node.created_at),
    }


def saga_node_to_row(node: SagaNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'group_id': node.group_id,
        'name': node.name,
        'summary': node.summary,
        'first_episode_uuid': node.first_episode_uuid,
        'last_episode_uuid': node.last_episode_uuid,
        'last_summarized_at': _timestamp(node.last_summarized_at),
        'last_summarized_episode_valid_at': _timestamp(node.last_summarized_episode_valid_at),
        'created_at': _timestamp(node.created_at),
    }


# ---------------------------------------------------------------------------
# Edge encoders (``uuid`` = source node uuid, ``edge_uuid`` = edge uuid)
# ---------------------------------------------------------------------------


def entity_edge_to_row(edge: EntityEdge) -> dict[str, Any]:
    return {
        **_edge_key(edge.uuid, edge.source_node_uuid),
        'target_node_uuid': edge.target_node_uuid,
        'group_id': edge.group_id,
        'name': edge.name,
        'fact': edge.fact,
        'fact_embedding': coerce_embedding(edge.fact_embedding),
        'episodes': list(edge.episodes),
        'attributes': _attributes(edge.attributes, ENTITY_EDGE_RESERVED_ATTRIBUTES),
        'created_at': _timestamp(edge.created_at),
        'valid_at': _timestamp(edge.valid_at),
        'invalid_at': _timestamp(edge.invalid_at),
        'expired_at': _timestamp(edge.expired_at),
        'reference_time': _timestamp(edge.reference_time),
    }


def _simple_edge_to_row(
    edge: EpisodicEdge | CommunityEdge | HasEpisodeEdge | NextEpisodeEdge,
) -> dict[str, Any]:
    return {
        **_edge_key(edge.uuid, edge.source_node_uuid),
        'target_node_uuid': edge.target_node_uuid,
        'group_id': edge.group_id,
        'created_at': _timestamp(edge.created_at),
    }


def episodic_edge_to_row(edge: EpisodicEdge) -> dict[str, Any]:
    return _simple_edge_to_row(edge)


def community_edge_to_row(edge: CommunityEdge) -> dict[str, Any]:
    return _simple_edge_to_row(edge)


def has_episode_edge_to_row(edge: HasEpisodeEdge) -> dict[str, Any]:
    return _simple_edge_to_row(edge)


def next_episode_edge_to_row(edge: NextEpisodeEdge) -> dict[str, Any]:
    return _simple_edge_to_row(edge)


# ---------------------------------------------------------------------------
# Node decoders (rows keyed by Graphiti field names)
# ---------------------------------------------------------------------------


def entity_node_from_row(row: Mapping[str, Any]) -> EntityNode:
    group_id = row['group_id']
    labels = [str(label) for label in _read_list(row.get('labels'))]
    # Mirror record_parsers.entity_node_from_record: drop the FalkorDB-era per-group label.
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)
    return EntityNode(
        uuid=row['uuid'],
        name=row['name'],
        group_id=group_id,
        labels=labels,
        name_embedding=_read_embedding(row.get('name_embedding')),
        summary=row.get('summary') or '',
        created_at=_required_datetime(row, 'created_at'),
        attributes=_read_attributes(row.get('attributes'), ENTITY_NODE_RESERVED_ATTRIBUTES),
    )


def episodic_node_from_row(row: Mapping[str, Any]) -> EpisodicNode:
    metadata = from_spanner_value(row.get('episode_metadata'))
    return EpisodicNode(
        uuid=row['uuid'],
        name=row['name'],
        group_id=row['group_id'],
        source=EpisodeType.from_str(row['source']),
        source_description=row['source_description'],
        content=row['content'],
        valid_at=_required_datetime(row, 'valid_at'),
        created_at=_required_datetime(row, 'created_at'),
        entity_edges=[str(item) for item in _read_list(row.get('entity_edges'))],
        episode_metadata=metadata if isinstance(metadata, dict) else None,
    )


def community_node_from_row(row: Mapping[str, Any]) -> CommunityNode:
    return CommunityNode(
        uuid=row['uuid'],
        name=row['name'],
        group_id=row['group_id'],
        name_embedding=_read_embedding(row.get('name_embedding')),
        summary=row.get('summary') or '',
        created_at=_required_datetime(row, 'created_at'),
    )


def saga_node_from_row(row: Mapping[str, Any]) -> SagaNode:
    return SagaNode(
        uuid=row['uuid'],
        name=row['name'],
        group_id=row['group_id'],
        summary=row.get('summary') or '',
        first_episode_uuid=row.get('first_episode_uuid'),
        last_episode_uuid=row.get('last_episode_uuid'),
        last_summarized_at=_read_datetime(row.get('last_summarized_at')),
        last_summarized_episode_valid_at=_read_datetime(
            row.get('last_summarized_episode_valid_at')
        ),
        created_at=_required_datetime(row, 'created_at'),
    )


# ---------------------------------------------------------------------------
# Edge decoders (rows keyed by Graphiti field names; raw table rows are renamed)
# ---------------------------------------------------------------------------


def entity_edge_from_row(row: Mapping[str, Any]) -> EntityEdge:
    record = edge_record_from_table_row(row)
    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        group_id=record['group_id'],
        name=record['name'],
        fact=record['fact'],
        fact_embedding=_read_embedding(record.get('fact_embedding')),
        episodes=[str(item) for item in _read_list(record.get('episodes'))],
        created_at=_required_datetime(record, 'created_at'),
        valid_at=_read_datetime(record.get('valid_at')),
        invalid_at=_read_datetime(record.get('invalid_at')),
        expired_at=_read_datetime(record.get('expired_at')),
        reference_time=_read_datetime(record.get('reference_time')),
        attributes=_read_attributes(record.get('attributes'), ENTITY_EDGE_RESERVED_ATTRIBUTES),
    )


def _simple_edge_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    record = edge_record_from_table_row(row)
    return {
        'uuid': record['uuid'],
        'source_node_uuid': record['source_node_uuid'],
        'target_node_uuid': record['target_node_uuid'],
        'group_id': record['group_id'],
        'created_at': _required_datetime(record, 'created_at'),
    }


def episodic_edge_from_row(row: Mapping[str, Any]) -> EpisodicEdge:
    return EpisodicEdge(**_simple_edge_fields(row))


def community_edge_from_row(row: Mapping[str, Any]) -> CommunityEdge:
    return CommunityEdge(**_simple_edge_fields(row))


def has_episode_edge_from_row(row: Mapping[str, Any]) -> HasEpisodeEdge:
    return HasEpisodeEdge(**_simple_edge_fields(row))


def next_episode_edge_from_row(row: Mapping[str, Any]) -> NextEpisodeEdge:
    return NextEpisodeEdge(**_simple_edge_fields(row))
