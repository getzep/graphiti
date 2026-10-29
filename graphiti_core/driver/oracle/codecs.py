# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from array import array
from datetime import datetime, timezone
from typing import Any

from graphiti_core.edges import (
    CommunityEdge,
    Edge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode

from .errors import InvalidEmbeddingDimension, NaiveDatetimeError

EMBEDDING_DIMENSION = 1024


def ensure_vector(value: Any | None) -> list[float] | None:
    if value is None:
        return None
    vector = [float(component) for component in value]
    if len(vector) != EMBEDDING_DIMENSION:
        raise InvalidEmbeddingDimension(expected=EMBEDDING_DIMENSION, actual=len(vector))
    return vector


def vector_to_bind(value: Any | None) -> array | None:
    vector = ensure_vector(value)
    return array('f', vector) if vector is not None else None


def ensure_utc_datetime(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NaiveDatetimeError('datetime must be timezone-aware')
    return parsed.astimezone(timezone.utc)


def optional_utc_datetime(value: datetime | str | None) -> datetime | None:
    return ensure_utc_datetime(value) if value is not None else None


def decode_oracle_datetime(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def optional_oracle_datetime(value: datetime | str | None) -> datetime | None:
    return decode_oracle_datetime(value) if value is not None else None


def encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def decode_json(value: Any | None, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def entity_node_to_binds(node: EntityNode) -> dict[str, Any]:
    created_at = ensure_utc_datetime(node.created_at)
    return {
        'uuid': node.uuid,
        'kind': 'ENTITY',
        'group_id': node.group_id,
        'name': node.name,
        'labels': encode_json(node.labels),
        'summary': node.summary,
        'properties': encode_json(node.attributes),
        'search_text': f'{node.name}\n{node.summary}'.strip(),
        'name_embedding': vector_to_bind(node.name_embedding),
        'created_at': created_at,
    }


def entity_node_from_record(record: dict[str, Any]) -> EntityNode:
    return EntityNode(
        uuid=str(record['uuid']),
        name=str(record['name']),
        group_id=str(record['group_id']),
        labels=list(decode_json(record.get('labels'), default=[])),
        summary=str(record.get('summary') or ''),
        attributes=dict(decode_json(record.get('properties'), default={})),
        name_embedding=ensure_vector(record.get('name_embedding')),
        created_at=decode_oracle_datetime(record['created_at']),
    )


def episode_node_to_binds(node: EpisodicNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'kind': 'EPISODIC',
        'group_id': node.group_id,
        'name': node.name,
        'labels': encode_json(node.labels),
        'source': node.source.value,
        'source_description': node.source_description,
        'content': node.content,
        'entity_edge_uuids': encode_json(node.entity_edges),
        'properties': encode_json(node.episode_metadata or {}),
        'search_text': f'{node.name}\n{node.content}'.strip(),
        'created_at': ensure_utc_datetime(node.created_at),
        'valid_at': ensure_utc_datetime(node.valid_at),
    }


def episode_node_from_record(record: dict[str, Any]) -> EpisodicNode:
    metadata = dict(decode_json(record.get('properties'), default={}))
    return EpisodicNode(
        uuid=str(record['uuid']),
        name=str(record['name']),
        group_id=str(record['group_id']),
        labels=list(decode_json(record.get('labels'), default=[])),
        source=EpisodeType.from_str(str(record['source'])),
        source_description=str(record.get('source_description') or ''),
        content=str(record.get('content') or ''),
        entity_edges=list(decode_json(record.get('entity_edge_uuids'), default=[])),
        episode_metadata=metadata or None,
        created_at=decode_oracle_datetime(record['created_at']),
        valid_at=decode_oracle_datetime(record['valid_at']),
    )


def community_node_to_binds(node: CommunityNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'kind': 'COMMUNITY',
        'group_id': node.group_id,
        'name': node.name,
        'labels': encode_json(node.labels),
        'summary': node.summary,
        'search_text': f'{node.name}\n{node.summary}'.strip(),
        'name_embedding': vector_to_bind(node.name_embedding),
        'created_at': ensure_utc_datetime(node.created_at),
    }


def community_node_from_record(record: dict[str, Any]) -> CommunityNode:
    return CommunityNode(
        uuid=str(record['uuid']),
        name=str(record['name']),
        group_id=str(record['group_id']),
        labels=list(decode_json(record.get('labels'), default=[])),
        summary=str(record.get('summary') or ''),
        name_embedding=ensure_vector(record.get('name_embedding')),
        created_at=decode_oracle_datetime(record['created_at']),
    )


def saga_node_to_binds(node: SagaNode) -> dict[str, Any]:
    return {
        'uuid': node.uuid,
        'kind': 'SAGA',
        'group_id': node.group_id,
        'name': node.name,
        'labels': encode_json(node.labels),
        'summary': node.summary,
        'search_text': f'{node.name}\n{node.summary}'.strip(),
        'created_at': ensure_utc_datetime(node.created_at),
        'first_episode_uuid': node.first_episode_uuid,
        'last_episode_uuid': node.last_episode_uuid,
        'last_summarized_at': optional_utc_datetime(node.last_summarized_at),
        'last_summarized_episode_uuid': node.last_summarized_episode_uuid,
        'last_summarized_episode_valid_at': optional_utc_datetime(
            node.last_summarized_episode_valid_at
        ),
    }


def saga_node_from_record(record: dict[str, Any]) -> SagaNode:
    return SagaNode(
        uuid=str(record['uuid']),
        name=str(record['name']),
        group_id=str(record['group_id']),
        labels=list(decode_json(record.get('labels'), default=[])),
        summary=str(record.get('summary') or ''),
        created_at=decode_oracle_datetime(record['created_at']),
        first_episode_uuid=record.get('first_episode_uuid'),
        last_episode_uuid=record.get('last_episode_uuid'),
        last_summarized_at=optional_oracle_datetime(record.get('last_summarized_at')),
        last_summarized_episode_uuid=record.get('last_summarized_episode_uuid'),
        last_summarized_episode_valid_at=optional_oracle_datetime(
            record.get('last_summarized_episode_valid_at')
        ),
    )


def edge_to_binds(edge: Edge, *, kind: str) -> dict[str, Any]:
    binds: dict[str, Any] = {
        'uuid': edge.uuid,
        'kind': kind,
        'source_uuid': edge.source_node_uuid,
        'target_uuid': edge.target_node_uuid,
        'group_id': edge.group_id,
        'name': kind,
        'fact': None,
        'episode_uuids': encode_json([]),
        'properties': encode_json({}),
        'search_text': kind,
        'fact_embedding': None,
        'created_at': ensure_utc_datetime(edge.created_at),
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'reference_time': None,
    }
    if isinstance(edge, EntityEdge):
        binds.update(
            {
                'name': edge.name,
                'fact': edge.fact,
                'episode_uuids': encode_json(edge.episodes),
                'properties': encode_json(edge.attributes),
                'search_text': f'{edge.name}\n{edge.fact}'.strip(),
                'fact_embedding': vector_to_bind(edge.fact_embedding),
                'expired_at': optional_utc_datetime(edge.expired_at),
                'valid_at': optional_utc_datetime(edge.valid_at),
                'invalid_at': optional_utc_datetime(edge.invalid_at),
                'reference_time': optional_utc_datetime(edge.reference_time),
            }
        )
    return binds


def edge_from_record(record: dict[str, Any]) -> Edge:
    kind = str(record['kind'])
    common = {
        'uuid': str(record['uuid']),
        'source_node_uuid': str(record['source_uuid']),
        'target_node_uuid': str(record['target_uuid']),
        'group_id': str(record['group_id']),
        'created_at': decode_oracle_datetime(record['created_at']),
    }
    if kind == 'RELATES_TO':
        return EntityEdge(
            **common,
            name=str(record.get('name') or 'RELATES_TO'),
            fact=str(record.get('fact') or ''),
            fact_embedding=ensure_vector(record.get('fact_embedding')),
            episodes=list(decode_json(record.get('episode_uuids'), default=[])),
            attributes=dict(decode_json(record.get('properties'), default={})),
            expired_at=optional_oracle_datetime(record.get('expired_at')),
            valid_at=optional_oracle_datetime(record.get('valid_at')),
            invalid_at=optional_oracle_datetime(record.get('invalid_at')),
            reference_time=optional_oracle_datetime(record.get('reference_time')),
        )
    edge_types: dict[str, type[Edge]] = {
        'MENTIONS': EpisodicEdge,
        'HAS_MEMBER': CommunityEdge,
        'HAS_EPISODE': HasEpisodeEdge,
        'NEXT_EPISODE': NextEpisodeEdge,
    }
    try:
        edge_type = edge_types[kind]
    except KeyError as error:
        raise ValueError(f'unsupported edge kind {kind}') from error
    return edge_type(**common)
