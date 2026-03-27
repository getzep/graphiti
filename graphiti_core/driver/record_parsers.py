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

from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode


def entity_node_from_record(record: Any) -> EntityNode:
    """Parse an entity node from a database record."""
    attributes = record['attributes']
    attributes.pop('uuid', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('name_embedding', None)
    attributes.pop('summary', None)
    attributes.pop('created_at', None)
    attributes.pop('labels', None)

    labels = record.get('labels', [])
    group_id = record.get('group_id')
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
        summary=record['summary'],
        attributes=attributes,
    )


def entity_edge_from_record(record: Any) -> EntityEdge:
    """Parse an entity edge from a database record."""
    attributes = record['attributes']
    attributes.pop('uuid', None)
    attributes.pop('source_node_uuid', None)
    attributes.pop('target_node_uuid', None)
    attributes.pop('fact', None)
    attributes.pop('fact_embedding', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('episodes', None)
    attributes.pop('created_at', None)
    attributes.pop('expired_at', None)
    attributes.pop('valid_at', None)
    attributes.pop('invalid_at', None)

    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=record['episodes'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
        expired_at=parse_db_date(record['expired_at']),
        valid_at=parse_db_date(record['valid_at']),
        invalid_at=parse_db_date(record['invalid_at']),
        attributes=attributes,
    )


def episodic_node_from_record(record: Any) -> EpisodicNode:
    """Parse an episodic node from a database record."""
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
    )


def community_node_from_record(record: Any) -> CommunityNode:
    """Parse a community node from a database record."""
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
        summary=record['summary'],
    )
