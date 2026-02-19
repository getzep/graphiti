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
from typing import Any

from graphiti_core.driver.record_parsers import entity_edge_from_record, entity_node_from_record
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode


def parse_kuzu_entity_node(record: Any) -> EntityNode:
    """Parse a Kuzu entity node record, deserializing JSON attributes."""
    if isinstance(record.get('attributes'), str):
        try:
            record['attributes'] = json.loads(record['attributes'])
        except (json.JSONDecodeError, TypeError):
            record['attributes'] = {}
    elif record.get('attributes') is None:
        record['attributes'] = {}
    return entity_node_from_record(record)


def parse_kuzu_entity_edge(record: Any) -> EntityEdge:
    """Parse a Kuzu entity edge record, deserializing JSON attributes."""
    if isinstance(record.get('attributes'), str):
        try:
            record['attributes'] = json.loads(record['attributes'])
        except (json.JSONDecodeError, TypeError):
            record['attributes'] = {}
    elif record.get('attributes') is None:
        record['attributes'] = {}
    return entity_edge_from_record(record)
