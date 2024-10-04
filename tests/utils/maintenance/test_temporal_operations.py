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

import unittest
from datetime import datetime

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.temporal_operations import (
    extract_date_strings_from_edge,
)


# Helper function to create test data
def create_test_data():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now, group_id='1')
    node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now, group_id='1')
    node3 = EntityNode(uuid='3', name='Node3', labels=['Person'], created_at=now, group_id='1')

    # Create edges
    existing_edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='KNOWS',
        fact='Node1 knows Node2',
        created_at=now,
        group_id='1',
    )
    existing_edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='2',
        target_node_uuid='3',
        name='LIKES',
        fact='Node2 likes Node3',
        created_at=now,
        group_id='1',
    )
    new_edge1 = EntityEdge(
        uuid='e3',
        source_node_uuid='1',
        target_node_uuid='3',
        name='WORKS_WITH',
        fact='Node1 works with Node3',
        created_at=now,
        group_id='1',
    )
    new_edge2 = EntityEdge(
        uuid='e4',
        source_node_uuid='1',
        target_node_uuid='2',
        name='DISLIKES',
        fact='Node1 dislikes Node2',
        created_at=now,
        group_id='1',
    )

    return {
        'nodes': [node1, node2, node3],
        'existing_edges': [existing_edge1, existing_edge2],
        'new_edges': [new_edge1, new_edge2],
    }


class TestExtractDateStringsFromEdge(unittest.TestCase):
    def generate_entity_edge(self, valid_at, invalid_at):
        return EntityEdge(
            source_node_uuid='1',
            target_node_uuid='2',
            name='KNOWS',
            fact='Node1 knows Node2',
            created_at=datetime.now(),
            valid_at=valid_at,
            invalid_at=invalid_at,
            group_id='1',
        )

    def test_both_dates_present(self):
        edge = self.generate_entity_edge(datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 2, 12, 0))
        result = extract_date_strings_from_edge(edge)
        expected = 'Start Date: 2024-01-01T12:00:00 (End Date: 2024-01-02T12:00:00)'
        self.assertEqual(result, expected)

    def test_only_valid_at_present(self):
        edge = self.generate_entity_edge(datetime(2024, 1, 1, 12, 0), None)
        result = extract_date_strings_from_edge(edge)
        expected = 'Start Date: 2024-01-01T12:00:00'
        self.assertEqual(result, expected)

    def test_only_invalid_at_present(self):
        edge = self.generate_entity_edge(None, datetime(2024, 1, 2, 12, 0))
        result = extract_date_strings_from_edge(edge)
        expected = ' (End Date: 2024-01-02T12:00:00)'
        self.assertEqual(result, expected)

    def test_no_dates_present(self):
        edge = self.generate_entity_edge(None, None)
        result = extract_date_strings_from_edge(edge)
        expected = ''
        self.assertEqual(result, expected)


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])
