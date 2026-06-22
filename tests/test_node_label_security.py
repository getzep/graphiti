import pytest
from pydantic import ValidationError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import NodeLabelValidationError
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_entity_node_save_query,
)
from graphiti_core.nodes import EntityNode


def test_entity_node_rejects_unsafe_labels():
    with pytest.raises(ValidationError, match='node_labels must start with a letter or underscore'):
        EntityNode(
            name='Alice',
            group_id='group',
            labels=['Entity`) WITH n MATCH (x) DETACH DELETE x //'],
        )


def test_entity_node_assignment_rejects_unsafe_labels():
    node = EntityNode(name='Alice', group_id='group', labels=['Person'])

    with pytest.raises(ValidationError, match='node_labels must start with a letter or underscore'):
        node.labels = ['Entity`) WITH n MATCH (x) DETACH DELETE x //']


def test_entity_node_save_query_rejects_unsafe_labels_when_validation_is_bypassed():
    with pytest.raises(
        NodeLabelValidationError, match='node_labels must start with a letter or underscore'
    ):
        get_entity_node_save_query(
            GraphProvider.NEO4J,
            'Entity:Entity`) WITH n MATCH (x) DETACH DELETE x //',
        )


def test_entity_node_save_bulk_query_rejects_unsafe_labels_when_validation_is_bypassed():
    with pytest.raises(
        NodeLabelValidationError, match='node_labels must start with a letter or underscore'
    ):
        get_entity_node_save_bulk_query(
            GraphProvider.FALKORDB,
            [
                {
                    'uuid': 'node-1',
                    'name': 'Alice',
                    'group_id': 'group',
                    'summary': 'summary',
                    'created_at': '2024-01-01T00:00:00Z',
                    'name_embedding': [0.1, 0.2],
                    'labels': ['Entity', 'Entity`) WITH n MATCH (x) DETACH DELETE x //'],
                }
            ],
        )
