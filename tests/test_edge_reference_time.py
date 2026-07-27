"""Regression tests for preserving EntityEdge.reference_time on reads."""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.record_parsers import entity_edge_from_record
from graphiti_core.edges import get_entity_edge_from_record
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query


REFERENCE_TIME = datetime(2024, 12, 9, tzinfo=timezone.utc)


@pytest.mark.parametrize('provider', list(GraphProvider))
def test_entity_edge_return_query_selects_reference_time(provider: GraphProvider):
    query = get_entity_edge_return_query(provider)

    assert 'e.reference_time AS reference_time' in query


@pytest.mark.parametrize('parser', [get_entity_edge_from_record, entity_edge_from_record])
def test_entity_edge_parser_preserves_reference_time(parser):
    record = {
        'uuid': 'edge-uuid',
        'source_node_uuid': 'source-uuid',
        'target_node_uuid': 'target-uuid',
        'fact': 'Source relates to target',
        'name': 'relates to',
        'group_id': 'test',
        'episodes': [],
        'created_at': REFERENCE_TIME,
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'reference_time': REFERENCE_TIME,
        'attributes': {},
    }

    if parser is get_entity_edge_from_record:
        edge = parser(record, GraphProvider.NEO4J)
    else:
        edge = parser(record)

    assert edge.reference_time == REFERENCE_TIME
