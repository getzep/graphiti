"""Unit tests for response formatting helpers.

These cover the case where a graph driver returns temporal values as its own
types rather than Python built-ins. Such values arrive inside ``attributes``,
an untyped dict Pydantic's JSON mode does not convert, so without sanitizing
them the formatted response cannot be JSON-encoded.

The fake driver types below stand in for ``neo4j.time`` so these tests run
without the neo4j driver installed -- this package only requires
``graphiti-core[falkordb]``.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from utils.formatting import format_fact_result, format_node_result, to_node_result

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class FakeDriverDateTime:
    """Stands in for neo4j.time.DateTime: not a datetime, but has to_native()."""

    def __init__(self, native: datetime):
        self._native = native

    def to_native(self) -> datetime:
        return self._native


class FakeDriverDuration:
    """Stands in for neo4j.time.Duration: no to_native(), no isoformat()."""

    def __str__(self) -> str:
        return 'P1DT2H'


class FakeUnconvertible:
    """A driver type whose to_native() raises; must degrade to str(), not blow up."""

    def to_native(self):
        raise ValueError('cannot convert')

    def __str__(self) -> str:
        return 'unconvertible'


def _edge(attributes: dict) -> EntityEdge:
    return EntityEdge(
        uuid='edge-uuid',
        group_id='group',
        source_node_uuid='src-uuid',
        target_node_uuid='tgt-uuid',
        name='RELATES_TO',
        fact='a relates to b',
        created_at=NOW,
        attributes=attributes,
    )


def _node(attributes: dict) -> EntityNode:
    return EntityNode(
        uuid='node-uuid',
        name='Entity',
        group_id='group',
        labels=['Entity'],
        created_at=NOW,
        summary='a summary',
        attributes=attributes,
    )


def test_format_fact_result_serializes_driver_datetime():
    result = format_fact_result(_edge({'reference_time': FakeDriverDateTime(NOW)}))

    assert result['attributes']['reference_time'] == NOW.isoformat()
    json.dumps(result)


def test_format_node_result_serializes_driver_datetime():
    result = format_node_result(_node({'observed_at': FakeDriverDateTime(NOW)}))

    assert result['attributes']['observed_at'] == NOW.isoformat()
    json.dumps(result)


def test_to_node_result_serializes_driver_datetime():
    result = to_node_result(_node({'observed_at': FakeDriverDateTime(NOW)}))

    assert result['attributes']['observed_at'] == NOW.isoformat()
    json.dumps(result)


def test_nested_containers_are_sanitized():
    attributes = {
        'history': [{'at': FakeDriverDateTime(NOW)}],
        'nested': {'inner': {'at': FakeDriverDateTime(NOW)}},
    }

    result = format_fact_result(_edge(attributes))

    assert result['attributes']['history'][0]['at'] == NOW.isoformat()
    assert result['attributes']['nested']['inner']['at'] == NOW.isoformat()
    json.dumps(result)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (FakeDriverDuration(), 'P1DT2H'),
        (FakeUnconvertible(), 'unconvertible'),
        (timedelta(days=1), str(timedelta(days=1))),
        (NOW, NOW.isoformat()),
    ],
)
def test_non_native_values_degrade_to_strings(value, expected):
    result = format_fact_result(_edge({'value': value}))

    assert result['attributes']['value'] == expected
    json.dumps(result)


def test_plain_values_pass_through_unchanged():
    attributes = {'count': 3, 'label': 'x', 'ok': True, 'missing': None}

    result = format_fact_result(_edge(attributes))

    assert result['attributes'] == attributes
    json.dumps(result)


def test_embeddings_are_still_stripped():
    edge_result = format_fact_result(_edge({'fact_embedding': [0.1, 0.2]}))
    node_result = format_node_result(_node({'name_embedding': [0.1, 0.2]}))
    typed_node_result = to_node_result(_node({'name_embedding': [0.1, 0.2]}))

    assert 'fact_embedding' not in edge_result['attributes']
    assert 'fact_embedding' not in edge_result
    assert 'name_embedding' not in node_result['attributes']
    assert 'name_embedding' not in node_result
    assert 'name_embedding' not in typed_node_result['attributes']


def test_typed_fields_are_still_serialized():
    result = format_fact_result(_edge({}))

    assert result['uuid'] == 'edge-uuid'
    assert result['fact'] == 'a relates to b'
    # Pydantic's JSON mode renders UTC as a trailing 'Z'; compare instants.
    assert datetime.fromisoformat(result['created_at'].replace('Z', '+00:00')) == NOW
    assert result['attributes'] == {}
