"""Database datetime round trips must not introduce a named UTC time zone."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz
from neo4j.time import DateTime

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j.operations.entity_edge_ops import Neo4jEntityEdgeOperations
from graphiti_core.driver.record_parsers import entity_edge_from_record
from graphiti_core.edges import get_entity_edge_from_record
from graphiti_core.helpers import parse_db_date
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

BOUNDARY = datetime(2026, 1, 11, microsecond=123456, tzinfo=timezone.utc)


@pytest.mark.parametrize('zone', [pytz.UTC, pytz.timezone('Asia/Bangkok'), pytz.FixedOffset(330)])
def test_neo4j_aware_dates_preserve_the_instant_in_fixed_utc(zone):
    original = BOUNDARY.astimezone(zone)
    stored = DateTime.from_native(original)

    parsed = parse_db_date(stored)

    assert parsed == BOUNDARY
    assert parsed.microsecond == BOUNDARY.microsecond
    # Neo4j serializes named UTC and fixed-offset UTC differently. Python
    # datetime equality alone cannot catch that storage representation change.
    assert parsed.tzinfo is timezone.utc
    reparsed = parse_db_date(DateTime.from_native(parsed))
    assert reparsed == BOUNDARY
    assert reparsed.tzinfo is timezone.utc


def test_fixed_utc_neo4j_date_remains_fixed_utc():
    parsed = parse_db_date(DateTime.from_native(BOUNDARY))
    assert parsed == BOUNDARY
    assert parsed.tzinfo is timezone.utc


def test_neo4j_local_datetime_remains_naive():
    local = BOUNDARY.replace(tzinfo=None)
    parsed = parse_db_date(DateTime.from_native(local))
    assert parsed == local
    assert parsed.tzinfo is None


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (
            '2026-01-11T05:30:00+05:30',
            datetime(2026, 1, 11, 5, 30, tzinfo=timezone(timedelta(hours=5.5))),
        ),
        ('2026-01-11T00:00:00', datetime(2026, 1, 11)),
        (BOUNDARY.astimezone(pytz.UTC), BOUNDARY.astimezone(pytz.UTC)),
        (None, None),
    ],
)
def test_non_neo4j_inputs_keep_existing_parse_behavior(value, expected):
    parsed = parse_db_date(value)
    assert parsed == expected
    if isinstance(expected, datetime):
        assert parsed.tzinfo == expected.tzinfo
    if isinstance(value, datetime):
        assert parsed is value


def named_utc_record():
    stored = DateTime.from_native(BOUNDARY.astimezone(pytz.UTC))
    return {
        'uuid': 'edge',
        'source_node_uuid': 'person',
        'target_node_uuid': 'project',
        'name': 'ASSIGNED_TO',
        'group_id': 'roundtrip-test',
        'fact': 'Kiran is assigned to Payments.',
        'fact_embedding': [0.0, 1.0],
        'episodes': [],
        'created_at': stored,
        'valid_at': stored,
        'invalid_at': stored,
        'reference_time': stored,
        'expired_at': None,
        'attributes': {},
    }


@pytest.mark.parametrize('parser', ['legacy', 'operations'])
@pytest.mark.parametrize('save_path', ['individual', 'legacy_bulk', 'operations_bulk'])
async def test_reloaded_edges_keep_fixed_utc_in_individual_and_bulk_payloads(parser, save_path):
    record = named_utc_record()
    edge = (
        get_entity_edge_from_record(record, GraphProvider.NEO4J)
        if parser == 'legacy'
        else entity_edge_from_record(record)
    )
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        graph_operations_interface=None,
        execute_query=AsyncMock(),
    )
    if save_path == 'individual':
        await edge.save(driver)
        payload = driver.execute_query.await_args.kwargs['edge_data']
    elif save_path == 'operations_bulk':
        await Neo4jEntityEdgeOperations().save_bulk(driver, [edge])
        payload = driver.execute_query.await_args.kwargs['entity_edges'][0]
    else:
        tx = SimpleNamespace(run=AsyncMock())
        embedder = MagicMock()
        await add_nodes_and_edges_bulk_tx(tx, [], [], [], [edge], embedder, driver)
        payload = next(
            call.kwargs['entity_edges'][0]
            for call in tx.run.await_args_list
            if 'entity_edges' in call.kwargs
        )
        embedder.create.assert_not_called()

    for field in ['created_at', 'valid_at', 'invalid_at']:
        assert payload[field] == BOUNDARY
        assert payload[field].tzinfo is timezone.utc
    assert payload['expired_at'] is None
