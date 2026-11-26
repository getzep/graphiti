"""Test temporal fulltext retrieval with ranked synthetic index hits, without a DB.

SQLite evaluates the WHERE fragment emitted by Graphiti. The executor models
Neo4j's procedure-level limit before that filter; it does not implement Lucene
ranking or validate a live Neo4j server.
"""

import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.edges import EntityEdge
from graphiti_core.graph_queries import get_relationships_query
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.search.search_utils import edge_fulltext_search

CUTOFF = datetime(2024, 6, 1, tzinfo=timezone.utc)


class RankedExecutor:
    provider = GraphProvider.NEO4J
    search_interface = None
    fulltext_syntax = ''

    def __init__(self, depth):
        self.cypher = ''
        self.edges = {}
        self.db = sqlite3.connect(':memory:')
        self.db.execute('CREATE TABLE edges(uuid TEXT, invalid_at REAL, score REAL)')
        # The obsolete facts are better index matches than the current fact.
        for i in range(depth):
            self.add(f'old-{i}', CUTOFF, 100 - i)
        self.add('current', None, 1)

    def add(self, uuid, invalid_at, score):
        self.edges[uuid] = EntityEdge(
            uuid=uuid,
            source_node_uuid='alice',
            target_node_uuid=uuid + '-city',
            name='LIVES_IN',
            group_id='group-a',
            fact='Alice lives in ' + uuid,
            episodes=[],
            created_at=CUTOFF,
            valid_at=CUTOFF - timedelta(days=1),
            invalid_at=invalid_at,
        )
        self.db.execute(
            'INSERT INTO edges VALUES (?, ?, ?)',
            (uuid, invalid_at.timestamp() if invalid_at else None, score),
        )

    async def execute_query(self, cypher, **params):
        self.cypher = cypher
        fragment = re.search(r'\bWHERE\b(.*?)\bWITH\b', cypher, re.S)
        # No group filter is passed to these calls, so only date predicates are present.
        predicate = fragment.group(1) if fragment else '1'
        bindings = {
            key: value.timestamp() if isinstance(value, datetime) else value
            for key, value in params.items()
        }
        source = 'edges'
        if '{limit: $limit}' in cypher:
            source = '(SELECT * FROM edges ORDER BY score DESC LIMIT :limit)'
        rows = self.db.execute(
            f'SELECT uuid FROM {source} AS e WHERE {predicate} ORDER BY score DESC LIMIT :limit',
            bindings,
        ).fetchall()
        return [self.edges[uuid].model_dump() for (uuid,) in rows], None, None


@pytest.fixture
def executor_factory():
    executors = []

    def create(depth):
        executor = RankedExecutor(depth)
        executors.append(executor)
        return executor

    yield create
    for executor in executors:
        executor.db.close()


def temporal_filter():
    return SearchFilters(
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=CUTOFF, comparison_operator=ComparisonOperator.greater_than)],
        ]
    )


async def retrieve(executor, filters, operations):
    if operations:
        return await Neo4jSearchOperations().edge_fulltext_search(
            executor, 'Alice lives in Berlin', filters, limit=20
        )
    return await edge_fulltext_search(executor, 'Alice lives in Berlin', filters, limit=20)


@pytest.mark.parametrize('depth', [19, 20, 50])
@pytest.mark.parametrize('operations', [False, True])
async def test_temporal_filter_finds_current_fact_past_obsolete_index_hits(
    executor_factory, depth, operations
):
    executor = executor_factory(depth)

    edges = await retrieve(executor, temporal_filter(), operations)

    assert [edge.uuid for edge in edges] == ['current']
    if depth >= 20:
        assert '{limit: $limit}' not in executor.cypher
    assert 'ORDER BY score DESC' in executor.cypher
    assert 'LIMIT $limit' in executor.cypher


@pytest.mark.parametrize('operations', [False, True])
async def test_temporal_filter_keeps_later_ending_fact(executor_factory, operations):
    executor = executor_factory(50)
    executor.add('later-ending', CUTOFF + timedelta(seconds=1), 2)

    edges = await retrieve(executor, temporal_filter(), operations)

    assert [edge.uuid for edge in edges] == ['later-ending', 'current']


@pytest.mark.parametrize('operations', [False, True])
async def test_temporal_filter_still_limits_eligible_hits_in_score_order(
    executor_factory, operations
):
    executor = executor_factory(0)
    for i in range(35):
        executor.add(f'eligible-{i}', None, 100 - i)

    edges = await retrieve(executor, temporal_filter(), operations)

    assert [edge.uuid for edge in edges] == [f'eligible-{i}' for i in range(20)]


@pytest.mark.parametrize('operations', [False, True])
async def test_unfiltered_search_preserves_index_limit(executor_factory, operations):
    executor = executor_factory(50)

    edges = await retrieve(executor, SearchFilters(), operations)

    assert [edge.uuid for edge in edges] == [f'old-{i}' for i in range(20)]
    assert '{limit: $limit}' in executor.cypher


@pytest.mark.parametrize(
    'provider', [GraphProvider.FALKORDB, GraphProvider.KUZU, GraphProvider.NEPTUNE]
)
def test_other_providers_preserve_relationship_query(provider):
    default_query = get_relationships_query('edge_name_and_fact', 20, provider)
    assert (
        get_relationships_query('edge_name_and_fact', 20, provider, apply_index_limit=False)
        == default_query
    )
