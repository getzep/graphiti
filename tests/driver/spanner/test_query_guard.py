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

# Hermetic tests for the execute_query Cypher guard, statement routing, and driver wiring.
# Constructing SpannerDriver opens no connection, so no emulator is needed here.

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.spanner.driver import classify_statement, ensure_not_cypher
from graphiti_core.driver.spanner.errors import (
    SpannerCypherNotSupportedError,
    SpannerDriverError,
    query_snippet,
)
from graphiti_core.driver.spanner.legacy import (
    SpannerLegacyGraphOperations,
    SpannerLegacySearchInterface,
)
from graphiti_core.driver.spanner.operations.base import spanner_driver, spanner_transaction
from graphiti_core.driver.spanner.session import SpannerSession
from graphiti_core.driver.spanner.transaction import SpannerTransaction
from graphiti_core.driver.spanner_driver import SpannerConfig, SpannerDriver

CYPHER = [
    'MATCH (n:Entity {uuid: $uuid}) RETURN n',
    '\n        MATCH (n:Entity)\n        WHERE n.uuid IN $uuids\n        RETURN n\n    ',
    'MERGE (n:Entity {uuid: "x"}) SET n.name = "y"',
    'UNWIND [1, 2] AS x RETURN x',
    'MATCH (n:Entity) DETACH DELETE n',
    'CREATE (n:Entity {uuid: "x"})',
    'match (n) return n',
    'CALL db.index.fulltext.queryNodes("node_name_and_summary", $query) YIELD node',
    'SELECT uuid FROM Entity WHERE uuid = $uuid',
]

NOT_CYPHER = [
    'GRAPH graphiti MATCH (n:Entity) WHERE n.uuid IN UNNEST(@uuids) RETURN n.uuid AS uuid',
    '  -- leading comment\nGRAPH graphiti MATCH (n) RETURN n.uuid AS uuid',
    '/* hint */ @{USE_ADDITIONAL_PARALLELISM=TRUE} GRAPH graphiti MATCH (n) RETURN n.uuid',
    'SELECT uuid, name FROM Entity WHERE uuid = @uuid',
    "SELECT JSON_VALUE(attributes, '$.age') AS age FROM Entity",
    'SELECT * FROM GRAPH_TABLE(graphiti MATCH (n:Entity) COLUMNS (n.uuid AS uuid))',
    'WITH t AS (SELECT 1 AS x) SELECT x FROM t',
    'DELETE FROM Entity WHERE group_id = @group_id',
    'INSERT OR UPDATE INTO Entity (uuid) VALUES (@uuid)',
    'CREATE TABLE Foo (a INT64) PRIMARY KEY (a)',
    'SELECT match_score FROM Scores',
]


@pytest.mark.parametrize('query', CYPHER)
def test_cypher_is_rejected(query: str):
    with pytest.raises(SpannerCypherNotSupportedError) as info:
        ensure_not_cypher(query)
    assert query_snippet(query) in str(info.value)
    assert isinstance(info.value, SpannerDriverError)
    assert info.value.query == query


@pytest.mark.parametrize('query', NOT_CYPHER)
def test_sql_and_gql_are_accepted(query: str):
    ensure_not_cypher(query)


def test_error_message_has_first_120_characters():
    query = 'MATCH (n:Entity) WHERE n.uuid = $uuid RETURN ' + ', '.join(
        f'n.field_{i} AS field_{i}' for i in range(40)
    )
    with pytest.raises(SpannerCypherNotSupportedError) as info:
        ensure_not_cypher(query)
    snippet = query_snippet(query)
    assert len(snippet) == 120
    assert str(info.value).endswith(snippet)
    assert query[:200] not in str(info.value)


@pytest.mark.parametrize(
    ('query', 'kind'),
    [
        ('SELECT 1', 'read'),
        ('  select 1', 'read'),
        ('WITH t AS (SELECT 1 AS x) SELECT x FROM t', 'read'),
        ('GRAPH graphiti MATCH (n) RETURN n.uuid', 'read'),
        ('-- c\nGRAPH graphiti MATCH (n) RETURN n.uuid', 'read'),
        ('(SELECT 1) UNION ALL (SELECT 2)', 'read'),
        ('INSERT INTO Entity (uuid) VALUES (@u)', 'dml'),
        ('UPDATE Entity SET name = @n WHERE uuid = @u', 'dml'),
        ('DELETE FROM Entity WHERE TRUE', 'dml'),
        ('CREATE INDEX X ON Entity(name)', 'ddl'),
        ('DROP INDEX X', 'ddl'),
        ('ALTER TABLE Entity ADD COLUMN x INT64', 'ddl'),
    ],
)
def test_classify_statement(query: str, kind: str):
    assert classify_statement(query) == kind


@pytest.fixture
def offline_driver(monkeypatch: pytest.MonkeyPatch) -> SpannerDriver:
    # monkeypatch keeps the emulator variable from leaking into other tests.
    monkeypatch.setenv('SPANNER_EMULATOR_HOST', 'localhost:1')
    return SpannerDriver(project='offline', instance='instance', database='database')


async def test_execute_query_rejects_cypher_before_any_io(offline_driver: SpannerDriver):
    with pytest.raises(SpannerCypherNotSupportedError):
        await offline_driver.execute_query('MATCH (n:Entity {uuid: $uuid}) RETURN n', uuid='x')
    await offline_driver.close()


async def test_execute_query_requires_types_for_none(offline_driver: SpannerDriver):
    with pytest.raises(ValueError, match="'cursor'"):
        await offline_driver.execute_query('SELECT 1 WHERE @cursor IS NULL', cursor=None)
    await offline_driver.close()


def test_driver_wiring(offline_driver: SpannerDriver):
    driver = offline_driver
    assert driver.provider == GraphProvider.SPANNER
    assert driver.fulltext_syntax == ''
    assert driver.default_group_id == ''
    assert driver._database == 'database'
    assert driver.database.database_id == 'database'
    assert driver.instance.instance_id == 'instance'
    assert driver.config.emulator_host == 'localhost:1'
    expected = {
        'entity_node_ops': EntityNodeOperations,
        'episode_node_ops': EpisodeNodeOperations,
        'community_node_ops': CommunityNodeOperations,
        'saga_node_ops': SagaNodeOperations,
        'entity_edge_ops': EntityEdgeOperations,
        'episodic_edge_ops': EpisodicEdgeOperations,
        'community_edge_ops': CommunityEdgeOperations,
        'has_episode_edge_ops': HasEpisodeEdgeOperations,
        'next_episode_edge_ops': NextEpisodeEdgeOperations,
        'search_ops': SearchOperations,
        'graph_ops': GraphMaintenanceOperations,
    }
    for name, abc in expected.items():
        ops = getattr(driver, name)
        assert isinstance(ops, abc), name
        assert type(ops).__name__.startswith('Spanner'), name
    assert isinstance(driver.graph_operations_interface, SpannerLegacyGraphOperations)
    assert isinstance(driver.search_interface, SpannerLegacySearchInterface)
    assert driver.clone('other') is driver
    assert driver.build_fulltext_query('Alice (Bob)') == 'alice OR bob'
    session = driver.session()
    assert isinstance(session, SpannerSession)
    assert session.provider == GraphProvider.SPANNER
    assert session.driver is driver


def test_with_database_points_at_another_database(offline_driver: SpannerDriver):
    other = offline_driver.with_database('other_db')
    assert other is not offline_driver
    assert other.database.database_id == 'other_db'
    assert other._database == 'other_db'
    assert other.config.database == 'other_db'
    assert offline_driver.database.database_id == 'database'
    assert other.instance is offline_driver.instance
    assert offline_driver.session('other_db').driver.database.database_id == 'other_db'


def test_config_from_kwargs_and_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('SPANNER_PROJECT', raising=False)
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'from-gcp-env')
    monkeypatch.setenv('SPANNER_EMULATOR_HOST', 'localhost:1')
    config = SpannerConfig(instance='i', database='d')
    assert config.project == 'from-gcp-env'
    assert config.emulator_host == 'localhost:1'
    assert config.graph_name == 'graphiti'
    monkeypatch.setenv('SPANNER_PROJECT', 'from-spanner-env')
    assert SpannerConfig(instance='i', database='d').project == 'from-spanner-env'
    assert SpannerConfig(project='explicit', instance='i', database='d').project == 'explicit'
    driver = SpannerDriver(config, embedding_dim=None, graph_name='g2')
    assert driver.embedding_dim is None
    assert driver.graph_name == 'g2'
    with pytest.raises(ValueError):
        SpannerConfig(instance='i', database='d', graph_name='bad-name')


def test_ops_helpers_validate_types(offline_driver: SpannerDriver):
    assert spanner_driver(offline_driver) is offline_driver
    with pytest.raises(SpannerDriverError):
        spanner_driver(object())  # type: ignore[arg-type]
    tx = SpannerTransaction(offline_driver)
    assert spanner_transaction(tx) is tx
    assert spanner_transaction(None) is None
