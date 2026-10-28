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

# SpannerDriver against the Spanner emulator (or a real instance); see conftest.py.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner -q

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.spanner_v1 import param_types

from graphiti_core.driver.spanner import SpannerDriver
from graphiti_core.driver.spanner.errors import SpannerCypherNotSupportedError, SpannerDriverError
from graphiti_core.driver.spanner.mutations import Mutation, delete, upsert
from graphiti_core.driver.spanner.rows import (
    COMMUNITY_EDGE_COLUMNS,
    COMMUNITY_NODE_COLUMNS,
    ENTITY_EDGE_COLUMNS,
    ENTITY_NODE_COLUMNS,
    EPISODIC_EDGE_COLUMNS,
    EPISODIC_NODE_COLUMNS,
    HAS_EPISODE_EDGE_COLUMNS,
    NEXT_EPISODE_EDGE_COLUMNS,
    SAGA_NODE_COLUMNS,
    community_edge_from_row,
    community_edge_to_row,
    community_node_from_row,
    community_node_to_row,
    entity_edge_from_row,
    entity_edge_to_row,
    entity_node_from_row,
    entity_node_to_row,
    episodic_edge_from_row,
    episodic_edge_to_row,
    episodic_node_from_row,
    episodic_node_to_row,
    has_episode_edge_from_row,
    has_episode_edge_to_row,
    next_episode_edge_from_row,
    next_episode_edge_to_row,
    saga_node_from_row,
    saga_node_to_row,
)
from graphiti_core.driver.spanner.schema import (
    INDEX_NAMES,
    KIND_INDEX,
    KIND_SEARCH_INDEX,
    SEARCH_INDEX_NAMES,
    build_ddl,
    build_drop_ddl,
    ddl_object_name,
    edge_select_columns,
    node_select_columns,
)
from graphiti_core.driver.spanner.transaction import SpannerTransaction
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from tests.driver.spanner.conftest import TEST_EMBEDDING_DIM

# Emulator-backed, like the repository's database-backed suites; skipped without the
# emulator by the fixtures, so no `integration` marker (which would exclude them from CI).
pytestmark = pytest.mark.asyncio(loop_scope='session')

T0 = datetime(2026, 10, 28, 12, 30, 15, 123456, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=1)
T2 = T0 + timedelta(hours=2)
# 0.5 and 0.25 are exact in FLOAT32, so embeddings compare equal after a round trip.
EMBEDDING = [0.5] * (TEST_EMBEDDING_DIM - 1) + [0.25]

TABLE_NAMES_SQL = "SELECT TABLE_NAME AS name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ''"


def _group() -> str:
    return f'group_{uuid4().hex[:8]}'


def _entity(group_id: str, name: str = 'Alice', **kwargs: Any) -> EntityNode:
    return EntityNode(
        name=name,
        group_id=group_id,
        labels=['Person'],
        summary=f'{name} is an engineer',
        created_at=T0,
        attributes={'age': 30},
        **kwargs,
    )


def _entity_edge(group_id: str, source: EntityNode, target: EntityNode) -> EntityEdge:
    return EntityEdge(
        group_id=group_id,
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='WORKS_WITH',
        fact=f'{source.name} works with {target.name}',
        fact_embedding=EMBEDDING,
        episodes=['episode-1'],
        created_at=T0,
        valid_at=T1,
        attributes={'since': 2020},
    )


def _upsert_entities(nodes: list[EntityNode]) -> Mutation:
    return upsert('Entity', [entity_node_to_row(node) for node in nodes], ENTITY_NODE_COLUMNS)


async def _entity_uuids(driver: SpannerDriver, uuids: list[str]) -> set[str]:
    records = await driver.read(
        'SELECT uuid FROM Entity WHERE uuid IN UNNEST(@uuids)', {'uuids': uuids}
    )
    return {record['uuid'] for record in records}


async def test_build_indices_and_constraints_is_idempotent(spanner_driver: SpannerDriver):
    expected = {ddl_object_name(s) for s in build_ddl(TEST_EMBEDDING_DIM, 'graphiti')}
    assert expected <= await spanner_driver.existing_schema_objects()
    assert await spanner_driver.build_indices_and_constraints() == []


async def test_close_is_fast_and_idempotent(spanner_driver: SpannerDriver):
    # Database.close() in google-cloud-spanner 3.71 can block for 10 minutes; see
    # driver.close_database_in_background.
    driver = SpannerDriver(spanner_driver.config)
    assert await driver.read('SELECT 1 AS ok') == [{'ok': 1}]
    started = time.perf_counter()
    await driver.close()
    await driver.close()
    assert time.perf_counter() - started < 10
    with pytest.raises(SpannerDriverError, match='closed'):
        await driver.read('SELECT 1 AS ok')


async def test_execute_query_rejects_cypher(spanner_driver: SpannerDriver):
    with pytest.raises(SpannerCypherNotSupportedError):
        await spanner_driver.execute_query('MATCH (n:Entity {uuid: $uuid}) RETURN n', uuid='x')


async def test_upsert_then_read_returns_plain_python_types(spanner_driver: SpannerDriver):
    node = _entity(_group(), name_embedding=EMBEDDING)
    await spanner_driver.apply_mutations([_upsert_entities([node])])

    records = await spanner_driver.read(
        'SELECT uuid, name, labels, attributes, created_at FROM Entity WHERE uuid = @uuid',
        {'uuid': node.uuid},
    )
    assert len(records) == 1
    record = records[0]
    assert type(record['attributes']) is dict
    assert record['attributes'] == {'age': 30}
    assert type(record['labels']) is list
    assert record['labels'] == ['Person', 'Entity']
    assert type(record['created_at']) is datetime
    assert record['created_at'] == T0
    assert record['created_at'].tzinfo is not None

    # A float query vector is inferred as ARRAY<FLOAT32>, matching the column type.
    distance = await spanner_driver.read(
        'SELECT COSINE_DISTANCE(name_embedding, @vector) AS distance FROM Entity '
        'WHERE uuid = @uuid',
        {'uuid': node.uuid, 'vector': EMBEDDING},
    )
    assert distance[0]['distance'] == pytest.approx(0.0, abs=1e-6)

    # NULL parameters need an explicit type.
    records = await spanner_driver.read(
        'SELECT @cursor IS NULL AS is_null', {'cursor': None}, {'cursor': param_types.STRING}
    )
    assert records == [{'is_null': True}]


async def test_execute_query_gql_and_dml(spanner_driver: SpannerDriver):
    group_id = _group()
    alice, bob = _entity(group_id, 'Alice'), _entity(group_id, 'Bob')
    await spanner_driver.apply_mutations([_upsert_entities([alice, bob])])

    records, header, summary = await spanner_driver.execute_query(
        'GRAPH graphiti MATCH (n:Entity) WHERE n.uuid IN UNNEST(@uuids) '
        'RETURN n.uuid AS uuid, n.name AS name ORDER BY name',
        uuids=[alice.uuid, bob.uuid],
    )
    assert header == ['uuid', 'name']
    assert summary is None
    assert sorted(records, key=lambda r: r['name']) == [
        {'uuid': alice.uuid, 'name': 'Alice'},
        {'uuid': bob.uuid, 'name': 'Bob'},
    ]

    result = await spanner_driver.execute_query(
        'UPDATE Entity SET summary = @summary WHERE uuid = @uuid',
        summary='updated',
        uuid=alice.uuid,
        routing_='w',
        database_='ignored',
    )
    assert result == ([], [], None)
    records, _, _ = await spanner_driver.execute_query(
        'SELECT summary FROM Entity WHERE uuid = @uuid', params={'uuid': alice.uuid}
    )
    assert records == [{'summary': 'updated'}]

    empty, header, _ = await spanner_driver.execute_query(
        'SELECT uuid FROM Entity WHERE uuid = @uuid', uuid='does-not-exist'
    )
    assert (empty, header) == ([], [])


async def test_transaction_commits_on_exit_and_discards_on_exception(
    spanner_driver: SpannerDriver,
):
    group_id = _group()
    committed, discarded = _entity(group_id, 'Kept'), _entity(group_id, 'Dropped')

    async with spanner_driver.transaction() as tx:
        assert isinstance(tx, SpannerTransaction)
        await spanner_driver.apply_mutations([_upsert_entities([committed])], tx=tx)
        assert len(tx.mutations) == 1
        # Buffered, not applied yet.
        assert await _entity_uuids(spanner_driver, [committed.uuid]) == set()
    assert await _entity_uuids(spanner_driver, [committed.uuid]) == {committed.uuid}

    with pytest.raises(RuntimeError, match='boom'):
        async with spanner_driver.transaction() as tx:
            await spanner_driver.apply_mutations([_upsert_entities([discarded])], tx=tx)
            raise RuntimeError('boom')
    assert await _entity_uuids(spanner_driver, [discarded.uuid]) == set()

    # Deletes are buffered the same way.
    async with spanner_driver.transaction() as tx:
        await spanner_driver.apply_mutations([delete('Entity', [committed.uuid])], tx=tx)
    assert await _entity_uuids(spanner_driver, [committed.uuid]) == set()


async def test_session_execute_write_passes_transaction_and_commits(
    spanner_driver: SpannerDriver,
):
    group_id = _group()
    node = _entity(group_id)
    seen: dict[str, Any] = {}

    async def write(tx: Any, nodes: list[EntityNode], driver: SpannerDriver | None = None) -> str:
        seen['tx'] = tx
        seen['driver'] = driver
        assert driver is not None
        await driver.apply_mutations([_upsert_entities(nodes)], tx=tx)
        return 'done'

    async with spanner_driver.session() as session:
        result = await session.execute_write(write, [node], driver=spanner_driver)
    assert result == 'done'
    assert isinstance(seen['tx'], SpannerTransaction)
    assert seen['driver'] is spanner_driver
    assert await _entity_uuids(spanner_driver, [node.uuid]) == {node.uuid}

    failed = _entity(group_id, 'Failed')

    async def failing(tx: Any, driver: SpannerDriver | None = None) -> None:
        assert driver is not None
        await driver.apply_mutations([_upsert_entities([failed])], tx=tx)
        raise ValueError('nope')

    with pytest.raises(ValueError, match='nope'):
        await spanner_driver.session().execute_write(failing, driver=spanner_driver)
    assert await _entity_uuids(spanner_driver, [failed.uuid]) == set()


async def test_execute_partitioned_dml_deletes_by_group(spanner_driver: SpannerDriver):
    keep, drop = _group(), _group()
    kept_nodes = [_entity(keep, f'keep{i}') for i in range(2)]
    dropped_nodes = [_entity(drop, f'drop{i}') for i in range(3)]
    edge = _entity_edge(drop, dropped_nodes[0], dropped_nodes[1])
    await spanner_driver.apply_mutations(
        [
            _upsert_entities(kept_nodes + dropped_nodes),
            upsert('RelatesTo', [entity_edge_to_row(edge)], ENTITY_EDGE_COLUMNS),
        ]
    )

    # Edge tables first, then node tables (see spec section 6).
    edges_deleted = await spanner_driver.execute_partitioned_dml(
        'DELETE FROM RelatesTo WHERE group_id = @group_id', {'group_id': drop}
    )
    nodes_deleted = await spanner_driver.execute_partitioned_dml(
        'DELETE FROM Entity WHERE group_id = @group_id', {'group_id': drop}
    )
    assert edges_deleted == 1
    assert nodes_deleted == 3
    records = await spanner_driver.read(
        'SELECT group_id, COUNT(*) AS n FROM Entity WHERE group_id IN UNNEST(@groups) '
        'GROUP BY group_id',
        {'groups': [keep, drop]},
    )
    assert records == [{'group_id': keep, 'n': 2}]


async def test_with_database_points_at_another_database(
    spanner_driver: SpannerDriver, scratch_driver: SpannerDriver
):
    assert scratch_driver.database.database_id != spanner_driver.database.database_id
    assert scratch_driver._database == scratch_driver.database.database_id
    assert scratch_driver.config.database == scratch_driver.database.database_id
    assert len(await spanner_driver.read(TABLE_NAMES_SQL)) == 9
    assert await scratch_driver.read(TABLE_NAMES_SQL) == []

    # DDL and DML through execute_query land in the scratch database only.
    await scratch_driver.execute_query('CREATE TABLE Probe (id INT64) PRIMARY KEY (id)')
    await scratch_driver.execute_query('INSERT INTO Probe (id) VALUES (@id)', id=7)
    records, _, _ = await scratch_driver.execute_query('SELECT id FROM Probe')
    assert records == [{'id': 7}]
    assert {r['name'] for r in await scratch_driver.read(TABLE_NAMES_SQL)} == {'Probe'}
    assert 'Probe' not in {r['name'] for r in await spanner_driver.read(TABLE_NAMES_SQL)}


async def test_schema_lifecycle_on_fresh_database(scratch_driver: SpannerDriver):
    driver = scratch_driver
    ddl = build_ddl(TEST_EMBEDDING_DIM, 'graphiti')
    assert await driver.build_indices_and_constraints() == ddl
    assert await driver.build_indices_and_constraints() == []

    dropped = await driver.delete_all_indexes()
    assert {ddl_object_name(s)[1] for s in dropped} == {*INDEX_NAMES, *SEARCH_INDEX_NAMES}
    reapplied = await driver.build_indices_and_constraints()
    assert {ddl_object_name(s) for s in reapplied} == {
        *((KIND_INDEX, name) for name in INDEX_NAMES),
        *((KIND_SEARCH_INDEX, name) for name in SEARCH_INDEX_NAMES),
    }

    group_id = _group()
    alice, bob = _entity(group_id, 'Alice'), _entity(group_id, 'Bob')
    edge = _entity_edge(group_id, alice, bob)
    await driver.apply_mutations(
        [
            _upsert_entities([alice, bob]),
            upsert('RelatesTo', [entity_edge_to_row(edge)], ENTITY_EDGE_COLUMNS),
        ]
    )
    assert await driver.clear_tables() >= 3
    assert await _entity_uuids(driver, [alice.uuid, bob.uuid]) == set()

    await driver.apply_mutations([_upsert_entities([alice])])
    rebuilt = await driver.build_indices_and_constraints(delete_existing=True)
    assert rebuilt == build_drop_ddl('graphiti') + ddl
    assert await _entity_uuids(driver, [alice.uuid]) == set()
    assert await driver.build_indices_and_constraints() == []


async def test_apply_mutations_chunks_large_batches(spanner_driver: SpannerDriver):
    group_id = _group()
    nodes = [
        _entity(group_id, f'n{i}') for i in range(spanner_driver.config.mutation_chunk_rows + 7)
    ]
    await spanner_driver.apply_mutations([_upsert_entities(nodes)])
    records = await spanner_driver.read(
        'SELECT COUNT(*) AS n FROM Entity WHERE group_id = @group_id', {'group_id': group_id}
    )
    assert records == [{'n': len(nodes)}]


async def test_edge_insert_without_source_node_fails(spanner_driver: SpannerDriver):
    group_id = _group()
    target = _entity(group_id, 'Target')
    await spanner_driver.apply_mutations([_upsert_entities([target])])
    missing = _entity(group_id, 'Missing')  # never written
    edge = _entity_edge(group_id, missing, target)
    with pytest.raises(GoogleAPICallError):
        await spanner_driver.apply_mutations(
            [upsert('RelatesTo', [entity_edge_to_row(edge)], ENTITY_EDGE_COLUMNS)]
        )


async def test_gql_edge_properties_use_graphiti_names(spanner_driver: SpannerDriver):
    group_id = _group()
    alice, bob = _entity(group_id, 'Alice'), _entity(group_id, 'Bob')
    edge = _entity_edge(group_id, alice, bob)
    await spanner_driver.apply_mutations(
        [
            _upsert_entities([alice, bob]),
            upsert('RelatesTo', [entity_edge_to_row(edge)], ENTITY_EDGE_COLUMNS),
        ]
    )
    records, header, _ = await spanner_driver.execute_query(
        'GRAPH graphiti MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) '
        'WHERE a.group_id = @group_id '
        'RETURN e.uuid AS uuid, e.source_node_uuid AS src, e.target_node_uuid AS dst',
        group_id=group_id,
    )
    assert header == ['uuid', 'src', 'dst']
    assert records == [{'uuid': edge.uuid, 'src': alice.uuid, 'dst': bob.uuid}]

    # Full-text SEARCH over the edge's TOKENLIST property also works in GQL.
    records, _, _ = await spanner_driver.execute_query(
        'GRAPH graphiti MATCH ()-[e:RELATES_TO]->() '
        'WHERE SEARCH(e.search_tokens, @query) AND e.group_id IN UNNEST(@group_ids) '
        'RETURN e.uuid AS uuid',
        query='works',
        group_ids=[group_id],
    )
    assert records == [{'uuid': edge.uuid}]


async def test_rows_round_trip_through_database(spanner_driver: SpannerDriver):
    group_id = _group()
    entity_a = _entity(group_id, 'Alice', name_embedding=EMBEDDING)
    entity_b = _entity(group_id, 'Bob')
    episode_1 = EpisodicNode(
        name='episode one',
        group_id=group_id,
        source=EpisodeType.message,
        source_description='chat',
        content='user: Alice met Bob',
        valid_at=T1,
        created_at=T0,
        entity_edges=['x', 'y'],
        episode_metadata={'channel': 'web'},
    )
    episode_2 = EpisodicNode(
        name='episode two',
        group_id=group_id,
        source=EpisodeType.text,
        source_description='doc',
        content='Bob wrote back',
        valid_at=T2,
        created_at=T0,
    )
    community = CommunityNode(
        name='engineers', group_id=group_id, name_embedding=EMBEDDING, summary='s', created_at=T0
    )
    saga = SagaNode(
        name='saga',
        group_id=group_id,
        summary='a saga',
        first_episode_uuid=episode_1.uuid,
        last_episode_uuid=episode_2.uuid,
        last_summarized_at=T2,
        last_summarized_episode_valid_at=T1,
        created_at=T0,
    )
    entity_edge = _entity_edge(group_id, entity_a, entity_b)
    simple = {'group_id': group_id, 'created_at': T0}
    mention = EpisodicEdge(
        source_node_uuid=episode_1.uuid, target_node_uuid=entity_a.uuid, **simple
    )
    member = CommunityEdge(
        source_node_uuid=community.uuid, target_node_uuid=entity_a.uuid, **simple
    )
    has_episode = HasEpisodeEdge(
        source_node_uuid=saga.uuid, target_node_uuid=episode_1.uuid, **simple
    )
    next_episode = NextEpisodeEdge(
        source_node_uuid=episode_1.uuid, target_node_uuid=episode_2.uuid, **simple
    )

    await spanner_driver.apply_mutations(
        [
            _upsert_entities([entity_a, entity_b]),
            upsert(
                'Episodic',
                [episodic_node_to_row(episode_1), episodic_node_to_row(episode_2)],
                EPISODIC_NODE_COLUMNS,
            ),
            upsert('Community', [community_node_to_row(community)], COMMUNITY_NODE_COLUMNS),
            upsert('Saga', [saga_node_to_row(saga)], SAGA_NODE_COLUMNS),
            upsert('RelatesTo', [entity_edge_to_row(entity_edge)], ENTITY_EDGE_COLUMNS),
            upsert('Mentions', [episodic_edge_to_row(mention)], EPISODIC_EDGE_COLUMNS),
            upsert('HasMember', [community_edge_to_row(member)], COMMUNITY_EDGE_COLUMNS),
            upsert('HasEpisode', [has_episode_edge_to_row(has_episode)], HAS_EPISODE_EDGE_COLUMNS),
            upsert(
                'NextEpisode', [next_episode_edge_to_row(next_episode)], NEXT_EPISODE_EDGE_COLUMNS
            ),
        ]
    )

    node_cases: list[tuple[str, Any, Any]] = [
        ('Entity', entity_a, entity_node_from_row),
        ('Entity', entity_b, entity_node_from_row),
        ('Episodic', episode_1, episodic_node_from_row),
        ('Episodic', episode_2, episodic_node_from_row),
        ('Community', community, community_node_from_row),
        ('Saga', saga, saga_node_from_row),
    ]
    for table, model, from_row in node_cases:
        records = await spanner_driver.read(
            f'SELECT {node_select_columns(table)} FROM {table} WHERE uuid = @uuid',
            {'uuid': model.uuid},
        )
        expected = model.model_dump()
        if table == 'Entity':
            expected['labels'] = model.labels + ['Entity']
        assert from_row(records[0]).model_dump() == expected, table

    edge_cases: list[tuple[str, Any, Any]] = [
        ('RelatesTo', entity_edge, entity_edge_from_row),
        ('Mentions', mention, episodic_edge_from_row),
        ('HasMember', member, community_edge_from_row),
        ('HasEpisode', has_episode, has_episode_edge_from_row),
        ('NextEpisode', next_episode, next_episode_edge_from_row),
    ]
    for table, model, from_row in edge_cases:
        records = await spanner_driver.read(
            f'SELECT {edge_select_columns(table)} FROM {table} WHERE edge_uuid = @uuid',
            {'uuid': model.uuid},
        )
        assert from_row(records[0]).model_dump() == model.model_dump(), table
        raw = await spanner_driver.read(
            f'SELECT * FROM {table} WHERE edge_uuid = @uuid', {'uuid': model.uuid}
        )
        assert from_row(raw[0]).model_dump() == model.model_dump(), table

    # The same edge read through GQL property names.
    fields = [
        'uuid',
        'source_node_uuid',
        'target_node_uuid',
        'group_id',
        'name',
        'fact',
        'fact_embedding',
        'episodes',
        'attributes',
        'created_at',
        'valid_at',
        'invalid_at',
        'expired_at',
        'reference_time',
    ]
    records, _, _ = await spanner_driver.execute_query(
        'GRAPH graphiti MATCH ()-[e:RELATES_TO]->() WHERE e.uuid = @uuid RETURN '
        + ', '.join(f'e.{field} AS {field}' for field in fields),
        uuid=entity_edge.uuid,
    )
    assert entity_edge_from_row(records[0]).model_dump() == entity_edge.model_dump()
