# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract, unit, and embedded integration tests for RelateDriver."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

relate = pytest.importorskip('relate')

from graphiti_core.driver.driver import GraphDriver, GraphProvider  # noqa: E402
from graphiti_core.driver.operations import (  # noqa: E402
    CommunityEdgeOperations,
    CommunityNodeOperations,
    EntityEdgeOperations,
    EntityNodeOperations,
    EpisodeNodeOperations,
    EpisodicEdgeOperations,
    GraphMaintenanceOperations,
    HasEpisodeEdgeOperations,
    NextEpisodeEdgeOperations,
    SagaNodeOperations,
    SearchOperations,
)
from graphiti_core.driver.relate import RELATE_SESSION_SOURCE_REVISION  # noqa: E402
from graphiti_core.driver.relate import _session as vendored_session  # noqa: E402
from graphiti_core.driver.relate.translation import (  # noqa: E402
    RELATE_MAX_BFS_DEPTH,
    translate_relate_query,
)
from graphiti_core.driver.relate_driver import RelateDriver, RelateDriverSession  # noqa: E402
from graphiti_core.edges import (  # noqa: E402
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError  # noqa: E402
from graphiti_core.models.nodes.node_db_queries import (  # noqa: E402
    get_entity_node_save_bulk_query,
    get_entity_node_update_bulk_query,
)
from graphiti_core.nodes import (  # noqa: E402
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    Node,
    SagaNode,
)
from graphiti_core.search.search_filters import SearchFilters  # noqa: E402
from graphiti_core.search.search_utils import (  # noqa: E402
    edge_bfs_search as utils_edge_bfs_search,
)
from graphiti_core.search.search_utils import (  # noqa: E402
    edge_fulltext_search as utils_edge_fulltext_search,
)
from graphiti_core.search.search_utils import (  # noqa: E402
    node_bfs_search as utils_node_bfs_search,
)
from graphiti_core.search.search_utils import (  # noqa: E402
    node_fulltext_search as utils_node_fulltext_search,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk  # noqa: E402

EXPECTED_SESSION_EXPORTS = {
    'Column',
    'CoreError',
    'LeaseOwnershipError',
    'Node',
    'Path',
    'QueryResult',
    'ReadOnlyStoreError',
    'Record',
    'Relationship',
    'RelateStore',
    'RelateTransaction',
    'ResultNormalizationError',
    'StoreClosedError',
    'StoreState',
    'Temporal',
    'TransactionOutcomeIndeterminateError',
    'TransactionState',
    'TransactionStateError',
    'UpdateStatistics',
    'Value',
    'Vector',
}


class _StubEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [1.0, 0.0, 0.0]

    async def create_batch(self, input_data_list):
        return [[1.0, 0.0, 0.0] for _ in input_data_list]


@pytest.fixture
async def driver():
    instance = await RelateDriver.open(':memory:')
    try:
        yield instance
    finally:
        await instance.close()


# --- Vendored _session contract & sync tests --------------------------------


def test_vendored_session_exports_and_revision():
    assert vendored_session._SOURCE_REVISION == '5117e8d'
    assert RELATE_SESSION_SOURCE_REVISION == '5117e8d'
    assert set(vendored_session.__all__) == EXPECTED_SESSION_EXPORTS | {'__version__'}
    assert {name for name in vars(vendored_session) if not name.startswith('_')} == (
        EXPECTED_SESSION_EXPORTS
    )
    assert not hasattr(vendored_session, 'connect')
    assert not hasattr(vendored_session, 'Connection')
    assert not hasattr(vendored_session, 'Cursor')


# --- Provider & operations identity tests -----------------------------------


async def test_provider_identity_and_operations(driver: RelateDriver):
    assert GraphDriver.RELATE == GraphProvider.RELATE
    assert GraphProvider.RELATE.value == 'relate'
    assert driver.provider == GraphDriver.RELATE
    assert driver.default_group_id == ''
    assert driver.fulltext_syntax == ''
    assert driver.aoss_client is None

    assert isinstance(driver.entity_node_ops, EntityNodeOperations)
    assert isinstance(driver.episode_node_ops, EpisodeNodeOperations)
    assert isinstance(driver.community_node_ops, CommunityNodeOperations)
    assert isinstance(driver.saga_node_ops, SagaNodeOperations)
    assert isinstance(driver.entity_edge_ops, EntityEdgeOperations)
    assert isinstance(driver.episodic_edge_ops, EpisodicEdgeOperations)
    assert isinstance(driver.community_edge_ops, CommunityEdgeOperations)
    assert isinstance(driver.has_episode_edge_ops, HasEpisodeEdgeOperations)
    assert isinstance(driver.next_episode_edge_ops, NextEpisodeEdgeOperations)
    assert isinstance(driver.search_ops, SearchOperations)
    assert isinstance(driver.graph_ops, GraphMaintenanceOperations)

    await driver.health_check()


# --- Query translation unit tests -------------------------------------------


def test_translate_vector_procedures():
    entity_query = """
        MERGE (n:Entity {uuid: $entity_data.uuid})
        SET n:Entity
        SET n = $entity_data
        WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)
        RETURN n.uuid AS uuid
    """
    translated_entity = translate_relate_query(entity_query)
    assert 'setNodeVectorProperty' not in translated_entity
    assert 'SET n.name_embedding = $entity_data.name_embedding' in translated_entity

    bulk_entity = """
        UNWIND $nodes AS node
        MERGE (n:Entity {uuid: node.uuid})
        SET n:$(node.labels)
        SET n = node
        WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
        RETURN n.uuid AS uuid
    """
    translated_bulk = translate_relate_query(bulk_entity)
    assert 'setNodeVectorProperty' not in translated_bulk
    assert 'SET n.name_embedding = node.name_embedding' in translated_bulk

    edge_query = """
        MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
        SET e = $edge_data
        WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)
        RETURN e.uuid AS uuid
    """
    translated_edge = translate_relate_query(edge_query)
    assert 'setRelationshipVectorProperty' not in translated_edge
    assert 'SET e.fact_embedding = $edge_data.fact_embedding' in translated_edge


def test_translate_fulltext_arity():
    nodes_query = (
        'CALL db.index.fulltext.queryNodes("node_name_and_summary", $query, {limit: $limit})\n'
        'YIELD node AS n, score'
    )
    assert translate_relate_query(nodes_query) == (
        'CALL db.index.fulltext.queryNodes("node_name_and_summary", $query)\nYIELD node AS n, score'
    )

    rels_query = (
        'CALL db.index.fulltext.queryRelationships("edge_name_and_fact", $query, {limit: $limit})\n'
        'YIELD relationship AS rel, score'
    )
    assert translate_relate_query(rels_query) == (
        'CALL db.index.fulltext.queryRelationships("edge_name_and_fact", $query)\n'
        'YIELD relationship AS rel, score'
    )


def test_translate_call_in_transactions_delete():
    query = """
        MATCH (n:Entity {group_id: $group_id})
        CALL (n) {
            DETACH DELETE n
        } IN TRANSACTIONS OF $batch_size ROWS
    """
    translated = translate_relate_query(query)
    assert 'IN TRANSACTIONS' not in translated
    assert 'DETACH DELETE n' in translated


def test_translate_bfs_depth_cap():
    deep = 'MATCH (origin)-[:RELATES_TO|MENTIONS*1..501]->(n:Entity) RETURN n'
    translated = translate_relate_query(deep)
    assert f'*1..{RELATE_MAX_BFS_DEPTH}' in translated

    shallow = 'MATCH (origin)-[:RELATES_TO|MENTIONS*1..3]->(n:Entity) RETURN n'
    assert translate_relate_query(shallow) == shallow


# --- Embedded integration tests ---------------------------------------------


async def test_indices_build_and_delete(driver: RelateDriver):
    await driver.build_indices_and_constraints()

    records, _, _ = await driver.execute_query('SHOW INDEXES')
    user_indexes = {r['name'] for r in records if r.get('type') != 'LOOKUP'}
    assert 'entity_uuid' in user_indexes
    assert 'node_name_and_summary' in user_indexes
    assert 'edge_name_and_fact' in user_indexes
    assert 'episode_content' in user_indexes
    assert 'community_name' in user_indexes

    await driver.delete_all_indexes()
    after_records, _, _ = await driver.execute_query('SHOW INDEXES')
    remaining_user = {r['name'] for r in after_records if r.get('type') != 'LOOKUP'}
    assert remaining_user == set()

    # Rebuild with delete_existing=True succeeds cleanly.
    await driver.build_indices_and_constraints(delete_existing=True)
    rebuilt_records, _, _ = await driver.execute_query('SHOW INDEXES')
    assert any(r['name'] == 'entity_uuid' for r in rebuilt_records)


async def test_node_crud_and_attribute_lookup(driver: RelateDriver):
    await driver.build_indices_and_constraints()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    entity = EntityNode(
        uuid='ent-1',
        name='Alice',
        group_id='grp-1',
        labels=['Entity', 'Person'],
        created_at=now,
        name_embedding=[1.0, 0.0, 0.0],
        summary='Alice summary',
        attributes={'role': 'engineer', 'level': 3},
    )
    await entity.save(driver)

    fetched = await EntityNode.get_by_uuid(driver, 'ent-1')
    assert fetched.name == 'Alice'
    assert set(fetched.labels) == {'Entity', 'Person'}
    assert fetched.attributes == {'role': 'engineer', 'level': 3}
    await fetched.load_name_embedding(driver)
    assert fetched.name_embedding == [1.0, 0.0, 0.0]

    by_attr = await EntityNode.get_by_attribute_values(
        driver, 'grp-1', {'role': 'engineer'}, labels=['Person']
    )
    assert [n.uuid for n in by_attr] == ['ent-1']

    episode = EpisodicNode(
        uuid='ep-1',
        name='Episode 1',
        group_id='grp-1',
        created_at=now,
        source=EpisodeType.message,
        source_description='chat',
        content='Alice works with Bob',
        valid_at=now,
        entity_edges=['edge-1'],
    )
    await driver.episode_node_ops.save(driver, episode)
    got_ep = await driver.episode_node_ops.get_by_uuid(driver, 'ep-1')
    assert got_ep.content == 'Alice works with Bob'
    assert got_ep.source == EpisodeType.message

    community = CommunityNode(
        uuid='comm-1',
        name='Engineering',
        group_id='grp-1',
        created_at=now,
        name_embedding=[0.0, 1.0, 0.0],
        summary='Engineering team',
    )
    await driver.community_node_ops.save(driver, community)
    got_comm = await driver.community_node_ops.get_by_uuid(driver, 'comm-1')
    assert got_comm.name == 'Engineering'
    assert got_comm.name_embedding == [0.0, 1.0, 0.0]

    saga = SagaNode(
        uuid='saga-1',
        name='Main Saga',
        group_id='grp-1',
        created_at=now,
        summary='Saga summary',
        first_episode_uuid='ep-1',
        last_episode_uuid='ep-1',
        last_summarized_at=now,
        last_summarized_episode_valid_at=now,
    )
    await driver.saga_node_ops.save(driver, saga)
    got_saga = await driver.saga_node_ops.get_by_uuid(driver, 'saga-1')
    assert got_saga.summary == 'Saga summary'
    assert got_saga.first_episode_uuid == 'ep-1'

    # Model-level delete_by_uuids and delete_by_group_id must not fail in explicit txn mode.
    await Node.delete_by_uuids(driver, ['comm-1'])
    with pytest.raises(NodeNotFoundError):
        await driver.community_node_ops.get_by_uuid(driver, 'comm-1')

    await Node.delete_by_group_id(driver, 'grp-1')
    with pytest.raises(NodeNotFoundError):
        await driver.entity_node_ops.get_by_uuid(driver, 'ent-1')


async def test_bulk_save_update_and_label_hygiene(driver: RelateDriver):
    now = datetime(2026, 1, 2, 8, 0, tzinfo=timezone.utc)
    n1 = EntityNode(
        uuid='b-1',
        name='Alice',
        group_id='grp-bulk',
        labels=['Entity', 'Person'],
        created_at=now,
        name_embedding=[1.0, 0.0, 0.0],
        summary='v1',
    )
    n2 = EntityNode(
        uuid='b-2',
        name='Bob',
        group_id='grp-bulk',
        labels=['Entity', 'Person'],
        created_at=now,
        name_embedding=[0.0, 1.0, 0.0],
        summary='v1',
    )
    ep = EpisodicNode(
        uuid='ep-bulk',
        name='Bulk Ep',
        group_id='grp-bulk',
        created_at=now,
        source=EpisodeType.text,
        source_description='doc',
        content='Alice mentors Bob',
        valid_at=now,
    )
    ep_edge = EpisodicEdge(
        uuid='ee-bulk',
        group_id='grp-bulk',
        source_node_uuid='ep-bulk',
        target_node_uuid='b-1',
        created_at=now,
    )
    ent_edge = EntityEdge(
        uuid='re-bulk',
        group_id='grp-bulk',
        source_node_uuid='b-1',
        target_node_uuid='b-2',
        name='MENTORS',
        fact='Alice mentors Bob',
        created_at=now,
        reference_time=now,
        fact_embedding=[1.0, 0.0, 0.0],
        episodes=['ep-bulk'],
    )

    await add_nodes_and_edges_bulk(
        driver,
        episodic_nodes=[ep],
        episodic_edges=[ep_edge],
        entity_nodes=[n1, n2],
        entity_edges=[ent_edge],
        embedder=_StubEmbedder(),
    )

    rows, _, _ = await driver.execute_query(
        "MATCH (n:Entity {uuid: 'b-1'}) RETURN labels(n) AS labels, n.labels AS stray"
    )
    assert set(rows[0]['labels']) == {'Entity', 'Person'}
    assert rows[0]['stray'] is None

    # Bulk update modifies existing nodes without creating missing ones.
    payload = [
        {
            'uuid': 'b-1',
            'name': 'Alice Updated',
            'group_id': 'grp-bulk',
            'summary': 'v2',
            'created_at': now,
            'name_embedding': [1.0, 0.0, 0.0],
            'labels': ['Entity', 'Person'],
        },
        {
            'uuid': 'non-existent',
            'name': 'Ghost',
            'group_id': 'grp-bulk',
            'summary': 'never',
            'created_at': now,
            'name_embedding': [0.0, 0.0, 1.0],
            'labels': ['Entity'],
        },
    ]
    save_q = get_entity_node_save_bulk_query(GraphProvider.RELATE, payload)
    update_q = get_entity_node_update_bulk_query(GraphProvider.RELATE, payload)
    assert 'MERGE' in save_q
    assert 'MERGE' not in update_q

    await driver.execute_query(update_q, nodes=payload)
    updated = await driver.entity_node_ops.get_by_uuid(driver, 'b-1')
    assert updated.name == 'Alice Updated'
    with pytest.raises(NodeNotFoundError):
        await driver.entity_node_ops.get_by_uuid(driver, 'non-existent')


async def test_edge_crud_and_embeddings(driver: RelateDriver):
    now = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    n1 = EntityNode(uuid='e-n1', name='A', group_id='g1', created_at=now)
    n2 = EntityNode(uuid='e-n2', name='B', group_id='g1', created_at=now)
    ep1 = EpisodicNode(
        uuid='e-ep1',
        name='Ep1',
        group_id='g1',
        created_at=now,
        source=EpisodeType.text,
        source_description='s',
        content='c1',
        valid_at=now,
    )
    ep2 = EpisodicNode(
        uuid='e-ep2',
        name='Ep2',
        group_id='g1',
        created_at=now + timedelta(minutes=1),
        source=EpisodeType.text,
        source_description='s',
        content='c2',
        valid_at=now + timedelta(minutes=1),
    )
    comm = CommunityNode(uuid='e-c1', name='C', group_id='g1', created_at=now)
    saga = SagaNode(uuid='e-s1', name='S', group_id='g1', created_at=now)

    await driver.entity_node_ops.save_bulk(driver, [n1, n2])
    await driver.episode_node_ops.save_bulk(driver, [ep1, ep2])
    await driver.community_node_ops.save(driver, comm)
    await driver.saga_node_ops.save(driver, saga)

    rel_edge = EntityEdge(
        uuid='rel-1',
        group_id='g1',
        source_node_uuid='e-n1',
        target_node_uuid='e-n2',
        name='KNOWS',
        fact='A knows B',
        created_at=now,
        reference_time=now,
        fact_embedding=[0.2, 0.8, 0.0],
        episodes=['e-ep1'],
    )
    await driver.entity_edge_ops.save(driver, rel_edge)
    got_rel = await driver.entity_edge_ops.get_by_uuid(driver, 'rel-1')
    assert got_rel.fact == 'A knows B'
    assert got_rel.reference_time == now
    await driver.entity_edge_ops.load_embeddings(driver, got_rel)
    assert got_rel.fact_embedding == pytest.approx([0.2, 0.8, 0.0])

    comm_edge = CommunityEdge(
        uuid='ce-1',
        group_id='g1',
        source_node_uuid='e-c1',
        target_node_uuid='e-n1',
        created_at=now,
    )
    await driver.community_edge_ops.save(driver, comm_edge)
    assert (await driver.community_edge_ops.get_by_uuid(driver, 'ce-1')).uuid == 'ce-1'

    has_ep = HasEpisodeEdge(
        uuid='he-1',
        group_id='g1',
        source_node_uuid='e-s1',
        target_node_uuid='e-ep1',
        created_at=now,
    )
    await driver.has_episode_edge_ops.save(driver, has_ep)
    assert (await driver.has_episode_edge_ops.get_by_uuid(driver, 'he-1')).uuid == 'he-1'

    next_ep = NextEpisodeEdge(
        uuid='ne-1',
        group_id='g1',
        source_node_uuid='e-ep1',
        target_node_uuid='e-ep2',
        created_at=now,
    )
    await driver.next_episode_edge_ops.save(driver, next_ep)
    assert (await driver.next_episode_edge_ops.get_by_uuid(driver, 'ne-1')).uuid == 'ne-1'

    await driver.entity_edge_ops.delete(driver, rel_edge)
    with pytest.raises(EdgeNotFoundError):
        await driver.entity_edge_ops.get_by_uuid(driver, 'rel-1')


async def test_search_ops_and_bfs_depth_clamping(driver: RelateDriver):
    await driver.build_indices_and_constraints()
    now = datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc)

    n1 = EntityNode(
        uuid='s-1',
        name='Alice',
        group_id='g-search',
        summary='Graph architect',
        created_at=now,
        name_embedding=[1.0, 0.0, 0.0],
    )
    n2 = EntityNode(
        uuid='s-2',
        name='Bob',
        group_id='g-search',
        summary='Rust engineer',
        created_at=now,
        name_embedding=[0.0, 1.0, 0.0],
    )
    ep = EpisodicNode(
        uuid='s-ep',
        name='EpSearch',
        group_id='g-search',
        created_at=now,
        source=EpisodeType.message,
        source_description='chat',
        content='Alice collaborated with Bob on Relate',
        valid_at=now,
    )
    comm = CommunityNode(
        uuid='s-comm',
        name='Database Team',
        group_id='g-search',
        created_at=now,
        name_embedding=[1.0, 0.0, 0.0],
        summary='Core DB builders',
    )
    await driver.entity_node_ops.save_bulk(driver, [n1, n2])
    await driver.episode_node_ops.save(driver, ep)
    await driver.community_node_ops.save(driver, comm)

    edge = EntityEdge(
        uuid='s-edge',
        group_id='g-search',
        source_node_uuid='s-1',
        target_node_uuid='s-2',
        name='COLLABORATES_WITH',
        fact='Alice collaborated with Bob',
        created_at=now,
        reference_time=now,
        fact_embedding=[1.0, 0.0, 0.0],
        episodes=['s-ep'],
    )
    ep_edge = EpisodicEdge(
        uuid='s-ep-edge',
        group_id='g-search',
        source_node_uuid='s-ep',
        target_node_uuid='s-1',
        created_at=now,
    )
    await driver.entity_edge_ops.save(driver, edge)
    await driver.episodic_edge_ops.save(driver, ep_edge)

    filters = SearchFilters()
    ft_nodes = await driver.search_ops.node_fulltext_search(
        driver, 'Alice', filters, group_ids=['g-search']
    )
    assert [n.uuid for n in ft_nodes] == ['s-1']

    ft_edges = await driver.search_ops.edge_fulltext_search(
        driver, 'collaborated', filters, group_ids=['g-search']
    )
    assert [e.uuid for e in ft_edges] == ['s-edge']

    ft_eps = await driver.search_ops.episode_fulltext_search(
        driver, 'Relate', filters, group_ids=['g-search']
    )
    assert [e.uuid for e in ft_eps] == ['s-ep']

    ft_comms = await driver.search_ops.community_fulltext_search(
        driver, 'Database', group_ids=['g-search']
    )
    assert [c.uuid for c in ft_comms] == ['s-comm']

    sim_nodes = await driver.search_ops.node_similarity_search(
        driver, [1.0, 0.0, 0.0], filters, group_ids=['g-search'], min_score=0.5
    )
    assert [n.uuid for n in sim_nodes] == ['s-1']

    sim_edges = await driver.search_ops.edge_similarity_search(
        driver, [1.0, 0.0, 0.0], 's-1', 's-2', filters, group_ids=['g-search'], min_score=0.5
    )
    assert [e.uuid for e in sim_edges] == ['s-edge']

    sim_comms = await driver.search_ops.community_similarity_search(
        driver, [1.0, 0.0, 0.0], group_ids=['g-search'], min_score=0.5
    )
    assert [c.uuid for c in sim_comms] == ['s-comm']

    # Deep BFS (> 500 hops) must clamp cleanly instead of raising TraversalDepthExceeded.
    bfs_nodes = await driver.search_ops.node_bfs_search(
        driver, ['s-1'], filters, max_depth=600, group_ids=['g-search']
    )
    assert [n.uuid for n in bfs_nodes] == ['s-2']

    bfs_edges = await driver.search_ops.edge_bfs_search(
        driver, ['s-1'], max_depth=600, search_filter=filters, group_ids=['g-search']
    )
    assert [e.uuid for e in bfs_edges] == ['s-edge']

    # Also verify generic search_utils paths with RelateDriver.
    assert [
        n.uuid
        for n in await utils_node_fulltext_search(driver, 'Alice', filters, group_ids=['g-search'])
    ] == ['s-1']
    assert [
        e.uuid
        for e in await utils_edge_fulltext_search(
            driver, 'collaborated', filters, group_ids=['g-search']
        )
    ] == ['s-edge']
    assert [
        n.uuid
        for n in await utils_node_bfs_search(driver, ['s-1'], filters, 600, group_ids=['g-search'])
    ] == ['s-2']
    assert [
        e.uuid
        for e in await utils_edge_bfs_search(driver, ['s-1'], 600, filters, group_ids=['g-search'])
    ] == ['s-edge']

    # Rerankers
    dist_ranked = await driver.search_ops.node_distance_reranker(driver, ['s-2', 's-1'], 's-1')
    assert [n.uuid for n in dist_ranked] == ['s-1', 's-2']

    mention_ranked = await driver.search_ops.episode_mentions_reranker(
        driver, ['s-2', 's-1'], min_score=1
    )
    assert [n.uuid for n in mention_ranked] == ['s-1']


async def test_transactions_sessions_and_file_persistence(tmp_path: Path):
    db_path = tmp_path / 'graphiti_relate_store'
    driver = await RelateDriver.open(db_path)
    try:
        # Commit via driver.transaction()
        async with driver.transaction() as tx:
            await tx.run("CREATE (:Entity {uuid: 'tx-ok', group_id: 'g1', name: 'Committed'})")

        # Rollback on exception via driver.transaction()
        with pytest.raises(RuntimeError, match='boom'):
            async with driver.transaction() as tx:
                await tx.run(
                    "CREATE (:Entity {uuid: 'tx-fail', group_id: 'g1', name: 'RolledBack'})"
                )
                raise RuntimeError('boom')

        # Commit and rollback via session.execute_write()
        async def write_ok(session: RelateDriverSession):
            await session.run(
                "CREATE (:Entity {uuid: 'sess-ok', group_id: 'g1', name: $name})",
                name='SessionOk',
            )

        async def write_fail(session: RelateDriverSession):
            await session.run(
                "CREATE (:Entity {uuid: 'sess-fail', group_id: 'g1', name: 'SessionFail'})"
            )
            raise ValueError('abort')

        async with driver.session() as session:
            assert isinstance(session, RelateDriverSession)
            await session.execute_write(write_ok)
            with pytest.raises(ValueError, match='abort'):
                await session.execute_write(write_fail)
    finally:
        await driver.close()

    # Reopen from disk and confirm only committed nodes survived.
    reopened = await RelateDriver.open(db_path)
    try:
        rows, _, _ = await reopened.execute_query(
            'MATCH (n:Entity) RETURN n.uuid AS uuid ORDER BY n.uuid'
        )
        assert [r['uuid'] for r in rows] == ['sess-ok', 'tx-ok']
    finally:
        await reopened.close()


async def test_error_logging_does_not_leak_parameter_values(driver: RelateDriver, caplog):
    secret = 'super-secret-token-value'
    with (
        caplog.at_level(logging.ERROR, logger='graphiti_core.driver.relate_driver'),
        pytest.raises(relate.RelateError),
    ):
        await driver.execute_query('THIS IS NOT VALID CYPHER', api_token=secret)

    assert secret not in caplog.text
    assert 'api_token' in caplog.text


async def test_session_run_dispatches_to_active_tx_or_driver():
    mock_driver = AsyncMock(spec=RelateDriver)
    mock_driver.execute_query = AsyncMock(return_value=([], [], None))
    session = RelateDriverSession(mock_driver)

    await session.run('RETURN 1', a=1)
    mock_driver.execute_query.assert_awaited_once_with('RETURN 1', a=1)
