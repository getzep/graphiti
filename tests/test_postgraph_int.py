"""Cross-check integration tests for the PostGraph driver.

Exercises the Graphiti model-level API (node.save, EntityNode.get_by_uuid, etc.)
against a live PostgreSQL instance via the PostGraph driver. Mirrors the patterns
in test_node_int.py and test_edge_int.py so the same operations verified for
Neo4j / FalkorDB are proven to work identically on PostgreSQL.

Set POSTGRAPH_TEST_DSN to a PostgreSQL connection string.
Tests are skipped when the database is unreachable.
"""

import os
from datetime import datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio

from graphiti_core.driver.postgraph import PostGraphDriver
from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import (
    clear_data,
    retrieve_episodes,
)

POSTGRAPH_DSN = os.environ.get('POSTGRAPH_TEST_DSN', 'postgresql://localhost/post_graph_test')

EMBEDDING_DIM = 1024
GROUP_ID = 'postgraph_test_group'
GROUP_ID_2 = 'postgraph_test_group_2'

now = datetime.now()
later = now + timedelta(days=1)


@pytest_asyncio.fixture()
async def pg_driver():
    driver = PostGraphDriver(dsn=POSTGRAPH_DSN, embedding_dim=EMBEDDING_DIM)
    try:
        await driver._ensure_pool()
        await driver.build_indices_and_constraints()
    except Exception as exc:
        pytest.skip(f'PostgreSQL not reachable at {POSTGRAPH_DSN}: {exc}')
    yield driver
    await clear_data(driver, [GROUP_ID, GROUP_ID_2])
    await driver.close()


def _embedding(seed: float = 0.5) -> list[float]:
    return [seed] * EMBEDDING_DIM


def _mock_embedder() -> EmbedderClient:
    mock = Mock(spec=EmbedderClient)
    rng = np.random.default_rng(42)
    cache = {}

    async def embed(input_data, **kwargs):
        key = str(input_data)
        if key not in cache:
            cache[key] = rng.uniform(0.0, 0.9, EMBEDDING_DIM).tolist()
        return cache[key]

    mock.create = embed
    return mock


embedder = _mock_embedder()


# ---------------------------------------------------------------------------
# Helper: count rows via the driver's SQL interface
# ---------------------------------------------------------------------------

async def pg_node_count(driver, uuids: list[str]) -> int:
    records, _, _ = await driver.execute_query(
        'SELECT count(*) AS count FROM entity_nodes WHERE uuid = ANY($uuids)',
        uuids=uuids,
    )
    return int(records[0]['count'])


async def pg_edge_count(driver, uuids: list[str]) -> int:
    total = 0
    for table in ('entity_edges', 'episodic_edges', 'community_edges'):
        records, _, _ = await driver.execute_query(
            f'SELECT count(*) AS count FROM {table} WHERE uuid = ANY($uuids)',
            uuids=uuids,
        )
        total += int(records[0]['count'])
    return total


# ===========================================================================
# Entity Node CRUD
# ===========================================================================

class TestEntityNodeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get_by_uuid(self, pg_driver):
        node = EntityNode(
            uuid=str(uuid4()),
            name='Alice',
            group_id=GROUP_ID,
            labels=['Person'],
            created_at=now,
            name_embedding=_embedding(0.3),
            summary='Alice is a software engineer',
            attributes={'role': 'engineer'},
        )
        await node.save(pg_driver)

        retrieved = await EntityNode.get_by_uuid(pg_driver, node.uuid)
        assert retrieved.uuid == node.uuid
        assert retrieved.name == 'Alice'
        assert retrieved.group_id == GROUP_ID
        assert retrieved.summary == 'Alice is a software engineer'
        assert retrieved.attributes == {'role': 'engineer'}

    @pytest.mark.asyncio
    async def test_get_by_uuids(self, pg_driver):
        nodes = []
        for name in ('Bob', 'Charlie'):
            n = EntityNode(
                name=name, group_id=GROUP_ID, labels=[], name_embedding=_embedding(),
            )
            await n.save(pg_driver)
            nodes.append(n)

        retrieved = await EntityNode.get_by_uuids(pg_driver, [n.uuid for n in nodes])
        assert len(retrieved) == 2
        assert {r.name for r in retrieved} == {'Bob', 'Charlie'}

    @pytest.mark.asyncio
    async def test_get_by_group_ids(self, pg_driver):
        n1 = EntityNode(name='G1', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        n2 = EntityNode(name='G2', group_id=GROUP_ID_2, labels=[], name_embedding=_embedding())
        await n1.save(pg_driver)
        await n2.save(pg_driver)

        group1 = await EntityNode.get_by_group_ids(pg_driver, [GROUP_ID])
        group2 = await EntityNode.get_by_group_ids(pg_driver, [GROUP_ID_2])
        assert all(n.group_id == GROUP_ID for n in group1)
        assert all(n.group_id == GROUP_ID_2 for n in group2)

    @pytest.mark.asyncio
    async def test_delete(self, pg_driver):
        node = EntityNode(name='ToDelete', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        await node.save(pg_driver)
        assert await pg_node_count(pg_driver, [node.uuid]) == 1

        await node.delete(pg_driver)
        assert await pg_node_count(pg_driver, [node.uuid]) == 0

    @pytest.mark.asyncio
    async def test_delete_by_group_id(self, pg_driver):
        n1 = EntityNode(name='Keep', group_id=GROUP_ID_2, labels=[], name_embedding=_embedding())
        n2 = EntityNode(name='Remove', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        await n1.save(pg_driver)
        await n2.save(pg_driver)

        await EntityNode.delete_by_group_id(pg_driver, GROUP_ID)
        assert await pg_node_count(pg_driver, [n2.uuid]) == 0
        assert await pg_node_count(pg_driver, [n1.uuid]) == 1

    @pytest.mark.asyncio
    async def test_name_embedding_roundtrip(self, pg_driver):
        emb = _embedding(0.7)
        node = EntityNode(name='Emb', group_id=GROUP_ID, labels=[], name_embedding=emb)
        await node.save(pg_driver)

        retrieved = await EntityNode.get_by_uuid(pg_driver, node.uuid)
        await retrieved.load_name_embedding(pg_driver)
        assert retrieved.name_embedding is not None
        assert np.allclose(retrieved.name_embedding, emb, atol=1e-5)


# ===========================================================================
# Episodic Node CRUD
# ===========================================================================

class TestEpisodicNodeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_driver):
        episode = EpisodicNode(
            name='ep1',
            group_id=GROUP_ID,
            created_at=now,
            source=EpisodeType.text,
            source_description='test doc',
            content='Alice met Bob at the conference.',
            valid_at=now,
            entity_edges=[],
        )
        await episode.save(pg_driver)

        retrieved = await EpisodicNode.get_by_uuid(pg_driver, episode.uuid)
        assert retrieved.uuid == episode.uuid
        assert retrieved.content == 'Alice met Bob at the conference.'
        assert retrieved.source == EpisodeType.text

    @pytest.mark.asyncio
    async def test_retrieve_episodes(self, pg_driver):
        for i in range(3):
            ep = EpisodicNode(
                name=f'ep_{i}',
                group_id=GROUP_ID,
                created_at=now,
                source=EpisodeType.text,
                source_description='doc',
                content=f'Content {i}',
                valid_at=now + timedelta(hours=i),
                entity_edges=[],
            )
            await ep.save(pg_driver)

        episodes = await retrieve_episodes(
            pg_driver,
            reference_time=now + timedelta(hours=10),
            last_n=2,
            group_ids=[GROUP_ID],
        )
        assert len(episodes) == 2


# ===========================================================================
# Community Node CRUD
# ===========================================================================

class TestCommunityNodeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_driver):
        community = CommunityNode(
            name='Tech Community',
            group_id=GROUP_ID,
            summary='A community about tech',
            name_embedding=_embedding(0.4),
        )
        await community.save(pg_driver)

        retrieved = await CommunityNode.get_by_uuid(pg_driver, community.uuid)
        assert retrieved.name == 'Tech Community'
        assert retrieved.summary == 'A community about tech'


# ===========================================================================
# Saga Node CRUD
# ===========================================================================

class TestSagaNodeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_driver):
        saga = SagaNode(
            name='Onboarding Saga',
            group_id=GROUP_ID,
            summary='Tracks user onboarding',
        )
        await saga.save(pg_driver)

        retrieved = await SagaNode.get_by_uuid(pg_driver, saga.uuid)
        assert retrieved.name == 'Onboarding Saga'
        assert retrieved.group_id == GROUP_ID


# ===========================================================================
# Entity Edge CRUD
# ===========================================================================

class TestEntityEdgeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_driver):
        alice = EntityNode(name='Alice', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        bob = EntityNode(name='Bob', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await alice.save(pg_driver)
        await bob.save(pg_driver)

        edge = EntityEdge(
            source_node_uuid=alice.uuid,
            target_node_uuid=bob.uuid,
            created_at=now,
            name='knows',
            fact='Alice knows Bob',
            episodes=[],
            expired_at=None,
            valid_at=now,
            invalid_at=None,
            group_id=GROUP_ID,
        )
        await edge.generate_embedding(embedder)
        await edge.save(pg_driver)

        retrieved = await EntityEdge.get_by_uuid(pg_driver, edge.uuid)
        assert retrieved.source_node_uuid == alice.uuid
        assert retrieved.target_node_uuid == bob.uuid
        assert retrieved.name == 'knows'
        assert retrieved.fact == 'Alice knows Bob'

    @pytest.mark.asyncio
    async def test_get_by_node_uuid(self, pg_driver):
        a = EntityNode(name='A', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='B', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await a.save(pg_driver)
        await b.save(pg_driver)

        edge = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='linked', fact='A linked B', episodes=[], group_id=GROUP_ID,
        )
        await edge.generate_embedding(embedder)
        await edge.save(pg_driver)

        from_a = await EntityEdge.get_by_node_uuid(pg_driver, a.uuid)
        assert len(from_a) >= 1
        assert any(e.uuid == edge.uuid for e in from_a)

        from_b = await EntityEdge.get_by_node_uuid(pg_driver, b.uuid)
        assert any(e.uuid == edge.uuid for e in from_b)
        assert from_b[0].source_node_uuid == a.uuid
        assert from_b[0].target_node_uuid == b.uuid

    @pytest.mark.asyncio
    async def test_fact_embedding_roundtrip(self, pg_driver):
        a = EntityNode(name='X', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='Y', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await a.save(pg_driver)
        await b.save(pg_driver)

        edge = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='rel', fact='X rel Y', episodes=[], group_id=GROUP_ID,
        )
        original_emb = await edge.generate_embedding(embedder)
        await edge.save(pg_driver)

        retrieved = await EntityEdge.get_by_uuid(pg_driver, edge.uuid)
        await retrieved.load_fact_embedding(pg_driver)
        assert retrieved.fact_embedding is not None
        assert np.allclose(retrieved.fact_embedding, original_emb, atol=1e-5)

    @pytest.mark.asyncio
    async def test_delete_node_cascades_edges(self, pg_driver):
        a = EntityNode(name='CascA', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='CascB', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await a.save(pg_driver)
        await b.save(pg_driver)

        edge = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='casc', fact='cascade test', episodes=[], group_id=GROUP_ID,
        )
        await edge.generate_embedding(embedder)
        await edge.save(pg_driver)
        assert await pg_edge_count(pg_driver, [edge.uuid]) == 1

        await a.delete(pg_driver)
        assert await pg_edge_count(pg_driver, [edge.uuid]) == 0


# ===========================================================================
# Episodic Edge CRUD
# ===========================================================================

class TestEpisodicEdgeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_driver):
        episode = EpisodicNode(
            name='ep', group_id=GROUP_ID, created_at=now,
            source=EpisodeType.message, source_description='chat',
            content='Hello world', valid_at=now, entity_edges=[],
        )
        entity = EntityNode(name='World', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        await episode.save(pg_driver)
        await entity.save(pg_driver)

        edge = EpisodicEdge(
            source_node_uuid=episode.uuid,
            target_node_uuid=entity.uuid,
            created_at=now,
            group_id=GROUP_ID,
        )
        await edge.save(pg_driver)

        retrieved = await EpisodicEdge.get_by_uuid(pg_driver, edge.uuid)
        assert retrieved.source_node_uuid == episode.uuid
        assert retrieved.target_node_uuid == entity.uuid

    @pytest.mark.asyncio
    async def test_get_episode_by_entity(self, pg_driver):
        episode = EpisodicNode(
            name='mention_ep', group_id=GROUP_ID, created_at=now,
            source=EpisodeType.text, source_description='doc',
            content='Mentions Alice', valid_at=now, entity_edges=[],
        )
        alice = EntityNode(name='Alice', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        await episode.save(pg_driver)
        await alice.save(pg_driver)

        mention = EpisodicEdge(
            source_node_uuid=episode.uuid, target_node_uuid=alice.uuid,
            created_at=now, group_id=GROUP_ID,
        )
        await mention.save(pg_driver)

        episodes = await EpisodicNode.get_by_entity_node_uuid(pg_driver, alice.uuid)
        assert len(episodes) >= 1
        assert any(ep.uuid == episode.uuid for ep in episodes)


# ===========================================================================
# Community Edge CRUD
# ===========================================================================

class TestCommunityEdgeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_driver):
        community = CommunityNode(
            name='C1', group_id=GROUP_ID, summary='comm', name_embedding=_embedding(),
        )
        entity = EntityNode(name='Member', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        await community.save(pg_driver)
        await entity.save(pg_driver)

        edge = CommunityEdge(
            source_node_uuid=community.uuid,
            target_node_uuid=entity.uuid,
            created_at=now,
            group_id=GROUP_ID,
        )
        await edge.save(pg_driver)

        retrieved = await CommunityEdge.get_by_uuid(pg_driver, edge.uuid)
        assert retrieved.source_node_uuid == community.uuid
        assert retrieved.target_node_uuid == entity.uuid


# ===========================================================================
# Search Operations
# ===========================================================================

class TestSearch:
    @pytest.mark.asyncio
    async def test_fulltext_node_search(self, pg_driver):
        node = EntityNode(
            name='PostgreSQL Expert',
            group_id=GROUP_ID,
            labels=[],
            name_embedding=_embedding(),
            summary='An expert in PostgreSQL databases',
        )
        await node.save(pg_driver)

        results = await pg_driver.search_ops.node_fulltext_search(
            pg_driver, 'PostgreSQL', SearchFilters(), group_ids=[GROUP_ID], limit=10,
        )
        assert len(results) >= 1
        assert any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_fulltext_edge_search(self, pg_driver):
        a = EntityNode(name='FA', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='FB', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await a.save(pg_driver)
        await b.save(pg_driver)

        edge = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='collaborates', fact='FA collaborates with FB on research',
            episodes=[], group_id=GROUP_ID,
        )
        await edge.generate_embedding(embedder)
        await edge.save(pg_driver)

        results = await pg_driver.search_ops.edge_fulltext_search(
            pg_driver, 'collaborates research', SearchFilters(), group_ids=[GROUP_ID], limit=10,
        )
        assert len(results) >= 1
        assert any(r.uuid == edge.uuid for r in results)

    @pytest.mark.asyncio
    async def test_similarity_node_search(self, pg_driver):
        emb = _embedding(0.8)
        node = EntityNode(
            name='SimilarNode', group_id=GROUP_ID, labels=[], name_embedding=emb,
        )
        await node.save(pg_driver)

        results = await pg_driver.search_ops.node_similarity_search(
            pg_driver, emb, SearchFilters(), group_ids=[GROUP_ID], limit=10, min_score=0.0,
        )
        assert len(results) >= 1
        assert any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_similarity_edge_search(self, pg_driver):
        a = EntityNode(name='SA', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='SB', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await a.save(pg_driver)
        await b.save(pg_driver)

        edge = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='sim_edge', fact='similarity test edge', episodes=[], group_id=GROUP_ID,
        )
        fact_emb = await edge.generate_embedding(embedder)
        await edge.save(pg_driver)

        results = await pg_driver.search_ops.edge_similarity_search(
            pg_driver, fact_emb, None, None, SearchFilters(),
            group_ids=[GROUP_ID], limit=10, min_score=0.0,
        )
        assert len(results) >= 1
        assert any(r.uuid == edge.uuid for r in results)

    @pytest.mark.asyncio
    async def test_bfs_node_search(self, pg_driver):
        a = EntityNode(name='BFS_A', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='BFS_B', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        c = EntityNode(name='BFS_C', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.7))
        await a.save(pg_driver)
        await b.save(pg_driver)
        await c.save(pg_driver)

        e1 = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='bfs1', fact='a to b', episodes=[], group_id=GROUP_ID,
        )
        e2 = EntityEdge(
            source_node_uuid=b.uuid, target_node_uuid=c.uuid,
            created_at=now, name='bfs2', fact='b to c', episodes=[], group_id=GROUP_ID,
        )
        await e1.generate_embedding(embedder)
        await e2.generate_embedding(embedder)
        await e1.save(pg_driver)
        await e2.save(pg_driver)

        results = await pg_driver.search_ops.node_bfs_search(
            pg_driver, [a.uuid], SearchFilters(), max_depth=2,
            group_ids=[GROUP_ID], limit=10,
        )
        found_names = {r.name for r in results}
        assert 'BFS_B' in found_names
        assert 'BFS_C' in found_names


# ===========================================================================
# Group ID (realm) Isolation
# ===========================================================================

class TestGroupIdIsolation:
    @pytest.mark.asyncio
    async def test_nodes_isolated_by_group(self, pg_driver):
        n1 = EntityNode(name='Realm1', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        n2 = EntityNode(name='Realm2', group_id=GROUP_ID_2, labels=[], name_embedding=_embedding())
        await n1.save(pg_driver)
        await n2.save(pg_driver)

        g1 = await EntityNode.get_by_group_ids(pg_driver, [GROUP_ID])
        g2 = await EntityNode.get_by_group_ids(pg_driver, [GROUP_ID_2])

        assert all(n.group_id == GROUP_ID for n in g1)
        assert all(n.group_id == GROUP_ID_2 for n in g2)
        assert any(n.name == 'Realm1' for n in g1)
        assert any(n.name == 'Realm2' for n in g2)

    @pytest.mark.asyncio
    async def test_search_isolated_by_group(self, pg_driver):
        n1 = EntityNode(
            name='IsolatedAlpha', group_id=GROUP_ID, labels=[],
            name_embedding=_embedding(0.9), summary='alpha search target',
        )
        n2 = EntityNode(
            name='IsolatedAlpha', group_id=GROUP_ID_2, labels=[],
            name_embedding=_embedding(0.9), summary='alpha search decoy',
        )
        await n1.save(pg_driver)
        await n2.save(pg_driver)

        results = await pg_driver.search_ops.node_fulltext_search(
            pg_driver, 'IsolatedAlpha', SearchFilters(), group_ids=[GROUP_ID], limit=10,
        )
        assert all(r.group_id == GROUP_ID for r in results)

    @pytest.mark.asyncio
    async def test_bfs_does_not_cross_groups(self, pg_driver):
        a = EntityNode(name='BFS_G1_A', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        b = EntityNode(name='BFS_G1_B', group_id=GROUP_ID, labels=[], name_embedding=_embedding(0.6))
        await a.save(pg_driver)
        await b.save(pg_driver)

        e1 = EntityEdge(
            source_node_uuid=a.uuid, target_node_uuid=b.uuid,
            created_at=now, name='g1_link', fact='within group', episodes=[], group_id=GROUP_ID,
        )
        await e1.generate_embedding(embedder)
        await e1.save(pg_driver)

        x = EntityNode(name='BFS_G2_X', group_id=GROUP_ID_2, labels=[], name_embedding=_embedding())
        await x.save(pg_driver)

        cross_edge = EntityEdge(
            source_node_uuid=b.uuid, target_node_uuid=x.uuid,
            created_at=now, name='cross', fact='crosses groups', episodes=[], group_id=GROUP_ID_2,
        )
        await cross_edge.generate_embedding(embedder)
        await cross_edge.save(pg_driver)

        results = await pg_driver.search_ops.node_bfs_search(
            pg_driver, [a.uuid], SearchFilters(), max_depth=3,
            group_ids=[GROUP_ID], limit=10,
        )
        found_names = {r.name for r in results}
        assert 'BFS_G1_B' in found_names
        assert 'BFS_G2_X' not in found_names

    @pytest.mark.asyncio
    async def test_clear_data_scoped_to_group(self, pg_driver):
        n1 = EntityNode(name='Keep', group_id=GROUP_ID_2, labels=[], name_embedding=_embedding())
        n2 = EntityNode(name='Remove', group_id=GROUP_ID, labels=[], name_embedding=_embedding())
        await n1.save(pg_driver)
        await n2.save(pg_driver)

        await clear_data(pg_driver, [GROUP_ID])

        assert await pg_node_count(pg_driver, [n2.uuid]) == 0
        assert await pg_node_count(pg_driver, [n1.uuid]) == 1


# ===========================================================================
# Graph Maintenance
# ===========================================================================

class TestGraphMaintenance:
    @pytest.mark.asyncio
    async def test_community_clustering(self, pg_driver):
        nodes = []
        for name in ('C_Alice', 'C_Bob', 'C_Charlie'):
            n = EntityNode(name=name, group_id=GROUP_ID, labels=[], name_embedding=_embedding())
            await n.save(pg_driver)
            nodes.append(n)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                e = EntityEdge(
                    source_node_uuid=nodes[i].uuid, target_node_uuid=nodes[j].uuid,
                    created_at=now, name=f'{nodes[i].name}_to_{nodes[j].name}',
                    fact=f'{nodes[i].name} knows {nodes[j].name}',
                    episodes=[], group_id=GROUP_ID,
                )
                await e.generate_embedding(embedder)
                await e.save(pg_driver)

        clusters = await pg_driver.graph_ops.get_community_clusters(
            pg_driver, group_ids=[GROUP_ID],
        )
        assert len(clusters) >= 1
        all_uuids = {n.uuid for cluster in clusters for n in cluster}
        assert all(node.uuid in all_uuids for node in nodes)


# ===========================================================================
# Document Scenario Cross-Check
# ===========================================================================

class TestDocumentScenario:
    """End-to-end scenario: ingest a set of 'documents' as episodes, link
    entities mentioned in them, then verify the graph is queryable."""

    @pytest.mark.asyncio
    async def test_multi_document_graph(self, pg_driver):
        docs = [
            ('Meeting Notes', 'Alice and Bob discussed the Q3 roadmap.'),
            ('Slack Thread', 'Bob asked Charlie for the API docs.'),
            ('Email', 'Charlie sent the design review to Alice.'),
        ]

        episodes = []
        for i, (name, content) in enumerate(docs):
            ep = EpisodicNode(
                name=name, group_id=GROUP_ID, created_at=now + timedelta(hours=i),
                source=EpisodeType.text, source_description=f'doc_{i}',
                content=content, valid_at=now + timedelta(hours=i), entity_edges=[],
            )
            await ep.save(pg_driver)
            episodes.append(ep)

        people = {}
        for name in ('Alice', 'Bob', 'Charlie'):
            n = EntityNode(name=name, group_id=GROUP_ID, labels=['Person'])
            await n.generate_name_embedding(embedder)
            await n.save(pg_driver)
            people[name] = n

        mentions = {
            0: ['Alice', 'Bob'],
            1: ['Bob', 'Charlie'],
            2: ['Charlie', 'Alice'],
        }
        for ep_idx, names in mentions.items():
            for name in names:
                mention = EpisodicEdge(
                    source_node_uuid=episodes[ep_idx].uuid,
                    target_node_uuid=people[name].uuid,
                    created_at=now, group_id=GROUP_ID,
                )
                await mention.save(pg_driver)

        relationships = [
            ('Alice', 'Bob', 'discussed', 'Alice and Bob discussed the Q3 roadmap'),
            ('Bob', 'Charlie', 'asked', 'Bob asked Charlie for the API docs'),
            ('Charlie', 'Alice', 'sent', 'Charlie sent the design review to Alice'),
        ]
        edges = []
        for src, tgt, name, fact in relationships:
            e = EntityEdge(
                source_node_uuid=people[src].uuid,
                target_node_uuid=people[tgt].uuid,
                created_at=now, name=name, fact=fact, episodes=[episodes[0].uuid],
                group_id=GROUP_ID, valid_at=now,
            )
            await e.generate_embedding(embedder)
            await e.save(pg_driver)
            edges.append(e)

        # Cross-check 1: all people retrievable
        all_people = await EntityNode.get_by_group_ids(pg_driver, [GROUP_ID])
        people_names = {n.name for n in all_people}
        assert {'Alice', 'Bob', 'Charlie'} <= people_names

        # Cross-check 2: episodes retrievable
        all_eps = await EpisodicNode.get_by_group_ids(pg_driver, [GROUP_ID])
        assert len(all_eps) >= 3

        # Cross-check 3: fulltext search finds edges
        results = await pg_driver.search_ops.edge_fulltext_search(
            pg_driver, 'roadmap', SearchFilters(), group_ids=[GROUP_ID], limit=10,
        )
        assert any('roadmap' in r.fact.lower() for r in results)

        # Cross-check 4: BFS from Alice finds Bob and Charlie within 2 hops
        bfs = await pg_driver.search_ops.node_bfs_search(
            pg_driver, [people['Alice'].uuid], SearchFilters(),
            max_depth=2, group_ids=[GROUP_ID], limit=10,
        )
        bfs_names = {n.name for n in bfs}
        assert 'Bob' in bfs_names
        assert 'Charlie' in bfs_names

        # Cross-check 5: similarity search finds nodes with similar embeddings
        alice_emb = people['Alice'].name_embedding
        sim = await pg_driver.search_ops.node_similarity_search(
            pg_driver, alice_emb, SearchFilters(),
            group_ids=[GROUP_ID], limit=10, min_score=0.0,
        )
        assert len(sim) >= 1

        # Cross-check 6: episode mentions link back to entities
        alice_episodes = await EpisodicNode.get_by_entity_node_uuid(
            pg_driver, people['Alice'].uuid,
        )
        assert len(alice_episodes) >= 2

        # Cross-check 7: edges queryable by node
        alice_edges = await EntityEdge.get_by_node_uuid(pg_driver, people['Alice'].uuid)
        assert len(alice_edges) >= 2
