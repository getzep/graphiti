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

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from graphiti_core.driver.oracle.codecs import encode_json, entity_node_to_binds
from graphiti_core.driver.oracle.config import Oracle26aiConfig
from graphiti_core.driver.oracle.driver import Oracle26aiDriver
from graphiti_core.driver.oracle.errors import GroupScopeConflict
from graphiti_core.driver.oracle.sql.edges import VALIDATE_ENDPOINTS
from graphiti_core.driver.oracle.sql.nodes import DELETE_NODE, DELETE_NODE_EDGES, MERGE_ENTITY_NODE
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.errors import GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from graphiti_core.search.search_filters import SearchFilters

ROOT = Path(__file__).parents[4]
NOW = datetime(2026, 9, 22, 12, tzinfo=UTC)
TEST_NAMESPACE = os.environ.get('ORACLE_TEST_NAMESPACE', 'graphiti-oracle-' + uuid4().hex)
VECTOR_A = [1.0] + [0.0] * 1023
VECTOR_B = [0.0, 1.0] + [0.0] * 1022


def stable_uuid(group_id: str, name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f'ontology-ai:live:{group_id}:{name}'))


@dataclass(frozen=True, slots=True)
class LiveFixture:
    group_id: str
    entities: tuple[EntityNode, EntityNode, EntityNode]
    episodes: tuple[EpisodicNode, EpisodicNode]
    community: CommunityNode
    saga: SagaNode
    entity_edges: tuple[EntityEdge, EntityEdge]
    mention_edges: tuple[EpisodicEdge, EpisodicEdge, EpisodicEdge]
    community_edge: CommunityEdge
    has_episode_edges: tuple[HasEpisodeEdge, HasEpisodeEdge]
    next_episode_edge: NextEpisodeEdge


def make_fixture(group_id: str) -> LiveFixture:
    entity_a = EntityNode(
        uuid=stable_uuid(group_id, 'entity-a'),
        name='Alpha live concept',
        group_id=group_id,
        labels=['Concept'],
        summary='Alpha is the first live validation concept.',
        attributes={'rank': 1, 'active': True},
        name_embedding=VECTOR_A,
        created_at=NOW,
    )
    entity_b = EntityNode(
        uuid=stable_uuid(group_id, 'entity-b'),
        name='Beta live concept',
        group_id=group_id,
        labels=['Concept'],
        summary='Beta connects Alpha to Gamma.',
        attributes={'rank': 2},
        name_embedding=VECTOR_B,
        created_at=NOW,
    )
    entity_c = EntityNode(
        uuid=stable_uuid(group_id, 'entity-c'),
        name='Gamma live concept',
        group_id=group_id,
        labels=['Concept'],
        summary='Gamma is reached through Beta.',
        attributes={'rank': 3},
        name_embedding=[-1.0] + [0.0] * 1023,
        created_at=NOW,
    )
    episode_a = EpisodicNode(
        uuid=stable_uuid(group_id, 'episode-a'),
        name='Outage detected',
        group_id=group_id,
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic live validation',
        content='{"event":"outage detected"}',
        entity_edges=[],
        episode_metadata={'sequence': 1},
        created_at=NOW,
        valid_at=NOW - timedelta(hours=2),
    )
    episode_b = EpisodicNode(
        uuid=stable_uuid(group_id, 'episode-b'),
        name='Outage restored',
        group_id=group_id,
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic live validation',
        content='{"event":"outage restored"}',
        entity_edges=[],
        episode_metadata={'sequence': 2},
        created_at=NOW + timedelta(minutes=1),
        valid_at=NOW - timedelta(hours=1),
    )
    community = CommunityNode(
        uuid=stable_uuid(group_id, 'community'),
        name='Live validation cluster',
        group_id=group_id,
        labels=[],
        summary='Cluster containing Alpha, Beta, and Gamma.',
        name_embedding=VECTOR_A,
        created_at=NOW,
    )
    saga = SagaNode(
        uuid=stable_uuid(group_id, 'saga'),
        name='live-validation-sequence',
        group_id=group_id,
        labels=[],
        summary='Two ordered outage events.',
        first_episode_uuid=episode_a.uuid,
        last_episode_uuid=episode_b.uuid,
        last_summarized_at=episode_b.created_at,
        last_summarized_episode_uuid=episode_b.uuid,
        last_summarized_episode_valid_at=episode_b.valid_at,
        created_at=NOW,
    )
    entity_edges = (
        EntityEdge(
            uuid=stable_uuid(group_id, 'edge-a-b'),
            source_node_uuid=entity_a.uuid,
            target_node_uuid=entity_b.uuid,
            group_id=group_id,
            name='CONNECTS_TO',
            fact='Alpha connects directly to Beta.',
            fact_embedding=VECTOR_A,
            episodes=[episode_a.uuid],
            attributes={'confidence': 1.0},
            created_at=NOW,
            valid_at=NOW,
            reference_time=NOW,
        ),
        EntityEdge(
            uuid=stable_uuid(group_id, 'edge-b-c'),
            source_node_uuid=entity_b.uuid,
            target_node_uuid=entity_c.uuid,
            group_id=group_id,
            name='CONNECTS_TO',
            fact='Beta connects directly to Gamma.',
            fact_embedding=VECTOR_B,
            episodes=[episode_b.uuid],
            created_at=NOW,
            valid_at=NOW,
            reference_time=NOW,
        ),
    )
    mention_edges = (
        EpisodicEdge(
            uuid=stable_uuid(group_id, 'mention-a-1'),
            source_node_uuid=episode_a.uuid,
            target_node_uuid=entity_a.uuid,
            group_id=group_id,
            created_at=NOW,
        ),
        EpisodicEdge(
            uuid=stable_uuid(group_id, 'mention-a-2'),
            source_node_uuid=episode_b.uuid,
            target_node_uuid=entity_a.uuid,
            group_id=group_id,
            created_at=NOW,
        ),
        EpisodicEdge(
            uuid=stable_uuid(group_id, 'mention-b-1'),
            source_node_uuid=episode_b.uuid,
            target_node_uuid=entity_b.uuid,
            group_id=group_id,
            created_at=NOW,
        ),
    )
    return LiveFixture(
        group_id=group_id,
        entities=(entity_a, entity_b, entity_c),
        episodes=(episode_a, episode_b),
        community=community,
        saga=saga,
        entity_edges=entity_edges,
        mention_edges=mention_edges,
        community_edge=CommunityEdge(
            uuid=stable_uuid(group_id, 'community-member'),
            source_node_uuid=community.uuid,
            target_node_uuid=entity_a.uuid,
            group_id=group_id,
            created_at=NOW,
        ),
        has_episode_edges=(
            HasEpisodeEdge(
                uuid=stable_uuid(group_id, 'has-episode-a'),
                source_node_uuid=saga.uuid,
                target_node_uuid=episode_a.uuid,
                group_id=group_id,
                created_at=NOW,
            ),
            HasEpisodeEdge(
                uuid=stable_uuid(group_id, 'has-episode-b'),
                source_node_uuid=saga.uuid,
                target_node_uuid=episode_b.uuid,
                group_id=group_id,
                created_at=NOW,
            ),
        ),
        next_episode_edge=NextEpisodeEdge(
            uuid=stable_uuid(group_id, 'next-episode'),
            source_node_uuid=episode_a.uuid,
            target_node_uuid=episode_b.uuid,
            group_id=group_id,
            created_at=NOW,
        ),
    )


async def seed_fixture(driver: Oracle26aiDriver, fixture: LiveFixture) -> None:
    await driver.entity_node_ops.save_bulk(driver, list(fixture.entities))
    await driver.episode_node_ops.save_bulk(driver, list(fixture.episodes))
    await driver.community_node_ops.save_bulk(driver, [fixture.community])
    await driver.saga_node_ops.save_bulk(driver, [fixture.saga])
    await driver.entity_edge_ops.save_bulk(driver, list(fixture.entity_edges))
    await driver.episodic_edge_ops.save_bulk(driver, list(fixture.mention_edges))
    await driver.community_edge_ops.save(driver, fixture.community_edge)
    await driver.has_episode_edge_ops.save_bulk(driver, list(fixture.has_episode_edges))
    await driver.next_episode_edge_ops.save_bulk(driver, [fixture.next_episode_edge])


async def live_driver_fixture(request):
    config = Oracle26aiConfig.model_validate(
        {
            'dsn': os.environ.get('ORACLE_DSN', 'myatp_low'),
            'user': os.environ.get('ORACLE_USER', 'admin'),
            'password': os.environ['ORACLE_PASSWORD'],
            'config_dir': str((ROOT / os.environ['ORACLE_CONFIG_DIR']).resolve()),
            'wallet_location': str((ROOT / os.environ['ORACLE_WALLET_DIR']).resolve()),
            'wallet_password': os.environ['ORACLE_WALLET_PASSWORD'],
        }
    )
    driver = Oracle26aiDriver(config)
    await driver.build_indices_and_constraints()
    namespace = re.sub(
        r'[^A-Za-z0-9_-]',
        '-',
        TEST_NAMESPACE,
    )[:64]
    group_id = f'{namespace}-{request.node.name}'[:240]
    await driver.graph_ops.clear_data(driver, [group_id])
    try:
        yield driver, make_fixture(group_id)
    finally:
        await driver.graph_ops.clear_data(driver, [group_id])
        await driver.close()


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_all_node_and_edge_families_round_trip(live_driver) -> None:
    driver, fixture = live_driver
    await seed_fixture(driver, fixture)

    assert {
        node.uuid
        for node in await driver.entity_node_ops.get_by_group_ids(
            driver,
            [fixture.group_id],
        )
    } == {node.uuid for node in fixture.entities}
    assert {
        node.uuid
        for node in await driver.episode_node_ops.get_by_uuids(
            driver, [node.uuid for node in fixture.episodes]
        )
    } == {node.uuid for node in fixture.episodes}
    assert (
        await driver.community_node_ops.get_by_uuid(driver, fixture.community.uuid)
    ).summary == fixture.community.summary
    assert (await driver.saga_node_ops.get_by_uuid(driver, fixture.saga.uuid)).model_dump() == (
        fixture.saga.model_dump()
    )

    entity_without_embedding = await driver.entity_node_ops.get_by_uuid(
        driver, fixture.entities[0].uuid
    )
    entity_without_embedding.name_embedding = None
    await driver.entity_node_ops.load_embeddings(driver, entity_without_embedding)
    assert entity_without_embedding.name_embedding == pytest.approx(VECTOR_A)
    community_without_embedding = await driver.community_node_ops.get_by_uuid(
        driver, fixture.community.uuid
    )
    community_without_embedding.name_embedding = None
    await driver.community_node_ops.load_name_embedding(driver, community_without_embedding)
    assert community_without_embedding.name_embedding == pytest.approx(VECTOR_A)

    edge_ops_and_values = (
        (driver.entity_edge_ops, fixture.entity_edges[0]),
        (driver.episodic_edge_ops, fixture.mention_edges[0]),
        (driver.community_edge_ops, fixture.community_edge),
        (driver.has_episode_edge_ops, fixture.has_episode_edges[0]),
        (driver.next_episode_edge_ops, fixture.next_episode_edge),
    )
    for operations, edge in edge_ops_and_values:
        restored = await operations.get_by_uuid(driver, edge.uuid)
        assert restored.model_dump() == edge.model_dump()

    assert {
        edge.uuid
        for edge in await driver.entity_edge_ops.get_between_nodes(
            driver,
            fixture.entities[0].uuid,
            fixture.entities[1].uuid,
        )
    } == {fixture.entity_edges[0].uuid}
    edge_without_embedding = await driver.entity_edge_ops.get_by_uuid(
        driver, fixture.entity_edges[0].uuid
    )
    edge_without_embedding.fact_embedding = None
    await driver.entity_edge_ops.load_embeddings(driver, edge_without_embedding)
    assert edge_without_embedding.fact_embedding == pytest.approx(VECTOR_A)
    assert {
        episode.uuid
        for episode in await driver.episode_node_ops.get_by_entity_node_uuid(
            driver, fixture.entities[0].uuid
        )
    } == {episode.uuid for episode in fixture.episodes}
    assert [
        episode.uuid
        for episode in await driver.episode_node_ops.retrieve_episodes(
            driver,
            NOW,
            last_n=2,
            group_ids=[fixture.group_id],
            source='json',
            saga=fixture.saga.name,
        )
    ] == [episode.uuid for episode in fixture.episodes]
    assert (
        await driver.saga_node_ops.get_previous_episode_uuid(
            driver,
            fixture.saga.uuid,
            fixture.episodes[1].uuid,
        )
        == fixture.episodes[0].uuid
    )
    assert [
        content
        for content, _, _, _ in await driver.saga_node_ops.get_episode_contents(
            driver,
            fixture.saga.uuid,
            since=NOW - timedelta(minutes=1),
        )
    ] == [episode.content for episode in fixture.episodes]


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_search_traversal_rerankers_and_graph_maintenance(live_driver) -> None:
    driver, fixture = live_driver
    await seed_fixture(driver, fixture)
    filters = SearchFilters()

    alpha_nodes = await driver.search_ops.node_fulltext_search(
        driver, 'Alpha', filters, [fixture.group_id], 10
    )
    assert alpha_nodes[0].uuid == fixture.entities[0].uuid
    assert {node.uuid for node in alpha_nodes} == {
        fixture.entities[0].uuid,
        fixture.entities[1].uuid,
    }
    assert [
        edge.uuid
        for edge in await driver.search_ops.edge_fulltext_search(
            driver, 'Alpha', filters, [fixture.group_id], 10
        )
    ] == [fixture.entity_edges[0].uuid]
    natural_language_edges = await driver.search_ops.edge_fulltext_search(
        driver,
        'O que conecta Alpha e Beta?',
        filters,
        [fixture.group_id],
        10,
    )
    assert natural_language_edges[0].uuid == fixture.entity_edges[0].uuid
    assert [
        episode.uuid
        for episode in await driver.search_ops.episode_fulltext_search(
            driver, 'detected', filters, [fixture.group_id], 10
        )
    ] == [fixture.episodes[0].uuid]
    assert [
        community.uuid
        for community in await driver.search_ops.community_fulltext_search(
            driver, 'Cluster Alpha', [fixture.group_id], 10
        )
    ] == [fixture.community.uuid]
    assert {
        node.uuid
        for node in await driver.search_ops.node_similarity_search(
            driver, VECTOR_A, filters, [fixture.group_id], 10, 0.99
        )
    } == {fixture.entities[0].uuid}
    assert {
        edge.uuid
        for edge in await driver.search_ops.edge_similarity_search(
            driver,
            VECTOR_A,
            fixture.entities[0].uuid,
            fixture.entities[1].uuid,
            filters,
            [fixture.group_id],
            10,
            0.99,
        )
    } == {fixture.entity_edges[0].uuid}
    assert [
        community.uuid
        for community in await driver.search_ops.community_similarity_search(
            driver, VECTOR_A, [fixture.group_id], 10, 0.99
        )
    ] == [fixture.community.uuid]
    community_embeddings = await driver.search_ops.get_embeddings_for_communities(
        driver, [fixture.community]
    )
    assert community_embeddings[fixture.community.uuid] == pytest.approx(VECTOR_A)
    assert {
        node.uuid
        for node in await driver.search_ops.node_bfs_search(
            driver,
            [fixture.entities[0].uuid],
            filters,
            2,
            [fixture.group_id],
            10,
        )
    } == {fixture.entities[1].uuid, fixture.entities[2].uuid}
    assert {
        edge.uuid
        for edge in await driver.search_ops.edge_bfs_search(
            driver,
            [fixture.entities[0].uuid],
            2,
            filters,
            [fixture.group_id],
            10,
        )
    } == {edge.uuid for edge in fixture.entity_edges}

    uuids, scores = await driver.search_ops.node_distance_reranker(
        driver,
        [node.uuid for node in fixture.entities],
        fixture.entities[0].uuid,
    )
    assert uuids == [node.uuid for node in fixture.entities]
    assert scores == [0.1, 1.0, 0.5]
    mentioned_uuids, mention_scores = await driver.search_ops.episode_mentions_reranker(
        driver,
        [node.uuid for node in fixture.entities],
    )
    assert mentioned_uuids == [
        fixture.entities[0].uuid,
        fixture.entities[1].uuid,
        fixture.entities[2].uuid,
    ]
    assert mention_scores == [2.0, 1.0, 0.0]

    assert {
        node.uuid
        for node in await driver.graph_ops.get_mentioned_nodes(driver, list(fixture.episodes))
    } == {fixture.entities[0].uuid, fixture.entities[1].uuid}
    assert [
        community.uuid
        for community in await driver.graph_ops.get_communities_by_nodes(
            driver, [fixture.entities[0]]
        )
    ] == [fixture.community.uuid]
    community, changed = await driver.graph_ops.determine_entity_community_result(
        driver, fixture.entities[0]
    )
    assert community is not None and community.uuid == fixture.community.uuid
    assert changed is False
    clusters = await driver.graph_ops.get_community_clusters(driver, [fixture.group_id])
    assert any(
        {node.uuid for node in cluster} == {node.uuid for node in fixture.entities}
        for cluster in clusters
    )
    readiness = await driver.schema_readiness()
    assert readiness.ready is True
    await driver.graph_ops.remove_communities(driver, [fixture.group_id])
    assert await driver.community_node_ops.get_by_group_ids(driver, [fixture.group_id]) == []


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_fulltext_barrier_makes_committed_node_and_fact_immediately_searchable(
    live_driver,
) -> None:
    driver, fixture = live_driver
    scoped = driver.clone(fixture.group_id)
    node_token = f'FreshnessNode{uuid4().hex}'
    fact_token = f'FreshnessFact{uuid4().hex}'
    source = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'freshness-source'),
        name=f'{node_token} source',
        group_id=fixture.group_id,
        labels=['FreshnessProbe'],
        summary='Committed node for deterministic Oracle Text validation.',
        created_at=NOW,
    )
    target = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'freshness-target'),
        name='Freshness target',
        group_id=fixture.group_id,
        labels=['FreshnessProbe'],
        summary='Target endpoint for deterministic Oracle Text validation.',
        created_at=NOW,
    )
    edge = EntityEdge(
        uuid=stable_uuid(fixture.group_id, 'freshness-edge'),
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id=fixture.group_id,
        name='FRESHNESS_PROBE',
        fact=f'{fact_token} is searchable after the explicit barrier.',
        episodes=[],
        created_at=NOW,
        valid_at=NOW,
        reference_time=NOW,
    )

    assert node_token != fact_token
    await driver.entity_node_ops.save_bulk(scoped, [source, target])
    await driver.entity_edge_ops.save(scoped, edge)

    synchronized = await driver.sync_fulltext_indexes()
    node_results = await driver.search_ops.node_fulltext_search(
        scoped,
        node_token,
        SearchFilters(),
        group_ids=None,
        limit=10,
    )
    edge_results = await driver.search_ops.edge_fulltext_search(
        scoped,
        fact_token,
        SearchFilters(),
        group_ids=None,
        limit=10,
    )

    assert synchronized == ('GRAPHITI_NODES_TEXT_IDX', 'GRAPHITI_EDGES_TEXT_IDX')
    assert [node.uuid for node in node_results] == [source.uuid]
    assert [result.uuid for result in edge_results] == [edge.uuid]


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_large_utf8_clob_and_json_payloads_round_trip_without_truncation(
    live_driver,
) -> None:
    driver, fixture = live_driver
    scoped = driver.clone(fixture.group_id)
    node_token = f'LargeNode{uuid4().hex}'
    episode_token = f'LargeEpisode{uuid4().hex}'
    community_token = f'LargeCommunity{uuid4().hex}'
    fact_token = f'LargeFact{uuid4().hex}'
    direct_boundary_token = f'DirectBoundary{uuid4().hex}'
    locator_boundary_token = f'LocatorBoundary{uuid4().hex}'
    direct_boundary_name = 'Direct CLOB boundary'
    section = 'Memória técnica de observação. ' * 600
    node_summary = ('Resumo técnico com ação e conexão. ' * 900) + node_token
    direct_boundary_summary = (
        'a'
        * (
            3999
            - len(direct_boundary_name.encode('utf-8'))
            - len(direct_boundary_token.encode('utf-8'))
            - 1
        )
    ) + f' {direct_boundary_token}'
    locator_boundary_summary = (
        'b' * (4001 - len(locator_boundary_token.encode('utf-8')) - 1)
    ) + f' {locator_boundary_token}'
    episode_content = json.dumps(
        {
            'description': ('Evidência UTF-8 ação observação. ' * 1200),
            'token': episode_token,
        },
        ensure_ascii=False,
        separators=(',', ':'),
    )
    episode_metadata = {'section': section, 'token': episode_token}
    community_summary = ('Comunidade técnica operação. ' * 600) + community_token
    edge_fact = ('Fato técnico ação conexão. ' * 1500) + fact_token
    edge_attributes = {'section': section, 'token': fact_token}

    assert len(node_summary.encode('utf-8')) > 4096
    assert len(f'{direct_boundary_name}\n{direct_boundary_summary}'.encode()) == 4000
    assert len(locator_boundary_summary.encode('utf-8')) == 4001
    assert len(episode_content.encode('utf-8')) > 32768
    assert len(encode_json(episode_metadata).encode('utf-8')) > 12000
    assert len(community_summary.encode('utf-8')) > 12000
    assert len(edge_fact.encode('utf-8')) > 32768
    assert len(encode_json(edge_attributes).encode('utf-8')) > 12000

    source = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'large-source'),
        name='Large payload source',
        group_id=fixture.group_id,
        labels=['LargePayload'],
        summary=node_summary,
        created_at=NOW,
    )
    target = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'large-target'),
        name='Large payload target',
        group_id=fixture.group_id,
        labels=['LargePayload'],
        summary='Target endpoint',
        created_at=NOW,
    )
    direct_boundary = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'direct-boundary'),
        name=direct_boundary_name,
        group_id=fixture.group_id,
        labels=['LargePayload'],
        summary=direct_boundary_summary,
        created_at=NOW,
    )
    locator_boundary = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'locator-boundary'),
        name='Temporary CLOB boundary',
        group_id=fixture.group_id,
        labels=['LargePayload'],
        summary=locator_boundary_summary,
        created_at=NOW,
    )
    episode = EpisodicNode(
        uuid=stable_uuid(fixture.group_id, 'large-episode'),
        name='Large UTF-8 episode',
        group_id=fixture.group_id,
        labels=['LargePayload'],
        source=EpisodeType.json,
        source_description='synthetic large-payload validation',
        content=episode_content,
        entity_edges=[],
        episode_metadata=episode_metadata,
        created_at=NOW,
        valid_at=NOW,
    )
    community = CommunityNode(
        uuid=stable_uuid(fixture.group_id, 'large-community'),
        name='Large payload community',
        group_id=fixture.group_id,
        labels=['LargePayload'],
        summary=community_summary,
        created_at=NOW,
    )
    edge = EntityEdge(
        uuid=stable_uuid(fixture.group_id, 'large-edge'),
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id=fixture.group_id,
        name='LARGE_PAYLOAD_RELATION',
        fact=edge_fact,
        episodes=[episode.uuid],
        attributes=edge_attributes,
        created_at=NOW,
        valid_at=NOW,
        reference_time=NOW,
    )

    await driver.entity_node_ops.save_bulk(
        scoped,
        [source, target, direct_boundary, locator_boundary],
    )
    await driver.episode_node_ops.save(scoped, episode)
    await driver.community_node_ops.save(scoped, community)
    await driver.entity_edge_ops.save(scoped, edge)
    await driver.sync_fulltext_indexes()

    persisted_source = await driver.entity_node_ops.get_by_uuid(scoped, source.uuid)
    persisted_direct_boundary = await driver.entity_node_ops.get_by_uuid(
        scoped, direct_boundary.uuid
    )
    persisted_locator_boundary = await driver.entity_node_ops.get_by_uuid(
        scoped, locator_boundary.uuid
    )
    persisted_episode = await driver.episode_node_ops.get_by_uuid(scoped, episode.uuid)
    persisted_community = await driver.community_node_ops.get_by_uuid(scoped, community.uuid)
    persisted_edge = await driver.entity_edge_ops.get_by_uuid(scoped, edge.uuid)

    assert persisted_source.summary == node_summary
    assert persisted_direct_boundary.summary == direct_boundary_summary
    assert persisted_locator_boundary.summary == locator_boundary_summary
    assert persisted_episode.content == episode_content
    assert persisted_episode.episode_metadata == episode_metadata
    assert persisted_community.summary == community_summary
    assert persisted_edge.fact == edge_fact
    assert persisted_edge.attributes == edge_attributes

    node_results = await driver.search_ops.node_fulltext_search(
        scoped, node_token, SearchFilters(), group_ids=None, limit=10
    )
    direct_boundary_results = await driver.search_ops.node_fulltext_search(
        scoped, direct_boundary_token, SearchFilters(), group_ids=None, limit=10
    )
    locator_boundary_results = await driver.search_ops.node_fulltext_search(
        scoped, locator_boundary_token, SearchFilters(), group_ids=None, limit=10
    )
    episode_results = await driver.search_ops.episode_fulltext_search(
        scoped, episode_token, SearchFilters(), group_ids=None, limit=10
    )
    community_results = await driver.search_ops.community_fulltext_search(
        scoped, community_token, group_ids=None, limit=10
    )
    edge_results = await driver.search_ops.edge_fulltext_search(
        scoped, fact_token, SearchFilters(), group_ids=None, limit=10
    )

    assert [result.uuid for result in node_results] == [source.uuid]
    assert [result.uuid for result in direct_boundary_results] == [direct_boundary.uuid]
    assert [result.uuid for result in locator_boundary_results] == [locator_boundary.uuid]
    assert [result.uuid for result in episode_results] == [episode.uuid]
    assert [result.uuid for result in community_results] == [community.uuid]
    assert [result.uuid for result in edge_results] == [edge.uuid]


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_delete_lifecycle_for_every_node_and_edge_family(live_driver) -> None:
    driver, fixture = live_driver
    await seed_fixture(driver, fixture)

    await driver.entity_edge_ops.delete(driver, fixture.entity_edges[0])
    await driver.entity_edge_ops.delete_by_uuids(driver, [fixture.entity_edges[1].uuid])
    await driver.episodic_edge_ops.delete(driver, fixture.mention_edges[0])
    await driver.episodic_edge_ops.delete_by_uuids(
        driver,
        [edge.uuid for edge in fixture.mention_edges[1:]],
    )
    await driver.community_edge_ops.delete_by_uuids(
        driver,
        [fixture.community_edge.uuid],
    )
    await driver.has_episode_edge_ops.delete(driver, fixture.has_episode_edges[0])
    await driver.has_episode_edge_ops.delete_by_uuids(
        driver,
        [fixture.has_episode_edges[1].uuid],
    )
    await driver.next_episode_edge_ops.delete(driver, fixture.next_episode_edge)

    with pytest.raises(GroupsEdgesNotFoundError):
        await driver.entity_edge_ops.get_by_group_ids(driver, [fixture.group_id])
    structural_edge_operations = (
        driver.episodic_edge_ops,
        driver.community_edge_ops,
        driver.has_episode_edge_ops,
        driver.next_episode_edge_ops,
    )
    for operations in structural_edge_operations:
        assert await operations.get_by_group_ids(driver, [fixture.group_id]) == []

    await driver.entity_node_ops.delete(driver, fixture.entities[0])
    await driver.entity_node_ops.delete_by_uuids(
        driver,
        [node.uuid for node in fixture.entities[1:]],
    )
    await driver.episode_node_ops.delete(driver, fixture.episodes[0])
    await driver.episode_node_ops.delete_by_uuids(driver, [fixture.episodes[1].uuid])
    await driver.community_node_ops.delete_by_uuids(driver, [fixture.community.uuid])
    await driver.saga_node_ops.delete(driver, fixture.saga)

    node_operations = (
        driver.entity_node_ops,
        driver.episode_node_ops,
        driver.community_node_ops,
        driver.saga_node_ops,
    )
    for operations in node_operations:
        assert await operations.get_by_group_ids(driver, [fixture.group_id]) == []


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_graphiti_0311_identity_saga_and_directed_filters(live_driver) -> None:
    driver, fixture = live_driver
    await seed_fixture(driver, fixture)
    scoped = driver.clone(fixture.group_id)

    exact = await EntityNode.get_by_names(
        scoped,
        [fixture.entities[0].name],
        [fixture.group_id],
    )
    assert [node.uuid for node in exact] == [fixture.entities[0].uuid]

    by_identity = await driver.entity_node_ops.get_by_attribute_values(
        scoped,
        fixture.group_id,
        {'rank': 1, 'active': True},
        ['Concept'],
    )
    assert [node.uuid for node in by_identity] == [fixture.entities[0].uuid]

    backfill = EpisodicNode(
        uuid=stable_uuid(fixture.group_id, 'episode-backfill'),
        name='Historical outage imported later',
        group_id=fixture.group_id,
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic backfill validation',
        content='{"event":"historical outage"}',
        created_at=NOW + timedelta(minutes=2),
        valid_at=NOW - timedelta(days=30),
    )
    await driver.episode_node_ops.save(scoped, backfill)
    await driver.has_episode_edge_ops.save(
        scoped,
        HasEpisodeEdge(
            uuid=stable_uuid(fixture.group_id, 'has-episode-backfill'),
            source_node_uuid=fixture.saga.uuid,
            target_node_uuid=backfill.uuid,
            group_id=fixture.group_id,
            created_at=backfill.created_at,
        ),
    )
    saga_rows = await driver.saga_node_ops.get_episode_contents(
        scoped,
        fixture.saga.uuid,
        since=NOW - timedelta(minutes=1),
    )
    assert [content for content, _, _, _ in saga_rows] == [
        fixture.episodes[0].content,
        fixture.episodes[1].content,
        backfill.content,
    ]
    assert [created_at for _, _, created_at, _ in saga_rows] == sorted(
        created_at for _, _, created_at, _ in saga_rows
    )

    directed = await driver.search_ops.edge_fulltext_search(
        scoped,
        'connects',
        SearchFilters(
            edge_source_node_uuids=[fixture.entities[1].uuid],
            edge_target_node_uuids=[fixture.entities[2].uuid],
        ),
        [fixture.group_id],
        10,
    )
    assert [edge.uuid for edge in directed] == [fixture.entity_edges[1].uuid]


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_known_uuid_group_isolation_rollback_and_reload(live_driver) -> None:
    driver, fixture = live_driver
    group_a = fixture.group_id
    group_b = f'{group_a}-other'
    scoped_a = driver.clone(group_a)
    scoped_b = driver.clone(group_b)
    await driver.graph_ops.clear_data(driver, [group_b])
    try:
        await seed_fixture(scoped_a, fixture)
        protected_uuid = fixture.entities[0].uuid

        with pytest.raises(NodeNotFoundError):
            await driver.entity_node_ops.get_by_uuid(scoped_b, protected_uuid)

        with pytest.raises(GroupScopeConflict):
            await driver.entity_node_ops.get_by_uuid(
                scoped_b,
                protected_uuid,
                group_id=group_a,
            )

        implicit_search = await driver.search_ops.node_fulltext_search(
            scoped_b,
            'Alpha',
            SearchFilters(),
            group_ids=None,
        )
        assert implicit_search == []

        await driver.graph_ops.clear_data(scoped_b)
        protected = await driver.entity_node_ops.get_by_uuid(scoped_a, protected_uuid)
        assert protected.group_id == group_a

        collision = fixture.entities[0].model_copy(update={'group_id': group_b})
        with pytest.raises(GroupScopeConflict):
            await driver.entity_node_ops.save(scoped_b, collision)

        await driver.entity_node_ops.delete_by_uuids(
            scoped_b,
            [protected_uuid],
            group_id=group_b,
        )
        protected = await driver.entity_node_ops.get_by_uuid(scoped_a, protected_uuid)
        assert protected.group_id == group_a

        with pytest.raises(GroupScopeConflict):
            await driver.entity_node_ops.delete(scoped_b, fixture.entities[0])

        with pytest.raises(GroupScopeConflict):
            await driver.graph_ops.get_mentioned_nodes(scoped_b, [fixture.episodes[0]])

        reranked, rerank_scores = await driver.search_ops.episode_mentions_reranker(
            scoped_b,
            [protected_uuid],
        )
        assert reranked == []
        assert rerank_scores == []

        transient = EntityNode(
            uuid=stable_uuid(group_b, 'transient'),
            name='Transient group B node',
            group_id=group_b,
            labels=['Concept'],
            summary='Must roll back before collision.',
            created_at=NOW,
        )
        with pytest.raises(GroupScopeConflict):
            await driver.entity_node_ops.save_bulk(scoped_b, [transient, collision])
        with pytest.raises(NodeNotFoundError):
            await driver.entity_node_ops.get_by_uuid(scoped_b, transient.uuid)

        await driver.entity_node_ops.save(scoped_a, fixture.entities[0])
        replayed = await driver.entity_node_ops.get_by_names(
            scoped_a,
            [fixture.entities[0].name],
            [group_a],
        )
        assert [node.uuid for node in replayed] == [protected_uuid]

        await driver.entity_node_ops.save(scoped_b, transient)
        await driver.entity_node_ops.delete(scoped_b, transient)
        with pytest.raises(NodeNotFoundError):
            await driver.entity_node_ops.get_by_uuid(scoped_b, transient.uuid)
        await driver.entity_node_ops.save(scoped_b, transient)
        reloaded = await driver.entity_node_ops.get_by_uuid(scoped_b, transient.uuid)
        assert reloaded.group_id == group_b
    finally:
        await driver.graph_ops.clear_data(driver, [group_b])


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_saga_composite_cursor_consumes_tied_timestamps_once(live_driver) -> None:
    driver, fixture = live_driver
    group_id = f'{fixture.group_id}-tied-saga'
    scoped = driver.clone(group_id)
    tied = NOW + timedelta(days=1)
    episode_uuids = sorted([stable_uuid(group_id, 'tie-a'), stable_uuid(group_id, 'tie-b')])
    episode_a = EpisodicNode(
        uuid=episode_uuids[0],
        name='Tie A',
        group_id=group_id,
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic live cursor validation',
        content='{"event":"first tied episode"}',
        created_at=tied,
        valid_at=tied,
    )
    episode_b = episode_a.model_copy(
        update={
            'uuid': episode_uuids[1],
            'name': 'Tie B',
            'content': '{"event":"second tied episode"}',
        }
    )
    saga = SagaNode(
        uuid=stable_uuid(group_id, 'tie-saga'),
        name='tie-safe-live',
        group_id=group_id,
        labels=[],
        created_at=tied,
        last_summarized_at=tied,
        last_summarized_episode_uuid=None,
    )
    links = [
        HasEpisodeEdge(
            uuid=stable_uuid(group_id, f'tie-link-{index}'),
            source_node_uuid=saga.uuid,
            target_node_uuid=episode.uuid,
            group_id=group_id,
            created_at=tied,
        )
        for index, episode in enumerate((episode_a, episode_b))
    ]

    await driver.graph_ops.clear_data(driver, [group_id])
    try:
        await driver.saga_node_ops.save(scoped, saga)
        await driver.episode_node_ops.save_bulk(scoped, [episode_a, episode_b])
        await driver.has_episode_edge_ops.save_bulk(scoped, links)

        first = await driver.saga_node_ops.get_episode_contents(
            scoped,
            saga.uuid,
            since=saga.last_summarized_at,
            after_uuid=saga.last_summarized_episode_uuid,
            limit=1,
        )
        second = await driver.saga_node_ops.get_episode_contents(
            scoped,
            saga.uuid,
            since=first[0][2],
            after_uuid=first[0][3],
            limit=1,
        )
        exhausted = await driver.saga_node_ops.get_episode_contents(
            scoped,
            saga.uuid,
            since=second[0][2],
            after_uuid=second[0][3],
            limit=1,
        )

        assert [first[0][3], second[0][3]] == [episode_a.uuid, episode_b.uuid]
        assert exhausted == []
    finally:
        await driver.graph_ops.clear_data(driver, [group_id])


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_read_committed_uuid_race_keeps_one_group_owner(live_driver) -> None:
    driver, fixture = live_driver
    group_a = f'{fixture.group_id}-race-a'
    group_b = f'{fixture.group_id}-race-b'
    shared_uuid = stable_uuid(fixture.group_id, 'read-committed-shared')
    scoped_a = driver.clone(group_a)
    scoped_b = driver.clone(group_b)
    node_a = EntityNode(
        uuid=shared_uuid,
        name='Race A',
        group_id=group_a,
        labels=['Concept'],
        summary='One owner must win.',
        created_at=NOW,
    )
    node_b = node_a.model_copy(update={'name': 'Race B', 'group_id': group_b})

    await driver.graph_ops.clear_data(driver, [group_a, group_b])
    try:
        results = await asyncio.gather(
            driver.entity_node_ops.save(scoped_a, node_a),
            driver.entity_node_ops.save(scoped_b, node_b),
            return_exceptions=True,
        )

        assert sum(result is None for result in results) == 1
        assert sum(isinstance(result, BaseException) for result in results) == 1
        records, _, _ = await driver.execute_query(
            'SELECT group_id FROM graphiti_nodes WHERE uuid = :uuid',
            uuid=shared_uuid,
        )
        assert len(records) == 1
        assert records[0]['group_id'] in {group_a, group_b}
    finally:
        await driver.graph_ops.clear_data(driver, [group_a, group_b])


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_endpoint_lock_blocks_delete_reinsert_aba(live_driver) -> None:
    driver, fixture = live_driver
    group_a = f'{fixture.group_id}-endpoint-a'
    group_b = f'{fixture.group_id}-endpoint-b'
    scoped_a = driver.clone(group_a)
    source = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'aba-source'),
        name='ABA source',
        group_id=group_a,
        labels=['Concept'],
        summary='Must remain stable through edge merge.',
        created_at=NOW,
    )
    target = EntityNode(
        uuid=stable_uuid(fixture.group_id, 'aba-target'),
        name='ABA target',
        group_id=group_a,
        labels=['Concept'],
        summary='Second locked endpoint.',
        created_at=NOW,
    )
    replacement = source.model_copy(update={'name': 'ABA replacement', 'group_id': group_b})
    edge = EntityEdge(
        uuid=stable_uuid(fixture.group_id, 'aba-edge'),
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id=group_a,
        name='LOCKS_WITH',
        fact='Source locks with target.',
        created_at=NOW,
    )
    validated = asyncio.Event()
    resume_edge = asyncio.Event()
    replacement_committed = asyncio.Event()

    class PausingTransaction:
        def __init__(self, inner) -> None:
            self.inner = inner

        async def run(self, query: str, **binds):
            result = await self.inner.run(query, **binds)
            if query == VALIDATE_ENDPOINTS:
                validated.set()
                await resume_edge.wait()
            return result

    async def save_edge() -> None:
        async with scoped_a.transaction() as tx:
            await driver.entity_edge_ops.save(
                scoped_a,
                edge,
                tx=PausingTransaction(tx),
            )

    async def replace_endpoint() -> None:
        await validated.wait()
        async with driver.transaction() as tx:
            await tx.run(
                DELETE_NODE_EDGES,
                uuid=source.uuid,
                scope_group_id=group_a,
            )
            await tx.run(
                DELETE_NODE,
                uuid=source.uuid,
                kind='ENTITY',
                scope_group_id=group_a,
            )
            await tx.run(MERGE_ENTITY_NODE, **entity_node_to_binds(replacement))
        replacement_committed.set()

    await driver.graph_ops.clear_data(driver, [group_a, group_b])
    try:
        await driver.entity_node_ops.save_bulk(scoped_a, [source, target])
        edge_task = asyncio.create_task(save_edge())
        await asyncio.wait_for(validated.wait(), timeout=5)
        replacement_task = asyncio.create_task(replace_endpoint())
        try:
            await asyncio.wait_for(replacement_committed.wait(), timeout=5)
            replacement_was_blocked = False
        except TimeoutError:
            replacement_was_blocked = True
        resume_edge.set()
        results = await asyncio.gather(edge_task, replacement_task, return_exceptions=True)
        rows, _, _ = await driver.execute_query(
            'SELECT e.group_id AS edge_group_id, n.group_id AS source_group_id '
            'FROM graphiti_edges e '
            'JOIN graphiti_nodes n ON n.uuid = e.source_uuid '
            'WHERE e.uuid = :uuid',
            uuid=edge.uuid,
        )

        assert results[0] is None
        assert replacement_was_blocked is True
        if results[1] is None:
            assert rows == []
        else:
            assert isinstance(results[1], BaseException)
            assert rows == [{'edge_group_id': group_a, 'source_group_id': group_a}]
    finally:
        resume_edge.set()
        await driver.graph_ops.clear_data(driver, [group_a, group_b])


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_live_opposite_direction_edge_locks_do_not_deadlock(live_driver) -> None:
    driver, fixture = live_driver
    group_id = f'{fixture.group_id}-lock-order'
    scoped = driver.clone(group_id)
    node_a = EntityNode(
        uuid=stable_uuid(group_id, 'lock-a'),
        name='Lock A',
        group_id=group_id,
        labels=['Concept'],
        summary='First endpoint.',
        created_at=NOW,
    )
    node_b = EntityNode(
        uuid=stable_uuid(group_id, 'lock-b'),
        name='Lock B',
        group_id=group_id,
        labels=['Concept'],
        summary='Second endpoint.',
        created_at=NOW,
    )
    forward = EntityEdge(
        uuid=stable_uuid(group_id, 'lock-forward'),
        source_node_uuid=node_a.uuid,
        target_node_uuid=node_b.uuid,
        group_id=group_id,
        name='FORWARD',
        fact='A points to B.',
        created_at=NOW,
    )
    reverse = EntityEdge(
        uuid=stable_uuid(group_id, 'lock-reverse'),
        source_node_uuid=node_b.uuid,
        target_node_uuid=node_a.uuid,
        group_id=group_id,
        name='REVERSE',
        fact='B points to A.',
        created_at=NOW,
    )

    await driver.graph_ops.clear_data(driver, [group_id])
    try:
        await driver.entity_node_ops.save_bulk(scoped, [node_a, node_b])
        await asyncio.wait_for(
            asyncio.gather(
                driver.entity_edge_ops.save(scoped, forward),
                driver.entity_edge_ops.save(scoped, reverse),
            ),
            timeout=10,
        )
        records, _, _ = await driver.execute_query(
            'SELECT COUNT(*) AS edge_count FROM graphiti_edges WHERE group_id = :group_id',
            group_id=group_id,
        )
        assert records == [{'edge_count': 2}]
    finally:
        await driver.graph_ops.clear_data(driver, [group_id])


pytestmark = [pytest.mark.integration, pytest.mark.oracle]
