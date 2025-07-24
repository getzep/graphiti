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

import logging
import sys
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.embedder.openai import OpenAIEmbedder
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from tests.helpers_test import drivers, get_driver

pytestmark = pytest.mark.integration

pytest_plugins = ('pytest_asyncio',)

group_id = f'test_group_{str(uuid4())}'


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'driver',
    drivers,
    ids=drivers,
)
async def test_episodic_edge(driver):
    graph_driver = get_driver(driver)
    embedder = OpenAIEmbedder()

    now = datetime.now()

    episode_node = EpisodicNode(
        name='test_episode',
        labels=[],
        created_at=now,
        valid_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Alice likes Bob',
        entity_edges=[],
        group_id=group_id,
    )

    node_count = await get_node_count(graph_driver, episode_node.uuid)
    assert node_count == 0
    await episode_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, episode_node.uuid)
    assert node_count == 1

    alice_node = EntityNode(
        name='Alice',
        labels=[],
        created_at=now,
        summary='Alice summary',
        group_id=group_id,
    )
    await alice_node.generate_name_embedding(embedder)

    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 0
    await alice_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 1

    episodic_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=alice_node.uuid,
        created_at=now,
        group_id=group_id,
    )

    edge_count = await get_edge_count(graph_driver, episodic_edge.uuid)
    assert edge_count == 0
    await episodic_edge.save(graph_driver)
    edge_count = await get_edge_count(graph_driver, episodic_edge.uuid)
    assert edge_count == 1

    retrieved = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge.uuid)
    assert retrieved.uuid == episodic_edge.uuid
    assert retrieved.source_node_uuid == episode_node.uuid
    assert retrieved.target_node_uuid == alice_node.uuid
    assert retrieved.created_at == now
    assert retrieved.group_id == group_id

    retrieved = await EpisodicEdge.get_by_uuids(graph_driver, [episodic_edge.uuid])
    assert len(retrieved) == 1
    assert retrieved[0].uuid == episodic_edge.uuid
    assert retrieved[0].source_node_uuid == episode_node.uuid
    assert retrieved[0].target_node_uuid == alice_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    retrieved = await EpisodicEdge.get_by_group_ids(graph_driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == episodic_edge.uuid
    assert retrieved[0].source_node_uuid == episode_node.uuid
    assert retrieved[0].target_node_uuid == alice_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    await episodic_edge.delete(graph_driver)
    edge_count = await get_edge_count(graph_driver, episodic_edge.uuid)
    assert edge_count == 0

    await episode_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, episode_node.uuid)
    assert node_count == 0

    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'driver',
    drivers,
    ids=drivers,
)
async def test_entity_edge(driver):
    graph_driver = get_driver(driver)
    embedder = OpenAIEmbedder()

    now = datetime.now()

    alice_node = EntityNode(
        name='Alice',
        labels=[],
        created_at=now,
        summary='Alice summary',
        group_id=group_id,
    )
    await alice_node.generate_name_embedding(embedder)

    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 0
    await alice_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 1

    bob_node = EntityNode(
        name='Bob', labels=[], created_at=now, summary='Bob summary', group_id=group_id
    )
    await bob_node.generate_name_embedding(embedder)

    node_count = await get_node_count(graph_driver, bob_node.uuid)
    assert node_count == 0
    await bob_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, bob_node.uuid)
    assert node_count == 1

    entity_edge = EntityEdge(
        source_node_uuid=alice_node.uuid,
        target_node_uuid=bob_node.uuid,
        created_at=now,
        name='likes',
        fact='Alice likes Bob',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    edge_embedding = await entity_edge.generate_embedding(embedder)

    edge_count = await get_edge_count(graph_driver, entity_edge.uuid)
    assert edge_count == 0
    await entity_edge.save(graph_driver)
    edge_count = await get_edge_count(graph_driver, entity_edge.uuid)
    assert edge_count == 1

    retrieved = await EntityEdge.get_by_uuid(graph_driver, entity_edge.uuid)
    assert retrieved.uuid == entity_edge.uuid
    assert retrieved.source_node_uuid == alice_node.uuid
    assert retrieved.target_node_uuid == bob_node.uuid
    assert retrieved.created_at == now
    assert retrieved.group_id == group_id

    retrieved = await EntityEdge.get_by_uuids(graph_driver, [entity_edge.uuid])
    assert len(retrieved) == 1
    assert retrieved[0].uuid == entity_edge.uuid
    assert retrieved[0].source_node_uuid == alice_node.uuid
    assert retrieved[0].target_node_uuid == bob_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    retrieved = await EntityEdge.get_by_group_ids(graph_driver, [group_id], limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == entity_edge.uuid
    assert retrieved[0].source_node_uuid == alice_node.uuid
    assert retrieved[0].target_node_uuid == bob_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    retrieved = await EntityEdge.get_by_node_uuid(graph_driver, alice_node.uuid)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == entity_edge.uuid
    assert retrieved[0].source_node_uuid == alice_node.uuid
    assert retrieved[0].target_node_uuid == bob_node.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    await entity_edge.load_fact_embedding(graph_driver)
    assert np.allclose(entity_edge.fact_embedding, edge_embedding)

    await entity_edge.delete(graph_driver)
    edge_count = await get_edge_count(graph_driver, entity_edge.uuid)
    assert edge_count == 0

    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 0

    await bob_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, bob_node.uuid)
    assert node_count == 0

    await graph_driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'driver',
    drivers,
    ids=drivers,
)
async def test_community_edge(driver):
    graph_driver = get_driver(driver)
    embedder = OpenAIEmbedder()

    now = datetime.now()

    community_node_1 = CommunityNode(
        name='Community A',
        group_id=group_id,
        summary='Community A summary',
    )
    await community_node_1.generate_name_embedding(embedder)
    node_count = await get_node_count(graph_driver, community_node_1.uuid)
    assert node_count == 0
    await community_node_1.save(graph_driver)
    node_count = await get_node_count(graph_driver, community_node_1.uuid)
    assert node_count == 1

    community_node_2 = CommunityNode(
        name='Community B',
        group_id=group_id,
        summary='Community B summary',
    )
    await community_node_2.generate_name_embedding(embedder)
    node_count = await get_node_count(graph_driver, community_node_2.uuid)
    assert node_count == 0
    await community_node_2.save(graph_driver)
    node_count = await get_node_count(graph_driver, community_node_2.uuid)
    assert node_count == 1

    alice_node = EntityNode(
        name='Alice', labels=[], created_at=now, summary='Alice summary', group_id=group_id
    )
    await alice_node.generate_name_embedding(embedder)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 0
    await alice_node.save(graph_driver)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 1

    community_edge = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=community_node_2.uuid,
        created_at=now,
        group_id=group_id,
    )
    edge_count = await get_edge_count(graph_driver, community_edge.uuid)
    assert edge_count == 0
    await community_edge.save(graph_driver)
    edge_count = await get_edge_count(graph_driver, community_edge.uuid)
    assert edge_count == 1

    retrieved = await CommunityEdge.get_by_uuid(graph_driver, community_edge.uuid)
    assert retrieved.uuid == community_edge.uuid
    assert retrieved.source_node_uuid == community_node_1.uuid
    assert retrieved.target_node_uuid == community_node_2.uuid
    assert retrieved.created_at == now
    assert retrieved.group_id == group_id

    retrieved = await CommunityEdge.get_by_uuids(graph_driver, [community_edge.uuid])
    assert len(retrieved) == 1
    assert retrieved[0].uuid == community_edge.uuid
    assert retrieved[0].source_node_uuid == community_node_1.uuid
    assert retrieved[0].target_node_uuid == community_node_2.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    retrieved = await CommunityEdge.get_by_group_ids(graph_driver, [group_id], limit=1)
    assert len(retrieved) == 1
    assert retrieved[0].uuid == community_edge.uuid
    assert retrieved[0].source_node_uuid == community_node_1.uuid
    assert retrieved[0].target_node_uuid == community_node_2.uuid
    assert retrieved[0].created_at == now
    assert retrieved[0].group_id == group_id

    await community_edge.delete(graph_driver)
    edge_count = await get_edge_count(graph_driver, community_edge.uuid)
    assert edge_count == 0

    await alice_node.delete(graph_driver)
    node_count = await get_node_count(graph_driver, alice_node.uuid)
    assert node_count == 0

    await community_node_1.delete(graph_driver)
    node_count = await get_node_count(graph_driver, community_node_1.uuid)
    assert node_count == 0

    await community_node_2.delete(graph_driver)
    node_count = await get_node_count(graph_driver, community_node_2.uuid)
    assert node_count == 0

    await graph_driver.close()


async def get_node_count(driver: GraphDriver, uuid: str):
    results, _, _ = await driver.execute_query(
        """
        MATCH (n {uuid: $uuid})
        RETURN COUNT(n) as count
        """,
        uuid=uuid,
    )
    return int(results[0]['count'])


async def get_edge_count(driver: GraphDriver, uuid: str):
    results, _, _ = await driver.execute_query(
        """
        MATCH (n)-[e {uuid: $uuid}]->(m)
        RETURN COUNT(e) as count
        UNION ALL
        MATCH (n)-[e:RELATES_TO]->(m {uuid: $uuid})-[e2:RELATES_TO]->(m2)
        RETURN COUNT(m) as count
        """,
        uuid=uuid,
    )
    return sum(int(result['count']) for result in results)
