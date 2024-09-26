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

import asyncio
import logging
import os
import sys
from datetime import datetime

import pytest
from dotenv import load_dotenv

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

pytestmark = pytest.mark.integration

pytest_plugins = ('pytest_asyncio',)

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4j_USER = os.getenv('NEO4J_USER')
NEO4j_PASSWORD = os.getenv('NEO4J_PASSWORD')


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


def format_context(facts):
    formatted_string = ''
    formatted_string += 'FACTS:\n'
    for fact in facts:
        formatted_string += f'  - {fact}\n'
    formatted_string += '\n'

    return formatted_string.strip()


@pytest.mark.asyncio
async def test_graphiti_init():
    logger = setup_logging()
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD)

    edges = await graphiti.search(
        'tania tetlow', center_node_uuid='4bf7ebb3-3a98-46c7-90a6-8e516c487961', group_ids=None
    )

    logger.info('\nQUERY: Tania Tetlow\n' + format_context([edge.fact for edge in edges]))

    edges = await graphiti.search('issues with higher ed', group_ids=None)

    logger.info('\nQUERY: issues with higher ed\n' + format_context([edge.fact for edge in edges]))

    results = await graphiti._search(
        'issues with higher ed', COMBINED_HYBRID_SEARCH_RRF, group_ids=None
    )
    pretty_results = {
        'edges': [edge.fact for edge in results.edges],
        'nodes': [node.name for node in results.nodes],
        'communities': [community.name for community in results.communities],
    }

    logger.info(pretty_results)
    await graphiti.close()


@pytest.mark.asyncio
async def test_graph_integration():
    client = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD)
    embedder = client.llm_client.get_embedder()
    driver = client.driver

    now = datetime.now()
    episode = EpisodicNode(
        name='test_episode',
        labels=[],
        created_at=now,
        valid_at=now,
        source='message',
        source_description='conversation message',
        content='Alice likes Bob',
        entity_edges=[],
    )

    alice_node = EntityNode(
        name='Alice',
        labels=[],
        created_at=now,
        summary='Alice summary',
    )

    bob_node = EntityNode(name='Bob', labels=[], created_at=now, summary='Bob summary')

    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episode.uuid, target_node_uuid=alice_node.uuid, created_at=now
    )

    episodic_edge_2 = EpisodicEdge(
        source_node_uuid=episode.uuid, target_node_uuid=bob_node.uuid, created_at=now
    )

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
    )

    await entity_edge.generate_embedding(embedder, client.llm_client.embedding_model)

    nodes = [episode, alice_node, bob_node]
    edges = [episodic_edge_1, episodic_edge_2, entity_edge]

    # test save
    await asyncio.gather(*[node.save(driver) for node in nodes])
    await asyncio.gather(*[edge.save(driver) for edge in edges])

    # test get
    assert await EpisodicNode.get_by_uuid(driver, episode.uuid) is not None
    assert await EntityNode.get_by_uuid(driver, alice_node.uuid) is not None
    assert await EpisodicEdge.get_by_uuid(driver, episodic_edge_1.uuid) is not None
    assert await EntityEdge.get_by_uuid(driver, entity_edge.uuid) is not None

    # test delete
    await asyncio.gather(*[node.delete(driver) for node in nodes])
    await asyncio.gather(*[edge.delete(driver) for edge in edges])
