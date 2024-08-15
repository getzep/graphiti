import os

import pytest
import asyncio
from dotenv import load_dotenv

from neo4j import AsyncGraphDatabase
from openai import OpenAI

from core.edges import EpisodicEdge, EntityEdge
from core.graphiti import Graphiti
from core.nodes import EpisodicNode, EntityNode
from datetime import datetime

pytest_plugins = ("pytest_asyncio",)

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4j_USER = os.getenv("NEO4J_USER")
NEO4j_PASSWORD = os.getenv("NEO4J_PASSWORD")


@pytest.mark.asyncio
async def test_graphiti_init():
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, None)
    await graphiti.build_indices()
    graphiti.close()


@pytest.mark.asyncio
async def test_graph_integration():
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4j_USER, NEO4j_PASSWORD),
    )
    embedder = OpenAI().embeddings

    now = datetime.now()
    episode = EpisodicNode(
        name="test_episode",
        labels=[],
        created_at=now,
        source="message",
        source_description="conversation message",
        content="Alice likes Bob",
        entity_edges=[],
    )

    alice_node = EntityNode(
        name="Alice",
        labels=[],
        created_at=now,
        summary="Alice summary",
    )

    bob_node = EntityNode(name="Bob", labels=[], created_at=now, summary="Bob summary")

    episodic_edge_1 = EpisodicEdge(
        source_node=episode, target_node=alice_node, created_at=now
    )

    episodic_edge_2 = EpisodicEdge(
        source_node=episode, target_node=bob_node, created_at=now
    )

    entity_edge = EntityEdge(
        source_node=alice_node,
        target_node=bob_node,
        created_at=now,
        name="likes",
        fact="Alice likes Bob",
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
    )

    entity_edge.generate_embedding(embedder)

    nodes = [episode, alice_node, bob_node]
    edges = [episodic_edge_1, episodic_edge_2, entity_edge]

    await asyncio.gather(*[node.save(driver) for node in nodes])
    await asyncio.gather(*[edge.save(driver) for edge in edges])
