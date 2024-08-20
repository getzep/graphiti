import logging
import sys
import os

import pytest

pytestmark = pytest.mark.integration

import asyncio
from dotenv import load_dotenv

from neo4j import AsyncGraphDatabase
from openai import OpenAI

from core.edges import EpisodicEdge, EntityEdge
from core.graphiti import Graphiti
from core.llm_client.config import EMBEDDING_DIM
from core.nodes import EpisodicNode, EntityNode
from datetime import datetime


pytest_plugins = ("pytest_asyncio",)

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4j_USER = os.getenv("NEO4J_USER")
NEO4j_PASSWORD = os.getenv("NEO4J_PASSWORD")


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


def format_context(context):
    formatted_string = ""
    for uuid, data in context.items():
        formatted_string += f"UUID: {uuid}\n"
        formatted_string += f"  Name: {data['name']}\n"
        formatted_string += f"  Summary: {data['summary']}\n"
        formatted_string += "  Facts:\n"
        for fact in data["facts"]:
            formatted_string += f"    - {fact}\n"
        formatted_string += "\n"
    return formatted_string.strip()


@pytest.mark.asyncio
async def test_graphiti_init():
    logger = setup_logging()
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, None)
    await graphiti.build_indices()

    context = await graphiti.search("Freakenomics guest")

    logger.info("QUERY: Freakenomics guest" + "RESULT:" + format_context(context))

    context = await graphiti.search("tania tetlow")

    logger.info("QUERY: Tania Tetlow" + "RESULT:" + format_context(context))

    context = await graphiti.search("issues with higher ed")

    logger.info("QUERY: issues with higher ed" + "RESULT:" + format_context(context))
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
        source_node_uuid=episode, target_node_uuid=alice_node, created_at=now
    )

    episodic_edge_2 = EpisodicEdge(
        source_node_uuid=episode, target_node_uuid=bob_node, created_at=now
    )

    entity_edge = EntityEdge(
        source_node_uuid=alice_node.uuid,
        target_node_uuid=bob_node.uuid,
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
