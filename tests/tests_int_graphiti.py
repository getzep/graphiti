import logging
import sys
import os

import pytest

from core.search.search import SearchConfig

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
    episodes = context["episodes"]
    nodes = context["nodes"]
    edges = context["edges"]

    "Entities:\n"
    for node in nodes:
        formatted_string += f"  UUID: {node.uuid}\n"
        formatted_string += f"    Name: {node.name}\n"
        formatted_string += f"    Summary: {node.summary}\n"

    formatted_string += "Facts:\n"
    for edge in edges:
        formatted_string += f"  - {edge.fact}\n"
    formatted_string += "\n"

    return formatted_string.strip()


@pytest.mark.asyncio
async def test_graphiti_init():
    logger = setup_logging()
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, None)

    search_config = SearchConfig()

    context = await graphiti.search("Freakenomics guest", datetime.now(), search_config)

    logger.info("\nQUERY: Freakenomics guest" + "\nRESULT:\n" + format_context(context))

    context = await graphiti.search("tania tetlow", datetime.now(), search_config)

    logger.info("\nQUERY: Tania Tetlow" + "\nRESULT:\n" + format_context(context))

    context = await graphiti.search(
        "issues with higher ed", datetime.now(), search_config
    )

    logger.info(
        "\nQUERY: issues with higher ed" + "\nRESULT:\n" + format_context(context)
    )
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
