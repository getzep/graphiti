from core import Graphiti
from core.utils.maintenance.graph_data_operations import clear_data
from dotenv import load_dotenv
import os
import asyncio
import logging
import sys

load_dotenv()

neo4j_uri = os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
neo4j_user = os.environ.get("NEO4J_USER") or "neo4j"
neo4j_password = os.environ.get("NEO4J_PASSWORD") or "password"


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


async def main():
    setup_logging()
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    await clear_data(client.driver)

    # await client.build_indices()
    await client.add_episode(
        name="Message 1",
        episode_body="Paul: I love apples",
        source_description="WhatsApp Message",
    )
    await client.add_episode(
        name="Message 2",
        episode_body="Paul: I own many bananas",
        source_description="WhatsApp Message",
    )
    await client.add_episode(
        name="Message 3",
        episode_body="Assistant: The best type of apples available are Fuji apples",
        source_description="WhatsApp Message",
    )
    await client.add_episode(
        name="Message 4",
        episode_body="Paul: Oh, I actually hate those",
        source_description="WhatsApp Message",
    )


asyncio.run(main())
