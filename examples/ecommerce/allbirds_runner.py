import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from core import Graphiti
from core.nodes import EpisodeType
from core.utils.bulk_utils import RawEpisode
from core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
neo4j_user = os.environ.get('NEO4J_USER') or 'neo4j'
neo4j_password = os.environ.get('NEO4J_PASSWORD') or 'password'

logger = logging.getLogger(__name__)


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


async def main(use_bulk: bool = True):
    setup_logging()
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, 'allbirds_products.json')
    products = json.load(open(json_file_path))['products']
    logger.info(products)

    if not use_bulk:
        for i, product in enumerate(products):
            await client.add_episode(
                name=f'Product {i}',
                episode_body=json.dumps(product),
                source=EpisodeType.json,
                reference_time=datetime.now(),
                source_description='Allbirds products',
            )

    episodes: list[RawEpisode] = [
        RawEpisode(
            name=f'Product {i}',
            content=str(product),
            source_description='Allbirds products',
            source=EpisodeType.json,
            reference_time=datetime.now(),
        )
        for i, product in enumerate(products)
    ]

    await client.add_episode_bulk(episodes)


asyncio.run(main())
