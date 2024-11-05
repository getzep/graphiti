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
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


shoe_conversation = [
    "SalesBot: Hi, I'm Allbirds Assistant! How can I help you today?",
    "John: Hi, I'm looking for a new pair of shoes.",
    'SalesBot: Of course! What kinde of material are you looking for?',
    "John: I'm looking for shoes made out of wool",
    """SalesBot: We have just what you are looking for, how do you like our Men's SuperLight Wool Runners 
    - Dark Grey (Medium Grey Sole)? They use the SuperLight Foam technology.""",
    """John: Oh, actually I bought those 2 months ago, but unfortunately found out that I was allergic to wool. 
    I think I will pass on those, maybe there is something with a retro look that you could suggest?""",
    """SalesBot: Im sorry to hear that! Would you be interested in Men's Couriers - 
    (Blizzard Sole) model? We have them in Natural Black and Basin Blue colors""",
    'John: Oh that is perfect, I LOVE the Natural Black color!. I will take those.',
]


async def add_messages(client: Graphiti):
    for i, message in enumerate(shoe_conversation):
        await client.add_episode(
            name=f'Message {i}',
            episode_body=message,
            source=EpisodeType.message,
            reference_time=datetime.now(timezone.utc),
            source_description='Shoe conversation',
        )


async def main():
    setup_logging()
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    await ingest_products_data(client)
    await add_messages(client)


async def ingest_products_data(client: Graphiti):
    script_dir = Path(__file__).parent
    json_file_path = script_dir / '../data/manybirds_products.json'

    with open(json_file_path) as file:
        products = json.load(file)['products']

    episodes: list[RawEpisode] = [
        RawEpisode(
            name=f'Product {i}',
            content=str(product),
            source_description='Allbirds products',
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )
        for i, product in enumerate(products)
    ]

    for episode in episodes:
        await client.add_episode(
            episode.name,
            episode.content,
            episode.source_description,
            episode.reference_time,
            episode.source,
        )


asyncio.run(main())
