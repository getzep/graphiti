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
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.llm_client import AnthropicClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from dotenv import load_dotenv

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
neo4j_user = os.environ.get('NEO4J_USER') or 'neo4j'
neo4j_password = os.environ.get('NEO4J_PASSWORD') or 'password'


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
    "SalesShoeBot: Hi, I'm the Sales Shoe Bot focused on selling Allbirds shoes. How can I help you today?",
    "John: Hi, I'm looking for a new pair of shoes.",
    'SalesShoeBot: Of course! What kinde of material are you looking for?',
    "John: I'm looking for shoes made out of wool",
    "SalesShoeBot: We have just what you are looking for, how do you like our Men's SuperLight Wool Runners - Dark Grey (Medium Grey Sole)? They use the SuperLight Foam technology.",
    'John: Oh, actually I bought those 2 months ago, but unfortunately found out that I was allergic to wool. I think I will pass on those, maybe there is something with a retro look that you could suggest?',
    "SalesShoeBot: Im sorry to hear that! Would you be interested in Men's Couriers - (Blizzard Sole) model? We have them in Natural Black and Basin Blue colors",
    'John: Oh that is perfect, I LOVE the Natural Black color!. I will take those.',
]


async def add_messages(client: Graphiti):
    for i, message in enumerate(shoe_conversation):
        await client.add_episode(
            name=f'Message {i}',
            episode_body=message,
            source=EpisodeType.message,
            reference_time=datetime.now(),
            source_description='Shoe conversation',
        )


async def main():
    setup_logging()
    llm_client = AnthropicClient(LLMConfig(api_key=os.environ.get('ANTHROPIC_API_KEY')))
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client)

    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    await ingest_products_data()
    await add_messages(client)


async def ingest_products_data():
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, 'allbirds_products.json')
    products = json.load(open(json_file_path))['products']
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