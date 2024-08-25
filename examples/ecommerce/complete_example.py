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
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

from core.llm_client.anthropic_client import AnthropicClient
from core.llm_client.config import LLMConfig
from core import Graphiti
from core.utils.maintenance.graph_data_operations import clear_data
from core.nodes import EpisodeType
from core.utils.bulk_utils import RawEpisode

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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


async def main():
    setup_logging()
    llm_client = AnthropicClient(LLMConfig(api_key=os.environ.get('ANTHROPIC_API_KEY')))
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client)

    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    await ingest_products_data()

    for i, message in enumerate(shoe_conversation):
        await client.add_episode(
            name=f'Message {i}',
            episode_body=message,
            source=EpisodeType.message,
            reference_time=datetime.now(),
            source_description='Shoe conversation',
        )


shoe_conversation = [
    "SalesShoeBot: Hi, I'm the Sales Shoe Bot. How can I help you today?",
    "Jane: I'm looking for a new pair of shoes.",
    'SalesShoeBot: Of course! We are selling Allbird shoes. What kinde of material are you looking for??',
    "Jane: I'm looking for shoes made out of Eucalyptus Tree Fiber",
    'SalesShoeBot: We have just what you are looking for, are you willing to consider the Tree Breezer?',
    "Jane: Sure, I'll take a look.",
    'Jane: Actually, a friend of mine bought these last year and they hated it, do you have something else? Maybe something made out of cotton?',
    'SalesShoeBot: Im sorry to hear that! We have Anytime no show soc in rugged beige?',
    'Jane: Oh that is perfect, I LOVE that color!.',
]


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
