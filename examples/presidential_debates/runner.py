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

from dotenv import load_dotenv

from examples.presidential_debates.parser import get_debate_messages
from graphiti_core import Graphiti
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

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
    messages = get_debate_messages()
    print(messages[:3])
    print(len(messages))
    # i need to create tuples from each 3 messages in the messages
    message_tuples = [messages[i : i + 5] for i in range(0, len(messages), 5)]
    # # neets to be october 3 2000
    now = datetime(2000, 10, 3)
    # episodes: list[BulkEpisode] = [
    #     BulkEpisode(
    #         name=f'Statement {i + 1}',
    #         content=f"{episode['role']}: {episode['statement']}",
    #         source_description='Bush Gore presidential_debates 2001',
    #         episode_type='string',
    #         reference_time=now + timedelta(seconds=i * 10),
    #     )
    #     for i, episode in enumerate(messages)
    # ]

    # await clear_data(client.driver)
    # await client.build_indices_and_constraints()
    # await client.add_episode_bulk(episodes)

    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    for i, episode_list in enumerate(message_tuples[:4]):
        combined_roles_and_statements_str = ''.join(
            f"{episode['role']}: {episode['statement']}\n" for episode in episode_list
        )
        await client.add_episode(
            name=f'Statement {i + 1}',
            episode_body=combined_roles_and_statements_str,
            source_description='Bush Gore depate 2001',
            reference_time=now + timedelta(seconds=i * 10),
        )


asyncio.run(main())
