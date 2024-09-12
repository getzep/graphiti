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

from dotenv import load_dotenv
from transcript_parser import parse_podcast_messages

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
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


async def main(use_bulk: bool = True):
    setup_logging()
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    messages = parse_podcast_messages()

    if not use_bulk:
        for i, message in enumerate(messages[3:130]):
            await client.add_episode(
                name=f'Message {i}',
                episode_body=f'{message.speaker_name} ({message.role}): {message.content}',
                reference_time=message.actual_timestamp,
                source_description='Podcast Transcript',
                group_id='1',
            )
        return

    episodes: list[RawEpisode] = [
        RawEpisode(
            name=f'Message {i}',
            content=f'{message.speaker_name} ({message.role}): {message.content}',
            source=EpisodeType.message,
            source_description='Podcast Transcript',
            reference_time=message.actual_timestamp,
        )
        for i, message in enumerate(messages[3:20])
    ]

    await client.add_episode_bulk(episodes, None)


asyncio.run(main(False))
