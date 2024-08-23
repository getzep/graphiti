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

from core import Graphiti
from core.utils.bulk_utils import BulkEpisode
from core.utils.maintenance.graph_data_operations import clear_data
from examples.romeo.parse import get_romeo_messages

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
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    messages = get_romeo_messages()
    now = datetime.now()
    current_index = 0
    # make it upload in chunks of 50
    episodes: list[BulkEpisode] = [
        BulkEpisode(
            name=f'Message {i}',
            content=f'{speaker}: {speech}',
            source_description='Podcast Transcript',
            episode_type='string',
            reference_time=now + timedelta(seconds=i * 10),
        )
        for i, (speaker, speech) in enumerate(messages[current_index : current_index + 50])
    ]
    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    await client.add_episode_bulk(episodes)
    current_index += 50
    while current_index < len(messages):
        episodes: list[BulkEpisode] = [
            BulkEpisode(
                name=f'Message {i}',
                content=f'{speaker}: {speech}',
                source_description='Podcast Transcript',
                episode_type='string',
                reference_time=now + timedelta(seconds=i * 10),
            )
            for i, (speaker, speech) in enumerate(messages[current_index : current_index + 50])
        ]
        await client.add_episode_bulk(episodes)
        current_index += 50

    # for i, (speaker, speech) in enumerate(get_romeo_messages()):
    #     await client.add_episode(
    #         name=f'Message {i}',
    #         episode_body=f'{speaker}: {speech}',
    #         source_description='',
    #         # reference_time=datetime.now() - timedelta(days=365 * 3),
    #         reference_time=datetime.now(),
    #     )
    # await client.add_episode(
    # 	name='Message 5',
    # 	episode_body='Jane: I  miss Paul',
    # 	source_description='WhatsApp Message',
    # 	reference_time=datetime.now(),
    # )
    # await client.add_episode(
    # 	name='Message 6',
    # 	episode_body='Jane: I dont miss Paul anymore, I hate him',
    # 	source_description='WhatsApp Message',
    # 	reference_time=datetime.now(),
    # )

    # await client.add_episode(
    #     name="Message 3",
    #     episode_body="Assistant: The best type of apples available are Fuji apples",
    #     source_description="WhatsApp Message",
    # )
    # await client.add_episode(
    #     name="Message 4",
    #     episode_body="Paul: Oh, I actually hate those",
    #     source_description="WhatsApp Message",
    # )


asyncio.run(main())
