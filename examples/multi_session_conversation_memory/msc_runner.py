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
import csv
from time import time

from dotenv import load_dotenv

from examples.multi_session_conversation_memory.parse_msc_messages import parse_msc_messages, ParsedMscMessage
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
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


async def add_conversation(graphiti: Graphiti, group_id: str, messages: list[ParsedMscMessage]):
    fields = ['Group id', 'Episode #', 'Word Count', 'Ingestion Duration (ms)', 'Ingestion Rate (words/ms)']
    csv_items: list[dict] = []

    for i, message in enumerate(messages):
        word_count = len(message.content.split(" "))
        start = time()
        await graphiti.add_episode(
            name=f'Message {group_id + "-" + str(i)}',
            episode_body=f'{message.speaker_name}: {message.content}',
            reference_time=message.actual_timestamp,
            source_description='Multi-Session Conversation',
            group_id=group_id,
        )
        end = time()

        duration = (end - start) * 1000

        csv_items.append({'Group id': group_id, 'Episode #': i, 'Word Count': word_count,
                          'Ingestion Duration (ms)': duration, 'Ingestion Rate (words/ms)': word_count / duration})

    with open('../data/msc_ingestion.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writerows(csv_items)


async def main():
    setup_logging()
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    msc_messages = parse_msc_messages()[5:10]

    await asyncio.gather(
        *[add_conversation(graphiti, str(group_id), messages) for group_id, messages in enumerate(msc_messages)])

    # build communities
    # await client.build_communities()


asyncio.run(main())
