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

from examples.multi_session_conversation_memory.parse_msc_messages import (
    ParsedMscMessage,
    parse_msc_messages,
)
from graphiti_core import Graphiti
from graphiti_core.helpers import semaphore_gather

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
    for i, message in enumerate(messages):
        await graphiti.add_episode(
            name=f'Message {group_id + "-" + str(i)}',
            episode_body=f'{message.speaker_name}: {message.content}',
            reference_time=message.actual_timestamp,
            source_description='Multi-Session Conversation',
            group_id=group_id,
        )


async def main():
    setup_logging()
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    msc_messages = parse_msc_messages()
    i = 0
    while i < len(msc_messages):
        msc_message_slice = msc_messages[i : i + 10]
        group_ids = range(len(msc_messages))[i : i + 10]

        await semaphore_gather(
            *[
                add_conversation(graphiti, str(group_id), messages)
                for group_id, messages in zip(group_ids, msc_message_slice)
            ]
        )

        i += 10


asyncio.run(main())
