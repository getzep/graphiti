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
import tempfile
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from redislite.async_falkordb_client import AsyncFalkorDB
from transcript_parser import parse_podcast_messages

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()


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


class Person(BaseModel):
    """A human person, fictional or nonfictional."""

    first_name: str | None = Field(..., description='First name')
    last_name: str | None = Field(..., description='Last name')
    occupation: str | None = Field(..., description="The person's work occupation")


class City(BaseModel):
    """A city"""

    country: str | None = Field(..., description='The country the city is in')


class IsPresidentOf(BaseModel):
    """Relationship between a person and the entity they are a president of"""


class InterpersonalRelationship(BaseModel):
    """A relationship between two people (e.g., knows, works with, interviewed)"""


class LocatedIn(BaseModel):
    """A relationship indicating something is located in or associated with a place"""


async def main(use_bulk: bool = False):
    setup_logging()

    # Configure LLM client
    llm_config = LLMConfig(model='gpt-4.1-mini', small_model='gpt-4.1-nano')
    llm_client = OpenAIClient(config=llm_config)

    # Use embedded FalkorDB (falkordblite) so the runner needs no external DB
    falkor_db_path = os.path.join(tempfile.gettempdir(), 'podcast_runner_falkordb.db')
    falkor_db = AsyncFalkorDB(dbfilename=falkor_db_path)
    falkor_driver = FalkorDriver(falkor_db=falkor_db)

    client = Graphiti(graph_driver=falkor_driver, llm_client=llm_client)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()
    messages = parse_podcast_messages()
    group_id = uuid4().hex

    raw_episodes: list[RawEpisode] = []
    for i, message in enumerate(messages[3:14]):
        raw_episodes.append(
            RawEpisode(
                name=f'Message {i}',
                content=f'{message.speaker_name} ({message.role}): {message.content}',
                reference_time=message.actual_timestamp,
                source=EpisodeType.message,
                source_description='Podcast Transcript',
            )
        )
    # Define edge types - note that some edge types are reused across multiple node type pairs
    # This tests the fix for preserving all signatures when edge types are shared
    edge_types = {
        'IS_PRESIDENT_OF': IsPresidentOf,
        'INTERPERSONAL_RELATIONSHIP': InterpersonalRelationship,
        'LOCATED_IN': LocatedIn,
    }

    # Edge type map with shared edge types across multiple node type pairs:
    # - INTERPERSONAL_RELATIONSHIP is used for both (Person, Person) and (Person, Entity)
    # - LOCATED_IN is used for both (Person, City) and (Entity, City)
    edge_type_map = {
        ('Person', 'Entity'): ['IS_PRESIDENT_OF', 'INTERPERSONAL_RELATIONSHIP'],
        ('Person', 'Person'): ['INTERPERSONAL_RELATIONSHIP'],  # Same type, different signature
        ('Person', 'City'): ['LOCATED_IN'],
        ('Entity', 'City'): ['LOCATED_IN'],  # Same type, different signature
    }

    if use_bulk:
        await client.add_episode_bulk(
            raw_episodes,
            group_id=group_id,
            entity_types={'Person': Person, 'City': City},
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            saga='Freakonomics Podcast',
        )
    else:
        for i, message in enumerate(messages[3:14]):
            episodes = await client.retrieve_episodes(
                message.actual_timestamp, 3, group_ids=[group_id]
            )
            episode_uuids = [episode.uuid for episode in episodes]

            await client.add_episode(
                name=f'Message {i}',
                episode_body=f'{message.speaker_name} ({message.role}): {message.content}',
                reference_time=message.actual_timestamp,
                source_description='Podcast Transcript',
                group_id=group_id,
                entity_types={'Person': Person, 'City': City},
                edge_types=edge_types,
                edge_type_map=edge_type_map,
                previous_episode_uuids=episode_uuids,
                saga='Freakonomics Podcast',
            )

    # Print token usage summary sorted by prompt type
    print('\n\nIngestion complete. Token usage by prompt type:')
    client.token_tracker.print_summary(sort_by='prompt_name')

    # Exercise search against the populated graph
    print('\n\nRunning search queries against the graph:')
    queries = [
        'Who is the president of Fordham University?',
        'What is the Freakonomics podcast about?',
        'Tania Tetlow',
    ]
    for query in queries:
        print(f'\nQuery: {query}')
        edge_results = await client.search(query, group_ids=[group_id], num_results=5)
        if not edge_results:
            print('  (no edge results)')
        for edge in edge_results:
            print(f'  - [{edge.name}] {edge.fact}')

        node_results = await client.search_(
            query,
            group_ids=[group_id],
            config=NODE_HYBRID_SEARCH_RRF.model_copy(update={'limit': 5}),
        )
        if not node_results.nodes:
            print('  (no node results)')
        for node in node_results.nodes:
            print(f'  * {node.name} ({", ".join(node.labels)})')


asyncio.run(main(False))
