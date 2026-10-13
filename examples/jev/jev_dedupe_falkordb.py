"""
Copyright 2026, Zep Software, Inc.

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
from datetime import datetime, timezone
from logging import INFO
from uuid import uuid4

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import (
    JevClient,
    LLMRuntime,
    LLMTransport,
    OpenAIClient,
    PromptRoutes,
    jev_prompt_overrides,
)
from graphiti_core.nodes import EntityNode, EpisodeType

#################################################
# CONFIGURATION
#################################################
# Set up logging and read the database and API
# settings from the environment.
#################################################

logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('JEV_API_KEY'):
    raise ValueError('OPENAI_API_KEY and JEV_API_KEY must be set')

falkor_username = os.environ.get('FALKORDB_USERNAME')
falkor_password = os.environ.get('FALKORDB_PASSWORD')
falkor_host = os.environ.get('FALKORDB_HOST', 'localhost')
falkor_port = int(os.environ.get('FALKORDB_PORT', '6379'))


async def main(falkor_driver: FalkorDriver, group_id: str):
    #################################################
    # INITIALIZATION
    #################################################
    # Use OpenAI as the default model. Route only
    # node dedupe and edge resolution to Jev.
    #################################################

    # Jev is a decision model that answers typed questions with probabilities.
    # Jev does not generate text.
    # Extraction, summaries, and all other prompts stay on gpt-6-luna.
    # Only node dedupe and edge resolution go to Jev.
    openai = LLMTransport(OpenAIClient(), models=['gpt-6-luna'])
    jev = LLMTransport(JevClient(), models=['jev-latest'])
    jev_dedupe = jev.model('jev-latest', prompt_overrides=jev_prompt_overrides())
    runtime = LLMRuntime(
        model=openai.model('gpt-6-luna'),
        routes=PromptRoutes(
            dedupe_nodes=PromptRoutes.DedupeNodes(nodes=jev_dedupe),
            dedupe_edges=PromptRoutes.DedupeEdges(resolve_edge=jev_dedupe),
        ),
    )

    graphiti = Graphiti(graph_driver=falkor_driver, llm_runtime=runtime)

    try:
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Each episode uses the same group and a later
        # reference time. Jev resolves the entities and
        # the facts of each episode.
        #################################################

        episodes = [
            {
                'content': (
                    'Alice Chen is a software engineer at Acme. '
                    'Alice manages the customer billing platform.'
                ),
                'reference_time': datetime(2025, 1, 1, tzinfo=timezone.utc),
                'description': 'Alice Chen joins Acme',
            },
            {
                'content': (
                    'Ms. Chen works for Acme as a software engineer. '
                    'Alice is responsible for managing the customer billing platform.'
                ),
                'reference_time': datetime(2025, 1, 4, tzinfo=timezone.utc),
                'description': 'Ms. Chen continues at Acme',
            },
            {
                'content': (
                    'On January 7, Alice Chen became a senior engineer at Acme. '
                    'On January 10, she left Acme and joined Globex as a software engineer. '
                    'She continues to manage the customer billing platform.'
                ),
                'reference_time': datetime(2025, 1, 10, tzinfo=timezone.utc),
                'description': 'Alice Chen joins Globex',
            },
        ]

        for index, episode in enumerate(episodes, start=1):
            name = f'Jev dedupe episode {index}'
            result = await graphiti.add_episode(
                name=name,
                episode_body=episode['content'],
                source=EpisodeType.text,
                source_description=episode['description'],
                reference_time=episode['reference_time'],
                group_id=group_id,
            )

            print(f'\n--- Episode {index}: {episode["description"]} ---')
            if result.nodes:
                print(f'  Entities ({len(result.nodes)}):')
                for node in result.nodes:
                    labels = ', '.join(node.labels) if node.labels else 'Entity'
                    print(f'    - {node.name} [{labels}]')
            else:
                print('  No entities returned.')

            if result.edges:
                print(f'  Facts ({len(result.edges)}):')
                for edge in result.edges:
                    print(f'    - {edge.fact}')
            else:
                print('  No facts returned.')

        #################################################
        # GROUP RESULTS
        #################################################

        edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id])
        nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])

        print('\nAll entity facts:')
        for edge in edges:
            temporal = []
            if edge.invalid_at:
                temporal.append(f'invalid_at={edge.invalid_at.isoformat()}')
            if edge.expired_at:
                temporal.append(f'expired_at={edge.expired_at.isoformat()}')
            status = ', '.join(temporal) if temporal else 'active'
            print(f'  - {edge.fact} [{status}]')

        print('\nEntity nodes:')
        for node in nodes:
            print(f'  - {node.name} (UUID: {node.uuid})')

        jev_usage = jev.client.token_tracker.get_usage()
        print('\nJev prompt calls:')
        for prompt_name in ('dedupe_nodes.nodes', 'dedupe_edges.resolve_edge'):
            usage = jev_usage.get(prompt_name)
            call_count = usage.call_count if usage else 0
            print(f'  - {prompt_name}: {call_count}')
    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    group_id = f'jev-demo-{uuid4()}'
    falkor_driver = FalkorDriver(
        host=falkor_host,
        port=falkor_port,
        username=falkor_username,
        password=falkor_password,
        database=group_id,
    )
    asyncio.run(main(falkor_driver, group_id))
