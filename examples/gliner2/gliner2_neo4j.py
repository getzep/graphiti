"""
Copyright 2025, Zep Software, Inc.

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
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gliner2_client import GLiNER2Client
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################
# GLiNER2 is a lightweight extraction model
# (205M-340M params) that runs locally on CPU.
# It handles entity and relation extraction,
# while an OpenAI client handles reasoning tasks
# (deduplication, summarization).
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
neo4j_uri = os.environ.get('NEO4J_URI')
neo4j_user = os.environ.get('NEO4J_USER')
neo4j_password = os.environ.get('NEO4J_PASSWORD')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

# GLiNER2 model configuration
gliner2_model = os.environ.get('GLINER2_MODEL', 'fastino/gliner2-base-v1')


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Set up a hybrid LLM client: GLiNER2 handles
    # entity and relation extraction locally, while
    # OpenAI handles deduplication, summarization,
    # and other reasoning tasks.
    #################################################

    # Create the OpenAI client for reasoning tasks
    openai_client = OpenAIClient(
        config=LLMConfig(
            api_key=os.environ.get('OPENAI_API_KEY'),
        ),
    )

    # Create the GLiNER2 hybrid client
    gliner2_client = GLiNER2Client(
        config=LLMConfig(model=gliner2_model),
        llm_client=openai_client,
        threshold=0.5,
    )

    # Initialize Graphiti with the GLiNER2 hybrid client
    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=gliner2_client,
    )

    try:
        #################################################
        # ADDING EPISODES
        #################################################
        # Entity and relation extraction from these
        # episodes will be handled by GLiNER2 locally.
        # Deduplication and summarization will be
        # delegated to OpenAI.
        #################################################

        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'California Politics {i}',
                episode_body=(
                    episode['content']
                    if isinstance(episode['content'], str)
                    else json.dumps(episode['content'])
                ),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: California Politics {i} ({episode["type"].value})')

        #################################################
        # SEARCH
        #################################################

        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # TOKEN USAGE
        #################################################
        # GLiNER2 token estimates are approximate.
        # The OpenAI client tracks actual API usage.
        #################################################

        print('\nGLiNER2 client token usage (estimated):')
        gliner2_client.token_tracker.print_summary()

        print('\nOpenAI client token usage:')
        openai_client.token_tracker.print_summary()

    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
