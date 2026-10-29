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
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.spanner_driver import SpannerDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Google Cloud Spanner Graph.
#
# Prerequisites in your Google Cloud project (see
# the "Google Cloud Spanner Graph" section of the
# main README for the gcloud commands):
#   - a Spanner instance on the Enterprise edition
#     (or the 90-day free trial instance)
#   - an empty database in that instance
#   - roles/spanner.databaseUser for the runtime
#     identity, plus roles/spanner.databaseAdmin
#     for whoever runs the schema creation below
# Credentials come from Application Default
# Credentials (gcloud auth application-default
# login, an attached service account, or
# GOOGLE_APPLICATION_CREDENTIALS).
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Spanner connection parameters
spanner_project = os.environ.get('SPANNER_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT')
spanner_instance = os.environ.get('SPANNER_INSTANCE')
spanner_database = os.environ.get('SPANNER_DATABASE')

if not spanner_project:
    raise ValueError('SPANNER_PROJECT (or GOOGLE_CLOUD_PROJECT) must be set')

if not spanner_instance:
    raise ValueError('SPANNER_INSTANCE must be set')

if not spanner_database:
    raise ValueError('SPANNER_DATABASE must be set')


def build_clients():
    """Gemini when GOOGLE_API_KEY is set (a Google-only stack), otherwise Graphiti's OpenAI defaults.

    Returns (llm_client, embedder, cross_encoder, embedding_dim). The embedding size is passed
    to the driver because the Spanner vector columns pin it.
    """
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key:
        return None, None, None, None

    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

    embedding_dim = 1536  # gemini-embedding-001 supports 768, 1536 and 3072
    return (
        GeminiClient(config=LLMConfig(api_key=google_api_key, model='gemini-2.5-flash')),
        GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=google_api_key,
                embedding_model='gemini-embedding-001',
                embedding_dim=embedding_dim,
            )
        ),
        GeminiRerankerClient(
            config=LLMConfig(api_key=google_api_key, model='gemini-2.5-flash-lite')
        ),
        embedding_dim,
    )


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Spanner and set up Graphiti's schema.
    # Spanner is schema-first: the tables, indexes and
    # the property graph must exist before any data
    # operation. build_indices_and_constraints() is
    # idempotent and only needs to be run once per
    # database.
    #################################################

    llm_client, embedder, cross_encoder, embedding_dim = build_clients()

    driver_kwargs = {}
    if embedding_dim is not None:
        driver_kwargs['embedding_dim'] = embedding_dim  # otherwise EMBEDDING_DIM (OpenAI default)
    driver = SpannerDriver(
        project=spanner_project,
        instance=spanner_instance,
        database=spanner_database,
        **driver_kwargs,
    )

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
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
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # Hybrid search over relationships (edges):
        # Spanner full-text search on the facts plus
        # cosine similarity over the fact embeddings,
        # fused with reciprocal rank fusion.
        #################################################

        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # Rerank results by graph distance from a node,
        # computed with a GQL traversal over the
        # property graph.
        #################################################

        if results:
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Who was the California Attorney General?', center_node_uuid=center_node_uuid
            )

            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Search recipes return nodes instead of edges.
        #################################################

        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5

        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Release the Spanner session pool and the
        # driver's worker threads.
        #################################################

        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
