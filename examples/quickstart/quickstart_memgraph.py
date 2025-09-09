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
from graphiti_core.driver.memgraph_driver import MemgraphDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Memgraph database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Memgraph connection parameters
# Make sure Memgraph is running (default port 7687, same as Neo4j)
memgraph_uri = os.environ.get('MEMGRAPH_URI', 'bolt://localhost:7687')
memgraph_user = os.environ.get('MEMGRAPH_USER', '')  # Memgraph often doesn't require auth by default
memgraph_password = os.environ.get('MEMGRAPH_PASSWORD', '')

if not memgraph_uri:
    raise ValueError('MEMGRAPH_URI must be set')


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Memgraph and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Memgraph driver
    memgraph_driver = MemgraphDriver(memgraph_uri, memgraph_user, memgraph_password)
    
    # Initialize Graphiti with Memgraph connection
    graphiti = Graphiti(graph_driver=memgraph_driver)

    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
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
                    'predecessor': 'Jerry Brown',
                },
                'type': EpisodeType.json,
                'description': 'politician info',
            },
            {
                'content': 'Jerry Brown was the predecessor of Gavin Newsom as Governor of California',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Memgraph Demo {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            logger.info(f'Added episode: Memgraph Demo {i} ({episode["type"].value})')

        logger.info('Episodes added successfully!')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        logger.info('\n=== BASIC SEARCH ===')
        search_query = 'Who was the California Attorney General?'
        logger.info(f'Search query: {search_query}')

        # Perform semantic search across edges (relationships)
        results = await graphiti.search(search_query)

        logger.info('Search results:')
        for result in results[:3]:  # Show top 3 results
            logger.info(f'UUID: {result.uuid}')
            logger.info(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                logger.info(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                logger.info(f'Valid until: {result.invalid_at}')
            logger.info('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        if results and len(results) > 0:
            logger.info('\n=== CENTER NODE SEARCH ===')
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid
            logger.info(f'Using center node UUID: {center_node_uuid}')

            # Perform graph-based search using the source node
            reranked_results = await graphiti.search(
                search_query, center_node_uuid=center_node_uuid
            )

            logger.info('Reranked search results:')
            for result in reranked_results[:3]:
                logger.info(f'UUID: {result.uuid}')
                logger.info(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    logger.info(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    logger.info(f'Valid until: {result.invalid_at}')
                logger.info('---')
        else:
            logger.info('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH WITH RECIPES
        #################################################
        # Graphiti provides predefined search configurations
        # (recipes) that optimize search for specific patterns
        # and use cases.
        #################################################

        logger.info('\n=== NODE SEARCH WITH RECIPES ===')
        recipe_query = 'California Governor'
        logger.info(f'Recipe search query: {recipe_query}')

        # Use hybrid search recipe for balanced semantic and keyword matching
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query=recipe_query,
            config=node_search_config,
        )

        logger.info('Node search results:')
        for node in node_search_results.nodes:
            logger.info(f'Node UUID: {node.uuid}')
            logger.info(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            logger.info(f'Content Summary: {node_summary}')
            logger.info(f'Node Labels: {", ".join(node.labels)}')
            logger.info(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                logger.info('Attributes:')
                for key, value in node.attributes.items():
                    logger.info(f'  {key}: {value}')
        #################################################
        # SUMMARY STATISTICS
        #################################################
        # Get overall statistics about the knowledge graph
        #################################################

        logger.info('\n=== SUMMARY ===')
        logger.info('Memgraph database populated successfully!')
        logger.info('Knowledge graph is ready for queries and exploration.')

    except Exception as e:
        logger.error(f'An error occurred: {e}')
        raise

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Memgraph when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        logger.info('Connection closed.')


if __name__ == '__main__':
    asyncio.run(main())
