"""
Quickstart example for Graphiti with Neptune Database using Gremlin query language.

This example demonstrates how to use Graphiti with AWS Neptune Database using
the Gremlin query language instead of openCypher.

Prerequisites:
1. AWS Neptune Database cluster (not Neptune Analytics - Gremlin is not supported)
2. AWS OpenSearch Service cluster for fulltext search
3. Environment variables:
   - OPENAI_API_KEY: Your OpenAI API key
   - NEPTUNE_HOST: Neptune Database endpoint (e.g., neptune-db://your-cluster.cluster-xxx.us-east-1.neptune.amazonaws.com)
   - NEPTUNE_AOSS_HOST: OpenSearch endpoint
4. AWS credentials configured (via ~/.aws/credentials or environment variables)

Note: Gremlin support in Graphiti is experimental and currently focuses on
basic graph operations. Some advanced features may still use OpenSearch for
fulltext and vector similarity searches.
"""

import asyncio
import logging
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.driver.driver import QueryLanguage
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.nodes import EpisodeType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Main function demonstrating Graphiti with Neptune Gremlin.
    """
    # Initialize Neptune driver with Gremlin query language
    # Note: Only Neptune Database supports Gremlin (not Neptune Analytics)
    driver = NeptuneDriver(
        host='neptune-db://your-cluster.cluster-xxx.us-east-1.neptune.amazonaws.com',
        aoss_host='your-aoss-cluster.us-east-1.aoss.amazonaws.com',
        port=8182,
        query_language=QueryLanguage.GREMLIN,  # Use Gremlin instead of Cypher
    )

    # Initialize LLM client
    llm_client = OpenAIClient()

    # Initialize Graphiti
    graphiti = Graphiti(driver, llm_client)

    logger.info('Initializing graph indices...')
    await graphiti.build_indices_and_constraints()

    # Add some episodes
    episodes = [
        'Kamala Harris is the Attorney General of California. She was previously '
        'the district attorney for San Francisco.',
        'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
    ]

    logger.info('Adding episodes to the knowledge graph...')
    for episode in episodes:
        await graphiti.add_episode(
            name='Kamala Harris Career',
            episode_body=episode,
            source_description='Wikipedia article on Kamala Harris',
            reference_time=datetime.now(),
            source=EpisodeType.text,
        )

    # Search the graph
    logger.info('\\nSearching for information about Kamala Harris...')
    results = await graphiti.search('What positions has Kamala Harris held?')

    logger.info('\\nSearch Results:')
    logger.info(f'Nodes: {len(results.nodes)}')
    for node in results.nodes:
        logger.info(f'  - {node.name}: {node.summary}')

    logger.info(f'\\nEdges: {len(results.edges)}')
    for edge in results.edges:
        logger.info(f'  - {edge.name}: {edge.fact}')

    # Note: With Gremlin, the underlying queries use Gremlin traversal syntax
    # instead of Cypher, but the high-level Graphiti API remains the same.
    # The driver automatically handles query translation based on query_language setting.

    logger.info('\\nClosing driver...')
    await driver.close()

    logger.info('Done!')


if __name__ == '__main__':
    """
    Example output:

    INFO:__main__:Initializing graph indices...
    INFO:__main__:Adding episodes to the knowledge graph...
    INFO:__main__:
    Searching for information about Kamala Harris...
    INFO:__main__:
    Search Results:
    INFO:__main__:Nodes: 3
    INFO:__main__:  - Kamala Harris: Former Attorney General of California
    INFO:__main__:  - California: US State
    INFO:__main__:  - San Francisco: City in California
    INFO:__main__:
    Edges: 2
    INFO:__main__:  - held_position: Kamala Harris was Attorney General of California
    INFO:__main__:  - previously_served_as: Kamala Harris was district attorney for San Francisco
    INFO:__main__:
    Closing driver...
    INFO:__main__:Done!
    """
    asyncio.run(main())
