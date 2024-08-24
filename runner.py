import asyncio
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from core import Graphiti
from core.llm_client.anthropic_client import AnthropicClient
from core.llm_client.config import LLMConfig
from core.utils.maintenance.graph_data_operations import clear_data

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


bmw_sales = [
    {
        'episode_body': 'Paul (buyer): Hi, I would like to buy a new car',
    },
    {
        'episode_body': 'Dan (salesperson): Sure, I can help you with that. What kind of car are you looking for?',
    },
    {
        'episode_body': 'Paul (buyer): I am considering a BMW 3 series',
    },
    {
        'episode_body': 'Dan (salesperson): Great choice, we currently have a 2024 BMW 3 series in stock, it is a great car and costs $50,000',
    },
    {
        'episode_body': "Paul (buyer): Ah, I see, I can't afford that, I am interested in something cheaper, and won't consider anything over $30,000",
    },
    {
        'episode_body': 'Dan (salesperson): Are you open to considering a BMW 2 series? It is a great car and costs $30,000',
    },
    {
        'episode_body': "Paul (buyer): Just looking it up, it looks solid. Can I book a test drive tomorrow? Let's say 10am?",
    },
    {
        'episode_body': 'Dan (salesperson): Absolutely, I will book a test drive for you tomorrow at 10am',
    },
]


dates_mentioned = [
    {
        'episode_body': 'Paul (user): I have graduated from Univerity of Toronto in 2022',
    },
    {
        'episode_body': 'Jane (user): How cool, I graduated from the same school in 1999',
    },
]

times_mentioned = [
    {
        'episode_body': 'Paul (user): 15 minutes ago we put a deposit on our new house',
    },
]

time_range_mentioned = [
    {
        'episode_body': 'Paul (user): I served as a US Marine in 2015-2019',
    },
]

relative_time_range_mentioned = [
    {
        'episode_body': 'Paul (user): I lived in Toronto for 10 years, until moving to Vancouver yesterday',
    },
]


async def main():
    setup_logging()
    llm_client = AnthropicClient(LLMConfig(api_key=os.environ.get('ANTHROPIC_API_KEY')))
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()

    for i, message in enumerate(bmw_sales):
        await client.add_episode(
            name=f'Message {i}',
            episode_body=message['episode_body'],
            source_description='',
            reference_time=datetime.now(),
        )


asyncio.run(main())
