import asyncio
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from graphiti_core import Graphiti
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


bmw_sales = [
    {
        'episode_body': 'Paul (buyer): Hi, I would like to buy a new car',
    },
    {
        'episode_body': 'Dan The Salesman (salesman): Sure, I can help you with that. What kind of car are you looking for?',
    },
    {
        'episode_body': 'Paul (buyer): I am looking for a new BMW',
    },
    {
        'episode_body': 'Dan The Salesman (salesman): Great choice! What kind of BMW are you looking for?',
    },
    {
        'episode_body': 'Paul (buyer): I am considering a BMW 3 series',
    },
    {
        'episode_body': 'Dan The Salesman (salesman): Great choice, we currently have a 2024 BMW 3 series in stock, it is a great car and costs $50,000',
    },
    {
        'episode_body': "Paul (buyer): Actually I am interested in something cheaper, I won't consider anything over $30,000",
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
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()

    for i, message in enumerate(bmw_sales):
        await client.add_episode(
            name=f'Message {i}',
            episode_body=message['episode_body'],
            source_description='',
            # reference_time=datetime.now() - timedelta(days=365 * 3),
            reference_time=datetime.now(),
        )
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
