import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from transcript_parser import parse_podcast_messages

from core import Graphiti
from core.utils.bulk_utils import BulkEpisode
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


async def main(use_bulk: bool = True):
	setup_logging()
	client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
	await clear_data(client.driver)
	await client.build_indices_and_constraints()
	messages = parse_podcast_messages()

	if not use_bulk:
		for i, message in enumerate(messages[3:14]):
			await client.add_episode(
				name=f'Message {i}',
				episode_body=f'{message.speaker_name} ({message.role}): {message.content}',
				reference_time=message.actual_timestamp,
				source_description='Podcast Transcript',
			)

	episodes: list[BulkEpisode] = [
		BulkEpisode(
			name=f'Message {i}',
			content=f'{message.speaker_name} ({message.role}): {message.content}',
			source_description='Podcast Transcript',
			episode_type='string',
			reference_time=message.actual_timestamp,
		)
		for i, message in enumerate(messages[3:14])
	]

	await client.add_episode_bulk(episodes)


asyncio.run(main(True))
