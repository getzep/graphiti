import asyncio
import logging
from datetime import datetime, timezone
from typing import LiteralString

from neo4j import AsyncDriver

from core.nodes import EpisodicNode

EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def build_indices_and_constraints(driver: AsyncDriver):
	constraints: list[LiteralString] = [
		"""
        CREATE CONSTRAINT entity_name IF NOT EXISTS
        FOR (n:Entity) REQUIRE n.name IS UNIQUE
        """,
		"""
        CREATE CONSTRAINT edge_facts IF NOT EXISTS
        FOR ()-[e:RELATES_TO]-() REQUIRE e.fact IS UNIQUE
        """,
	]

	range_indices: list[LiteralString] = [
		'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
		'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
		'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
		'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
		'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
		'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
		'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
		'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
		'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
		'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
		'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
		'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
		'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
	]

	fulltext_indices: list[LiteralString] = [
		'CREATE FULLTEXT INDEX name_and_summary IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.summary]',
		'CREATE FULLTEXT INDEX name_and_fact IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact]',
	]

	vector_indices: list[LiteralString] = [
		"""
        CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
        FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
        OPTIONS {indexConfig: {
         `vector.dimensions`: 1024,
         `vector.similarity_function`: 'cosine'
        }}
        """,
		"""
        CREATE VECTOR INDEX name_embedding IF NOT EXISTS
        FOR (n:Entity) ON (n.name_embedding)
        OPTIONS {indexConfig: {
         `vector.dimensions`: 1024,
         `vector.similarity_function`: 'cosine'
        }}
        """,
	]
	index_queries: list[LiteralString] = (
		constraints + range_indices + fulltext_indices + vector_indices
	)

	await asyncio.gather(*[driver.execute_query(query) for query in index_queries])


async def clear_data(driver: AsyncDriver):
	async with driver.session() as session:

		async def delete_all(tx):
			await tx.run('MATCH (n) DETACH DELETE n')

		await session.execute_write(delete_all)


async def retrieve_episodes(
	driver: AsyncDriver,
	reference_time: datetime,
	last_n: int = EPISODE_WINDOW_LEN,
	sources: list[str] | None = 'messages',
) -> list[EpisodicNode]:
	"""Retrieve the last n episodic nodes from the graph"""
	result = await driver.execute_query(
		"""
        MATCH (e:Episodic) WHERE e.valid_at <= $reference_time
        RETURN e.content as content,
            e.created_at as created_at,
            e.valid_at as valid_at,
            e.uuid as uuid,
            e.name as name,
            e.source_description as source_description,
            e.source as source
        ORDER BY e.created_at DESC
        LIMIT $num_episodes
        """,
		reference_time=reference_time,
		num_episodes=last_n,
	)
	episodes = [
		EpisodicNode(
			content=record['content'],
			created_at=datetime.fromtimestamp(
				record['created_at'].to_native().timestamp(), timezone.utc
			),
			valid_at=(
				datetime.fromtimestamp(
					record['valid_at'].to_native().timestamp(),
					timezone.utc,
				)
				if record['valid_at'] is not None
				else None
			),
			uuid=record['uuid'],
			source=record['source'],
			name=record['name'],
			source_description=record['source_description'],
		)
		for record in result.records
	]
	return list(reversed(episodes))  # Return in chronological order
