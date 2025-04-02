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

import logging
from datetime import datetime, timezone

from neo4j import AsyncDriver
from typing_extensions import LiteralString

from graphiti_core.helpers import DEFAULT_DATABASE, semaphore_gather
from graphiti_core.nodes import EpisodeType, EpisodicNode

EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def build_indices_and_constraints(driver: AsyncDriver, delete_existing: bool = False):
    if delete_existing:
        records, _, _ = await driver.execute_query(
            """
        SHOW INDEXES YIELD name
        """,
            database_=DEFAULT_DATABASE,
        )
        index_names = [record['name'] for record in records]
        await semaphore_gather(
            *[
                driver.execute_query(
                    """DROP INDEX $name""",
                    name=name,
                    _database=DEFAULT_DATABASE,
                )
                for name in index_names
            ]
        )

    range_indices: list[LiteralString] = [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
        'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
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
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS 
        FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS 
        FOR (n:Community) ON EACH [n.name, n.group_id]""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS 
        FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
    ]

    index_queries: list[LiteralString] = range_indices + fulltext_indices

    await semaphore_gather(
        *[
            driver.execute_query(
                query,
                _database=DEFAULT_DATABASE,
            )
            for query in index_queries
        ]
    )


async def clear_data(driver: AsyncDriver, group_ids: list[str] | None = None):
    async with driver.session() as session:

        async def delete_all(tx):
            await tx.run('MATCH (n) DETACH DELETE n')

        async def delete_group_ids(tx):
            await tx.run(
                'MATCH (n:Entity|Episodic|Community) WHERE n.group_id IN $group_ids DETACH DELETE n',
                group_ids=group_ids,
            )

        if group_ids is None:
            await session.execute_write(delete_all)
        else:
            await session.execute_write(delete_group_ids)


async def retrieve_episodes(
    driver: AsyncDriver,
    reference_time: datetime,
    last_n: int = EPISODE_WINDOW_LEN,
    group_ids: list[str] | None = None,
) -> list[EpisodicNode]:
    """
    Retrieve the last n episodic nodes from the graph.

    Args:
        driver (AsyncDriver): The Neo4j driver instance.
        reference_time (datetime): The reference time to filter episodes. Only episodes with a valid_at timestamp
                                   less than or equal to this reference_time will be retrieved. This allows for
                                   querying the graph's state at a specific point in time.
        last_n (int, optional): The number of most recent episodes to retrieve, relative to the reference_time.
        group_ids (list[str], optional): The list of group ids to return data from.

    Returns:
        list[EpisodicNode]: A list of EpisodicNode objects representing the retrieved episodes.
    """
    result = await driver.execute_query(
        """
        MATCH (e:Episodic) WHERE e.valid_at <= $reference_time 
        AND ($group_ids IS NULL) OR e.group_id in $group_ids
        RETURN e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.group_id AS group_id,
            e.name AS name,
            e.source_description AS source_description,
            e.source AS source
        ORDER BY e.created_at DESC
        LIMIT $num_episodes
        """,
        reference_time=reference_time,
        num_episodes=last_n,
        group_ids=group_ids,
        _database=DEFAULT_DATABASE,
    )
    episodes = [
        EpisodicNode(
            content=record['content'],
            created_at=datetime.fromtimestamp(
                record['created_at'].to_native().timestamp(), timezone.utc
            ),
            valid_at=(record['valid_at'].to_native()),
            uuid=record['uuid'],
            group_id=record['group_id'],
            source=EpisodeType.from_str(record['source']),
            name=record['name'],
            source_description=record['source_description'],
        )
        for record in result.records
    ]
    return list(reversed(episodes))  # Return in chronological order
