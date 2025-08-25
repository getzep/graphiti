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
from datetime import datetime

from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import semaphore_gather
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
)
from graphiti_core.nodes import EpisodeType, EpisodicNode, get_episodic_node_from_record

EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def build_indices_and_constraints(driver: GraphDriver, delete_existing: bool = False):
    if driver.provider == GraphProvider.NEPTUNE:
        return  # Neptune does not need indexes built
    if delete_existing:
        records, _, _ = await driver.execute_query(
            """
            SHOW INDEXES YIELD name
            """,
        )
        index_names = [record['name'] for record in records]
        await semaphore_gather(
            *[
                driver.execute_query(
                    """DROP INDEX $name""",
                    name=name,
                )
                for name in index_names
            ]
        )
    range_indices: list[LiteralString] = get_range_indices(driver.provider)

    fulltext_indices: list[LiteralString] = get_fulltext_indices(driver.provider)

    index_queries: list[LiteralString] = range_indices + fulltext_indices

    await semaphore_gather(
        *[
            driver.execute_query(
                query,
            )
            for query in index_queries
        ]
    )


async def clear_data(driver: GraphDriver, group_ids: list[str] | None = None):
    async with driver.session() as session:

        async def delete_all(tx):
            await tx.run('MATCH (n) DETACH DELETE n')

        async def delete_group_ids(tx):
            await tx.run(
                'MATCH (n) WHERE (n:Entity OR n:Episodic OR n:Community) AND n.group_id IN $group_ids DETACH DELETE n',
                group_ids=group_ids,
            )

        if group_ids is None:
            await session.execute_write(delete_all)
        else:
            await session.execute_write(delete_group_ids)


async def retrieve_episodes(
    driver: GraphDriver,
    reference_time: datetime,
    last_n: int = EPISODE_WINDOW_LEN,
    group_ids: list[str] | None = None,
    source: EpisodeType | None = None,
) -> list[EpisodicNode]:
    """
    Retrieve the last n episodic nodes from the graph.

    Args:
        driver (Driver): The Neo4j driver instance.
        reference_time (datetime): The reference time to filter episodes. Only episodes with a valid_at timestamp
                                   less than or equal to this reference_time will be retrieved. This allows for
                                   querying the graph's state at a specific point in time.
        last_n (int, optional): The number of most recent episodes to retrieve, relative to the reference_time.
        group_ids (list[str], optional): The list of group ids to return data from.

    Returns:
        list[EpisodicNode]: A list of EpisodicNode objects representing the retrieved episodes.
    """
    group_id_filter: LiteralString = (
        '\nAND e.group_id IN $group_ids' if group_ids and len(group_ids) > 0 else ''
    )
    source_filter: LiteralString = '\nAND e.source = $source' if source is not None else ''

    query: LiteralString = (
        """
                MATCH (e:Episodic)
                WHERE e.valid_at <= $reference_time
                """
        + group_id_filter
        + source_filter
        + """
        RETURN
        """
        + (
            EPISODIC_NODE_RETURN_NEPTUNE
            if driver.provider == GraphProvider.NEPTUNE
            else EPISODIC_NODE_RETURN
        )
        + """
        ORDER BY e.valid_at DESC
        LIMIT $num_episodes
        """
    )
    result, _, _ = await driver.execute_query(
        query,
        reference_time=reference_time,
        source=source.name if source is not None else None,
        num_episodes=last_n,
        group_ids=group_ids,
    )

    episodes = [get_episodic_node_from_record(record) for record in result]
    return list(reversed(episodes))  # Return in chronological order


async def build_dynamic_indexes(driver: GraphDriver, group_id: str):
    # Make sure indices exist for this group_id in Neo4j
    if driver.provider == GraphProvider.NEO4J:
        await semaphore_gather(
            driver.execute_query(
                """CREATE FULLTEXT INDEX $episode_content IF NOT EXISTS
FOR (e:"""
                + 'Episodic_'
                + group_id.replace('-', '')
                + """) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
                episode_content='episode_content_' + group_id.replace('-', ''),
            ),
            driver.execute_query(
                """CREATE FULLTEXT INDEX $node_name_and_summary IF NOT EXISTS FOR (n:"""
                + 'Entity_'
                + group_id.replace('-', '')
                + """) ON EACH [n.name, n.summary, n.group_id]""",
                node_name_and_summary='node_name_and_summary_' + group_id.replace('-', ''),
            ),
            driver.execute_query(
                """CREATE FULLTEXT INDEX $community_name IF NOT EXISTS
                                                         FOR (n:"""
                + 'Community_'
                + group_id.replace('-', '')
                + """) ON EACH [n.name, n.group_id]""",
                community_name='Community_' + group_id.replace('-', ''),
            ),
            driver.execute_query(
                """CREATE VECTOR INDEX $group_entity_vector IF NOT EXISTS
                                                        FOR (n:"""
                + 'Entity_'
                + group_id.replace('-', '')
                + """)
                               ON n.embedding
                               OPTIONS { indexConfig: {
                                `vector.dimensions`: 1024,
                                `vector.similarity_function`: 'cosine'
                               }}""",
                group_entity_vector='group_entity_vector_' + group_id.replace('-', ''),
            ),
        )
