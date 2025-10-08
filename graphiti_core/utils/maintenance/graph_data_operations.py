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
    if driver.aoss_client:
        await driver.create_aoss_indices()  # pyright: ignore[reportAttributeAccessIssue]
        return
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

    # Don't create fulltext indices if OpenSearch is being used
    if not driver.aoss_client:
        fulltext_indices: list[LiteralString] = get_fulltext_indices(driver.provider)

    if driver.provider == GraphProvider.KUZU:
        # Skip creating fulltext indices if they already exist. Need to do this manually
        # until Kuzu supports `IF NOT EXISTS` for indices.
        result, _, _ = await driver.execute_query('CALL SHOW_INDEXES() RETURN *;')
        if len(result) > 0:
            fulltext_indices = []

        # Only load the `fts` extension if it's not already loaded, otherwise throw an error.
        result, _, _ = await driver.execute_query('CALL SHOW_LOADED_EXTENSIONS() RETURN *;')
        if len(result) == 0:
            fulltext_indices.insert(
                0,
                """
                INSTALL fts;
                LOAD fts;
                """,
            )

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
            if driver.aoss_client:
                await driver.clear_aoss_indices()

        async def delete_group_ids(tx):
            labels = ['Entity', 'Episodic', 'Community']
            if driver.provider == GraphProvider.KUZU:
                labels.append('RelatesToNode_')

            for label in labels:
                await tx.run(
                    f"""
                    MATCH (n:{label})
                    WHERE n.group_id IN $group_ids
                    DETACH DELETE n
                    """,
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

    query_params: dict = {}
    query_filter = ''
    if group_ids and len(group_ids) > 0:
        query_filter += '\nAND e.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    if source is not None:
        query_filter += '\nAND e.source = $source'
        query_params['source'] = source.name

    query: LiteralString = (
        """
                        MATCH (e:Episodic)
                        WHERE e.valid_at <= $reference_time
                        """
        + query_filter
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
        num_episodes=last_n,
        **query_params,
    )

    episodes = [get_episodic_node_from_record(record) for record in result]
    return list(reversed(episodes))  # Return in chronological order
