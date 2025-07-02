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

from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import parse_db_date, semaphore_gather
from graphiti_core.nodes import EpisodeType, EpisodicNode

EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def build_indices_and_constraints(driver: GraphDriver, delete_existing: bool = False):
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
                'MATCH (n:Entity|Episodic|Community) WHERE n.group_id IN $group_ids DETACH DELETE n',
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
                                MATCH (e:Episodic) WHERE e.valid_at <= $reference_time
                                """
        + group_id_filter
        + source_filter
        + """
        RETURN e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.group_id AS group_id,
            e.name AS name,
            e.source_description AS source_description,
            e.source AS source
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

    episodes = [
        EpisodicNode(
            content=record['content'],
            created_at=parse_db_date(record['created_at'])
            or datetime.min.replace(tzinfo=timezone.utc),
            valid_at=parse_db_date(record['valid_at']) or datetime.min.replace(tzinfo=timezone.utc),
            uuid=record['uuid'],
            group_id=record['group_id'],
            source=EpisodeType.from_str(record['source']),
            name=record['name'],
            source_description=record['source_description'],
        )
        for record in result
    ]
    return list(reversed(episodes))  # Return in chronological order
