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
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
)
from graphiti_core.nodes import EpisodeType, EpisodicNode, get_episodic_node_from_record

EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def clear_data(driver: GraphDriver, group_ids: list[str] | None = None):
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.clear_data(driver, group_ids)
        except NotImplementedError:
            pass

    async with driver.session() as session:

        async def delete_all(tx):
            await tx.run('MATCH (n) DETACH DELETE n')

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
    saga: str | None = None,
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
        source (EpisodeType, optional): Filter episodes by source type.
        saga (str, optional): If provided, only retrieve episodes that belong to the saga with this name.

    Returns:
        list[EpisodicNode]: A list of EpisodicNode objects representing the retrieved episodes.
    """
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.retrieve_episodes(
                driver, reference_time, last_n, group_ids, source, saga
            )
        except NotImplementedError:
            pass

    # If saga is provided, retrieve episodes from that saga only
    if saga is not None:
        group_id = group_ids[0] if group_ids else None
        source_filter = 'AND e.source = $source' if source is not None else ''

        records, _, _ = await driver.execute_query(
            f"""
            MATCH (s:Saga {{name: $saga_name, group_id: $group_id}})-[:HAS_EPISODE]->(e:Episodic)
            WHERE e.valid_at <= $reference_time
            {source_filter}
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
            """,
            saga_name=saga,
            group_id=group_id,
            reference_time=reference_time,
            source=source.name if source else None,
            num_episodes=last_n,
        )

        episodes = [get_episodic_node_from_record(record) for record in records]
        return list(reversed(episodes))  # Return in chronological order

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
