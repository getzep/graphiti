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

import asyncio
import logging
from datetime import datetime, timezone

from typing_extensions import LiteralString

from graphiti_core.driver import Driver
from graphiti_core.helpers import DEFAULT_DATABASE
from graphiti_core.nodes import EpisodeType, EpisodicNode

EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def build_indices_and_constraints(driver: Driver, delete_existing: bool = False):
    if delete_existing:
        await asyncio.gather(*driver.delete_all_indexes(DEFAULT_DATABASE))

    range_indices = [
        {"type": "node", "name": "entity_uuid", "label": "Entity", "property": "uuid"},
        {
            "type": "node",
            "name": "episode_uuid",
            "label": "Episodic",
            "property": "uuid",
        },
        {
            "type": "node",
            "name": "community_uuid",
            "label": "Community",
            "property": "uuid",
        },
        {
            "type": "relationship",
            "name": "relation_uuid",
            "label": "RELATES_TO",
            "property": "uuid",
        },
        {
            "type": "relationship",
            "name": "mention_uuid",
            "label": "MENTIONS",
            "property": "uuid",
        },
        {
            "type": "relationship",
            "name": "has_member_uuid",
            "label": "HAS_MEMBER",
            "property": "uuid",
        },
        {
            "type": "node",
            "name": "entity_group_id",
            "label": "Entity",
            "property": "group_id",
        },
        {
            "type": "node",
            "name": "episode_group_id",
            "label": "Episodic",
            "property": "group_id",
        },
        {
            "type": "relationship",
            "name": "relation_group_id",
            "label": "RELATES_TO",
            "property": "group_id",
        },
        {
            "type": "relationship",
            "name": "mention_group_id",
            "label": "MENTIONS",
            "property": "group_id",
        },
        {
            "type": "node",
            "name": "name_entity_index",
            "label": "Entity",
            "property": "name",
        },
        {
            "type": "node",
            "name": "created_at_entity_index",
            "label": "Entity",
            "property": "created_at",
        },
        {
            "type": "node",
            "name": "created_at_episodic_index",
            "label": "Episodic",
            "property": "created_at",
        },
        {
            "type": "node",
            "name": "valid_at_episodic_index",
            "label": "Episodic",
            "property": "valid_at",
        },
        {
            "type": "relationship",
            "name": "name_edge_index",
            "label": "RELATES_TO",
            "property": "name",
        },
        {
            "type": "relationship",
            "name": "created_at_edge_index",
            "label": "RELATES_TO",
            "property": "created_at",
        },
        {
            "type": "relationship",
            "name": "expired_at_edge_index",
            "label": "RELATES_TO",
            "property": "expired_at",
        },
        {
            "type": "relationship",
            "name": "valid_at_edge_index",
            "label": "RELATES_TO",
            "property": "valid_at",
        },
        {
            "type": "relationship",
            "name": "invalid_at_edge_index",
            "label": "RELATES_TO",
            "property": "invalid_at",
        },
    ]

    fulltext_indices = [
        {
            "type": "node_fulltext",
            "name": "node_name_and_summary",
            "label": "Entity",
            "properties": ["name", "summary", "group_id"],
        },
        {
            "type": "node_fulltext",
            "name": "community_name",
            "label": "Community",
            "properties": ["name", "group_id"],
        },
        {
            "type": "relationship_fulltext",
            "name": "edge_name_and_fact",
            "label": "RELATES_TO",
            "properties": ["name", "fact", "group_id"],
        },
    ]

    index_queries: list[LiteralString] = range_indices + fulltext_indices

    await asyncio.gather(
        *[
            (
                driver.create_node_fulltext_index(
                    label=index["label"],
                    properties=index["properties"],
                    index_name=index["name"],
                    database_=DEFAULT_DATABASE,
                )
                if index["type"] == "node_fulltext"
                else (
                    driver.create_relationship_fulltext_index(
                        label=index["label"],
                        properties=index["properties"],
                        index_name=index["name"],
                        database_=DEFAULT_DATABASE,
                    )
                    if index["type"] == "relationship_fulltext"
                    else (
                        driver.create_node_index(
                            label=index["label"],
                            property=index["property"],
                            index_name=index["name"],
                            database_=DEFAULT_DATABASE,
                        )
                        if index["type"] == "node"
                        else driver.create_relationship_index(
                            label=index["label"],
                            property=index["property"],
                            index_name=index["name"],
                            database_=DEFAULT_DATABASE,
                        )
                    )
                )
            )
            for index in index_queries
        ]
    )


async def clear_data(driver: Driver):
    async with driver.session() as session:

        async def delete_all(tx):
            await tx.run("MATCH (n) DETACH DELETE n")

        await session.execute_write(delete_all)


async def retrieve_episodes(
    driver: Driver,
    reference_time: datetime,
    last_n: int = EPISODE_WINDOW_LEN,
    group_ids: list[str] | None = None,
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
            content=record["content"],
            created_at=datetime.fromtimestamp(
                record["created_at"].to_native().timestamp(), timezone.utc
            ),
            valid_at=(record["valid_at"].to_native()),
            uuid=record["uuid"],
            group_id=record["group_id"],
            source=EpisodeType.from_str(record["source"]),
            name=record["name"],
            source_description=record["source_description"],
        )
        for record in result.records
    ]
    return list(reversed(episodes))  # Return in chronological order
