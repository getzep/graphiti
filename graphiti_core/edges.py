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
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver
from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError
from graphiti_core.helpers import DEFAULT_DATABASE, parse_db_date
from graphiti_core.models.edges.edge_db_queries import (
    COMMUNITY_EDGE_SAVE,
    ENTITY_EDGE_SAVE,
    EPISODIC_EDGE_SAVE,
)
from graphiti_core.nodes import Node

logger = logging.getLogger(__name__)


class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...

    async def delete(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
        DELETE e
        """,
            uuid=self.uuid,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Deleted Edge: {self.uuid}')

        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str): ...


class EpisodicEdge(Edge):
    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            EPISODIC_EDGE_SAVE,
            episode_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved edge to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Episodic)-[e:MENTIONS {uuid: $uuid}]->(m:Entity)
        RETURN
            e.uuid As uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid, 
            m.uuid AS target_node_uuid, 
            e.created_at AS created_at
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
        WHERE e.uuid IN $uuids
        RETURN
            e.uuid As uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid, 
            m.uuid AS target_node_uuid, 
            e.created_at AS created_at
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuids[0])
        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        created_at: datetime | None = None,
    ):
        cursor_query: LiteralString = 'AND e.created_at < $created_at' if created_at else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
        WHERE e.group_id IN $group_ids
        """
            + cursor_query
            + """
        RETURN
            e.uuid As uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid, 
            m.uuid AS target_node_uuid, 
            e.created_at AS created_at
        ORDER BY e.uuid DESC 
        """
            + limit_query,
            group_ids=group_ids,
            created_at=created_at,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges


class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(description='fact representing the edge and nodes that it connects')
    fact_embedding: list[float] | None = Field(default=None, description='embedding of the fact')
    episodes: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )
    expired_at: datetime | None = Field(
        default=None, description='datetime of when the node was invalidated'
    )
    valid_at: datetime | None = Field(
        default=None, description='datetime of when the fact became true'
    )
    invalid_at: datetime | None = Field(
        default=None, description='datetime of when the fact stopped being true'
    )

    async def generate_embedding(self, embedder: EmbedderClient):
        start = time()

        text = self.fact.replace('\n', ' ')
        self.fact_embedding = await embedder.create(input_data=[text])

        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.fact_embedding

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            ENTITY_EDGE_SAVE,
            source_uuid=self.source_node_uuid,
            target_uuid=self.target_node_uuid,
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            fact=self.fact,
            fact_embedding=self.fact_embedding,
            episodes=self.episodes,
            created_at=self.created_at,
            expired_at=self.expired_at,
            valid_at=self.valid_at,
            invalid_at=self.invalid_at,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved edge to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
        RETURN
            e.uuid AS uuid,
            n.uuid AS source_node_uuid,
            m.uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.group_id AS group_id,
            e.fact AS fact,
            e.fact_embedding AS fact_embedding,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        if len(uuids) == 0:
            return []

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        WHERE e.uuid IN $uuids
        RETURN
            e.uuid AS uuid,
            n.uuid AS source_node_uuid,
            m.uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.group_id AS group_id,
            e.fact AS fact,
            e.fact_embedding AS fact_embedding,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        created_at: datetime | None = None,
    ):
        cursor_query: LiteralString = 'AND e.created_at < $created_at' if created_at else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        WHERE e.group_id IN $group_ids
        """
            + cursor_query
            + """
        RETURN
            e.uuid AS uuid,
            n.uuid AS source_node_uuid,
            m.uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.group_id AS group_id,
            e.fact AS fact,
            e.fact_embedding AS fact_embedding,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at
        ORDER BY e.uuid DESC 
        """
            + limit_query,
            group_ids=group_ids,
            created_at=created_at,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    @classmethod
    async def get_by_node_uuid(cls, driver: AsyncDriver, node_uuid: str):
        query: LiteralString = """
        MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(m:Entity)
        RETURN DISTINCT
            e.uuid AS uuid,
            n.uuid AS source_node_uuid,
            m.uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.group_id AS group_id,
            e.fact AS fact,
            e.fact_embedding AS fact_embedding,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at
        """
        records, _, _ = await driver.execute_query(
            query, node_uuid=node_uuid, database_=DEFAULT_DATABASE, routing_='r'
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        return edges


class CommunityEdge(Edge):
    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            COMMUNITY_EDGE_SAVE,
            community_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved edge to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community)-[e:HAS_MEMBER {uuid: $uuid}]->(m:Entity | Community)
        RETURN
            e.uuid As uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid, 
            m.uuid AS target_node_uuid, 
            e.created_at AS created_at
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity | Community)
        WHERE e.uuid IN $uuids
        RETURN
            e.uuid As uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid, 
            m.uuid AS target_node_uuid, 
            e.created_at AS created_at
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        created_at: datetime | None = None,
    ):
        cursor_query: LiteralString = 'AND e.created_at < $created_at' if created_at else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity | Community)
        WHERE e.group_id IN $group_ids
        """
            + cursor_query
            + """
        RETURN
            e.uuid As uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid, 
            m.uuid AS target_node_uuid, 
            e.created_at AS created_at
        ORDER BY e.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            created_at=created_at,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges


# Edge helpers
def get_episodic_edge_from_record(record: Any) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=record['created_at'].to_native(),
    )


def get_entity_edge_from_record(record: Any) -> EntityEdge:
    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        name=record['name'],
        group_id=record['group_id'],
        episodes=record['episodes'],
        fact_embedding=record['fact_embedding'],
        created_at=record['created_at'].to_native(),
        expired_at=parse_db_date(record['expired_at']),
        valid_at=parse_db_date(record['valid_at']),
        invalid_at=parse_db_date(record['invalid_at']),
    )


def get_community_edge_from_record(record: Any):
    return CommunityEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=record['created_at'].to_native(),
    )
