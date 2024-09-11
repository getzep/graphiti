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

from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.llm_client.config import EMBEDDING_DIM
from graphiti_core.nodes import Node

logger = logging.getLogger(__name__)


class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: uuid4().hex)
    group_id: str | None = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...

    async def delete(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (n)-[e {uuid: $uuid}]->(m)
        DELETE e
        """,
            uuid=self.uuid,
        )

        logger.info(f'Deleted Edge: {self.uuid}')

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
            """
        MATCH (episode:Episodic {uuid: $episode_uuid}) 
        MATCH (node:Entity {uuid: $entity_uuid}) 
        MERGE (episode)-[r:MENTIONS {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN r.uuid AS uuid""",
            episode_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
        )

        logger.info(f'Saved edge to neo4j: {self.uuid}')

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
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        logger.info(f'Found Edge: {uuid}')
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]


class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(description='fact representing the edge and nodes that it connects')
    fact_embedding: list[float] | None = Field(default=None, description='embedding of the fact')
    episodes: list[str] | None = Field(
        default=None,
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

    async def generate_embedding(self, embedder, model='text-embedding-3-small'):
        start = time()

        text = self.fact.replace('\n', ' ')
        embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
        self.fact_embedding = embedding[:EMBEDDING_DIM]

        end = time()
        logger.info(f'embedded {text} in {end - start} ms')

        return embedding

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (source:Entity {uuid: $source_uuid}) 
        MATCH (target:Entity {uuid: $target_uuid}) 
        MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
        SET r = {uuid: $uuid, name: $name, group_id: $group_id, fact: $fact, fact_embedding: $fact_embedding, 
        episodes: $episodes, created_at: $created_at, expired_at: $expired_at, 
        valid_at: $valid_at, invalid_at: $invalid_at}
        RETURN r.uuid AS uuid""",
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
        )

        logger.info(f'Saved edge to neo4j: {self.uuid}')

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
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        logger.info(f'Found Edge: {uuid}')
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]


class CommunityEdge(Edge):
    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (community:Community {uuid: $community_uuid}) 
        MATCH (node:Entity | Community {uuid: $entity_uuid}) 
        MERGE (community)-[r:HAS_MEMBER {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN r.uuid AS uuid""",
            community_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
        )

        logger.info(f'Saved edge to neo4j: {self.uuid}')

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
        )

        edges = [get_community_edge_from_record(record) for record in records]

        logger.info(f'Found Edge: {uuid}')

        return edges[0]


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
