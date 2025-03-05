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
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver
from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import DEFAULT_DATABASE
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_SAVE,
    ENTITY_NODE_SAVE,
    EPISODIC_NODE_SAVE,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class EpisodeType(Enum):
    """
    Enumeration of different types of episodes that can be processed.

    This enum defines the various sources or formats of episodes that the system
    can handle. It's used to categorize and potentially handle different types
    of input data differently.

    Attributes:
    -----------
    message : str
        Represents a standard message-type episode. The content for this type
        should be formatted as "actor: content". For example, "user: Hello, how are you?"
        or "assistant: I'm doing well, thank you for asking."
    json : str
        Represents an episode containing a JSON string object with structured data.
    text : str
        Represents a plain text episode.
    """

    message = 'message'
    json = 'json'
    text = 'text'

    @staticmethod
    def from_str(episode_type: str):
        if episode_type == 'message':
            return EpisodeType.message
        if episode_type == 'json':
            return EpisodeType.json
        if episode_type == 'text':
            return EpisodeType.text
        logger.error(f'Episode type: {episode_type} not implemented')
        raise NotImplementedError


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...

    async def delete(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (n:Entity|Episodic|Community {uuid: $uuid})
        DETACH DELETE n
        """,
            uuid=self.uuid,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Deleted Node: {self.uuid}')

        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def delete_by_group_id(cls, driver: AsyncDriver, group_id: str):
        await driver.execute_query(
            """
        MATCH (n:Entity|Episodic|Community {group_id: $group_id})
        DETACH DELETE n
        """,
            group_id=group_id,
            database_=DEFAULT_DATABASE,
        )

        return 'SUCCESS'

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str): ...

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]): ...


class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    content: str = Field(description='raw episode data')
    valid_at: datetime = Field(
        description='datetime of when the original document was created',
    )
    entity_edges: list[str] = Field(
        description='list of entity edges referenced in this episode',
        default_factory=list,
    )

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            EPISODIC_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            source_description=self.source_description,
            content=self.content,
            entity_edges=self.entity_edges,
            created_at=self.created_at,
            valid_at=self.valid_at,
            source=self.source.value,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved Node to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic {uuid: $uuid})
            RETURN e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        if len(episodes) == 0:
            raise NodeNotFoundError(uuid)

        return episodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic) WHERE e.uuid IN $uuids
            RETURN DISTINCT
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

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
        MATCH (e:Episodic) WHERE e.group_id IN $group_ids
        """
            + cursor_query
            + """
            RETURN DISTINCT
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        ORDER BY e.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            created_at=created_at,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes


class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.name_embedding

    async def save(self, driver: AsyncDriver):
        entity_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'name_embedding': self.name_embedding,
            'group_id': self.group_id,
            'summary': self.summary,
            'created_at': self.created_at,
        }

        entity_data.update(self.attributes or {})

        result = await driver.execute_query(
            ENTITY_NODE_SAVE,
            labels=self.labels + ['Entity'],
            entity_data=entity_data,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved Node to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity {uuid: $uuid})
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity) WHERE n.uuid IN $uuids
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        return nodes

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        created_at: datetime | None = None,
    ):
        cursor_query: LiteralString = 'AND n.created_at < $created_at' if created_at else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity) WHERE n.group_id IN $group_ids
        """
            + cursor_query
            + """
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
        ORDER BY n.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            created_at=created_at,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        return nodes


class CommunityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='region summary of member nodes', default_factory=str)

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            COMMUNITY_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            summary=self.summary,
            name_embedding=self.name_embedding,
            created_at=self.created_at,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved Node to neo4j: {self.uuid}')

        return result

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.name_embedding

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community {uuid: $uuid})
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_community_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community) WHERE n.uuid IN $uuids
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        created_at: datetime | None = None,
    ):
        cursor_query: LiteralString = 'AND n.created_at < $created_at' if created_at else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community) WHERE n.group_id IN $group_ids
        """
            + cursor_query
            + """
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        ORDER BY n.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            created_at=created_at,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities


# Node helpers
def get_episodic_node_from_record(record: Any) -> EpisodicNode:
    return EpisodicNode(
        content=record['content'],
        created_at=record['created_at'].to_native().timestamp(),
        valid_at=(record['valid_at'].to_native()),
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
    )


def get_entity_node_from_record(record: Any) -> EntityNode:
    entity_node = EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        labels=record['labels'],
        created_at=record['created_at'].to_native(),
        summary=record['summary'],
        attributes=record['attributes'],
    )

    del entity_node.attributes['uuid']
    del entity_node.attributes['name']
    del entity_node.attributes['group_id']
    del entity_node.attributes['name_embedding']
    del entity_node.attributes['summary']
    del entity_node.attributes['created_at']

    return entity_node


def get_community_node_from_record(record: Any) -> CommunityNode:
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        created_at=record['created_at'].to_native(),
        summary=record['summary'],
    )
