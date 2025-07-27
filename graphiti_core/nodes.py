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
from typing import Any, ClassVar
from uuid import uuid4
import json

from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_SAVE,
    ENTITY_NODE_SAVE,
    EPISODIC_NODE_SAVE,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)

ENTITY_NODE_RETURN: LiteralString = """
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes"""


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

    helix_to_uuid: ClassVar[dict[str, str]] = {}
    uuid_to_helix: ClassVar[dict[str, str]] = {}

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        result = await driver.execute_query(
            """
        MATCH (n:Entity|Episodic|Community {uuid: $uuid})
        DETACH DELETE n
        """,
            uuid=self.uuid,
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
    async def delete_by_group_id(cls, driver: GraphDriver, group_id: str):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="deleteGroup",
                group_id=group_id,
            )
            return 'SUCCESS'
        
        await driver.execute_query(
            """
        MATCH (n:Entity|Episodic|Community {group_id: $group_id})
        DETACH DELETE n
        """,
            group_id=group_id,
        )

        return 'SUCCESS'

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str): ...

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]): ...


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

    async def save(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            stored_episode = None
            if self.uuid in Node.uuid_to_helix:
                stored_episode = await driver.execute_query(
                    "",
                    query="getEpisode",
                    episode_id=Node.uuid_to_helix.get(self.uuid),
                )
                stored_episode = stored_episode.get('episode', None)

            if stored_episode is not None:
                query = "updateEpisode"
                helix_id = Node.uuid_to_helix.get(self.uuid)
            else:
                query = "createEpisode"
                helix_id = self.uuid
            
            result = await driver.execute_query(
                "",
                query=query,
                episode_id=helix_id,
                name=self.name,
                group_id=self.group_id,
                source_description=self.source_description,
                content=self.content,
                entity_edges=self.entity_edges,
                created_at=self.created_at,
                valid_at=self.valid_at,
                source=self.source.value,
                labels=self.labels,
            )

            if query == "createEpisode":
                helix_id = result.get('episode', {}).get('id', None)
                if helix_id is None:
                    raise ValueError('Failed to create episode node')
                Node.uuid_to_helix[self.uuid] = helix_id
                Node.helix_to_uuid[helix_id] = self.uuid

            if query == "updateEpisode":
                logger.debug(f'Updated Episode Node: {self.uuid}')
            else:
                logger.debug(f'Created Episode Node to Graph: {self.uuid}')

            return {'uuid': self.uuid}
        
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
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    async def delete(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="deleteEpisode",
                episode_id=Node.uuid_to_helix.get(self.uuid),
            )

            helix_id = Node.uuid_to_helix.get(self.uuid)

            Node.uuid_to_helix.pop(self.uuid, None)
            if helix_id is not None:
                Node.helix_to_uuid.pop(helix_id, None)

            logger.debug(f'Deleted Episode Node: {self.uuid}')

            return result
            
        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            if uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(uuid)

            result = await driver.execute_query(
                "",
                query="getEpisode",
                episode_id=Node.uuid_to_helix.get(uuid),
            )

            if result is None:
                raise NodeNotFoundError(uuid)

            result = result.get('episode', None)

            if result is None:
                raise NodeNotFoundError(uuid)

            result['uuid'] = uuid
            helix_id = result.get('id')

            Node.helix_to_uuid[helix_id] = uuid

            return get_episodic_node_from_record(result)

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
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        if len(episodes) == 0:
            raise NodeNotFoundError(uuid)

        return episodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == 'helixdb':
            results = []
            for uuid in uuids:
                if uuid not in Node.uuid_to_helix:
                    continue

                result = await driver.execute_query(
                    "",
                    query="getEpisode",
                    episode_id=Node.uuid_to_helix.get(uuid),
                )

                if result is None:
                    continue
                
                result = result.get('episode', None)

                if result is None:
                    continue

                result['uuid'] = uuid
                results.append(result)

                helix_id = result.get('id')
                Node.helix_to_uuid[helix_id] = uuid

            episodes = [get_episodic_node_from_record(result) for result in results]
            return episodes

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
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        if driver.provider == 'helixdb':
            query = "getEpisodesbyGroup"
            if limit is not None:
                query = "getEpisodesbyGroupLimit"

            results = []
            for group_id in group_ids:
                result = await driver.execute_query(
                    "",
                    query=query,
                    limit=limit,
                    group_id=group_id,
                )
                result = result.get('episodes', [])

                for episode in result:
                    helix_id = episode.get('id', None)
                    if helix_id is None or helix_id not in Node.helix_to_uuid:
                        continue

                    episode['uuid'] = Node.helix_to_uuid.get(helix_id)

                    Node.uuid_to_helix[episode['uuid']] = helix_id

                    results.append(episode)

            if uuid_cursor is not None:
                results = [episode for episode in results if episode.get('uuid', '') < uuid_cursor]

            results.sort(key=lambda x: x.get('uuid', ''), reverse=True)

            episodes = [get_episodic_node_from_record(record) for record in results]

            return episodes

        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
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
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

    @classmethod
    async def get_by_entity_node_uuid(cls, driver: GraphDriver, entity_node_uuid: str):
        if driver.provider == 'helixdb':
            if entity_node_uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(entity_node_uuid)

            result = await driver.execute_query(
                "",
                query="getEpisodebyEntity",
                entity_id=Node.uuid_to_helix.get(entity_node_uuid),
            )

            result = result.get('episodes', [])

            episodes = []
            for episode in result:
                helix_id = episode.get('id', None)
                if helix_id is None or helix_id not in Node.helix_to_uuid:
                    continue

                episode['uuid'] = Node.helix_to_uuid.get(helix_id)

                if episode['uuid'] not in Node.uuid_to_helix:
                    Node.uuid_to_helix[episode['uuid']] = helix_id

                episodes.append(episode)

            episodes = [get_episodic_node_from_record(record) for record in episodes]
            return episodes

        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
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
            entity_node_uuid=entity_node_uuid,
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

    async def load_name_embedding(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="loadEntityEmbedding",
                entity_id=Node.uuid_to_helix.get(self.uuid),
            )
            self.name_embedding = result.get('embedding', None)[0].get('name_embedding', None)
            return

        query: LiteralString = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN n.name_embedding AS name_embedding
        """
        records, _, _ = await driver.execute_query(query, uuid=self.uuid, routing_='r')

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    async def save(self, driver: GraphDriver):
        entity_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'name_embedding': self.name_embedding,
            'group_id': self.group_id,
            'summary': self.summary,
            'created_at': self.created_at,
        }

        entity_data.update(self.attributes or {})

        if driver.provider == 'helixdb':
            stored_entity = None
            if self.uuid in Node.uuid_to_helix:
                stored_entity = await driver.execute_query(
                    "",
                    query="getEntity",
                    entity_id=Node.uuid_to_helix.get(self.uuid),
                )
                stored_entity = stored_entity.get('entity', None)

            if stored_entity is not None:
                query = "updateEntity"
                helix_id = Node.uuid_to_helix.get(self.uuid)
            else:
                query = "createEntity"
                helix_id = self.uuid

            result = await driver.execute_query(
                "",
                query=query,
                entity_id=helix_id,
                name=entity_data['name'],
                name_embedding=entity_data['name_embedding'],
                group_id=entity_data['group_id'],
                summary=entity_data['summary'],
                created_at=entity_data['created_at'],
                labels=self.labels,
                attributes=json.dumps(self.attributes),
            )

            if query == 'createEntity':
                helix_id = result.get('entity', {}).get('id', None)
                if helix_id is None:
                    raise ValueError('Failed to create entity node')
                Node.uuid_to_helix[self.uuid] = helix_id
                Node.helix_to_uuid[helix_id] = self.uuid

            if query == 'updateEntity':
                logger.debug(f'Updated Entity Node: {self.uuid}')
            else:
                logger.debug(f'Created Entity Node to Graph: {self.uuid}')

            return {'uuid': result.get('entity', {}).get('id', None)}

        result = await driver.execute_query(
            ENTITY_NODE_SAVE,
            labels=self.labels + ['Entity'],
            entity_data=entity_data,
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    async def delete(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="deleteEntity",
                entity_id=Node.uuid_to_helix.get(self.uuid),
            )

            helix_id = Node.uuid_to_helix.get(self.uuid)

            Node.uuid_to_helix.pop(self.uuid, None)
            if helix_id is not None:
                Node.helix_to_uuid.pop(helix_id, None)

            logger.debug(f'Deleted Entity Node: {self.uuid}')

            return result
        else:
            return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            if uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(uuid)

            result = await driver.execute_query(
                "",
                query="getEntity",
                entity_id=Node.uuid_to_helix.get(uuid),
            )

            if result is None:
                raise NodeNotFoundError(uuid)

            embedding = await driver.execute_query(
                "",
                query="loadEntityEmbedding",
                entity_id=Node.uuid_to_helix.get(uuid),
            )
            embedding = embedding.get('embedding', None)[0].get('name_embedding', None)

            result = result.get('entity', None)

            if result is None:
                raise NodeNotFoundError(uuid)

            result['uuid'] = uuid
            result['name_embedding'] = embedding
            result['attributes'] = json.loads(result.get('attributes', '{}'))
            helix_id = result.get('id')

            Node.helix_to_uuid[helix_id] = uuid

            return get_entity_node_from_record(result)

        query = (
            """
                                                                                MATCH (n:Entity {uuid: $uuid})
                                                                                """
            + ENTITY_NODE_RETURN
        )
        records, _, _ = await driver.execute_query(
            query,
            uuid=uuid,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == 'helixdb':
            results = []
            for uuid in uuids:
                if uuid not in Node.uuid_to_helix:
                    continue

                result = await driver.execute_query(
                    "",
                    query="getEntity",
                    entity_id=Node.uuid_to_helix.get(uuid),
                )

                if result is None:
                    continue

                embedding = await driver.execute_query(
                    "",
                    query="loadEntityEmbedding",
                    entity_id=Node.uuid_to_helix.get(uuid),
                )
                embedding = embedding.get('embedding', None)[0].get('name_embedding', None)

                result = result.get('entity', None)

                if result is None:
                    continue

                result['uuid'] = uuid
                result['name_embedding'] = embedding
                result['attributes'] = json.loads(result.get('attributes', '{}'))
                results.append(result)

                helix_id = result.get('id')
                Node.helix_to_uuid[helix_id] = uuid

            nodes = [get_entity_node_from_record(result) for result in results]

            return nodes

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity) WHERE n.uuid IN $uuids
        """
            + ENTITY_NODE_RETURN,
            uuids=uuids,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        return nodes

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
        with_embeddings: bool = False,
    ):
        if driver.provider == 'helixdb':
            query = "getEntitiesbyGroup"
            if limit is not None:
                query = "getEntitiesbyGroupLimit"
            
            results = []
            for group_id in group_ids:
                result = await driver.execute_query(
                    "",
                    query=query,
                    limit=limit,
                    group_id=group_id,
                )
                result = result.get('entities', [])
                for entity in result:
                    helix_id = entity.get('id', None)
                    if helix_id is None or helix_id not in Node.helix_to_uuid:
                        continue

                    embedding = await driver.execute_query(
                        "",
                        query="loadEntityEmbedding",
                        entity_id=helix_id,
                    )
                    embedding = embedding.get('embedding', None)[0].get('name_embedding', None)

                    entity['uuid'] = Node.helix_to_uuid.get(helix_id)
                    entity['attributes'] = json.loads(entity.get('attributes', '{}'))
                    entity['name_embedding'] = embedding

                    if entity['uuid'] not in Node.uuid_to_helix:
                        Node.uuid_to_helix[entity['uuid']] = helix_id

                    results.append(entity)

            if uuid_cursor is not None:
                results = [entity for entity in results if entity.get('uuid', '') < uuid_cursor]

            results.sort(key=lambda x: x.get('uuid', ''), reverse=True)

            nodes = [get_entity_node_from_record(record) for record in results]

            return nodes

        cursor_query: LiteralString = 'AND n.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        with_embeddings_query: LiteralString = (
            """,
                n.name_embedding AS name_embedding
                """
            if with_embeddings
            else ''
        )

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity) WHERE n.group_id IN $group_ids
        """
            + cursor_query
            + ENTITY_NODE_RETURN
            + with_embeddings_query
            + """
        ORDER BY n.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        return nodes


class CommunityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='region summary of member nodes', default_factory=str)

    async def save(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            stored_community = None
            if self.uuid in Node.uuid_to_helix:
                stored_community = await driver.execute_query(
                    "",
                    query="getCommunity",
                    community_id=Node.uuid_to_helix.get(self.uuid),
                )
                stored_community = stored_community.get('community', None)

            if stored_community is not None:
                query = "updateCommunity"
                helix_id = Node.uuid_to_helix.get(self.uuid)
            else:
                query = "createCommunity"
                helix_id = self.uuid
            
            result = await driver.execute_query(
                "",
                query=query,
                community_id=helix_id,
                name=self.name,
                group_id=self.group_id,
                summary=self.summary,
                name_embedding=self.name_embedding,
                created_at=self.created_at,
                labels=self.labels,
            )

            if query == 'createCommunity':
                helix_id = result.get('community', {}).get('id', None)
                if helix_id is None:
                    raise ValueError('Failed to create community node')
                Node.uuid_to_helix[self.uuid] = helix_id
                Node.helix_to_uuid[helix_id] = self.uuid
            
            if query == 'updateCommunity':
                logger.debug(f'Updated Community Node: {self.uuid}')
            else:
                logger.debug(f'Created Community Node to Graph: {self.uuid}')

            return {'uuid': result.get('community', {}).get('id', None)}

        result = await driver.execute_query(
            COMMUNITY_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            summary=self.summary,
            name_embedding=self.name_embedding,
            created_at=self.created_at,
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    async def delete(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="deleteCommunity",
                community_id=Node.uuid_to_helix.get(self.uuid),
            )

            helix_id = Node.uuid_to_helix.get(self.uuid)

            Node.uuid_to_helix.pop(self.uuid, None)
            if helix_id is not None:
                Node.helix_to_uuid.pop(helix_id, None)

            logger.debug(f'Deleted Community Node: {self.uuid}')

            return result
        else:
            return await super().delete(driver)

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.name_embedding

    async def load_name_embedding(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="loadCommunityEmbedding",
                community_id=Node.uuid_to_helix.get(self.uuid),
            )

            self.name_embedding = result.get('embedding', None)[0].get('name_embedding', None)
            return

        query: LiteralString = """
            MATCH (c:Community {uuid: $uuid})
            RETURN c.name_embedding AS name_embedding
        """
        records, _, _ = await driver.execute_query(query, uuid=self.uuid, routing_='r')

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            if uuid not in Node.uuid_to_helix:
                raise NodeNotFoundError(uuid)

            result = await driver.execute_query(
                "",
                query="getCommunity",
                community_id=Node.uuid_to_helix.get(uuid),
            )

            if result is None:
                raise NodeNotFoundError(uuid)

            embedding = await driver.execute_query(
                "",
                query="loadCommunityEmbedding",
                community_id=Node.uuid_to_helix.get(uuid),
            )
            embedding = embedding.get('embedding', None)[0].get('name_embedding', None)

            result = result.get('community', None)

            if result is None:
                raise NodeNotFoundError(uuid)

            result['uuid'] = uuid
            result['name_embedding'] = embedding
            helix_id = result.get('id')

            Node.helix_to_uuid[helix_id] = uuid

            return get_community_node_from_record(result)

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community {uuid: $uuid})
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        """,
            uuid=uuid,
            routing_='r',
        )

        nodes = [get_community_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == 'helixdb':
            results = []
            for uuid in uuids:
                if uuid not in Node.uuid_to_helix:
                    continue

                result = await driver.execute_query(
                    "",
                    query="getCommunity",
                    community_id=Node.uuid_to_helix.get(uuid),
                )

                if result is None:
                    continue

                embedding = await driver.execute_query(
                    "",
                    query="loadCommunityEmbedding",
                    community_id=Node.uuid_to_helix.get(uuid),
                )
                embedding = embedding.get('embedding', None)[0].get('name_embedding', None)

                result = result.get('community', None)

                if result is None:
                    continue

                result['uuid'] = uuid
                result['name_embedding'] = embedding
                results.append(result)

                helix_id = result.get('id')
                Node.helix_to_uuid[helix_id] = uuid

            nodes = [get_community_node_from_record(result) for result in results]

            return nodes

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community) WHERE n.uuid IN $uuids
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        """,
            uuids=uuids,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        if driver.provider == 'helixdb':
            query = "getCommunitiesbyGroup"
            if limit is not None:
                query = "getCommunitiesbyGroupLimit"
            
            results = []
            for group_id in group_ids:
                result = await driver.execute_query(
                    "",
                    query=query,
                    limit=limit,
                    group_id=group_id,
                )
                result = result.get('communities', [])
                for community in result:
                    helix_id = community.get('id', None)
                    if helix_id is None or helix_id not in Node.helix_to_uuid:
                        continue

                    embedding = await driver.execute_query(
                        "",
                        query="loadCommunityEmbedding",
                        community_id=helix_id,
                    )
                    embedding = embedding.get('embedding', None)[0].get('name_embedding', None)

                    community['uuid'] = Node.helix_to_uuid.get(helix_id)
                    community['name_embedding'] = embedding

                    if community['uuid'] not in Node.uuid_to_helix:
                        Node.uuid_to_helix[community['uuid']] = helix_id

                    results.append(community)

            if uuid_cursor is not None:
                results = [community for community in results if community.get('uuid', '') < uuid_cursor]

            results.sort(key=lambda x: x.get('uuid', ''), reverse=True)

            communities = [get_community_node_from_record(record) for record in results]

            return communities

        cursor_query: LiteralString = 'AND n.uuid < $uuid' if uuid_cursor else ''
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
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        ORDER BY n.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities


# Node helpers
def get_episodic_node_from_record(record: Any) -> EpisodicNode:
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
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
        name_embedding=record.get('name_embedding'),
        group_id=record['group_id'],
        labels=record['labels'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
        summary=record['summary'],
        attributes=record['attributes'],
    )

    entity_node.attributes.pop('uuid', None)
    entity_node.attributes.pop('name', None)
    entity_node.attributes.pop('group_id', None)
    entity_node.attributes.pop('name_embedding', None)
    entity_node.attributes.pop('summary', None)
    entity_node.attributes.pop('created_at', None)

    return entity_node


def get_community_node_from_record(record: Any) -> CommunityNode:
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
        summary=record['summary'],
    )


async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    if not nodes:  # Handle empty list case
        return
    name_embeddings = await embedder.create_batch([node.name for node in nodes])
    for node, name_embedding in zip(nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding
