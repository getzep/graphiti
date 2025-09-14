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

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.driver.driver import (
    COMMUNITY_INDEX_NAME,
    ENTITY_EDGE_INDEX_NAME,
    ENTITY_INDEX_NAME,
    EPISODE_INDEX_NAME,
    GraphDriver,
    GraphProvider,
)
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    COMMUNITY_NODE_RETURN_NEPTUNE,
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
    get_community_node_save_query,
    get_entity_node_return_query,
    get_entity_node_save_query,
    get_episode_node_save_query,
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
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        match driver.provider:
            case GraphProvider.NEO4J:
                records, _, _ = await driver.execute_query(
                    """
                    MATCH (n {uuid: $uuid})
                    WHERE n:Entity OR n:Episodic OR n:Community
                    OPTIONAL MATCH (n)-[r]-()
                    WITH collect(r.uuid) AS edge_uuids, n
                    DETACH DELETE n
                    RETURN edge_uuids
                    """,
                    uuid=self.uuid,
                )

                edge_uuids: list[str] = records[0].get('edge_uuids', []) if records else []

                if driver.aoss_client:
                    # Delete the node from OpenSearch indices
                    for index in (EPISODE_INDEX_NAME, ENTITY_INDEX_NAME, COMMUNITY_INDEX_NAME):
                        await driver.aoss_client.delete(
                            index=index,
                            id=self.uuid,
                            params={'routing': self.group_id},
                        )

                    # Bulk delete the detached edges
                    if edge_uuids:
                        actions = []
                        for eid in edge_uuids:
                            actions.append(
                                {'delete': {'_index': ENTITY_EDGE_INDEX_NAME, '_id': eid}}
                            )

                        await driver.aoss_client.bulk(body=actions)

            case GraphProvider.KUZU:
                for label in ['Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{uuid: $uuid}})
                        DETACH DELETE n
                        """,
                        uuid=self.uuid,
                    )
                # Entity edges are actually nodes in Kuzu, so simple `DETACH DELETE` will not work.
                # Explicitly delete the "edge" nodes first, then the entity node.
                await driver.execute_query(
                    """
                    MATCH (n:Entity {uuid: $uuid})-[:RELATES_TO]->(e:RelatesToNode_)
                    DETACH DELETE e
                    """,
                    uuid=self.uuid,
                )
                await driver.execute_query(
                    """
                    MATCH (n:Entity {uuid: $uuid})
                    DETACH DELETE n
                    """,
                    uuid=self.uuid,
                )
            case _:  # FalkorDB, Neptune
                for label in ['Entity', 'Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{uuid: $uuid}})
                        DETACH DELETE n
                        """,
                        uuid=self.uuid,
                    )

        logger.debug(f'Deleted Node: {self.uuid}')

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def delete_by_group_id(cls, driver: GraphDriver, group_id: str, batch_size: int = 100):
        match driver.provider:
            case GraphProvider.NEO4J:
                async with driver.session() as session:
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community {group_id: $group_id})
                        CALL {
                            WITH n
                            DETACH DELETE n
                        } IN TRANSACTIONS OF $batch_size ROWS
                        """,
                        group_id=group_id,
                        batch_size=batch_size,
                    )

                if driver.aoss_client:
                    await driver.aoss_client.delete_by_query(
                        index=EPISODE_INDEX_NAME,
                        body={'query': {'term': {'group_id': group_id}}},
                        params={'routing': group_id},
                    )

                    await driver.aoss_client.delete_by_query(
                        index=ENTITY_INDEX_NAME,
                        body={'query': {'term': {'group_id': group_id}}},
                        params={'routing': group_id},
                    )

                    await driver.aoss_client.delete_by_query(
                        index=COMMUNITY_INDEX_NAME,
                        body={'query': {'term': {'group_id': group_id}}},
                        params={'routing': group_id},
                    )

                    await driver.aoss_client.delete_by_query(
                        index=ENTITY_EDGE_INDEX_NAME,
                        body={'query': {'term': {'group_id': group_id}}},
                        params={'routing': group_id},
                    )

            case GraphProvider.KUZU:
                for label in ['Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{group_id: $group_id}})
                        DETACH DELETE n
                        """,
                        group_id=group_id,
                    )
                # Entity edges are actually nodes in Kuzu, so simple `DETACH DELETE` will not work.
                # Explicitly delete the "edge" nodes first, then the entity node.
                await driver.execute_query(
                    """
                    MATCH (n:Entity {group_id: $group_id})-[:RELATES_TO]->(e:RelatesToNode_)
                    DETACH DELETE e
                    """,
                    group_id=group_id,
                )
                await driver.execute_query(
                    """
                    MATCH (n:Entity {group_id: $group_id})
                    DETACH DELETE n
                    """,
                    group_id=group_id,
                )
            case _:  # FalkorDB, Neptune
                for label in ['Entity', 'Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{group_id: $group_id}})
                        DETACH DELETE n
                        """,
                        group_id=group_id,
                    )

    @classmethod
    async def delete_by_uuids(cls, driver: GraphDriver, uuids: list[str], batch_size: int = 100):
        match driver.provider:
            case GraphProvider.FALKORDB:
                for label in ['Entity', 'Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label})
                        WHERE n.uuid IN $uuids
                        DETACH DELETE n
                        """,
                        uuids=uuids,
                    )
            case GraphProvider.KUZU:
                for label in ['Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label})
                        WHERE n.uuid IN $uuids
                        DETACH DELETE n
                        """,
                        uuids=uuids,
                    )
                # Entity edges are actually nodes in Kuzu, so simple `DETACH DELETE` will not work.
                # Explicitly delete the "edge" nodes first, then the entity node.
                await driver.execute_query(
                    """
                    MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)
                    WHERE n.uuid IN $uuids
                    DETACH DELETE e
                    """,
                    uuids=uuids,
                )
                await driver.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE n.uuid IN $uuids
                    DETACH DELETE n
                    """,
                    uuids=uuids,
                )
            case _:  # Neo4J, Neptune
                async with driver.session() as session:
                    # Collect all edge UUIDs before deleting nodes
                    result = await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community)
                        WHERE n.uuid IN $uuids
                        MATCH (n)-[r]-()
                        RETURN collect(r.uuid) AS edge_uuids
                        """,
                        uuids=uuids,
                    )

                    record = await result.single()
                    edge_uuids: list[str] = (
                        record['edge_uuids'] if record and record['edge_uuids'] else []
                    )

                    # Now delete the nodes in batches
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community)
                        WHERE n.uuid IN $uuids
                        CALL {
                            WITH n
                            DETACH DELETE n
                        } IN TRANSACTIONS OF $batch_size ROWS
                        """,
                        uuids=uuids,
                        batch_size=batch_size,
                    )

                if driver.aoss_client:
                    for index in (EPISODE_INDEX_NAME, ENTITY_INDEX_NAME, COMMUNITY_INDEX_NAME):
                        await driver.aoss_client.delete_by_query(
                            index=index,
                            body={'query': {'terms': {'uuid': uuids}}},
                        )

                    if edge_uuids:
                        actions = [
                            {'delete': {'_index': ENTITY_EDGE_INDEX_NAME, '_id': eid}}
                            for eid in edge_uuids
                        ]
                        await driver.aoss_client.bulk(body=actions)

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
        episode_args = {
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'source_description': self.source_description,
            'content': self.content,
            'entity_edges': self.entity_edges,
            'created_at': self.created_at,
            'valid_at': self.valid_at,
            'source': self.source.value,
        }

        if driver.aoss_client:
            await driver.save_to_aoss(  # pyright: ignore reportAttributeAccessIssue
                'episodes',
                [episode_args],
            )

        result = await driver.execute_query(
            get_episode_node_save_query(driver.provider), **episode_args
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            ),
            uuid=uuid,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        if len(episodes) == 0:
            raise NodeNotFoundError(uuid)

        return episodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.uuid IN $uuids
            RETURN DISTINCT
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            ),
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
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN DISTINCT
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            )
            + """
            ORDER BY uuid DESC
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
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN DISTINCT
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            ),
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
        if driver.provider == GraphProvider.NEPTUNE:
            query: LiteralString = """
                MATCH (n:Entity {uuid: $uuid})
                RETURN [x IN split(n.name_embedding, ",") | toFloat(x)] as name_embedding
            """
        elif driver.aoss_client:
            resp = await driver.aoss_client.search(
                body={
                    'query': {'multi_match': {'query': self.uuid, 'fields': ['uuid']}},
                    'size': 1,
                },
                index=ENTITY_INDEX_NAME,
                params={'routing': self.group_id},
            )

            if resp['hits']['hits']:
                self.name_embedding = resp['hits']['hits'][0]['_source']['name_embedding']
                return
            else:
                raise NodeNotFoundError(self.uuid)

        else:
            query: LiteralString = """
                MATCH (n:Entity {uuid: $uuid})
                RETURN n.name_embedding AS name_embedding
            """
        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

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

        if driver.provider == GraphProvider.KUZU:
            entity_data['attributes'] = json.dumps(self.attributes)
            entity_data['labels'] = list(set(self.labels + ['Entity']))
            result = await driver.execute_query(
                get_entity_node_save_query(driver.provider, labels=''),
                **entity_data,
            )
        else:
            entity_data.update(self.attributes or {})
            labels = ':'.join(self.labels + ['Entity'])

            if driver.aoss_client:
                await driver.save_to_aoss(ENTITY_INDEX_NAME, [entity_data])  # pyright: ignore reportAttributeAccessIssue

            result = await driver.execute_query(
                get_entity_node_save_query(driver.provider, labels, bool(driver.aoss_client)),
                entity_data=entity_data,
            )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity {uuid: $uuid})
            RETURN
            """
            + get_entity_node_return_query(driver.provider),
            uuid=uuid,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """
            + get_entity_node_return_query(driver.provider),
            uuids=uuids,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

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
            MATCH (n:Entity)
            WHERE n.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + get_entity_node_return_query(driver.provider)
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

        nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

        return nodes


class CommunityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='region summary of member nodes', default_factory=str)

    async def save(self, driver: GraphDriver):
        if driver.provider == GraphProvider.NEPTUNE:
            await driver.save_to_aoss(  # pyright: ignore reportAttributeAccessIssue
                'communities',
                [{'name': self.name, 'uuid': self.uuid, 'group_id': self.group_id}],
            )
        result = await driver.execute_query(
            get_community_node_save_query(driver.provider),  # type: ignore
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            summary=self.summary,
            name_embedding=self.name_embedding,
            created_at=self.created_at,
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.name_embedding

    async def load_name_embedding(self, driver: GraphDriver):
        if driver.provider == GraphProvider.NEPTUNE:
            query: LiteralString = """
                MATCH (c:Community {uuid: $uuid})
                RETURN [x IN split(c.name_embedding, ",") | toFloat(x)] as name_embedding
            """
        else:
            query: LiteralString = """
            MATCH (c:Community {uuid: $uuid})
            RETURN c.name_embedding AS name_embedding
            """

        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Community {uuid: $uuid})
            RETURN
            """
            + (
                COMMUNITY_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else COMMUNITY_NODE_RETURN
            ),
            uuid=uuid,
            routing_='r',
        )

        nodes = [get_community_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.uuid IN $uuids
            RETURN
            """
            + (
                COMMUNITY_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else COMMUNITY_NODE_RETURN
            ),
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
        cursor_query: LiteralString = 'AND c.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + (
                COMMUNITY_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else COMMUNITY_NODE_RETURN
            )
            + """
            ORDER BY c.uuid DESC
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


def get_entity_node_from_record(record: Any, provider: GraphProvider) -> EntityNode:
    if provider == GraphProvider.KUZU:
        attributes = json.loads(record['attributes']) if record['attributes'] else {}
    else:
        attributes = record['attributes']
        attributes.pop('uuid', None)
        attributes.pop('name', None)
        attributes.pop('group_id', None)
        attributes.pop('name_embedding', None)
        attributes.pop('summary', None)
        attributes.pop('created_at', None)
        attributes.pop('labels', None)

    labels = record.get('labels', [])
    group_id = record.get('group_id')
    if 'Entity_' + group_id.replace('-', '') in labels:
        labels.remove('Entity_' + group_id.replace('-', ''))

    entity_node = EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(record['created_at']),  # type: ignore
        summary=record['summary'],
        attributes=attributes,
    )

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
