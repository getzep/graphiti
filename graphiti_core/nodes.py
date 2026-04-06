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
    document : str
        Represents a document episode (e.g., imported documents, knowledge base content).
    """

    message = 'message'
    json = 'json'
    text = 'text'
    document = 'document'

    @staticmethod
    def from_str(episode_type: str):
        if episode_type == 'message':
            return EpisodeType.message
        if episode_type == 'json':
            return EpisodeType.json
        if episode_type == 'text':
            return EpisodeType.text
        if episode_type == 'document':
            return EpisodeType.document
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
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_delete(self, driver)

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
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_delete_by_group_id(
                cls, driver, group_id, batch_size
            )

        match driver.provider:
            case GraphProvider.NEO4J:
                async with driver.session() as session:
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community {group_id: $group_id})
                        CALL (n) {
                            DETACH DELETE n
                        } IN TRANSACTIONS OF $batch_size ROWS
                        """,
                        group_id=group_id,
                        batch_size=batch_size,
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
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_delete_by_uuids(
                cls, driver, uuids, group_id=None, batch_size=batch_size
            )

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
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community)
                        WHERE n.uuid IN $uuids
                        MATCH (n)-[r]-()
                        RETURN collect(r.uuid) AS edge_uuids
                        """,
                        uuids=uuids,
                    )

                    # Now delete the nodes in batches
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community)
                        WHERE n.uuid IN $uuids
                        CALL (n) {
                            DETACH DELETE n
                        } IN TRANSACTIONS OF $batch_size ROWS
                        """,
                        uuids=uuids,
                        batch_size=batch_size,
                    )

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str): ...

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]): ...


class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    # EasyOps fix: content can be None when using file storage
    # (content is stored externally and database field is NULL)
    content: str | None = Field(default=None, description='raw episode data')
    valid_at: datetime = Field(
        description='datetime of when the original document was created',
    )
    entity_edges: list[str] = Field(
        description='list of entity edges referenced in this episode',
        default_factory=list,
    )

    # Document mode fields (EasyOps extension)
    summary: str | None = Field(
        default=None,
        description='LLM-generated summary of the document content (document mode only)',
    )
    tags: list[str] | None = Field(
        default=None,
        description='LLM-extracted tags/keywords from the document (document mode only)',
    )
    content_embedding: list[float] | None = Field(
        default=None,
        description='Vector embedding of the content for semantic search (document mode only)',
    )

    # Content storage fields (file storage support)
    content_hash: str | None = Field(
        default=None,
        description='SHA-256 hash of the content for deduplication',
    )
    content_storage_type: str | None = Field(
        default=None,
        description='Storage type: redis, local, oss, or s3',
    )
    content_file_path: str | None = Field(
        default=None,
        description='File path for stored content (None for redis mode)',
    )
    content_file_size: int | None = Field(
        default=None,
        description='Content size in bytes',
    )

    # User isolation field
    user_id: str | None = Field(
        default=None,
        description='User ID for data isolation within a group',
    )

    async def save(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.episodic_node_save(self, driver)

        # Store content to file storage (if using FalkorDriver with content_storage)
        content_for_db = self.content
        if hasattr(driver, 'content_storage') and hasattr(driver, 'content_storage_type'):
            # Store content (storage layer handles deduplication)
            storage_result = await driver.content_storage.store(
                episode_uuid=self.uuid,
                group_id=self.group_id,
                content=self.content,
            )

            # Update storage fields
            self.content_hash = storage_result.content_hash
            self.content_storage_type = driver.content_storage_type
            self.content_file_path = storage_result.file_path
            self.content_file_size = storage_result.file_size

            # For file-based storage, set content=None (don't store in FalkorDB)
            if driver.content_storage_type != 'redis':
                content_for_db = None

        episode_args = {
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'source_description': self.source_description,
            'content': content_for_db,  # None for file storage, content for redis
            'entity_edges': self.entity_edges,
            'created_at': self.created_at,
            'valid_at': self.valid_at,
            'source': self.source.value,
            # Document mode fields
            'summary': self.summary,
            'tags': self.tags or [],
            'content_embedding': self.content_embedding,
            # Content storage fields
            'content_hash': self.content_hash,
            'content_storage_type': self.content_storage_type,
            'content_file_path': self.content_file_path,
            'content_file_size': self.content_file_size,
            # User isolation
            'user_id': self.user_id,
        }

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

        episode = episodes[0]

        # Auto-load content from storage if needed (file storage mode)
        if not episode.content:
            episode.content = await driver.load_episode_content(episode)

        return episode

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

        # Auto-load content from storage if needed (file storage mode)
        for episode in episodes:
            if not episode.content:
                episode.content = await driver.load_episode_content(episode)

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

        # Auto-load content from storage if needed (file storage mode)
        for episode in episodes:
            if not episode.content:
                episode.content = await driver.load_episode_content(episode)

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

        # Auto-load content from storage if needed (file storage mode)
        for episode in episodes:
            if not episode.content:
                episode.content = await driver.load_episode_content(episode)

        return episodes


class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary_embedding: list[float] | None = Field(default=None, description='embedding of the summary for semantic search')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )
    reasoning: str | None = Field(
        default=None,
        description='LLM reasoning for entity extraction and type classification (Chain of Thought)'
    )
    type_scores: dict[str, Any] | None = Field(
        default=None,
        description='Type classification scores from LLM. Format: {type_name: {score: float, reasoning: str}}'
    )
    type_confidence: float | None = Field(
        default=None,
        description='Confidence score (0.0-1.0) for the final type classification'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient):
        """生成 name_embedding 和 summary_embedding（双向量策略）

        解决语义稀释问题：英文 name + 中文 summary 拼接会导致相似度下降。
        分别存储两个向量，搜索时取 max(name_sim, summary_sim) 作为得分。
        """
        start = time()

        # name_embedding: 仅用 name 生成
        name_text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[name_text])

        # summary_embedding: 用 summary 生成（如果有 summary）
        if self.summary:
            summary_text = self.summary.replace('\n', ' ')
            self.summary_embedding = await embedder.create(input_data=[summary_text])
        else:
            self.summary_embedding = None

        end = time()
        logger.debug(f'embedded {name_text[:50]}... in {end - start} ms')

        return self.name_embedding

    async def load_name_embedding(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_load_embeddings(self, driver)

        if driver.provider == GraphProvider.NEPTUNE:
            query: LiteralString = """
                MATCH (n:Entity {uuid: $uuid})
                RETURN [x IN split(n.name_embedding, ",") | toFloat(x)] as name_embedding
            """

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
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_save(self, driver)

        entity_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'name_embedding': self.name_embedding,
            'summary_embedding': self.summary_embedding,
            'group_id': self.group_id,
            'summary': self.summary,
            'created_at': self.created_at,
            'reasoning': self.reasoning,
            # EasyOps: Save type classification scores
            'type_scores': json.dumps(self.type_scores) if self.type_scores else None,
            'type_confidence': self.type_confidence,
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

            result = await driver.execute_query(
                get_entity_node_save_query(driver.provider, labels),
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
        # Document mode fields
        summary=record.get('summary'),
        tags=record.get('tags'),
        # Content storage fields
        content_hash=record.get('content_hash'),
        content_storage_type=record.get('content_storage_type'),
        content_file_path=record.get('content_file_path'),
        content_file_size=record.get('content_file_size'),
        # User isolation
        user_id=record.get('user_id'),
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
        attributes.pop('summary_embedding', None)  # EasyOps fix: exclude from prompt context
        attributes.pop('summary', None)
        attributes.pop('created_at', None)
        attributes.pop('labels', None)

    # Extract reasoning from attributes before removing it
    reasoning = attributes.pop('reasoning', None)

    # EasyOps: Extract type classification scores from attributes
    type_scores_raw = attributes.pop('type_scores', None)
    type_scores = None
    if type_scores_raw:
        if isinstance(type_scores_raw, str):
            try:
                type_scores = json.loads(type_scores_raw)
            except json.JSONDecodeError:
                type_scores = None
        elif isinstance(type_scores_raw, dict):
            type_scores = type_scores_raw
    type_confidence = attributes.pop('type_confidence', None)

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
        reasoning=reasoning,
        type_scores=type_scores,
        type_confidence=type_confidence,
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
    """批量生成实体节点的 name_embedding 和 summary_embedding（双向量策略）

    解决语义稀释问题：英文 name + 中文 summary 拼接会导致相似度下降。
    分别存储两个向量，搜索时取 max(name_sim, summary_sim) 作为得分。
    """
    # filter out falsey values from nodes
    filtered_nodes = [node for node in nodes if node.name]

    if not filtered_nodes:
        return

    # 1. 生成 name_embedding（仅用 name）
    name_texts = [node.name for node in filtered_nodes]
    name_embeddings = await embedder.create_batch(name_texts)
    for node, name_embedding in zip(filtered_nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding

    # 2. 生成 summary_embedding（仅用 summary，如果有的话）
    nodes_with_summary = [node for node in filtered_nodes if node.summary]
    if nodes_with_summary:
        summary_texts = [node.summary for node in nodes_with_summary]
        summary_embeddings = await embedder.create_batch(summary_texts)
        for node, summary_embedding in zip(nodes_with_summary, summary_embeddings, strict=True):
            node.summary_embedding = summary_embedding
