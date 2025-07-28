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
from datetime import datetime, timezone
from time import time
import json
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.models.edges.edge_db_queries import (
    COMMUNITY_EDGE_SAVE,
    ENTITY_EDGE_SAVE,
    EPISODIC_EDGE_SAVE,
)
from graphiti_core.nodes import Node, EpisodicNode, EntityNode, CommunityNode

logger = logging.getLogger(__name__)

ENTITY_EDGE_RETURN: LiteralString = """
        RETURN
            e.uuid AS uuid,
            startNode(e).uuid AS source_node_uuid,
            endNode(e).uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.group_id AS group_id,
            e.fact AS fact,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            properties(e) AS attributes"""


class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        result = await driver.execute_query(
            """
        MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
        DELETE e
        """,
            uuid=self.uuid,
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
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str): ...


class EpisodicEdge(Edge):
    async def save(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            stored_edge = await driver.execute_query(
                "",
                query="getEpisodeEdge",
                uuid=self.uuid,
            )
            stored_edge = stored_edge.get('episode_edge', None)

            if stored_edge is not None and len(stored_edge) > 0:
                query = "updateEpisodeEdge"
                helix_id = stored_edge[0].get('id')
            else:
                query = "createEpisodeEdge"
                helix_id = self.uuid

            result = await driver.execute_query(
                "",
                query=query,
                episodeEdge_id=helix_id,
                uuid=self.uuid,
                episode_uuid=self.source_node_uuid,
                entity_uuid=self.target_node_uuid,
                group_id=self.group_id,
                created_at=self.created_at,
            )

            if query == 'createEpisodeEdge':
                helix_id = result.get('episode_edge', None).get('id', None)
                if helix_id is None:
                    raise ValueError('Failed to create episode edge')

            if query == 'updateEpisodeEdge':
                logger.debug(f'Updated Episode Edge: {self.uuid}')
            else:
                logger.debug(f'Created Episode Edge: {self.uuid}')

            return {'uuid': self.uuid}

        result = await driver.execute_query(
            EPISODIC_EDGE_SAVE,
            episode_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
        )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    async def delete(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="deleteEpisodeEdge",
                uuid=self.uuid,
            )

            logger.debug(f'Deleted Episode Edge: {self.uuid}')

            return result

        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="getEpisodeEdge",
                uuid=uuid,
            )

            if not isinstance(result, dict):
                raise EdgeNotFoundError(uuid)

            source = result.get('episode', [])
            target = result.get('entity', [])
            result = result.get('episode_edge', [])

            if len(source) == 0 or len(target) == 0 or len(result) == 0:
                raise EdgeNotFoundError(uuid)

            source = source[0]
            target = target[0]
            result = result[0]

            result['source_node_uuid'] = source.get('uuid')
            result['target_node_uuid'] = target.get('uuid')

            return get_episodic_edge_from_record(result)

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
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == 'helixdb':
            results = []
            for uuid in uuids:
                result = await driver.execute_query(
                    "",
                    query="getEpisodeEdge",
                    uuid=uuid,
                )
                
                if not isinstance(result, dict):
                    continue
                
                source = result.get('episode', [])
                target = result.get('entity', [])
                result = result.get('episode_edge', [])
                
                if len(source) == 0 or len(target) == 0 or len(result) == 0:
                    continue

                source = source[0]
                target = target[0]
                result = result[0]

                result['source_node_uuid'] = source.get('uuid')
                result['target_node_uuid'] = target.get('uuid')
                results.append(result)

            edges = [get_episodic_edge_from_record(result) for result in results]

            return edges

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
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuids[0])
        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        if driver.provider == 'helixdb':
            query = "getEpisodeEdgesbyGroup"
            if limit is not None:
                query = "getEpisodeEdgesbyGroupLimit"
            
            results = []
            for group_id in group_ids:
                result = await driver.execute_query(
                    "",
                    query=query,
                    limit=limit,
                    group_id=group_id,
                )

                sources = result.get('episodes', [])
                targets = result.get('entities', [])
                result = result.get('episode_edges', [])

                for i in range(len(result)):
                    episode_edge = result[i]
                    source = sources[i]
                    target = targets[i]
                    helix_id = episode_edge.get('id')
                    if helix_id is None:
                        continue

                    episode_edge['source_node_uuid'] = source.get('uuid')
                    episode_edge['target_node_uuid'] = target.get('uuid')

                    results.append(episode_edge)
                    
            if uuid_cursor is not None:
                results = [episode_edge for episode_edge in results if episode_edge.get('uuid', '') < uuid_cursor]

            results.sort(key=lambda x: x.get('uuid', ''), reverse=True)

            episode_edges = [get_episodic_edge_from_record(record) for record in results]

            return episode_edges

        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
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
            uuid=uuid_cursor,
            limit=limit,
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
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the edge. Dependent on edge name'
    )

    async def generate_embedding(self, embedder: EmbedderClient):
        start = time()

        text = self.fact.replace('\n', ' ')
        self.fact_embedding = await embedder.create(input_data=[text])

        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.fact_embedding

    async def load_fact_embedding(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="loadFactEmbedding", 
                uuid=self.uuid,
            )

            result = result.get('embedding', None)

            if result is None or len(result) == 0:
                raise EdgeNotFoundError(self.uuid)

            self.fact_embedding = result[0].get('fact_embedding', None)
            return

        query: LiteralString = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            RETURN e.fact_embedding AS fact_embedding
        """
        records, _, _ = await driver.execute_query(query, uuid=self.uuid, routing_='r')

        if len(records) == 0:
            raise EdgeNotFoundError(self.uuid)

        self.fact_embedding = records[0]['fact_embedding']

    async def save(self, driver: GraphDriver):
        edge_data: dict[str, Any] = {
            'source_uuid': self.source_node_uuid,
            'target_uuid': self.target_node_uuid,
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'fact': self.fact,
            'fact_embedding': self.fact_embedding,
            'episodes': self.episodes,
            'created_at': self.created_at,
            'expired_at': self.expired_at,
            'valid_at': self.valid_at,
            'invalid_at': self.invalid_at,
        }

        edge_data.update(self.attributes or {})

        if driver.provider == 'helixdb':
            stored_fact = await driver.execute_query(
                "",
                query="getFact",
                uuid=self.uuid,
            )
            stored_fact = stored_fact.get('fact', None)

            if stored_fact is not None:
                query = "updateFact"
                helix_id = stored_fact.get('id')
            else:
                query = "createFact"
                helix_id = self.uuid

            valid_date = edge_data['valid_at']
            if valid_date is None:
                valid_date = datetime.fromtimestamp(0, timezone.utc).isoformat()
            invalid_date = edge_data['invalid_at']
            if invalid_date is None:
                invalid_date = datetime.fromtimestamp(0, timezone.utc).isoformat()
            expired_date = edge_data['expired_at']
            if expired_date is None:
                expired_date = datetime.fromtimestamp(0, timezone.utc).isoformat()

            result = await driver.execute_query(
                "",
                query=query,
                fact_id=helix_id,
                uuid=self.uuid,
                name=edge_data['name'],
                fact=edge_data['fact'],
                fact_embedding=edge_data['fact_embedding'],
                group_id=edge_data['group_id'],
                source_uuid=self.source_node_uuid,
                target_uuid=self.target_node_uuid,
                episodes=edge_data['episodes'],
                created_at=edge_data['created_at'],
                valid_at=valid_date,
                invalid_at=invalid_date,
                expired_at=expired_date,
                attributes=json.dumps(self.attributes),
            )

            if query == 'createFact':
                helix_id = result.get('fact_node', {}).get('id', None)
                if helix_id is None:
                    raise ValueError('Failed to create entity edge')

            if query == 'updateFact':
                logger.debug(f'Updated Entity Edge: {self.uuid}')
            else:
                logger.debug(f'Created Entity Edge to Graph: {self.uuid}')

            return {'uuid': self.uuid}

        result = await driver.execute_query(
            ENTITY_EDGE_SAVE,
            edge_data=edge_data,
        )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    async def delete(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="deleteFact",
                uuid=self.uuid,
            )

            logger.debug(f'Deleted Entity Edge: {self.uuid}')

            return result

        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb': 
            result = await driver.execute_query(
                "",
                query="getFact",
                uuid=uuid,
            )
            
            if result is None:
                raise EdgeNotFoundError(uuid)
            
            embedding = await driver.execute_query(
                "",
                query="loadFactEmbedding",
                uuid=uuid,
            )
            embedding = embedding.get('embedding', [{}])

            if len(embedding) == 0:
                embedding = None
            else:
                embedding = embedding[0].get('fact_embedding', None)

            source = result.get('source', [])
            target = result.get('target', [])
            result = result.get('fact', None)
            
            if result is None or len(source) == 0 or len(target) == 0:
                raise EdgeNotFoundError(uuid)

            source = source[0]
            target = target[0]
            
            result['fact_embedding'] = embedding
            result['valid_at'] = check_null_dates(result.get('valid_at', None))
            result['invalid_at'] = check_null_dates(result.get('invalid_at', None))
            result['expired_at'] = check_null_dates(result.get('expired_at', None))
            result['attributes'] = json.loads(result.get('attributes', '{}'))
            result['source_node_uuid'] = source.get('uuid')
            result['target_node_uuid'] = target.get('uuid')

            return get_entity_edge_from_record(result)

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
        """
            + ENTITY_EDGE_RETURN,
            uuid=uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == 'helixdb':
            results = []
            for uuid in uuids:
                result = await driver.execute_query(
                    "",
                    query="getFact",
                    uuid=uuid,
                )
                
                if result is None:
                    continue
                
                embedding = await driver.execute_query(
                    "",
                    query="loadFactEmbedding",
                    uuid=uuid,
                )
                embedding = embedding.get('embedding', [{}])

                if len(embedding) == 0:
                    embedding = None
                else:
                    embedding = embedding[0].get('fact_embedding', None)

                source = result.get('source', [])
                target = result.get('target', [])
                result = result.get('fact', None)
                
                if result is None or len(source) == 0 or len(target) == 0:
                    continue

                source = source[0]
                target = target[0]
                
                result['fact_embedding'] = embedding
                result['valid_at'] = check_null_dates(result.get('valid_at', None))
                result['invalid_at'] = check_null_dates(result.get('invalid_at', None))
                result['expired_at'] = check_null_dates(result.get('expired_at', None))
                result['attributes'] = json.loads(result.get('attributes', '{}'))
                result['source_node_uuid'] = source.get('uuid')
                result['target_node_uuid'] = target.get('uuid')
                results.append(result)

            edges = [get_entity_edge_from_record(result) for result in results]

            return edges
        
        if len(uuids) == 0:
            return []

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        WHERE e.uuid IN $uuids
        """
            + ENTITY_EDGE_RETURN,
            uuids=uuids,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        return edges

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
            query = "getFactsbyGroup"
            if limit is not None:
                query = "getFactsbyGroupLimit"
            
            results = []
            for group_id in group_ids:
                result = await driver.execute_query(
                    "",
                    query=query,
                    limit=limit,
                    group_id=group_id,
                )
                sources = result.get('sources', [])
                targets = result.get('targets', [])
                result = result.get('facts', [])

                for i in range(len(result)):
                    fact = result[i]
                    source = sources[i]
                    target = targets[i]

                    helix_id = fact.get('id', None)
                    if helix_id is None:
                        continue
                    
                    embedding = await driver.execute_query(
                        "",
                        query="loadFactEmbedding",
                        uuid=fact.get('uuid'),
                    )
                    embedding = embedding.get('embedding', [{}])

                    if len(embedding) == 0:
                        embedding = None
                    else:
                        embedding = embedding[0].get('fact_embedding', None)
                    
                    fact['fact_embedding'] = embedding
                    fact['valid_at'] = check_null_dates(fact.get('valid_at', None))
                    fact['invalid_at'] = check_null_dates(fact.get('invalid_at', None))
                    fact['expired_at'] = check_null_dates(fact.get('expired_at', None))
                    fact['attributes'] = json.loads(fact.get('attributes', '{}'))
                    fact['source_node_uuid'] = source.get('uuid')
                    fact['target_node_uuid'] = target.get('uuid')

                    results.append(fact)

            if uuid_cursor is not None:
                results = [fact for fact in results if fact.get('uuid', '') < uuid_cursor]
            
            results.sort(key=lambda x: x.get('uuid', ''), reverse=True)

            edges = [get_entity_edge_from_record(result) for result in results]

            return edges

        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        with_embeddings_query: LiteralString = (
            """,
                e.fact_embedding AS fact_embedding
                """
            if with_embeddings
            else ''
        )

        query: LiteralString = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + ENTITY_EDGE_RETURN
            + with_embeddings_query
            + """
        ORDER BY e.uuid DESC 
        """
            + limit_query
        )

        records, _, _ = await driver.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    @classmethod
    async def get_by_node_uuid(cls, driver: GraphDriver, node_uuid: str):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="getFactsbyEntity",
                uuid=node_uuid,
            )
            
            sources = result.get('sources', [])
            targets = result.get('targets', [])
            result = result.get('facts', [])
            
            results = []
            for i in range(len(result)):
                fact = result[i]
                source = sources[i]
                target = targets[i]

                helix_id = fact.get('id', None)
                if helix_id is None:
                    continue
                
                embedding = await driver.execute_query(
                    "",
                    query="loadFactEmbedding",
                    fact_id=helix_id,
                )
                embedding = embedding.get('embedding', [{}])

                if len(embedding) == 0:
                    embedding = None
                else:
                    embedding = embedding[0].get('fact_embedding', None)
                
                fact['fact_embedding'] = embedding
                fact['valid_at'] = check_null_dates(fact.get('valid_at', None))
                fact['invalid_at'] = check_null_dates(fact.get('invalid_at', None))
                fact['expired_at'] = check_null_dates(fact.get('expired_at', None))
                fact['attributes'] = json.loads(fact.get('attributes', '{}'))
                fact['source_node_uuid'] = source.get('uuid')
                fact['target_node_uuid'] = target.get('uuid')

                results.append(fact)

            edges = [get_entity_edge_from_record(result) for result in results]

            return edges

        query: LiteralString = (
            """
                                                                    MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(m:Entity)
                                                                    """
            + ENTITY_EDGE_RETURN
        )
        records, _, _ = await driver.execute_query(query, node_uuid=node_uuid, routing_='r')

        edges = [get_entity_edge_from_record(record) for record in records]

        return edges


class CommunityEdge(Edge):
    async def save(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            stored_edge = await driver.execute_query(
                "",
                query="getCommunityEdge",
                uuid=self.uuid,
            )
            stored_edge = stored_edge.get('community_edge', None)

            if stored_edge is not None and len(stored_edge) > 0:
                query = "updateCommunityEdge"
                helix_id = stored_edge[0].get('id', None)
            else:
                query = "createCommunityEdge"
                helix_id = self.uuid

            result = await driver.execute_query(
                "",
                query=query,
                communityEdge_id=helix_id,
                community_uuid=self.source_node_uuid,
                entity_uuid=self.target_node_uuid,  
                group_id=self.group_id,
                created_at=self.created_at,
                uuid=self.uuid,
            )

            if query == "createCommunityEdge":
                helix_id = result.get('community_edge', None).get('id', None)
                if helix_id is None:
                    raise ValueError(f"Failed to create community edge")

            if query == "updateCommunityEdge":
                logger.debug(f'Updated community edge: {self.uuid}')
            else:
                logger.debug(f'Created community edge: {self.uuid}')

            return {'uuid': self.uuid}
   
        result = await driver.execute_query(
            COMMUNITY_EDGE_SAVE,
            community_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
        )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    async def delete(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="deleteCommunityEdge",
                uuid=self.uuid,
            )

            logger.debug(f'Deleted community edge: {self.uuid}')

            return result

        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            result = await driver.execute_query(
                "",
                query="getCommunityEdge",
                uuid=uuid,
            )

            if not isinstance(result, dict):
                raise EdgeNotFoundError(uuid)

            source = result.get('community', [])
            target = result.get('entity', [])
            result = result.get('community_edge', [])

            if len(source) == 0 or len(target) == 0 or len(result) == 0:
                raise EdgeNotFoundError(uuid)

            source = source[0]
            target = target[0]
            result = result[0]

            result['source_node_uuid'] = source.get('uuid')
            result['target_node_uuid'] = target.get('uuid')

            return get_community_edge_from_record(result)

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
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if driver.provider == 'helixdb':
            results = []
            for uuid in uuids:
                result = await driver.execute_query(
                    "",
                    query="getCommunityEdge",
                    uuid=uuid,
                )

                if not isinstance(result, dict):
                    continue

                source = result.get('community', [])
                target = result.get('entity', [])
                result = result.get('community_edge', [])

                if len(source) == 0 or len(target) == 0 or len(result) == 0:
                    continue

                source = source[0]
                target = target[0]
                result = result[0]

                result['source_node_uuid'] = source.get('uuid')
                result['target_node_uuid'] = target.get('uuid')
                results.append(result)

            edges = [get_community_edge_from_record(result) for result in results]

            return edges
                
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
            routing_='r',
        )

        edges = [get_community_edge_from_record(record) for record in records]

        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        if driver.provider == 'helixdb':
            query = "getCommunityEdgesbyGroup"
            if limit is not None:
                query = "getCommunityEdgesbyGroupLimit"

            results = []
            for group_id in group_ids:
                result = await driver.execute_query(
                    "",
                    query=query,
                    limit=limit,
                    group_id=group_id,
                )

                sources = result.get('communities', [])
                targets = result.get('entities', [])
                result = result.get('community_edges', [])

                for i in range(len(result)):
                    community_edge = result[i]
                    source = sources[i]
                    target = targets[i]
                    helix_id = community_edge.get('id')
                    if helix_id is None:
                        continue

                    community_edge['source_node_uuid'] = source.get('uuid')
                    community_edge['target_node_uuid'] = target.get('uuid')

                    results.append(community_edge)

            if uuid_cursor is not None:
                results = [community_edge for community_edge in results if community_edge['uuid'] < uuid_cursor]
            
            results.sort(key=lambda x: x.get('uuid', ''), reverse=True)

            community_edges = [get_community_edge_from_record(record) for record in results]

            return community_edges
                

        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
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
            uuid=uuid_cursor,
            limit=limit,
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
        created_at=parse_db_date(record['created_at']),  # type: ignore
    )


def get_entity_edge_from_record(record: Any) -> EntityEdge:
    edge = EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=record['episodes'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
        expired_at=parse_db_date(record['expired_at']),
        valid_at=parse_db_date(record['valid_at']),
        invalid_at=parse_db_date(record['invalid_at']),
        attributes=record['attributes'],
    )

    edge.attributes.pop('uuid', None)
    edge.attributes.pop('source_node_uuid', None)
    edge.attributes.pop('target_node_uuid', None)
    edge.attributes.pop('fact', None)
    edge.attributes.pop('name', None)
    edge.attributes.pop('group_id', None)
    edge.attributes.pop('episodes', None)
    edge.attributes.pop('created_at', None)
    edge.attributes.pop('expired_at', None)
    edge.attributes.pop('valid_at', None)
    edge.attributes.pop('invalid_at', None)

    return edge


def get_community_edge_from_record(record: Any):
    return CommunityEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
    )


async def create_entity_edge_embeddings(embedder: EmbedderClient, edges: list[EntityEdge]):
    if len(edges) == 0:
        return
    fact_embeddings = await embedder.create_batch([edge.fact for edge in edges])
    for edge, fact_embedding in zip(edges, fact_embeddings, strict=True):
        edge.fact_embedding = fact_embedding

def check_null_dates(helix_date: str | None):
    if helix_date is None:
        return None
    date = datetime.fromisoformat(helix_date)
    if date == datetime.fromtimestamp(0, timezone.utc):
        return None
    return date.isoformat()