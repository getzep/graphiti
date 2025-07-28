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

    helix_to_uuid: ClassVar[dict[str, str]] = {}
    uuid_to_helix: ClassVar[dict[str, str]] = {}

    async def save(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.source_node_uuid not in EpisodicNode.uuid_to_helix:
                raise NodeNotFoundError(self.source_node_uuid)
            if self.target_node_uuid not in EntityNode.uuid_to_helix:
                raise NodeNotFoundError(self.target_node_uuid)

            stored_edge = None
            if self.uuid in EpisodicEdge.uuid_to_helix:
                stored_edge = await driver.execute_query(
                    "",
                    query="getEpisodeEdge",
                    episodeEdge_id=EpisodicEdge.uuid_to_helix.get(self.uuid),
                )
                stored_edge = stored_edge.get('episode_edge', None)

            if stored_edge is not None:
                query = "updateEpisodeEdge"
                helix_id = EpisodicEdge.uuid_to_helix.get(self.uuid)
            else:
                query = "createEpisodeEdge"
                helix_id = self.uuid

            result = await driver.execute_query(
                "",
                query=query,
                episodeEdge_id=helix_id,
                episode_id=EpisodicNode.uuid_to_helix.get(self.source_node_uuid),
                entity_id=EntityNode.uuid_to_helix.get(self.target_node_uuid),
                group_id=self.group_id,
                created_at=self.created_at,
            )

            if query == 'createEpisodeEdge':
                helix_id = result.get('episode_edge', {}).get('id', None)
                if helix_id is None:
                    raise ValueError('Failed to create episode edge')

                EpisodicEdge.uuid_to_helix[self.uuid] = helix_id
                EpisodicEdge.helix_to_uuid[helix_id] = self.uuid

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
            if self.uuid not in EpisodicEdge.uuid_to_helix:
                raise EdgeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="deleteEpisodeEdge",
                episodeEdge_id=EpisodicEdge.uuid_to_helix.get(self.uuid),
            )

            helix_id = EpisodicEdge.uuid_to_helix.get(self.uuid)

            EpisodicEdge.uuid_to_helix.pop(self.uuid, None)
            if helix_id is not None:
                EpisodicEdge.helix_to_uuid.pop(helix_id, None)

            logger.debug(f'Deleted Episode Edge: {self.uuid}')

            return result

        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            if uuid not in EpisodicEdge.uuid_to_helix:
                raise EdgeNotFoundError(uuid)
            
            result = await driver.execute_query(
                "",
                query="getEpisodeEdge",
                episodeEdge_id=EpisodicEdge.uuid_to_helix.get(uuid),
            )

            if result is None:
                raise EdgeNotFoundError(uuid)

            result = result.get('episode_edge', None)

            if result is None:
                raise EdgeNotFoundError(uuid)

            result['uuid'] = uuid
            result['source_node_uuid'] = EpisodicNode.helix_to_uuid.get(result.get('from_node'))
            result['target_node_uuid'] = EntityNode.helix_to_uuid.get(result.get('to_node'))
            helix_id = result.get('id')

            EpisodicEdge.helix_to_uuid[helix_id] = uuid
            EpisodicEdge.uuid_to_helix[uuid] = helix_id

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
                if uuid not in EpisodicEdge.uuid_to_helix:
                    continue
                    
                result = await driver.execute_query(
                    "",
                    query="getEpisodeEdge",
                    episodeEdge_id=EpisodicEdge.uuid_to_helix.get(uuid),
                )
                
                if result is None:
                    continue
                
                result = result.get('episode_edge', None)
                
                if result is None:
                    continue

                result['uuid'] = uuid
                result['source_node_uuid'] = EpisodicNode.helix_to_uuid.get(result.get('from_node'))
                result['target_node_uuid'] = EntityNode.helix_to_uuid.get(result.get('to_node'))
                results.append(result)

                helix_id = result.get('id')
                EpisodicEdge.helix_to_uuid[helix_id] = uuid
                EpisodicEdge.uuid_to_helix[uuid] = helix_id

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
                result = result.get('episode_edges', [])

                for episode_edge in result:
                    helix_id = episode_edge.get('id')
                    if helix_id is None or helix_id not in EpisodicEdge.helix_to_uuid:
                        continue

                    episode_edge['uuid'] = EpisodicEdge.helix_to_uuid.get(helix_id)
                    episode_edge['source_node_uuid'] = EpisodicNode.helix_to_uuid.get(episode_edge.get('from_node'))
                    episode_edge['target_node_uuid'] = EntityNode.helix_to_uuid.get(episode_edge.get('to_node'))

                    EpisodicEdge.uuid_to_helix[episode_edge['uuid']] = helix_id
                    EpisodicEdge.helix_to_uuid[helix_id] = episode_edge['uuid']

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

    helix_to_uuid: ClassVar[dict[str, str]] = {}
    uuid_to_helix: ClassVar[dict[str, str]] = {}

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
            if self.uuid not in EntityEdge.uuid_to_helix:
                raise EdgeNotFoundError(self.uuid)
            
            result = await driver.execute_query(
                "",
                query="loadFactEmbedding",
                fact_id=EntityEdge.uuid_to_helix.get(self.uuid),
            )
            self.fact_embedding = result.get('embedding', None)[0].get('fact_embedding', None)
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
            stored_fact = None
            if self.uuid in EntityEdge.uuid_to_helix:
                stored_fact = await driver.execute_query(
                    "",
                    query="getFact",
                    fact_id=EntityEdge.uuid_to_helix.get(self.uuid),
                )
                stored_fact = stored_fact.get('fact', None)

            if stored_fact is not None:
                query = "updateFact"
                helix_id = EntityEdge.uuid_to_helix.get(self.uuid)
            else:
                query = "createFact"
                helix_id = self.uuid

            if self.source_node_uuid not in EntityNode.uuid_to_helix:
                raise NodeNotFoundError(self.source_node_uuid)
            if self.target_node_uuid not in EntityNode.uuid_to_helix:
                raise NodeNotFoundError(self.target_node_uuid)

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
                name=edge_data['name'],
                fact=edge_data['fact'],
                fact_embedding=edge_data['fact_embedding'],
                group_id=edge_data['group_id'],
                source_uuid=EntityNode.uuid_to_helix.get(self.source_node_uuid),
                target_uuid=EntityNode.uuid_to_helix.get(self.target_node_uuid),
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

                EntityEdge.uuid_to_helix[self.uuid] = helix_id
                EntityEdge.helix_to_uuid[helix_id] = self.uuid

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
            if self.uuid not in EntityEdge.uuid_to_helix:
                raise EdgeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="deleteFact",
                fact_id=EntityEdge.uuid_to_helix.get(self.uuid),
            )

            helix_id = EntityEdge.uuid_to_helix.get(self.uuid)

            EntityEdge.uuid_to_helix.pop(self.uuid, None)
            if helix_id is not None:
                EntityEdge.helix_to_uuid.pop(helix_id, None)

            logger.debug(f'Deleted Entity Edge: {self.uuid}')

            return result

        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            if uuid not in EntityEdge.uuid_to_helix:
                raise EdgeNotFoundError(uuid)
            
            result = await driver.execute_query(
                "",
                query="getFact",
                fact_id=EntityEdge.uuid_to_helix.get(uuid),
            )
            
            if result is None:
                raise EdgeNotFoundError(uuid)
            
            embedding = await driver.execute_query(
                "",
                query="loadFactEmbedding",
                fact_id=EntityEdge.uuid_to_helix.get(uuid),
            )
            embedding = embedding.get('embedding', None)[0].get('fact_embedding', None)

            source = result.get('source', None)
            target = result.get('target', None)
            result = result.get('fact', None)
            
            if result is None:
                raise EdgeNotFoundError(uuid)
            
            result['uuid'] = uuid
            result['fact_embedding'] = embedding
            result['valid_at'] = check_null_dates(result.get('valid_at', None))
            result['invalid_at'] = check_null_dates(result.get('invalid_at', None))
            result['expired_at'] = check_null_dates(result.get('expired_at', None))
            result['attributes'] = json.loads(result.get('attributes', '{}'))
            result['source_node_uuid'] = EntityNode.helix_to_uuid.get(source[0].get('from_node'))
            result['target_node_uuid'] = EntityNode.helix_to_uuid.get(target[0].get('to_node'))
            helix_id = result.get('id')

            EntityEdge.helix_to_uuid[helix_id] = uuid
            EntityEdge.uuid_to_helix[uuid] = helix_id

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
                if uuid not in EntityEdge.uuid_to_helix:
                    continue
            
                result = await driver.execute_query(
                    "",
                    query="getFact",
                    fact_id=EntityEdge.uuid_to_helix.get(uuid),
                )
                
                if result is None:
                    continue
                
                embedding = await driver.execute_query(
                    "",
                    query="loadFactEmbedding",
                    fact_id=EntityEdge.uuid_to_helix.get(uuid),
                )
                embedding = embedding.get('embedding', None)[0].get('fact_embedding', None)

                source = result.get('source', None)
                target = result.get('target', None)
                result = result.get('fact', None)
                
                if result is None:
                    continue
                
                result['uuid'] = uuid
                result['fact_embedding'] = embedding
                result['valid_at'] = check_null_dates(result.get('valid_at', None))
                result['invalid_at'] = check_null_dates(result.get('invalid_at', None))
                result['expired_at'] = check_null_dates(result.get('expired_at', None))
                result['attributes'] = json.loads(result.get('attributes', '{}'))
                result['source_node_uuid'] = EntityNode.helix_to_uuid.get(source[0].get('from_node'))
                result['target_node_uuid'] = EntityNode.helix_to_uuid.get(target[0].get('to_node'))
                results.append(result)

                helix_id = result.get('id')
                EntityEdge.helix_to_uuid[helix_id] = uuid
                EntityEdge.uuid_to_helix[uuid] = helix_id

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
                sources = result.get('source', [])
                targets = result.get('target', [])
                result = result.get('facts', [])

                for i in range(len(result)):
                    fact = result[i]
                    source = sources[i].get('from_node')
                    target = targets[i].get('to_node')

                    if source not in EntityNode.helix_to_uuid:
                        continue
                    if target not in EntityNode.helix_to_uuid:
                        continue

                    helix_id = fact.get('id', None)
                    if helix_id is None or helix_id not in EntityEdge.helix_to_uuid:
                        continue
                    
                    embedding = await driver.execute_query(
                        "",
                        query="loadFactEmbedding",
                        fact_id=helix_id,
                    )
                    embedding = embedding.get('embedding', None)[0].get('fact_embedding', None)
                    
                    fact['uuid'] = EntityEdge.helix_to_uuid.get(helix_id)
                    fact['fact_embedding'] = embedding
                    fact['valid_at'] = check_null_dates(fact.get('valid_at', None))
                    fact['invalid_at'] = check_null_dates(fact.get('invalid_at', None))
                    fact['expired_at'] = check_null_dates(fact.get('expired_at', None))
                    fact['attributes'] = json.loads(fact.get('attributes', '{}'))
                    fact['source_node_uuid'] = EntityNode.helix_to_uuid.get(source)
                    fact['target_node_uuid'] = EntityNode.helix_to_uuid.get(target)

                    EntityEdge.uuid_to_helix[fact['uuid']] = helix_id
                    EntityEdge.helix_to_uuid[helix_id] = fact['uuid']

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
            if node_uuid not in EntityNode.uuid_to_helix:
                raise NodeNotFoundError(node_uuid)
            
            result = await driver.execute_query(
                "",
                query="getFactsbyEntity",
                entity_id=EntityNode.uuid_to_helix.get(node_uuid),
            )
            
            sources = result.get('source', [])
            targets = result.get('target', [])
            result = result.get('facts', [])
            
            results = []
            for i in range(len(result)):
                fact = result[i]
                source = sources[i].get('from_node')
                target = targets[i].get('to_node')

                if source not in EntityNode.helix_to_uuid:
                    continue
                if target not in EntityNode.helix_to_uuid:
                    continue
                
                helix_id = fact.get('id', None)
                if helix_id is None or helix_id not in EntityEdge.helix_to_uuid:
                    continue
                
                embedding = await driver.execute_query(
                    "",
                    query="loadFactEmbedding",
                    fact_id=helix_id,
                )
                embedding = embedding.get('embedding', None)[0].get('fact_embedding', None)
                
                fact['uuid'] = EntityEdge.helix_to_uuid.get(helix_id)
                fact['fact_embedding'] = embedding
                fact['valid_at'] = check_null_dates(fact.get('valid_at', None))
                fact['invalid_at'] = check_null_dates(fact.get('invalid_at', None))
                fact['expired_at'] = check_null_dates(fact.get('expired_at', None))
                fact['attributes'] = json.loads(fact.get('attributes', '{}'))
                fact['source_node_uuid'] = EntityNode.helix_to_uuid.get(source)
                fact['target_node_uuid'] = EntityNode.helix_to_uuid.get(target)

                EntityEdge.uuid_to_helix[fact['uuid']] = helix_id
                EntityEdge.helix_to_uuid[helix_id] = fact['uuid']

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

    helix_to_uuid: ClassVar[dict[str, str]] = {}
    uuid_to_helix: ClassVar[dict[str, str]] = {}

    async def save(self, driver: GraphDriver):
        if driver.provider == 'helixdb':
            if self.source_node_uuid not in CommunityNode.uuid_to_helix:
                raise NodeNotFoundError(self.source_node_uuid)
            if self.target_node_uuid not in EntityNode.uuid_to_helix:
                raise NodeNotFoundError(self.target_node_uuid)

            stored_edge = None
            if self.uuid in CommunityEdge.uuid_to_helix:
                stored_edge = await driver.execute_query(
                    "",
                    query="getCommunityEdge",
                    community_id=CommunityEdge.uuid_to_helix.get(self.uuid),
                )
                stored_edge = stored_edge.get('community_edge', None)

            if stored_edge is not None:
                query = "updateCommunityEdge"
                helix_id = CommunityEdge.uuid_to_helix.get(self.uuid)
            else:
                query = "createCommunityEdge"
                helix_id = self.uuid

            result = await driver.execute_query(
                "",
                query=query,
                communityEdge_id=helix_id,
                community_id=CommunityNode.uuid_to_helix.get(self.source_node_uuid),
                entity_id=EntityNode.uuid_to_helix.get(self.target_node_uuid),  
                group_id=self.group_id,
                created_at=self.created_at,
            )

            if query == "createCommunityEdge":
                helix_id = result.get('community_edge', None).get('id', None)
                if helix_id is not None:
                    raise ValueError(f"Failed to create community edge")
            
                CommunityEdge.uuid_to_helix[self.uuid] = helix_id
                CommunityEdge.helix_to_uuid[helix_id] = self.uuid

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
            if self.uuid not in CommunityEdge.uuid_to_helix:
                raise EdgeNotFoundError(self.uuid)

            result = await driver.execute_query(
                "",
                query="deleteCommunityEdge",
                communityEdge_id=CommunityEdge.uuid_to_helix.get(self.uuid),
            )

            helix_id = CommunityEdge.uuid_to_helix.get(self.uuid)

            CommunityEdge.uuid_to_helix.pop(self.uuid)
            if helix_id is not None:
                CommunityEdge.helix_to_uuid.pop(helix_id)

            logger.debug(f'Deleted community edge: {self.uuid}')

            return result

        return await super().delete(driver)

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        if driver.provider == 'helixdb':
            if uuid not in CommunityEdge.uuid_to_helix:
                raise EdgeNotFoundError(uuid)

            result = await driver.execute_query(
                "",
                query="getCommunityEdge",
                community_id=CommunityEdge.uuid_to_helix.get(uuid),
            )

            if result is None:
                raise EdgeNotFoundError(uuid)

            result = result.get('community_edge', None)

            if result is None:
                raise EdgeNotFoundError(uuid)

            result['uuid'] = uuid
            result['source_node_uuid'] = CommunityNode.helix_to_uuid.get(result.get('from_node'))
            result['target_node_uuid'] = EntityNode.helix_to_uuid.get(result.get('to_node'))
            helix_id = result.get('id')

            CommunityEdge.helix_to_uuid[helix_id] = uuid
            CommunityEdge.uuid_to_helix[uuid] = helix_id

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
                if uuid not in CommunityEdge.uuid_to_helix:
                    continue

                result = await driver.execute_query(
                    "",
                    query="getCommunityEdge",
                    community_id=CommunityEdge.uuid_to_helix.get(uuid),
                )

                if result is None:
                    continue

                result = result.get('community_edge', None)

                if result is None:
                    continue

                result['uuid'] = uuid
                result['source_node_uuid'] = CommunityNode.helix_to_uuid.get(result.get('from_node'))
                result['target_node_uuid'] = EntityNode.helix_to_uuid.get(result.get('to_node'))
                results.append(result)

                helix_id = result.get('id')
                CommunityEdge.helix_to_uuid[helix_id] = uuid
                CommunityEdge.uuid_to_helix[uuid] = helix_id

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
                result = result.get('community_edges', [])

                for community_edge in result:
                    helix_id = community_edge.get('id')
                    if helix_id is None or helix_id not in CommunityEdge.helix_to_uuid:
                        continue

                    community_edge['uuid'] = CommunityEdge.helix_to_uuid.get(helix_id)
                    community_edge['source_node_uuid'] = CommunityNode.helix_to_uuid.get(community_edge.get('from_node'))
                    community_edge['target_node_uuid'] = EntityNode.helix_to_uuid.get(community_edge.get('to_node'))

                    CommunityEdge.uuid_to_helix[community_edge['uuid']] = helix_id
                    CommunityEdge.helix_to_uuid[helix_id] = community_edge['uuid']

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