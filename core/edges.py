import logging
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from uuid import uuid4

from neo4j import AsyncDriver
from pydantic import BaseModel, Field

from core.llm_client.config import EMBEDDING_DIM
from core.nodes import Node

logger = logging.getLogger(__name__)


class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: uuid4().hex)
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False


class EpisodicEdge(Edge):
    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (episode:Episodic {uuid: $episode_uuid}) 
        MATCH (node:Entity {uuid: $entity_uuid}) 
        MERGE (episode)-[r:MENTIONS {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, created_at: $created_at}
        RETURN r.uuid AS uuid""",
            episode_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            created_at=self.created_at,
        )

        logger.info(f"Saved edge to neo4j: {self.uuid}")

        return result


# TODO: Neo4j doesn't support variables for edge types and labels.
#  Right now we have all edge nodes as type RELATES_TO


class EntityEdge(Edge):
    name: str = Field(description="name of the edge, relation name")
    fact: str = Field(
        description="fact representing the edge and nodes that it connects"
    )
    fact_embedding: list[float] | None = Field(
        default=None, description="embedding of the fact"
    )
    episodes: list[str] | None = Field(
        default=None,
        description="list of episode ids that reference these entity edges",
    )
    expired_at: datetime | None = Field(
        default=None, description="datetime of when the node was invalidated"
    )
    valid_at: datetime | None = Field(
        default=None, description="datetime of when the fact became true"
    )
    invalid_at: datetime | None = Field(
        default=None, description="datetime of when the fact stopped being true"
    )

    async def generate_embedding(self, embedder, model="text-embedding-3-small"):
        start = time()

        text = self.fact.replace("\n", " ")
        embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
        self.fact_embedding = embedding[:EMBEDDING_DIM]

        end = time()
        logger.info(f"embedded {text} in {end-start} ms")

        return embedding

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (source:Entity {uuid: $source_uuid}) 
        MATCH (target:Entity {uuid: $target_uuid}) 
        MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
        SET r = {uuid: $uuid, name: $name, fact: $fact, fact_embedding: $fact_embedding, 
        episodes: $episodes, created_at: $created_at, expired_at: $expired_at, 
        valid_at: $valid_at, invalid_at: $invalid_at}
        RETURN r.uuid AS uuid""",
            source_uuid=self.source_node_uuid,
            target_uuid=self.target_node_uuid,
            uuid=self.uuid,
            name=self.name,
            fact=self.fact,
            fact_embedding=self.fact_embedding,
            episodes=self.episodes,
            created_at=self.created_at,
            expired_at=self.expired_at,
            valid_at=self.valid_at,
            invalid_at=self.invalid_at,
        )

        logger.info(f"Saved edge to neo4j: {self.uuid}")

        return result
