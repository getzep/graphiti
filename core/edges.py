from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from datetime import datetime
from neo4j import AsyncDriver
from uuid import uuid1
import logging

from core.nodes import Node

logger = logging.getLogger(__name__)


class Edge(BaseModel, ABC):
    uuid: Field(default_factory=lambda: uuid1().hex)
    source_node: Node
    target_node: Node
    transaction_from: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...


class EpisodicEdge(Edge):
    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (episode:Episodic {uuid: $episode_uuid}) 
        MATCH (node:Semantic {uuid: $semantic_uuid}) 
        MERGE (episode)-[r:MENTIONS {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, transaction_from: $transaction_from}
        RETURN r.uuid AS uuid""",
            episode_uuid=self.source_node.uuid,
            semantic_uuid=self.target_node.uuid,
            uuid=self.uuid,
            transaction_from=self.transaction_from,
        )

        logger.info(f"Saved edge to neo4j: {self.uuid}")

        return result


# TODO: Neo4j doesn't support variables for edge types and labels.
#  Right now we have all edge nodes as type RELATES_TO


class SemanticEdge(Edge):
    name: str
    fact: str
    fact_embedding: list[int] = None
    episodes: list[str] = None  # list of episodes that reference these semantic edges
    transaction_to: datetime = None  # datetime of when the node was invalidated
    valid_from: datetime = None  # datetime of when the fact became true
    valid_to: datetime = None  # datetime of when the fact stopped being true

    def generate_embedding(self, embedder, model="text-embedding-3-large"):
        text = self.fact.replace("\n", " ")
        embedding = embedder.create(input=[text], model=model).data[0].embedding
        self.fact_embedding = embedding

        return embedding

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (source:Semantic {uuid: $source_uuid}) 
        MATCH (target:Semantic {uuid: $target_uuid}) 
        MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
        SET r = {uuid: $uuid, name: $name, fact: $fact, fact_embedding: $fact_embedding, 
        episodes: $episodes, transaction_from: $transaction_from, transaction_to: $transaction_to, 
        valid_from: $valid_from, valid_to: $valid_to}
        RETURN r.uuid AS uuid""",
            source_uuid=self.source_node.uuid,
            target_uuid=self.target_node.uuid,
            uuid=self.uuid,
            name=self.name,
            fact=self.fact,
            fact_embedding=self.fact_embedding,
            episodes=self.episodes,
            transaction_from=self.transaction_from,
            transaction_to=self.transaction_to,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")

        return result
