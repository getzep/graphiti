from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid1

from openai import OpenAI
from pydantic import BaseModel
from neo4j import AsyncDriver
import logging

logger = logging.getLogger(__name__)


class Node(BaseModel, ABC):
    uuid: str | None
    name: str
    labels: list[str]
    transaction_from: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...


class EpisodicNode(Node):
    source: str  # source type
    source_description: str  # description of the data source
    content: str  # raw episode data
    semantic_edges: list[str]  # list of semantic edges referenced in this episode
    valid_from: datetime = None  # datetime of when the original document was created

    async def save(self, driver: AsyncDriver):
        if self.uuid is None:
            uuid = uuid1()
            logger.info(f"Created uuid: {uuid} for node with name: {self.name}")
            self.uuid = str(uuid)

        result = await driver.execute_query(
            """
        MERGE (n:Episodic {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, source_description: $source_description, content: $content, 
        semantic_edges: $semantic_edges, transaction_from: $transaction_from, valid_from: $valid_from}
        RETURN n.uuid AS uuid""",
            uuid=self.uuid,
            name=self.name,
            source_description=self.source_description,
            content=self.content,
            semantic_edges=self.semantic_edges,
            transaction_from=self.transaction_from,
            valid_from=self.valid_from,
            _database="neo4j",
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")
        print(self.uuid)

        return result


class SemanticNode(Node):
    summary: str  # regional summary of surrounding edges

    async def refresh_summary(self, driver: AsyncDriver, llm_client: OpenAI): ...

    async def save(self, driver: AsyncDriver):
        if self.uuid is None:
            uuid = uuid1()
            logger.info(f"Created uuid: {uuid} for node with name: {self.name}")
            self.uuid = str(uuid)

        result = await driver.execute_query(
            """
        MERGE (n:Semantic {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, summary: $summary, transaction_from: $transaction_from}
        RETURN n.uuid AS uuid""",
            uuid=self.uuid,
            name=self.name,
            summary=self.summary,
            transaction_from=self.transaction_from,
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")

        return result
