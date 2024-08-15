from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid1

from openai import OpenAI
from pydantic import BaseModel, Field
from neo4j import AsyncDriver
import logging

logger = logging.getLogger(__name__)


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: uuid1().hex)
    name: str
    labels: list[str]
    created_at: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...


class EpisodicNode(Node):
    source: str  # source type
    source_description: str  # description of the data source
    content: str  # raw episode data
    entity_edges: list[str]  # list of entity edge ids referenced in this episode
    valid_at: datetime = None  # datetime of when the original document was created

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MERGE (n:Episodic {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, source_description: $source_description, content: $content, 
        entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
        RETURN n.uuid AS uuid""",
            uuid=self.uuid,
            name=self.name,
            source_description=self.source_description,
            content=self.content,
            entity_edges=self.entity_edges,
            created_at=self.created_at,
            valid_at=self.valid_at,
            _database="neo4j",
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")
        print(self.uuid)

        return result


class EntityNode(Node):
    summary: str  # regional summary of surrounding edges

    async def refresh_summary(self, driver: AsyncDriver, llm_client: OpenAI): ...

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MERGE (n:Entity {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, summary: $summary, created_at: $created_at}
        RETURN n.uuid AS uuid""",
            uuid=self.uuid,
            name=self.name,
            summary=self.summary,
            created_at=self.created_at,
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")

        return result
