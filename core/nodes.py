from abc import ABC, abstractmethod
from pydantic import Field
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, Field
from neo4j import AsyncDriver
import logging

logger = logging.getLogger(__name__)


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    labels: list[str] = Field(default_factory=list)
    created_at: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...


class EpisodicNode(Node):
    source: str = Field(description="source type")
    source_description: str = Field(description="description of the data source")
    content: str = Field(description="raw episode data")
    entity_edges: list[str] = Field(
        description="list of entity edges referenced in this episode",
        default_factory=list,
    )
    valid_at: datetime | None = Field(
        description="datetime of when the original document was created",
        default=None,
    )

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
    summary: str = Field(description="regional summary of surrounding edges")

    async def update_summary(self, driver: AsyncDriver): ...

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
