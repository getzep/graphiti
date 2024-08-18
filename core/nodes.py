from abc import ABC, abstractmethod
from pydantic import Field
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, Field
from neo4j import AsyncDriver
import logging

from core.llm_client.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    labels: list[str] = Field(default_factory=list)
    created_at: datetime

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False


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
        SET n = {uuid: $uuid, name: $name, source_description: $source_description, source: $source, content: $content, 
        entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
        RETURN n.uuid AS uuid""",
            uuid=self.uuid,
            name=self.name,
            source_description=self.source_description,
            content=self.content,
            entity_edges=self.entity_edges,
            created_at=self.created_at,
            valid_at=self.valid_at,
            source=self.source,
            _database="neo4j",
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")

        return result


class EntityNode(Node):
    name_embedding: list[float] | None = Field(
        default=None, description="embedding of the name"
    )
    summary: str = Field(description="regional summary of surrounding edges")

    async def update_summary(self, driver: AsyncDriver): ...

    async def refresh_summary(self, driver: AsyncDriver, llm_client: OpenAI): ...

    async def generate_name_embedding(self, embedder, model="text-embedding-3-small"):
        text = self.name.replace("\n", " ")
        embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
        self.name_embedding = embedding[:EMBEDDING_DIM]

        return embedding

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MERGE (n:Entity {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, name_embedding: $name_embedding, summary: $summary, created_at: $created_at}
        RETURN n.uuid AS uuid""",
            uuid=self.uuid,
            name=self.name,
            summary=self.summary,
            name_embedding=self.name_embedding,
            created_at=self.created_at,
        )

        logger.info(f"Saved Node to neo4j: {self.uuid}")

        return result
