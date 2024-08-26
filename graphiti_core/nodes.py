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
from datetime import datetime
from enum import Enum
from time import time
from uuid import uuid4

from neo4j import AsyncDriver
from openai import OpenAI
from pydantic import BaseModel, Field

from graphiti_core.llm_client.config import EMBEDDING_DIM

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
            source=self.source.value,
            _database='neo4j',
        )

        logger.info(f'Saved Node to neo4j: {self.uuid}')

        return result


class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)

    async def update_summary(self, driver: AsyncDriver): ...

    async def refresh_summary(self, driver: AsyncDriver, llm_client: OpenAI): ...

    async def generate_name_embedding(self, embedder, model='text-embedding-3-small'):
        start = time()
        text = self.name.replace('\n', ' ')
        embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
        self.name_embedding = embedding[:EMBEDDING_DIM]
        end = time()
        logger.info(f'embedded {text} in {end - start} ms')

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

        logger.info(f'Saved Node to neo4j: {self.uuid}')

        return result
