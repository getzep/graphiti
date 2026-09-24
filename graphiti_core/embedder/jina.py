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

from collections.abc import Iterable

from openai import AsyncOpenAI
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'jina-embeddings-v5-text-nano'


class JinaEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None
    task: str = Field(default='retrieval.passage')


class JinaEmbedder(EmbedderClient):
    """
    Jina AI Embedder Client
    
    Supports jina-embeddings-v5-text-nano (768d) and jina-embeddings-v5-text-small (1024d).
    """

    def __init__(self, config: JinaEmbedderConfig | None = None):
        if config is None:
            config = JinaEmbedderConfig()
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url='https://api.jina.ai/v1',
        )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data,
            model=self.config.embedding_model,
            extra_body={'task': self.config.task},
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(
            input=input_data_list,
            model=self.config.embedding_model,
            extra_body={'task': self.config.task},
        )
        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]
