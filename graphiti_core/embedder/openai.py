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
import logging
from time import perf_counter

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'
logger = logging.getLogger(__name__)


class OpenAIEmbedderConfig(EmbedderConfig):
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None


class OpenAIEmbedder(EmbedderClient):
    """
    OpenAI Embedder Client

    This client supports both AsyncOpenAI and AsyncAzureOpenAI clients.
    """

    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        if config is None:
            config = OpenAIEmbedderConfig()
        self.config = config

        if client is not None:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        started_at = perf_counter()
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        embedding = result.data[0].embedding[: self.config.embedding_dim]
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info(
            'Embedder timing model=%s dimensions=%s batch_size=1 elapsed_ms=%.1f',
            self.config.embedding_model,
            self.config.embedding_dim,
            elapsed_ms,
        )
        return embedding

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        started_at = perf_counter()
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        embeddings = [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info(
            'Embedder timing model=%s dimensions=%s batch_size=%s elapsed_ms=%.1f',
            self.config.embedding_model,
            self.config.embedding_dim,
            len(input_data_list),
            elapsed_ms,
        )
        return embeddings
