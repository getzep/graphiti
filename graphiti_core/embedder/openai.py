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

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'


EMBEDDER_CONNECT_TIMEOUT = 30.0
EMBEDDER_READ_TIMEOUT = 120.0


class OpenAIEmbedderConfig(EmbedderConfig):
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    """Maximum number of input texts per embeddings request.

    Some OpenAI-compatible providers cap the number of texts per request
    (for example, DashScope caps batches at 10 texts). Set ``batch_size``
    to divide larger batches into several requests. ``None`` (the default)
    sends one request per batch, which matches the official OpenAI API.
    """


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
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=httpx.Timeout(EMBEDDER_CONNECT_TIMEOUT, read=EMBEDDER_READ_TIMEOUT),
            )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        batch_size = self.config.batch_size if self.config else None

        if batch_size is not None and len(input_data_list) > batch_size:
            # The provider caps the number of texts per request: divide the
            # input into requests of at most batch_size texts.
            all_embeddings = []
            for i in range(0, len(input_data_list), batch_size):
                batch = input_data_list[i : i + batch_size]
                result = await self.client.embeddings.create(
                    input=batch, model=self.config.embedding_model
                )
                all_embeddings.extend(
                    embedding.embedding[: self.config.embedding_dim] for embedding in result.data
                )
            return all_embeddings

        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]
