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
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'


class OpenAIEmbedderConfig(EmbedderConfig):
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    """Requested output dimensionality.

    When set, it is forwarded to the OpenAI ``dimensions`` parameter (supported by
    ``text-embedding-3-*`` models) so the API returns a natively reduced, renormalized
    vector instead of one that is naively truncated. Leave as ``None`` (the default) for
    backwards-compatible behavior and for models/endpoints that do not support the
    parameter (e.g. ``text-embedding-ada-002`` or custom ``base_url`` deployments). When
    set, prefer matching :attr:`embedding_dim` so downstream storage stays consistent.
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
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    def _create_kwargs(self, input_data: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            'input': input_data,
            'model': self.config.embedding_model,
        }
        if self.config.dimensions is not None:
            kwargs['dimensions'] = self.config.dimensions
        return kwargs

    def _postprocess(self, embedding: list[float]) -> list[float]:
        # A natively requested ``dimensions`` value already yields a correctly sized,
        # renormalized vector, so only fall back to truncation when it is not used.
        if self.config.dimensions is not None:
            return embedding
        return embedding[: self.config.embedding_dim]

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(**self._create_kwargs(input_data))
        return self._postprocess(result.data[0].embedding)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(**self._create_kwargs(input_data_list))
        return [self._postprocess(embedding.embedding) for embedding in result.data]
