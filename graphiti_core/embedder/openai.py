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

import numpy as np
from collections.abc import Iterable

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'


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

    async def _validate_embedding(self, embedding: list[float], label: str) -> list[float]:
        """Validate embedding for NaN/Inf values. Retry once if invalid."""
        arr = np.asarray(embedding, dtype=np.float64)
        if np.all(np.isfinite(arr)):
            return embedding

        bad_count = int(np.sum(~np.isfinite(arr)))
        logger.warning(
            f'Embedding contains {bad_count}/{len(embedding)} NaN/Inf values '
            f'(label: {label}). Retrying...'
        )
        raise ValueError(
            f'Embedding returned {bad_count}/{len(embedding)} non-finite values. '
            f'The provider may have returned a corrupt response.'
        )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        embedding = result.data[0].embedding[: self.config.embedding_dim]
        try:
            return await self._validate_embedding(embedding, 'create')
        except ValueError:
            # Retry once: NaN could be transient provider-side corruption
            logger.info('Retrying embedding call after NaN detection...')
            result = await self.client.embeddings.create(
                input=input_data, model=self.config.embedding_model
            )
            embedding = result.data[0].embedding[: self.config.embedding_dim]
            await self._validate_embedding(embedding, 'create_retry')
            return embedding

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        embeddings = [
            embedding.embedding[: self.config.embedding_dim] for embedding in result.data
        ]
        try:
            validated = []
            for i, emb in enumerate(embeddings):
                validated.append(await self._validate_embedding(emb, f'batch[{i}]'))
            return validated
        except ValueError:
            pass

        # Retry full batch once
        logger.info(f'Retrying batch embedding call after NaN detection...')
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        embeddings_retry = [
            e.embedding[: self.config.embedding_dim] for e in result.data
        ]
        validated = []
        for j, emb_retry in enumerate(embeddings_retry):
            await self._validate_embedding(emb_retry, f'batch_retry[{j}]')
            validated.append(emb_retry)
        return validated
