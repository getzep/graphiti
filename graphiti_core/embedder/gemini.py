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
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
else:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            'google-genai is required for GeminiEmbedder. '
            'Install it with: pip install graphiti-core[google-genai]'
        ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'text-embedding-001'  # gemini-embedding-001 or text-embedding-005

DEFAULT_BATCH_SIZE = 100


class GeminiEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None


class GeminiEmbedder(EmbedderClient):
    """
    Google Gemini Embedder Client
    """

    def __init__(
        self,
        config: GeminiEmbedderConfig | None = None,
        client: 'genai.Client | None' = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the GeminiEmbedder with the provided configuration and client.

        Args:
            config (GeminiEmbedderConfig | None): The configuration for the GeminiEmbedder, including API key, model, base URL, temperature, and max tokens.
            client (genai.Client | None): An optional async client instance to use. If not provided, a new genai.Client is created.
            batch_size (int | None): An optional batch size to use. If not provided, the default batch size will be used.
        """
        if config is None:
            config = GeminiEmbedderConfig()

        self.config = config

        if client is None:
            self.client = genai.Client(api_key=config.api_key)
        else:
            self.client = client

        if batch_size is None and self.config.embedding_model == 'gemini-embedding-001':
            # Gemini API has a limit on the number of instances per request
            # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
            self.batch_size = 1
        elif batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data using Google's Gemini embedding model.

        Args:
            input_data: The input data to create embeddings for. Can be a string, list of strings,
                       or an iterable of integers or iterables of integers.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Generate embeddings
        result = await self.client.aio.models.embed_content(
            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
            contents=[input_data],  # type: ignore[arg-type]  # mypy fails on broad union type
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dim),
        )

        if not result.embeddings or len(result.embeddings) == 0 or not result.embeddings[0].values:
            raise ValueError('No embeddings returned from Gemini API in create()')

        return result.embeddings[0].values

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of input data using Google's Gemini embedding model.

        This method handles batching to respect the Gemini API's limits on the number
        of instances that can be processed in a single request.

        Args:
            input_data_list: A list of strings to create embeddings for.

        Returns:
            A list of embedding vectors (each vector is a list of floats).
        """
        if not input_data_list:
            return []

        batch_size = self.batch_size
        all_embeddings = []

        # Process inputs in batches
        for i in range(0, len(input_data_list), batch_size):
            batch = input_data_list[i : i + batch_size]

            try:
                # Generate embeddings for this batch
                result = await self.client.aio.models.embed_content(
                    model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
                    contents=batch,  # type: ignore[arg-type]  # mypy fails on broad union type
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.config.embedding_dim
                    ),
                )

                if not result.embeddings or len(result.embeddings) == 0:
                    raise Exception('No embeddings returned')

                # Process embeddings from this batch
                for embedding in result.embeddings:
                    if not embedding.values:
                        raise ValueError('Empty embedding values returned')
                    all_embeddings.append(embedding.values)

            except Exception as e:
                # If batch processing fails, fall back to individual processing
                logger.warning(
                    f'Batch embedding failed for batch {i // batch_size + 1}, falling back to individual processing: {e}'
                )

                for item in batch:
                    try:
                        # Process each item individually
                        result = await self.client.aio.models.embed_content(
                            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
                            contents=[item],  # type: ignore[arg-type]  # mypy fails on broad union type
                            config=types.EmbedContentConfig(
                                output_dimensionality=self.config.embedding_dim
                            ),
                        )

                        if not result.embeddings or len(result.embeddings) == 0:
                            raise ValueError('No embeddings returned from Gemini API')
                        if not result.embeddings[0].values:
                            raise ValueError('Empty embedding values returned')

                        all_embeddings.append(result.embeddings[0].values)

                    except Exception as individual_error:
                        logger.error(f'Failed to embed individual item: {individual_error}')
                        raise individual_error

        return all_embeddings
