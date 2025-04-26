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

from google import genai  # type: ignore
from google.genai import types  # type: ignore
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'embedding-001'


class GeminiEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None


class GeminiEmbedder(EmbedderClient):
    """
    Google Gemini Embedder Client
    """

    def __init__(self, config: GeminiEmbedderConfig | None = None):
        if config is None:
            config = GeminiEmbedderConfig()
        self.config = config

        # Configure the Gemini API
        self.client = genai.Client(
            api_key=config.api_key,
        )

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
            contents=[input_data],
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dim),
        )

        return result.embeddings[0].values

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        # Generate embeddings
        result = await self.client.aio.models.embed_content(
            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
            contents=input_data_list,
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dim),
        )

        return [embedding.values for embedding in result.embeddings]
