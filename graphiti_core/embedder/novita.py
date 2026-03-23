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

import os
from collections.abc import Iterable

from openai import AsyncOpenAI

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'qwen/qwen3-embedding-0.6b'
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_BASE_URL = 'https://api.novita.ai/openai'


class NovitaEmbedderConfig(EmbedderConfig):
    """Configuration for Novita AI embedder."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL


class NovitaEmbedder(EmbedderClient):
    """
    Novita AI Embedder Client.

    This client provides access to Novita AI's embedding models through their
    OpenAI-compatible API endpoint.

    Novita AI offers cost-effective embedding models including:
    - qwen/qwen3-embedding-0.6b (default) - 1024 dimensions, 8K max input

    Attributes:
        client: The AsyncOpenAI client configured for Novita AI.
        config: The embedder configuration.

    Example:
        ```python
        from graphiti_core.embedder.novita import NovitaEmbedder, NovitaEmbedderConfig

        # Using environment variable NOVITA_API_KEY
        embedder = NovitaEmbedder()

        # Or with explicit configuration
        embedder = NovitaEmbedder(
            config=NovitaEmbedderConfig(
                api_key='your-novita-api-key',
                embedding_model='qwen/qwen3-embedding-0.6b',
                embedding_dim=1024,
            )
        )
        ```
    """

    def __init__(
        self,
        config: NovitaEmbedderConfig | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """
        Initialize the Novita AI embedder client.

        Args:
            config: Embedder configuration including api_key and model.
                    If not provided, defaults are used.
            client: Optional pre-configured AsyncOpenAI client.
        """
        if config is None:
            config = NovitaEmbedderConfig()

        # Set Novita-specific defaults if not provided
        if config.api_key is None:
            config.api_key = os.environ.get('NOVITA_API_KEY')

        self.config = config

        if client is not None:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create an embedding for the input data.

        Args:
            input_data: The text or tokens to embed. If a list is provided,
                        only the first item is embedded (use create_batch for
                        multiple embeddings).

        Returns:
            The embedding vector.
        """
        if isinstance(input_data, str):
            input_list = [input_data]
        elif isinstance(input_data, list):
            input_list = [str(i) for i in input_data if i]
        else:
            input_list = [str(i) for i in input_data if i is not None]

        input_list = [i for i in input_list if i]
        if len(input_list) == 0:
            return []

        result = await self.client.embeddings.create(
            input=input_list, model=self.config.embedding_model
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for multiple inputs.

        Args:
            input_data_list: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]
