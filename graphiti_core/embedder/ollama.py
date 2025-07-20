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
from typing import Any

import httpx
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'nomic-embed-text'
DEFAULT_BASE_URL = 'http://localhost:11434'


class OllamaEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    base_url: str = Field(default=DEFAULT_BASE_URL)


class OllamaEmbedder(EmbedderClient):
    """
    Ollama Embedder Client
    
    Uses Ollama's native API endpoint for embeddings.
    """

    def __init__(self, config: OllamaEmbedderConfig | None = None):
        if config is None:
            config = OllamaEmbedderConfig()
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.embed_url = f"{self.base_url}/api/embed"

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data using Ollama's embedding model.

        Args:
            input_data: The input data to create embeddings for. Can be a string, list of strings,
                       or an iterable of integers or iterables of integers.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Convert input to string if needed
        if isinstance(input_data, str):
            text_input = input_data
        elif isinstance(input_data, list) and len(input_data) > 0:
            if isinstance(input_data[0], str):
                # For list of strings, take the first one for single embedding
                text_input = input_data[0]
            else:
                # Convert other types to string
                text_input = str(input_data[0])
        else:
            text_input = str(input_data)

        payload = {
            "model": self.config.embedding_model,
            "input": text_input
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.embed_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise Exception(f"Ollama API error {response.status_code}: {error_text}")

                result = response.json()

                if "embeddings" not in result:
                    raise Exception(f"No embeddings in response: {result}")

                embeddings = result["embeddings"]
                if not embeddings or len(embeddings) == 0:
                    raise Exception("Empty embeddings returned")

                # Return the first embedding, truncated to the configured dimension
                embedding = embeddings[0]
                return embedding[: self.config.embedding_dim]

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error creating Ollama embedding: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Ollama API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"Error creating Ollama embedding: {e}")
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create batch embeddings using Ollama's embedding model.
        
        Note: Ollama doesn't support batch embeddings natively, so we process them sequentially.
        """
        embeddings = []
        
        for text in input_data_list:
            try:
                embedding = await self.create(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error creating embedding for text '{text[:50]}...': {e}")
                raise
        
        return embeddings
