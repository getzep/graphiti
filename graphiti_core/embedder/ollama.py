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
    from ollama import AsyncClient
else:
    try:
        from ollama import AsyncClient
    except ImportError:
        raise ImportError(
            'ollama is required for OllamaEmbedder. '
            'Install it with: pip install graphiti-core[ollama]'
        ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'bge-m3:567m'
DEFAULT_BATCH_SIZE = 100

class OllamaEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None
    base_url: str | None = None


class OllamaEmbedder(EmbedderClient):
    """
    Ollama Embedder Client
    """
    def __init__(
        self,
        config: OllamaEmbedderConfig | None = None,
        client: AsyncClient | None = None,
        batch_size: int | None = None,
    ):
        if config is None:
            config = OllamaEmbedderConfig()

        self.config = config

        if client is None:
            # AsyncClient doesn't necessarily accept api_key; pass host via headers if needed
            try:
                host = config.base_url.rstrip('/v1') if config.base_url else None
                self.client = AsyncClient(host=host)
            except TypeError as e:
                logger.warning(f"Error creating AsyncClient: {e}")
                self.client = AsyncClient()
        else:
            self.client = client

        if batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
        """Create a single embedding for the input using Ollama.

        Ollama's embed endpoint accepts either a single string or list of strings.
        We normalize to a single-item list and return the first embedding vector.
        """
        # Ollama's embed returns an object with 'embedding' or similar fields
        try:
            # Support call with client.embed for async client
            result = await self.client.embed(model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL, input=input_data)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f'Ollama embed error: {e}')
            raise

        # Extract embedding and coerce to list[float]
        values: list[float] | None = None

        if hasattr(result, 'embeddings'):
            emb = result.embeddings
            if isinstance(emb, list) and len(emb) > 0:
                values = emb[0] if isinstance(emb[0], list | tuple) else emb # type: ignore
        elif isinstance(result, dict):
            if 'embedding' in result and isinstance(result['embedding'], list | tuple):
                values = list(result['embedding'])  # type: ignore
            elif 'embeddings' in result and isinstance(result['embeddings'], list) and len(result['embeddings']) > 0:
                first = result['embeddings'][0]
                if isinstance(first, dict) and 'embedding' in first and isinstance(first['embedding'], list | tuple):
                    values = list(first['embedding'])
                elif isinstance(first, list | tuple):
                    values = list(first)

        # If result itself is a list (some clients return list for single input)
        if values is None and isinstance(result, list | tuple):
            # assume it's already the embedding vector
            values = list(result)  # type: ignore
        if not values:
            raise ValueError('No embeddings returned from Ollama API in create()')

        return values

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(input_data_list), self.batch_size):
            batch = input_data_list[i : i + self.batch_size]
            try:
                result = await self.client.embed(model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL, input=batch)

                # result may be dict with 'embeddings' list or single 'embedding'
                if isinstance(result, dict) and 'embeddings' in result:
                    for emb in result['embeddings']:
                        if isinstance(emb, dict) and 'embedding' in emb and isinstance(emb['embedding'], list | tuple):
                            all_embeddings.append(list(emb['embedding']))
                        elif isinstance(emb, list | tuple):
                            all_embeddings.append(list(emb))
                        else:
                            # unexpected shape
                            raise ValueError('Unexpected embedding shape in batch result')
                else:
                    # Fallback: maybe result itself is a list of vectors
                    if isinstance(result, list):
                        all_embeddings.extend(result)
                    else:
                        # Single embedding returned for whole batch; if so, duplicate per item
                        embedding = None
                        if isinstance(result, dict) and 'embedding' in result:
                            embedding = result['embedding']
                        if embedding is None:
                            raise ValueError('No embeddings returned')
                        for _ in batch:
                            all_embeddings.append(embedding)

            except Exception as e:
                logger.warning(f'Batch embedding failed for batch {i // self.batch_size + 1}, falling back to individual processing: {e}')
                for item in batch:
                    try:
                        single = await self.client.embed(model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL, input=item)
                        emb = None
                        if hasattr(result, 'embeddings'):
                            _emb = result.embeddings
                            if isinstance(_emb, list) and len(_emb) > 0:
                                emb = _emb[0] if isinstance(_emb[0], list | tuple) else _emb # type: ignore
                        elif isinstance(single, dict) and 'embedding' in single:
                            emb = single['embedding']
                        elif isinstance(single, dict) and 'embeddings' in single:
                            emb = single['embeddings']
                        elif isinstance(single, list | tuple):
                            emb = single[0] if single else None # type: ignore
                        if not emb:
                            raise ValueError('No embeddings returned from Ollama API')
                        all_embeddings.append(emb) # type: ignore
                    except Exception as individual_error:
                        logger.error(f'Failed to embed individual item: {individual_error}')
                        raise individual_error

        return all_embeddings
