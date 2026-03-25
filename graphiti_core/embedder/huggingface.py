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

import asyncio
from collections.abc import Iterable
from functools import partial

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        'sentence-transformers is required for HuggingFaceEmbedder. '
        'Install it with: pip install graphiti-core[sentence-transformers]'
    ) from None

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class HuggingFaceEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


class HuggingFaceEmbedder(EmbedderClient):
    """
    HuggingFace Embedder Client using sentence-transformers.

    Runs locally — no API key required. The model is downloaded from HuggingFace
    Hub on first use and cached locally.

    Example usage:
        embedder = HuggingFaceEmbedder()  # uses all-MiniLM-L6-v2 by default
        embedder = HuggingFaceEmbedder(HuggingFaceEmbedderConfig(
            embedding_model='BAAI/bge-m3',
            embedding_dim=1024,
        ))
    """

    def __init__(self, config: HuggingFaceEmbedderConfig | None = None):
        if config is None:
            config = HuggingFaceEmbedderConfig()
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        loop = asyncio.get_event_loop()
        text = input_data if isinstance(input_data, str) else list(input_data)  # type: ignore[arg-type]
        embedding = await loop.run_in_executor(None, partial(self.model.encode, text))
        if hasattr(embedding, '__len__') and hasattr(embedding[0], '__len__'):
            return embedding[0].tolist()[: self.config.embedding_dim]
        return embedding.tolist()[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, partial(self.model.encode, input_data_list)
        )
        return [emb.tolist()[: self.config.embedding_dim] for emb in embeddings]
