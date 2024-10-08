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

from typing import Iterable, List

import voyageai  # type: ignore
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'voyage-3'


class VoyageAIEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None


class VoyageAIEmbedder(EmbedderClient):
    """
    VoyageAI Embedder Client
    """

    def __init__(self, config: VoyageAIEmbedderConfig | None = None):
        if config is None:
            config = VoyageAIEmbedderConfig()
        self.config = config
        self.client = voyageai.AsyncClient(api_key=config.api_key)

    async def create(
            self, input: str | List[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embed(input, model=self.config.embedding_model)
        return result.embeddings[0][: self.config.embedding_dim]
