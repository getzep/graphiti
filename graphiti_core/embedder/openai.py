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
import math
import logging

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'

# Conservative default to avoid exceeding provider token limits. This is a
# character-based heuristic; models count tokens, but char-based truncation
# reduces risk of excessively long inputs when tokenizers are unavailable.
DEFAULT_MAX_INPUT_CHARS = 3000
DEFAULT_BATCH_WINDOW = 16
logger = logging.getLogger(__name__)


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

    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        if not isinstance(text, str) or max_chars <= 0:
            return [str(text)]
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        chunks: list[str] = []
        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + max_chars)
            # try to break on whitespace for nicer chunks
            if end < L:
                idx = text.rfind('\n', start, end)
                if idx <= start:
                    idx = text.rfind(' ', start, end)
                if idx > start:
                    end = idx
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # ensure progress
            if end == start:
                end = start + max_chars
            start = end
        return chunks

    def _average_embeddings(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        avg = [0.0] * dim
        for v in vectors:
            # ignore malformed vectors
            if not v or len(v) != dim:
                continue
            for i, val in enumerate(v):
                avg[i] += val
        n = len(vectors)
        if n == 0:
            return []
        for i in range(dim):
            avg[i] = avg[i] / float(n)
        return avg

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        # If input_data is a simple string, guard against overly long inputs
        if isinstance(input_data, str):
            max_chars = getattr(self.config, "max_input_chars", DEFAULT_MAX_INPUT_CHARS)
            try:
                if len(input_data) <= max_chars:
                    result = await self.client.embeddings.create(
                        input=input_data, model=self.config.embedding_model
                    )
                    return result.data[0].embedding[: self.config.embedding_dim]

                # chunk large texts, embed in batches and average embeddings
                chunks = self._chunk_text(input_data, max_chars)
                vectors: list[list[float]] = []
                window = DEFAULT_BATCH_WINDOW
                for i in range(0, len(chunks), window):
                    batch = chunks[i : i + window]
                    resp = await self.client.embeddings.create(input=batch, model=self.config.embedding_model)
                    for d in resp.data:
                        vectors.append(d.embedding[: self.config.embedding_dim])
                avg = self._average_embeddings(vectors)
                return avg
            except Exception as e:
                logger.warning("OpenAIEmbedder.create failed; falling back to direct call: %s", e)
                # fallback: try direct call which will raise the original error
                result = await self.client.embeddings.create(
                    input=input_data, model=self.config.embedding_model
                )
                return result.data[0].embedding[: self.config.embedding_dim]

        # For batch or other iterables, delegate to create_batch for safer handling
        return await self.create_batch(list(input_data))

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        # Process each item with create() to ensure long inputs are chunked/averaged.
        out: list[list[float]] = []
        for item in input_data_list:
            try:
                vec = await self.create(item)
                out.append(vec)
            except Exception as e:
                logger.warning("create_batch: embedding failed for item; returning empty vector: %s", e)
                out.append([])
        return out
