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
import asyncio
from typing import List, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'

# Defaults
DEFAULT_MAX_INPUT_CHARS = 3000
DEFAULT_MAX_INPUT_TOKENS = 2048
DEFAULT_BATCH_WINDOW = 16
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 1.0
DEFAULT_AGGREGATION = 'average'  # 'average' or 'first'

logger = logging.getLogger(__name__)

# Optional tokeniser support (tiktoken). If unavailable, we fall back to
# character-based chunking.
try:
    import tiktoken  # type: ignore

    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False


class EmbeddingTooLargeError(Exception):
    pass


class OpenAIEmbedderConfig(EmbedderConfig):
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None
    # Tunables
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    batch_window: int = DEFAULT_BATCH_WINDOW
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE
    aggregation: str = DEFAULT_AGGREGATION


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

    def _num_tokens(self, text: str) -> Optional[int]:
        if not TIKTOKEN_AVAILABLE:
            return None
        try:
            enc = tiktoken.encoding_for_model(str(self.config.embedding_model))
        except Exception:
            try:
                enc = tiktoken.get_encoding('cl100k_base')
            except Exception:
                return None
        return len(enc.encode(text))

    def _token_chunk_text(self, text: str, max_tokens: int) -> List[str]:
        # Requires tiktoken
        if not TIKTOKEN_AVAILABLE or max_tokens <= 0:
            return [text]
        try:
            try:
                enc = tiktoken.encoding_for_model(str(self.config.embedding_model))
            except Exception:
                enc = tiktoken.get_encoding('cl100k_base')
            tokens = enc.encode(text)
            chunks: List[str] = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i : i + max_tokens]
                chunks.append(enc.decode(chunk_tokens))
            return chunks
        except Exception:
            return [text]

    def _chunk_text(self, text: str, max_chars: int, max_tokens: Optional[int] = None) -> List[str]:
        # Prefer token-aware chunking when possible
        if max_tokens and TIKTOKEN_AVAILABLE:
            token_chunks = self._token_chunk_text(text, max_tokens)
            if len(token_chunks) > 1:
                return token_chunks
        # fallback to char-based chunking
        if not isinstance(text, str) or max_chars <= 0:
            return [str(text)]
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + max_chars)
            if end < L:
                idx = text.rfind('\n', start, end)
                if idx <= start:
                    idx = text.rfind(' ', start, end)
                if idx > start:
                    end = idx
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == start:
                end = start + max_chars
            start = end
        return chunks

    async def _embed_batch_with_retry(self, batch: List[str]):
        last_exc: Optional[Exception] = None
        max_retries = int(getattr(self.config, 'max_retries', DEFAULT_MAX_RETRIES))
        base = float(getattr(self.config, 'retry_backoff_base', DEFAULT_RETRY_BACKOFF_BASE))
        for attempt in range(max_retries):
            try:
                resp = await self.client.embeddings.create(input=batch, model=self.config.embedding_model)
                return resp
            except Exception as e:  # noqa: BLE001 - wrap and retry conservatively
                last_exc = e
                err = str(e).lower()
                # If the provider complains about token limits, propagate a special
                # signal so caller can re-chunk more aggressively.
                if 'exceed' in err or 'token' in err or 'length' in err:
                    raise EmbeddingTooLargeError(e)
                if attempt < max_retries - 1:
                    sleep_time = base * (2 ** attempt)
                    logger.warning('embedder: attempt %d failed, retrying in %.1fs: %s', attempt + 1, sleep_time, e)
                    await asyncio.sleep(sleep_time)
                    continue
                break
        # no retry left
        raise last_exc

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

    def _average_embeddings(self, vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        avg = [0.0] * dim
        count = 0
        for v in vectors:
            if not v or len(v) != dim:
                continue
            count += 1
            for i, val in enumerate(v):
                avg[i] += val
        if count == 0:
            return []
        for i in range(dim):
            avg[i] = avg[i] / float(count)
        return avg

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> List[float]:
        # Handle simple string inputs with chunking + aggregation
        if isinstance(input_data, str):
            max_chars = int(getattr(self.config, 'max_input_chars', DEFAULT_MAX_INPUT_CHARS))
            max_tokens = int(getattr(self.config, 'max_input_tokens', DEFAULT_MAX_INPUT_TOKENS)) if getattr(self.config, 'max_input_tokens', None) else None
            batch_window = int(getattr(self.config, 'batch_window', DEFAULT_BATCH_WINDOW))
            agg = getattr(self.config, 'aggregation', DEFAULT_AGGREGATION)

            # Try progressively smaller chunk sizes if provider rejects the request
            current_max_tokens = max_tokens
            current_max_chars = max_chars
            min_chars = 256
            while True:
                try:
                    chunks = self._chunk_text(input_data, current_max_chars, current_max_tokens)
                    vectors: List[List[float]] = []
                    for i in range(0, len(chunks), batch_window):
                        batch = chunks[i : i + batch_window]
                        resp = await self._embed_batch_with_retry(batch)
                        for d in resp.data:
                            vectors.append(d.embedding[: self.config.embedding_dim])

                    if agg == 'first' and vectors:
                        return vectors[0]
                    # default: average
                    return self._average_embeddings(vectors)

                except EmbeddingTooLargeError as e:
                    # provider refused due to size; reduce chunking targets and retry
                    logger.warning('embedder: provider rejected large input, reducing chunk size and retrying')
                    if current_max_tokens:
                        current_max_tokens = max(128, current_max_tokens // 2)
                    if current_max_chars:
                        current_max_chars = max(min_chars, current_max_chars // 2)
                    # if we've reduced to minimum and still failing, re-raise
                    if (current_max_tokens is not None and current_max_tokens <= 128) and current_max_chars <= min_chars:
                        raise
                    continue

                except Exception as e:
                    logger.warning('OpenAIEmbedder.create failed; falling back to direct call: %s', e)
                    result = await self.client.embeddings.create(
                        input=input_data, model=self.config.embedding_model
                    )
                    return result.data[0].embedding[: self.config.embedding_dim]

        # For batch or other iterables, delegate to create_batch for safer handling
        return await self.create_batch(list(input_data))

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for item in input_data_list:
            try:
                vec = await self.create(item)
                out.append(vec)
            except Exception as e:
                logger.warning('create_batch: embedding failed for item; returning empty vector: %s', e)
                out.append([])
        return out
