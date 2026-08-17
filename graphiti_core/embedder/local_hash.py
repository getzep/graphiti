"""Dependency-free local embeddings for development and private deployments.

The embedder uses signed feature hashing over word tokens and Unicode character
n-grams.  It is deterministic, works reasonably well for both Chinese and
Latin text, and avoids sending source documents to a second external model.
It is intentionally a lightweight fallback rather than a replacement for a
production embedding model.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable

from pydantic import Field, model_validator

from .client import EMBEDDING_DIM, EmbedderClient, EmbedderConfig


class LocalHashEmbedderConfig(EmbedderConfig):
    """Configuration for :class:`LocalHashEmbedder`."""

    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True, ge=32)
    min_ngram: int = Field(default=2, ge=1)
    max_ngram: int = Field(default=4, ge=1)

    @model_validator(mode='after')
    def validate_ngram_range(self) -> LocalHashEmbedderConfig:
        if self.max_ngram < self.min_ngram:
            raise ValueError('max_ngram must be greater than or equal to min_ngram')
        return self


class LocalHashEmbedder(EmbedderClient):
    """Create stable local vectors with no model download or API dependency."""

    def __init__(self, config: LocalHashEmbedderConfig | None = None):
        self.config = config or LocalHashEmbedderConfig()

    @staticmethod
    def _as_text(input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> str:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return '\n'.join(str(item) for item in input_data)
        return ' '.join(str(item) for item in input_data)

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        normalized = unicodedata.normalize('NFKC', text).casefold()
        compact = re.sub(r'\s+', '', normalized)

        for token in re.findall(r'\w+', normalized, flags=re.UNICODE):
            if token:
                yield f'w:{token}', 2.0

        for size in range(self.config.min_ngram, self.config.max_ngram + 1):
            if len(compact) < size:
                continue
            weight = 1.0 / math.sqrt(size)
            for index in range(len(compact) - size + 1):
                yield f'c{size}:{compact[index : index + size]}', weight

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.config.embedding_dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode('utf-8'), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder='big', signed=False)
            bucket = value % self.config.embedding_dim
            # Use a bit outside the bucket mask for the sign. With power-of-two
            # dimensions, using the low bit would otherwise give every feature in
            # a bucket the same sign and turn collisions into a systematic bias.
            sign = -1.0 if (value >> 32) & 1 else 1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        return self._embed(self._as_text(input_data))

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input_data_list]
