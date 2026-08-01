"""Unit tests for DrevoEmbedder (thin preset over OpenAIEmbedder)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.embedder.drevo import (
    DEFAULT_DREVO_EMBEDDER_API_KEY,
    DEFAULT_DREVO_EMBEDDER_BASE_URL,
    DrevoEmbedder,
)
from graphiti_core.embedder.openai import OpenAIEmbedderConfig


def _client_with(embeddings) -> SimpleNamespace:
    """Stub AsyncOpenAI whose embeddings.create returns the given vectors."""
    create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=e) for e in embeddings])
    )
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


class TestDrevoEmbedderDefaults:
    def test_fills_local_endpoint_and_placeholder_key(self):
        embedder = DrevoEmbedder()
        assert str(embedder.client.base_url).startswith('http://localhost:8080')
        assert embedder.client.api_key == DEFAULT_DREVO_EMBEDDER_API_KEY

    def test_preserves_explicit_config(self):
        embedder = DrevoEmbedder(
            config=OpenAIEmbedderConfig(base_url='http://remote-drevo:9/v1', api_key='k')
        )
        assert str(embedder.client.base_url).startswith('http://remote-drevo:9')
        assert embedder.client.api_key == 'k'

    def test_default_base_url_constant(self):
        assert DEFAULT_DREVO_EMBEDDER_BASE_URL == 'http://localhost:8080/v1'


@pytest.mark.asyncio
class TestDrevoEmbedderCreate:
    async def test_create_returns_truncated_embedding(self):
        # embedding_dim defaults to 1024; a longer vector is truncated.
        long_vector = [float(i) for i in range(1030)]
        embedder = DrevoEmbedder(client=_client_with([long_vector]))

        result = await embedder.create('hello')

        assert len(result) == 1024
        assert result == long_vector[:1024]

    async def test_create_batch(self):
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        embedder = DrevoEmbedder(client=_client_with(vectors))

        result = await embedder.create_batch(['a', 'b'])

        assert result == vectors
        # The configured model is forwarded to drevo's OpenAI-compatible endpoint.
        call = embedder.client.embeddings.create.await_args
        assert call.kwargs['model'] == embedder.config.embedding_model
