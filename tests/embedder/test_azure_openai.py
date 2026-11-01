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

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from tests.embedder.embedder_fixtures import create_embedding_values


def _mock_embedding(multiplier: float = 0.1, dimension: int = 1536) -> MagicMock:
    mock_embedding = MagicMock()
    mock_embedding.embedding = create_embedding_values(multiplier, dimension)
    return mock_embedding


def _mock_azure_client(data: list[Any]) -> Any:
    """Build a mocked Azure client whose embeddings.create returns ``data``."""
    response = MagicMock()
    response.data = data
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_create_truncates_to_embedding_dim() -> None:
    """When ``embedding_dim`` is set, ``create`` truncates the embedding to it,
    matching OpenAIEmbedder / VoyageAIEmbedder behavior."""
    client = _mock_azure_client([_mock_embedding(dimension=1536)])
    embedder = AzureOpenAIEmbedderClient(azure_client=client, embedding_dim=384)

    result = await embedder.create('hello world')

    assert len(result) == 384


@pytest.mark.asyncio
async def test_create_without_embedding_dim_returns_full_output() -> None:
    """Default (``embedding_dim=None``) preserves the model's full output dimension."""
    client = _mock_azure_client([_mock_embedding(dimension=1536)])
    embedder = AzureOpenAIEmbedderClient(azure_client=client)

    result = await embedder.create('hello world')

    assert len(result) == 1536


@pytest.mark.asyncio
async def test_create_batch_truncates_each_to_embedding_dim() -> None:
    """``create_batch`` truncates every embedding to ``embedding_dim`` when set."""
    client = _mock_azure_client(
        [_mock_embedding(0.1, 1536), _mock_embedding(0.2, 1536), _mock_embedding(0.3, 1536)]
    )
    embedder = AzureOpenAIEmbedderClient(azure_client=client, embedding_dim=256)

    result = await embedder.create_batch(['a', 'b', 'c'])

    assert len(result) == 3
    assert all(len(embedding) == 256 for embedding in result)
