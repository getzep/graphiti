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

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from tests.embedder.embedder_fixtures import create_embedding_values


def create_azure_embedding(multiplier: float = 0.1, index: int = 0) -> MagicMock:
    """Create a mock Azure OpenAI embedding with a value multiplier and index."""
    mock_embedding = MagicMock()
    mock_embedding.embedding = create_embedding_values(multiplier)
    mock_embedding.index = index
    return mock_embedding


@pytest.fixture
def mock_azure_client() -> Generator[Any, Any, None]:
    """Create a mocked Azure OpenAI client."""
    with patch('openai.AsyncAzureOpenAI') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.embeddings = MagicMock()
        mock_instance.embeddings.create = AsyncMock()
        yield mock_instance


@pytest.fixture
def azure_embedder(mock_azure_client: Any) -> AzureOpenAIEmbedderClient:
    """Create an Azure OpenAI embedder with a mocked client."""
    return AzureOpenAIEmbedderClient(azure_client=mock_azure_client, model='text-embedding-3-small')


@pytest.mark.asyncio
async def test_create_returns_the_first_embedding(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """A single-input create returns that input's embedding."""
    mock_result = MagicMock()
    mock_result.data = [create_azure_embedding(0.1, index=0)]
    mock_azure_client.embeddings.create.return_value = mock_result

    result = await azure_embedder.create('Input text')

    assert result == create_embedding_values(0.1)


@pytest.mark.asyncio
async def test_create_batch_processes_multiple_inputs(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """A batch returns one embedding per input, in input order."""
    mock_result = MagicMock()
    mock_result.data = [
        create_azure_embedding(0.1, index=0),
        create_azure_embedding(0.2, index=1),
        create_azure_embedding(0.3, index=2),
    ]
    mock_azure_client.embeddings.create.return_value = mock_result

    result = await azure_embedder.create_batch(['Input 1', 'Input 2', 'Input 3'])

    assert len(result) == 3
    assert result == [
        create_embedding_values(0.1),
        create_embedding_values(0.2),
        create_embedding_values(0.3),
    ]


@pytest.mark.asyncio
async def test_create_batch_pairs_embeddings_by_index(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """Embeddings must be returned in INPUT order, keyed by each item's index.

    Reading `data` positionally silently attaches each embedding to the wrong
    input, which is invisible downstream: every vector looks well-formed and
    the wrong node is retrieved for the wrong text.
    """
    mock_result = MagicMock()
    mock_result.data = [
        create_azure_embedding(0.3, index=2),
        create_azure_embedding(0.1, index=0),
        create_azure_embedding(0.2, index=1),
    ]
    mock_azure_client.embeddings.create.return_value = mock_result

    result = await azure_embedder.create_batch(['Input 1', 'Input 2', 'Input 3'])

    assert result == [
        create_embedding_values(0.1),  # Input 1 -> index 0
        create_embedding_values(0.2),  # Input 2 -> index 1
        create_embedding_values(0.3),  # Input 3 -> index 2
    ]


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
