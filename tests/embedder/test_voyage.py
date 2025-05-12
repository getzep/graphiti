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

from graphiti_core.embedder.voyage import (
    DEFAULT_EMBEDDING_MODEL,
    VoyageAIEmbedder,
    VoyageAIEmbedderConfig,
)
from tests.embedder.embedder_fixtures import create_embedding_values


@pytest.fixture
def mock_voyageai_response() -> MagicMock:
    """Create a mock VoyageAI embeddings response."""
    mock_result = MagicMock()
    mock_result.embeddings = [create_embedding_values()]
    return mock_result


@pytest.fixture
def mock_voyageai_batch_response() -> MagicMock:
    """Create a mock VoyageAI batch embeddings response."""
    mock_result = MagicMock()
    mock_result.embeddings = [
        create_embedding_values(0.1),
        create_embedding_values(0.2),
        create_embedding_values(0.3),
    ]
    return mock_result


@pytest.fixture
def mock_voyageai_client() -> Generator[Any, Any, None]:
    """Create a mocked VoyageAI client."""
    with patch('voyageai.AsyncClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.embed = AsyncMock()
        yield mock_instance


@pytest.fixture
def voyageai_embedder(mock_voyageai_client: Any) -> VoyageAIEmbedder:
    """Create a VoyageAIEmbedder with a mocked client."""
    config = VoyageAIEmbedderConfig(api_key='test_api_key')
    client = VoyageAIEmbedder(config=config)
    client.client = mock_voyageai_client
    return client


@pytest.mark.asyncio
async def test_create_calls_api_correctly(
    voyageai_embedder: VoyageAIEmbedder,
    mock_voyageai_client: Any,
    mock_voyageai_response: MagicMock,
) -> None:
    """Test that create method correctly calls the API and processes the response."""
    # Setup
    mock_voyageai_client.embed.return_value = mock_voyageai_response

    # Call method
    result = await voyageai_embedder.create('Test input')

    # Verify API is called with correct parameters
    mock_voyageai_client.embed.assert_called_once()
    args, kwargs = mock_voyageai_client.embed.call_args
    assert args[0] == ['Test input']
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL

    # Verify result is processed correctly
    expected_result = [
        float(x)
        for x in mock_voyageai_response.embeddings[0][: voyageai_embedder.config.embedding_dim]
    ]
    assert result == expected_result


@pytest.mark.asyncio
async def test_create_batch_processes_multiple_inputs(
    voyageai_embedder: VoyageAIEmbedder,
    mock_voyageai_client: Any,
    mock_voyageai_batch_response: MagicMock,
) -> None:
    """Test that create_batch method correctly processes multiple inputs."""
    # Setup
    mock_voyageai_client.embed.return_value = mock_voyageai_batch_response
    input_batch = ['Input 1', 'Input 2', 'Input 3']

    # Call method
    result = await voyageai_embedder.create_batch(input_batch)

    # Verify API is called with correct parameters
    mock_voyageai_client.embed.assert_called_once()
    args, kwargs = mock_voyageai_client.embed.call_args
    assert args[0] == input_batch
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL

    # Verify all results are processed correctly
    assert len(result) == 3
    expected_results = [
        [
            float(x)
            for x in mock_voyageai_batch_response.embeddings[0][
                : voyageai_embedder.config.embedding_dim
            ]
        ],
        [
            float(x)
            for x in mock_voyageai_batch_response.embeddings[1][
                : voyageai_embedder.config.embedding_dim
            ]
        ],
        [
            float(x)
            for x in mock_voyageai_batch_response.embeddings[2][
                : voyageai_embedder.config.embedding_dim
            ]
        ],
    ]
    assert result == expected_results


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
