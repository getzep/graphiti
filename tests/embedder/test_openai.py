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

from graphiti_core.embedder.openai import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbedder,
    OpenAIEmbedderConfig,
)
from tests.embedder.embedder_fixtures import create_embedding_values


def create_openai_embedding(multiplier: float = 0.1) -> MagicMock:
    """Create a mock OpenAI embedding with specified value multiplier."""
    mock_embedding = MagicMock()
    mock_embedding.embedding = create_embedding_values(multiplier)
    return mock_embedding


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Create a mock OpenAI embeddings response."""
    mock_result = MagicMock()
    mock_result.data = [create_openai_embedding()]
    return mock_result


@pytest.fixture
def mock_openai_batch_response() -> MagicMock:
    """Create a mock OpenAI batch embeddings response."""
    mock_result = MagicMock()
    mock_result.data = [
        create_openai_embedding(0.1),
        create_openai_embedding(0.2),
        create_openai_embedding(0.3),
    ]
    return mock_result


@pytest.fixture
def mock_openai_client() -> Generator[Any, Any, None]:
    """Create a mocked OpenAI client."""
    with patch('openai.AsyncOpenAI') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.embeddings = MagicMock()
        mock_instance.embeddings.create = AsyncMock()
        yield mock_instance


@pytest.fixture
def openai_embedder(mock_openai_client: Any) -> OpenAIEmbedder:
    """Create an OpenAIEmbedder with a mocked client."""
    config = OpenAIEmbedderConfig(api_key='test_api_key')
    client = OpenAIEmbedder(config=config)
    client.client = mock_openai_client
    return client


@pytest.mark.asyncio
async def test_create_calls_api_correctly(
    openai_embedder: OpenAIEmbedder, mock_openai_client: Any, mock_openai_response: MagicMock
) -> None:
    """Test that create method correctly calls the API and processes the response."""
    # Setup
    mock_openai_client.embeddings.create.return_value = mock_openai_response

    # Call method
    result = await openai_embedder.create('Test input')

    # Verify API is called with correct parameters
    mock_openai_client.embeddings.create.assert_called_once()
    _, kwargs = mock_openai_client.embeddings.create.call_args
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['input'] == 'Test input'

    # Verify result is processed correctly
    assert result == mock_openai_response.data[0].embedding[: openai_embedder.config.embedding_dim]


@pytest.mark.asyncio
async def test_create_batch_processes_multiple_inputs(
    openai_embedder: OpenAIEmbedder, mock_openai_client: Any, mock_openai_batch_response: MagicMock
) -> None:
    """Test that create_batch method correctly processes multiple inputs."""
    # Setup
    mock_openai_client.embeddings.create.return_value = mock_openai_batch_response
    input_batch = ['Input 1', 'Input 2', 'Input 3']

    # Call method
    result = await openai_embedder.create_batch(input_batch)

    # Verify API is called with correct parameters
    mock_openai_client.embeddings.create.assert_called_once()
    _, kwargs = mock_openai_client.embeddings.create.call_args
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['input'] == input_batch

    # Verify all results are processed correctly
    assert len(result) == 3
    assert result == [
        mock_openai_batch_response.data[0].embedding[: openai_embedder.config.embedding_dim],
        mock_openai_batch_response.data[1].embedding[: openai_embedder.config.embedding_dim],
        mock_openai_batch_response.data[2].embedding[: openai_embedder.config.embedding_dim],
    ]


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
