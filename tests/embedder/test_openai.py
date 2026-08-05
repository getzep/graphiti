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
    # dimensions is not forwarded unless explicitly configured
    assert 'dimensions' not in kwargs

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


@pytest.mark.asyncio
async def test_create_forwards_dimensions_and_skips_truncation(
    mock_openai_client: Any, mock_openai_response: MagicMock
) -> None:
    """When ``dimensions`` is configured it is forwarded to the API and the natively
    sized vector is returned without the lossy truncation step."""
    # Setup: request a native dimensionality larger than the default embedding_dim
    native_vector = mock_openai_response.data[0].embedding
    config = OpenAIEmbedderConfig(api_key='test_api_key', dimensions=len(native_vector))
    embedder = OpenAIEmbedder(config=config)
    embedder.client = mock_openai_client
    mock_openai_client.embeddings.create.return_value = mock_openai_response

    # Call method
    result = await embedder.create('Test input')

    # Verify dimensions is forwarded to the API
    _, kwargs = mock_openai_client.embeddings.create.call_args
    assert kwargs['dimensions'] == len(native_vector)

    # Verify the full native vector is returned, not truncated to embedding_dim
    assert result == native_vector
    assert len(result) > embedder.config.embedding_dim


@pytest.mark.asyncio
async def test_create_batch_forwards_dimensions(
    mock_openai_client: Any, mock_openai_batch_response: MagicMock
) -> None:
    """create_batch forwards ``dimensions`` and returns untruncated native vectors."""
    native_len = len(mock_openai_batch_response.data[0].embedding)
    config = OpenAIEmbedderConfig(api_key='test_api_key', dimensions=native_len)
    embedder = OpenAIEmbedder(config=config)
    embedder.client = mock_openai_client
    mock_openai_client.embeddings.create.return_value = mock_openai_batch_response

    result = await embedder.create_batch(['Input 1', 'Input 2', 'Input 3'])

    _, kwargs = mock_openai_client.embeddings.create.call_args
    assert kwargs['dimensions'] == native_len
    assert result == [data.embedding for data in mock_openai_batch_response.data]


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
