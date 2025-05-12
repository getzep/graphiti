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

from graphiti_core.embedder.gemini import (
    DEFAULT_EMBEDDING_MODEL,
    GeminiEmbedder,
    GeminiEmbedderConfig,
)
from tests.embedder.embedder_fixtures import create_embedding_values


def create_gemini_embedding(multiplier: float = 0.1) -> MagicMock:
    """Create a mock Gemini embedding with specified value multiplier."""
    mock_embedding = MagicMock()
    mock_embedding.values = create_embedding_values(multiplier)
    return mock_embedding


@pytest.fixture
def mock_gemini_response() -> MagicMock:
    """Create a mock Gemini embeddings response."""
    mock_result = MagicMock()
    mock_result.embeddings = [create_gemini_embedding()]
    return mock_result


@pytest.fixture
def mock_gemini_batch_response() -> MagicMock:
    """Create a mock Gemini batch embeddings response."""
    mock_result = MagicMock()
    mock_result.embeddings = [
        create_gemini_embedding(0.1),
        create_gemini_embedding(0.2),
        create_gemini_embedding(0.3),
    ]
    return mock_result


@pytest.fixture
def mock_gemini_client() -> Generator[Any, Any, None]:
    """Create a mocked Gemini client."""
    with patch('google.genai.Client') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.aio = MagicMock()
        mock_instance.aio.models = MagicMock()
        mock_instance.aio.models.embed_content = AsyncMock()
        yield mock_instance


@pytest.fixture
def gemini_embedder(mock_gemini_client: Any) -> GeminiEmbedder:
    """Create a GeminiEmbedder with a mocked client."""
    config = GeminiEmbedderConfig(api_key='test_api_key')
    client = GeminiEmbedder(config=config)
    client.client = mock_gemini_client
    return client


@pytest.mark.asyncio
async def test_create_calls_api_correctly(
    gemini_embedder: GeminiEmbedder, mock_gemini_client: Any, mock_gemini_response: MagicMock
) -> None:
    """Test that create method correctly calls the API and processes the response."""
    # Setup
    mock_gemini_client.aio.models.embed_content.return_value = mock_gemini_response

    # Call method
    result = await gemini_embedder.create('Test input')

    # Verify API is called with correct parameters
    mock_gemini_client.aio.models.embed_content.assert_called_once()
    _, kwargs = mock_gemini_client.aio.models.embed_content.call_args
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['contents'] == ['Test input']

    # Verify result is processed correctly
    assert result == mock_gemini_response.embeddings[0].values


@pytest.mark.asyncio
async def test_create_batch_processes_multiple_inputs(
    gemini_embedder: GeminiEmbedder, mock_gemini_client: Any, mock_gemini_batch_response: MagicMock
) -> None:
    """Test that create_batch method correctly processes multiple inputs."""
    # Setup
    mock_gemini_client.aio.models.embed_content.return_value = mock_gemini_batch_response
    input_batch = ['Input 1', 'Input 2', 'Input 3']

    # Call method
    result = await gemini_embedder.create_batch(input_batch)

    # Verify API is called with correct parameters
    mock_gemini_client.aio.models.embed_content.assert_called_once()
    _, kwargs = mock_gemini_client.aio.models.embed_content.call_args
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['contents'] == input_batch

    # Verify all results are processed correctly
    assert len(result) == 3
    assert result == [
        mock_gemini_batch_response.embeddings[0].values,
        mock_gemini_batch_response.embeddings[1].values,
        mock_gemini_batch_response.embeddings[2].values,
    ]


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
