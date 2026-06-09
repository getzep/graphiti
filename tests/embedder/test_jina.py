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

from graphiti_core.embedder.jina import (
    DEFAULT_EMBEDDING_MODEL,
    JinaEmbedder,
    JinaEmbedderConfig,
)
from tests.embedder.embedder_fixtures import create_embedding_values


@pytest.fixture
def mock_jina_response() -> MagicMock:
    """Create a mock Jina embeddings response."""
    mock_result = MagicMock()
    mock_data = MagicMock()
    mock_data.embedding = create_embedding_values()
    mock_result.data = [mock_data]
    return mock_result


@pytest.fixture
def mock_jina_batch_response() -> MagicMock:
    """Create a mock Jina batch embeddings response."""
    mock_result = MagicMock()
    mock_data_1 = MagicMock()
    mock_data_1.embedding = create_embedding_values(0.1)
    mock_data_2 = MagicMock()
    mock_data_2.embedding = create_embedding_values(0.2)
    mock_data_3 = MagicMock()
    mock_data_3.embedding = create_embedding_values(0.3)
    mock_result.data = [mock_data_1, mock_data_2, mock_data_3]
    return mock_result


@pytest.fixture
def mock_jina_client() -> Generator[Any, Any, None]:
    """Create a mocked Jina client."""
    with patch('openai.AsyncOpenAI') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.embeddings = MagicMock()
        mock_instance.embeddings.create = AsyncMock()
        yield mock_instance


@pytest.fixture
def jina_embedder(mock_jina_client: Any) -> JinaEmbedder:
    """Create a JinaEmbedder with a mocked client."""
    config = JinaEmbedderConfig(api_key='test_api_key')
    client = JinaEmbedder(config=config)
    client.client = mock_jina_client
    return client


@pytest.mark.asyncio
async def test_create_calls_api_correctly(
    jina_embedder: JinaEmbedder,
    mock_jina_client: Any,
    mock_jina_response: MagicMock,
) -> None:
    """Test that create method correctly calls the API and processes the response."""
    # Setup
    mock_jina_client.embeddings.create.return_value = mock_jina_response

    # Call method
    result = await jina_embedder.create('Test input')

    # Verify API is called with correct parameters
    mock_jina_client.embeddings.create.assert_called_once()
    _, kwargs = mock_jina_client.embeddings.create.call_args
    assert kwargs['input'] == 'Test input'
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['extra_body'] == {'task': 'retrieval.passage'}

    # Verify result is processed correctly
    expected_result = mock_jina_response.data[0].embedding[: jina_embedder.config.embedding_dim]
    assert result == expected_result


@pytest.mark.asyncio
async def test_create_batch_processes_multiple_inputs(
    jina_embedder: JinaEmbedder,
    mock_jina_client: Any,
    mock_jina_batch_response: MagicMock,
) -> None:
    """Test that create_batch method correctly processes multiple inputs."""
    # Setup
    mock_jina_client.embeddings.create.return_value = mock_jina_batch_response
    input_batch = ['Input 1', 'Input 2', 'Input 3']

    # Call method
    result = await jina_embedder.create_batch(input_batch)

    # Verify API is called with correct parameters
    mock_jina_client.embeddings.create.assert_called_once()
    _, kwargs = mock_jina_client.embeddings.create.call_args
    assert kwargs['input'] == input_batch
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['extra_body'] == {'task': 'retrieval.passage'}

    # Verify all results are processed correctly
    assert len(result) == 3
    expected_results = [
        mock_jina_batch_response.data[0].embedding[: jina_embedder.config.embedding_dim],
        mock_jina_batch_response.data[1].embedding[: jina_embedder.config.embedding_dim],
        mock_jina_batch_response.data[2].embedding[: jina_embedder.config.embedding_dim],
    ]
    assert result == expected_results


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
