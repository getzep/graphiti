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

# Running tests: poetry run pytest -xvs tests/embedder/test_embedder_voyage.py
# Running tests with coverage: poetry run pytest -xvs tests/embedder/test_voyage.py --cov=graphiti_core.embedder.voyage --cov-report=term-missing

from typing import NamedTuple

import pytest
from pytest_mock import MockerFixture

from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.embedder.voyage import (
    DEFAULT_EMBEDDING_MODEL,
    VoyageAIEmbedder,
    VoyageAIEmbedderConfig,
)


class EmbeddingResponse(NamedTuple):
    embeddings: list[list[float]]


@pytest.mark.asyncio
async def test_voyageai_embedder_initialization(mocker: MockerFixture) -> None:
    """Test that VoyageAIEmbedder initializes correctly"""
    mock_voyage = mocker.patch('graphiti_core.embedder.voyage.voyageai')
    mock_client = mocker.AsyncMock()
    mock_voyage.AsyncClient.return_value = mock_client

    # Create embedder
    embedder = VoyageAIEmbedder(config=VoyageAIEmbedderConfig(api_key='test_key'))

    # Verify client initialization
    mock_voyage.AsyncClient.assert_called_once_with(api_key='test_key')
    assert embedder.config.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert embedder.config.embedding_dim == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_voyageai_embedder_custom_config(mocker: MockerFixture) -> None:
    """Test VoyageAIEmbedder with custom configuration"""
    mock_voyage = mocker.patch('graphiti_core.embedder.voyage.voyageai')
    mock_client = mocker.AsyncMock()
    mock_voyage.AsyncClient.return_value = mock_client

    # Create custom config
    custom_config = VoyageAIEmbedderConfig(
        api_key='test_key',
        embedding_model='custom-model',
        embedding_dim=512,
    )

    # Create embedder with custom config
    embedder = VoyageAIEmbedder(config=custom_config)

    # Verify config is used
    assert embedder.config.embedding_model == 'custom-model'
    assert embedder.config.embedding_dim == 512


@pytest.mark.asyncio
async def test_voyageai_embedder_create_with_string(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test creating embeddings with a string input"""
    mock_voyage = mocker.patch('graphiti_core.embedder.voyage.voyageai')
    mock_client = mocker.AsyncMock()
    mock_voyage.AsyncClient.return_value = mock_client

    # Create mock response
    embeddings: list[list[float]] = [mock_embedding_values]
    mock_response = EmbeddingResponse(embeddings=embeddings)
    mock_client.embed.return_value = mock_response

    # Create embedder
    embedder = VoyageAIEmbedder(config=VoyageAIEmbedderConfig(api_key='test_key'))

    # Call create method
    result = await embedder.create('test input')

    # Verify API call
    mock_client.embed.assert_called_once_with(['test input'], model=DEFAULT_EMBEDDING_MODEL)

    # Verify result
    assert len(result) == EMBEDDING_DIM
    assert result == mock_embedding_values


@pytest.mark.asyncio
async def test_voyageai_embedder_create_with_list(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test creating embeddings with a list input"""
    mock_voyage = mocker.patch('graphiti_core.embedder.voyage.voyageai')
    mock_client = mocker.AsyncMock()
    mock_voyage.AsyncClient.return_value = mock_client

    # Create mock response
    embeddings: list[list[float]] = [mock_embedding_values]
    mock_response = EmbeddingResponse(embeddings=embeddings)
    mock_client.embed.return_value = mock_response

    # Create embedder
    embedder = VoyageAIEmbedder(
        config=VoyageAIEmbedderConfig(
            api_key='test_key',
            embedding_model='custom-model',
        )
    )

    # Call create method with list
    result = await embedder.create(['input1', 'input2'])

    # Verify API call with correct inputs
    mock_client.embed.assert_called_once_with(['input1', 'input2'], model='custom-model')

    # Verify result
    assert len(result) == EMBEDDING_DIM
    assert result == mock_embedding_values


@pytest.mark.asyncio
async def test_voyageai_embedder_with_empty_input(mocker: MockerFixture) -> None:
    """Test behavior with empty input"""
    mock_voyage = mocker.patch('graphiti_core.embedder.voyage.voyageai')
    mock_client = mocker.AsyncMock()
    mock_voyage.AsyncClient.return_value = mock_client

    # Create embedder
    embedder = VoyageAIEmbedder(config=VoyageAIEmbedderConfig(api_key='test_key'))

    # Call create method with empty list
    result = await embedder.create([])

    # Should return empty list without API call
    assert result == []
    mock_client.embed.assert_not_called()


@pytest.mark.asyncio
async def test_voyageai_embedder_with_mixed_types(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test handling of mixed input types

    Note: All non-string inputs are converted to strings (e.g., integers like 123 become '123').
    None values and empty strings are filtered out.
    """
    mock_voyage = mocker.patch('graphiti_core.embedder.voyage.voyageai')
    mock_client = mocker.AsyncMock()
    mock_voyage.AsyncClient.return_value = mock_client

    # Create mock response
    embeddings: list[list[float]] = [mock_embedding_values]
    mock_response = EmbeddingResponse(embeddings=embeddings)
    mock_client.embed.return_value = mock_response

    # Create embedder
    embedder = VoyageAIEmbedder(config=VoyageAIEmbedderConfig(api_key='test_key'))

    # Call create method with mixed types
    result = await embedder.create([123, 'text', None, ''])  # type: ignore[arg-type]

    # Should filter out None and empty strings
    mock_client.embed.assert_called_once_with(['123', 'text'], model=DEFAULT_EMBEDDING_MODEL)

    # Verify result
    assert len(result) == EMBEDDING_DIM
    assert result == mock_embedding_values
