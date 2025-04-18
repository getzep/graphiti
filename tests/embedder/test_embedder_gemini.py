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

# Running tests: poetry run pytest -xvs tests/embedder/test_embedder_gemini.py
# Running tests with coverage: poetry run pytest -xvs tests/embedder/test_gemini.py --cov=graphiti_core.embedder.gemini --cov-report=term-missing

import pytest
from pytest_mock import MockerFixture

from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.embedder.gemini import (
    DEFAULT_EMBEDDING_MODEL,
    GeminiEmbedder,
    GeminiEmbedderConfig,
)


@pytest.mark.asyncio
async def test_gemini_embedder_initialization(mocker: MockerFixture) -> None:
    """Test that GeminiEmbedder initializes correctly with API key"""
    mock_genai = mocker.patch('graphiti_core.embedder.gemini.genai')
    mock_client = mocker.MagicMock()
    mock_genai.Client.return_value = mock_client

    embedder = GeminiEmbedder(config=GeminiEmbedderConfig(api_key='test_key'))

    # Verify client initialization
    mock_genai.Client.assert_called_once_with(api_key='test_key')
    assert embedder.config.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert embedder.config.embedding_dim == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_gemini_embedder_custom_config(mocker: MockerFixture) -> None:
    """Test GeminiEmbedder with custom configuration"""
    mock_genai = mocker.patch('graphiti_core.embedder.gemini.genai')
    mock_client = mocker.MagicMock()
    mock_genai.Client.return_value = mock_client

    custom_config = GeminiEmbedderConfig(
        api_key='test_key',
        embedding_model='custom-model',
        embedding_dim=512,
    )
    embedder = GeminiEmbedder(config=custom_config)

    # Verify client initialization with custom config
    assert embedder.config.embedding_model == 'custom-model'
    assert embedder.config.embedding_dim == 512


@pytest.mark.asyncio
async def test_gemini_embedder_create_with_string(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test creating embeddings with a string input"""
    mock_genai = mocker.patch('graphiti_core.embedder.gemini.genai')
    mock_client = mocker.MagicMock()
    mock_genai.Client.return_value = mock_client

    # Set up mock response
    mock_result = mocker.MagicMock()
    mock_embedding = mocker.MagicMock()
    mock_embedding.values = mock_embedding_values
    mock_result.embeddings = [mock_embedding]

    # Set up async mock for embed_content
    mock_aio = mocker.MagicMock()
    mock_models = mocker.MagicMock()
    mock_embed_content = mocker.AsyncMock(return_value=mock_result)
    mock_models.embed_content = mock_embed_content
    mock_aio.models = mock_models
    mock_client.aio = mock_aio

    # Create embedder
    embedder = GeminiEmbedder(config=GeminiEmbedderConfig(api_key='test_key'))

    # Call create method
    result = await embedder.create('test input')

    # Verify API call
    mock_embed_content.assert_called_once()
    _, kwargs = mock_embed_content.call_args
    assert kwargs['model'] == DEFAULT_EMBEDDING_MODEL
    assert kwargs['contents'] == ['test input']
    assert 'config' in kwargs

    # Verify result
    assert len(result) == EMBEDDING_DIM
    assert all(x == 0.1 for x in result)


@pytest.mark.asyncio
async def test_gemini_embedder_create_with_list(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test creating embeddings with a list input

    Note: The input list is intentionally wrapped in another list as the Gemini API's
    embed_content method expects a list of contents, even for single inputs.
    """
    mock_genai = mocker.patch('graphiti_core.embedder.gemini.genai')
    mock_client = mocker.MagicMock()
    mock_genai.Client.return_value = mock_client

    # Set up mock response
    mock_result = mocker.MagicMock()
    mock_embedding = mocker.MagicMock()
    mock_embedding.values = mock_embedding_values
    mock_result.embeddings = [mock_embedding]

    # Set up async mock for embed_content
    mock_aio = mocker.MagicMock()
    mock_models = mocker.MagicMock()
    mock_embed_content = mocker.AsyncMock(return_value=mock_result)
    mock_models.embed_content = mock_embed_content
    mock_aio.models = mock_models
    mock_client.aio = mock_aio

    # Create embedder with custom model
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key='test_key',
            embedding_model='custom-model',
        )
    )

    # Call create method with list
    test_input = ['input1', 'input2']
    result = await embedder.create(test_input)

    # Verify API call with custom model
    mock_embed_content.assert_called_once()
    _, kwargs = mock_embed_content.call_args
    assert kwargs['model'] == 'custom-model'
    assert kwargs['contents'] == [test_input]  # The list is wrapped in another list

    # Verify result
    assert len(result) == EMBEDDING_DIM
