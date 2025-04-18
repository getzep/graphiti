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

# Running tests: poetry run pytest -xvs tests/embedder/test_openai.py
# Running tests with coverage: poetry run pytest -xvs tests/embedder/test_openai.py --cov=graphiti_core.embedder.openai --cov-report=term-missing

from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.embedder.openai import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbedder,
    OpenAIEmbedderConfig,
)


@pytest.fixture
def mock_openai_client(mocker: MockerFixture, mock_embedding_values: list[float]) -> AsyncMock:
    mock_client = mocker.AsyncMock()
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock()]
    mock_response.data[0].embedding = mock_embedding_values
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


@pytest.mark.asyncio
async def test_openai_embedder_create_with_string(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test creating embeddings with a string input"""
    mock_openai = mocker.patch('graphiti_core.embedder.openai.AsyncOpenAI')
    mock_client = mocker.AsyncMock()
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock()]
    mock_response.data[0].embedding = mock_embedding_values
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    # Test with explicit model (testing behavior)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(api_key='test_key', embedding_model='text-embedding-3-small')
    )

    result = await embedder.create('test input')

    # Check result
    assert len(result) == EMBEDDING_DIM  # Should be truncated to embedding_dim
    assert all(isinstance(x, float) for x in result)

    # Verify API call
    mock_client.embeddings.create.assert_called_once_with(
        input='test input', model='text-embedding-3-small'
    )


@pytest.mark.asyncio
async def test_openai_embedder_create_with_list(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test creating embeddings with a list input"""
    mock_openai = mocker.patch('graphiti_core.embedder.openai.AsyncOpenAI')
    mock_client = mocker.AsyncMock()
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock()]
    mock_response.data[0].embedding = mock_embedding_values
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    # Only specifying API key to test default model behavior
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key='test_key'))

    result = await embedder.create(['test input 1', 'test input 2'])

    # Check result
    assert len(result) == EMBEDDING_DIM

    # Verify API call uses the default model from imported constant
    mock_client.embeddings.create.assert_called_once_with(
        input=['test input 1', 'test input 2'], model=DEFAULT_EMBEDDING_MODEL
    )


@pytest.mark.asyncio
async def test_openai_embedder_with_custom_client(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test using a custom OpenAI client"""
    mock_client = mocker.AsyncMock()
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock()]
    mock_response.data[0].embedding = mock_embedding_values
    mock_client.embeddings.create.return_value = mock_response

    # Test with default config to ensure it uses default model
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(), client=mock_client)

    result = await embedder.create('test input')

    # Check client was used
    mock_client.embeddings.create.assert_called_once_with(
        input='test input', model=DEFAULT_EMBEDDING_MODEL
    )
    assert len(result) == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_openai_embedder_default_config(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test that the default configuration works as expected"""
    mock_openai = mocker.patch('graphiti_core.embedder.openai.AsyncOpenAI')
    mock_client = mocker.AsyncMock()
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock()]
    mock_response.data[0].embedding = mock_embedding_values
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    # Create with completely default config
    embedder = OpenAIEmbedder()

    result = await embedder.create('test input')

    # Verify default model is used
    mock_client.embeddings.create.assert_called_once_with(
        input='test input', model=DEFAULT_EMBEDDING_MODEL
    )
    assert len(result) == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_openai_embedder_with_env_api_key(
    mocker: MockerFixture,
    mock_embedding_values: list[float],
) -> None:
    """Test that the embedder falls back to environment variables for API key"""
    # Mock the environment variable being set
    env_api_key = 'env_test_key'
    mocker.patch.dict('os.environ', {'OPENAI_API_KEY': env_api_key})

    # Import the actual OpenAI client here
    from openai import AsyncOpenAI

    # Create a direct instance of OpenAI client to verify env var behavior
    real_client = AsyncOpenAI(api_key=None)

    # Verify the client picked up the env var
    assert real_client.api_key == env_api_key

    # Now patch the embeddings.create method to avoid actual API calls
    mock_create = mocker.AsyncMock()
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock()]
    mock_response.data[0].embedding = mock_embedding_values
    mock_create.return_value = mock_response
    mocker.patch.object(real_client.embeddings, 'create', mock_create)

    # Patch the AsyncOpenAI constructor to return our real client with mocked create method
    mocker.patch('graphiti_core.embedder.openai.AsyncOpenAI', return_value=real_client)

    # Create the embedder with None api_key
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=None))

    # Verify the embedder's client has the correct API key
    assert embedder.client.api_key == env_api_key

    # Test that it works when creating embeddings
    await embedder.create('test input')

    # Verify the API call was made with the right parameters
    mock_create.assert_called_once_with(input='test input', model=DEFAULT_EMBEDDING_MODEL)
