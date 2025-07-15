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

# Running tests: pytest -xvs tests/embedder/test_gemini.py

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from embedder_fixtures import create_embedding_values

from graphiti_core.embedder.gemini import (
    DEFAULT_EMBEDDING_MODEL,
    GeminiEmbedder,
    GeminiEmbedderConfig,
)


def create_gemini_embedding(multiplier: float = 0.1, dimension: int = 1536) -> MagicMock:
    """Create a mock Gemini embedding with specified value multiplier and dimension."""
    mock_embedding = MagicMock()
    mock_embedding.values = create_embedding_values(multiplier, dimension)
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


class TestGeminiEmbedderInitialization:
    """Tests for GeminiEmbedder initialization."""

    @patch('google.genai.Client')
    def test_init_with_config(self, mock_client):
        """Test initialization with a config object."""
        config = GeminiEmbedderConfig(
            api_key='test_api_key', embedding_model='custom-model', embedding_dim=768
        )
        embedder = GeminiEmbedder(config=config)

        assert embedder.config == config
        assert embedder.config.embedding_model == 'custom-model'
        assert embedder.config.api_key == 'test_api_key'
        assert embedder.config.embedding_dim == 768

    @patch('google.genai.Client')
    def test_init_without_config(self, mock_client):
        """Test initialization without a config uses defaults."""
        embedder = GeminiEmbedder()

        assert embedder.config is not None
        assert embedder.config.embedding_model == DEFAULT_EMBEDDING_MODEL

    @patch('google.genai.Client')
    def test_init_with_partial_config(self, mock_client):
        """Test initialization with partial config."""
        config = GeminiEmbedderConfig(api_key='test_api_key')
        embedder = GeminiEmbedder(config=config)

        assert embedder.config.api_key == 'test_api_key'
        assert embedder.config.embedding_model == DEFAULT_EMBEDDING_MODEL


class TestGeminiEmbedderCreate:
    """Tests for GeminiEmbedder create method."""

    @pytest.mark.asyncio
    async def test_create_calls_api_correctly(
        self,
        gemini_embedder: GeminiEmbedder,
        mock_gemini_client: Any,
        mock_gemini_response: MagicMock,
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
    @patch('google.genai.Client')
    async def test_create_with_custom_model(
        self, mock_client_class, mock_gemini_client: Any, mock_gemini_response: MagicMock
    ) -> None:
        """Test create method with custom embedding model."""
        # Setup embedder with custom model
        config = GeminiEmbedderConfig(api_key='test_api_key', embedding_model='custom-model')
        embedder = GeminiEmbedder(config=config)
        embedder.client = mock_gemini_client
        mock_gemini_client.aio.models.embed_content.return_value = mock_gemini_response

        # Call method
        await embedder.create('Test input')

        # Verify custom model is used
        _, kwargs = mock_gemini_client.aio.models.embed_content.call_args
        assert kwargs['model'] == 'custom-model'

    @pytest.mark.asyncio
    @patch('google.genai.Client')
    async def test_create_with_custom_dimension(
        self, mock_client_class, mock_gemini_client: Any
    ) -> None:
        """Test create method with custom embedding dimension."""
        # Setup embedder with custom dimension
        config = GeminiEmbedderConfig(api_key='test_api_key', embedding_dim=768)
        embedder = GeminiEmbedder(config=config)
        embedder.client = mock_gemini_client

        # Setup mock response with custom dimension
        mock_response = MagicMock()
        mock_response.embeddings = [create_gemini_embedding(0.1, 768)]
        mock_gemini_client.aio.models.embed_content.return_value = mock_response

        # Call method
        result = await embedder.create('Test input')

        # Verify custom dimension is used in config
        _, kwargs = mock_gemini_client.aio.models.embed_content.call_args
        assert kwargs['config'].output_dimensionality == 768

        # Verify result has correct dimension
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_create_with_different_input_types(
        self,
        gemini_embedder: GeminiEmbedder,
        mock_gemini_client: Any,
        mock_gemini_response: MagicMock,
    ) -> None:
        """Test create method with different input types."""
        mock_gemini_client.aio.models.embed_content.return_value = mock_gemini_response

        # Test with string
        await gemini_embedder.create('Test string')

        # Test with list of strings
        await gemini_embedder.create(['Test', 'List'])

        # Test with iterable of integers
        await gemini_embedder.create([1, 2, 3])

        # Verify all calls were made
        assert mock_gemini_client.aio.models.embed_content.call_count == 3

    @pytest.mark.asyncio
    async def test_create_no_embeddings_error(
        self, gemini_embedder: GeminiEmbedder, mock_gemini_client: Any
    ) -> None:
        """Test create method handling of no embeddings response."""
        # Setup mock response with no embeddings
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_gemini_client.aio.models.embed_content.return_value = mock_response

        # Call method and expect exception
        with pytest.raises(ValueError) as exc_info:
            await gemini_embedder.create('Test input')

        assert 'No embeddings returned from Gemini API in create()' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_no_values_error(
        self, gemini_embedder: GeminiEmbedder, mock_gemini_client: Any
    ) -> None:
        """Test create method handling of embeddings with no values."""
        # Setup mock response with embedding but no values
        mock_embedding = MagicMock()
        mock_embedding.values = None
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_gemini_client.aio.models.embed_content.return_value = mock_response

        # Call method and expect exception
        with pytest.raises(ValueError) as exc_info:
            await gemini_embedder.create('Test input')

        assert 'No embeddings returned from Gemini API in create()' in str(exc_info.value)


class TestGeminiEmbedderCreateBatch:
    """Tests for GeminiEmbedder create_batch method."""

    @pytest.mark.asyncio
    async def test_create_batch_processes_multiple_inputs(
        self,
        gemini_embedder: GeminiEmbedder,
        mock_gemini_client: Any,
        mock_gemini_batch_response: MagicMock,
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

    @pytest.mark.asyncio
    async def test_create_batch_single_input(
        self,
        gemini_embedder: GeminiEmbedder,
        mock_gemini_client: Any,
        mock_gemini_response: MagicMock,
    ) -> None:
        """Test create_batch method with single input."""
        mock_gemini_client.aio.models.embed_content.return_value = mock_gemini_response
        input_batch = ['Single input']

        result = await gemini_embedder.create_batch(input_batch)

        assert len(result) == 1
        assert result[0] == mock_gemini_response.embeddings[0].values

    @pytest.mark.asyncio
    async def test_create_batch_empty_input(
        self, gemini_embedder: GeminiEmbedder, mock_gemini_client: Any
    ) -> None:
        """Test create_batch method with empty input."""
        # Setup mock response with no embeddings
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_gemini_client.aio.models.embed_content.return_value = mock_response

        input_batch = []

        result = await gemini_embedder.create_batch(input_batch)
        assert result == []
        mock_gemini_client.aio.models.embed_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_batch_no_embeddings_error(
        self, gemini_embedder: GeminiEmbedder, mock_gemini_client: Any
    ) -> None:
        """Test create_batch method handling of no embeddings response."""
        # Setup mock response with no embeddings
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_gemini_client.aio.models.embed_content.return_value = mock_response

        input_batch = ['Input 1', 'Input 2']

        with pytest.raises(ValueError) as exc_info:
            await gemini_embedder.create_batch(input_batch)

        assert 'No embeddings returned from Gemini API' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_batch_empty_values_error(
        self, gemini_embedder: GeminiEmbedder, mock_gemini_client: Any
    ) -> None:
        """Test create_batch method handling of embeddings with empty values."""
        # Setup mock response with embeddings but empty values
        mock_embedding1 = MagicMock()
        mock_embedding1.values = [0.1, 0.2, 0.3]  # Valid values
        mock_embedding2 = MagicMock()
        mock_embedding2.values = None  # Empty values

        # Mock response for the initial batch call
        mock_batch_response = MagicMock()
        mock_batch_response.embeddings = [mock_embedding1, mock_embedding2]

        # Mock response for individual processing of 'Input 1'
        mock_individual_response_1 = MagicMock()
        mock_individual_response_1.embeddings = [mock_embedding1]

        # Mock response for individual processing of 'Input 2' (which has empty values)
        mock_individual_response_2 = MagicMock()
        mock_individual_response_2.embeddings = [mock_embedding2]

        # Set side_effect for embed_content to control return values for each call
        mock_gemini_client.aio.models.embed_content.side_effect = [
            mock_batch_response,  # First call for the batch
            mock_individual_response_1,  # Second call for individual item 1
            mock_individual_response_2,  # Third call for individual item 2
        ]

        input_batch = ['Input 1', 'Input 2']

        with pytest.raises(ValueError) as exc_info:
            await gemini_embedder.create_batch(input_batch)

        assert 'Empty embedding values returned' in str(exc_info.value)

    @pytest.mark.asyncio
    @patch('google.genai.Client')
    async def test_create_batch_with_custom_model_and_dimension(
        self, mock_client_class, mock_gemini_client: Any
    ) -> None:
        """Test create_batch method with custom model and dimension."""
        # Setup embedder with custom settings
        config = GeminiEmbedderConfig(
            api_key='test_api_key', embedding_model='custom-batch-model', embedding_dim=512
        )
        embedder = GeminiEmbedder(config=config)
        embedder.client = mock_gemini_client

        # Setup mock response
        mock_response = MagicMock()
        mock_response.embeddings = [
            create_gemini_embedding(0.1, 512),
            create_gemini_embedding(0.2, 512),
        ]
        mock_gemini_client.aio.models.embed_content.return_value = mock_response

        input_batch = ['Input 1', 'Input 2']
        result = await embedder.create_batch(input_batch)

        # Verify custom settings are used
        _, kwargs = mock_gemini_client.aio.models.embed_content.call_args
        assert kwargs['model'] == 'custom-batch-model'
        assert kwargs['config'].output_dimensionality == 512

        # Verify results have correct dimension
        assert len(result) == 2
        assert all(len(embedding) == 512 for embedding in result)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
