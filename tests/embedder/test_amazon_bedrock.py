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

# Running tests: pytest -xvs tests/embedder/test_amazon_bedrock.py

import json
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.embedder.amazon_bedrock import (
    DEFAULT_EMBEDDING_MODEL,
    AmazonBedrockEmbedder,
    AmazonBedrockEmbedderConfig,
)


@pytest.fixture
def mock_boto3_client():
    """Create a mocked boto3 bedrock-runtime client."""
    with patch('boto3.client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_bedrock_response():
    """Create a mock Bedrock embeddings response."""
    mock_response = {'body': MagicMock()}
    mock_response_body = {'embedding': [0.1] * 1024}
    mock_response['body'].read.return_value.decode.return_value = json.dumps(mock_response_body)
    return mock_response


@pytest.fixture
def bedrock_embedder(mock_boto3_client):
    """Create an AmazonBedrockEmbedder with a mocked client."""
    config = AmazonBedrockEmbedderConfig(model=DEFAULT_EMBEDDING_MODEL, region='us-east-1')
    embedder = AmazonBedrockEmbedder(config=config)
    embedder.client = mock_boto3_client
    return embedder


class TestAmazonBedrockEmbedderInitialization:
    """Tests for AmazonBedrockEmbedder initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = AmazonBedrockEmbedderConfig(model='custom.model:0', region='eu-west-1')
        embedder = AmazonBedrockEmbedder(config=config)

        assert embedder.config.model == 'custom.model:0'
        assert embedder.config.region == 'eu-west-1'

    def test_init_without_config(self):
        """Test initialization without a config uses defaults."""
        embedder = AmazonBedrockEmbedder()

        assert embedder.config.model == DEFAULT_EMBEDDING_MODEL
        assert embedder.config.region == 'us-east-1'

    def test_config_defaults(self):
        """Test that config has correct default values."""
        config = AmazonBedrockEmbedderConfig()

        assert config.model == DEFAULT_EMBEDDING_MODEL
        assert config.region == 'us-east-1'


class TestAmazonBedrockEmbedderCreate:
    """Tests for AmazonBedrockEmbedder create method."""

    @pytest.mark.asyncio
    async def test_create_with_string_input(
        self, bedrock_embedder, mock_boto3_client, mock_bedrock_response
    ):
        """Test create method with string input."""
        # Setup
        mock_boto3_client.invoke_model.return_value = mock_bedrock_response
        input_text = 'Test input string'

        # Call method
        result = await bedrock_embedder.create(input_text)

        # Verify API call
        mock_boto3_client.invoke_model.assert_called_once()
        call_args = mock_boto3_client.invoke_model.call_args

        assert call_args[1]['modelId'] == DEFAULT_EMBEDDING_MODEL
        assert call_args[1]['accept'] == 'application/json'
        assert call_args[1]['contentType'] == 'application/json'

        # Verify request body
        body = json.loads(call_args[1]['body'])
        assert body['inputText'] == input_text

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_create_with_list_input(
        self, bedrock_embedder, mock_boto3_client, mock_bedrock_response
    ):
        """Test create method with list of strings input."""
        # Setup
        mock_boto3_client.invoke_model.return_value = mock_bedrock_response
        input_list = ['First string', 'Second string', 'Third string']

        # Call method
        result = await bedrock_embedder.create(input_list)

        # Verify request body contains joined strings
        call_args = mock_boto3_client.invoke_model.call_args
        body = json.loads(call_args[1]['body'])
        assert body['inputText'] == 'First string Second string Third string'

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_create_with_other_input(
        self, bedrock_embedder, mock_boto3_client, mock_bedrock_response
    ):
        """Test create method with non-string, non-list input."""
        # Setup
        mock_boto3_client.invoke_model.return_value = mock_bedrock_response
        input_data = 12345  # Integer input

        # Call method
        result = await bedrock_embedder.create(input_data)

        # Verify request body contains string representation
        call_args = mock_boto3_client.invoke_model.call_args
        body = json.loads(call_args[1]['body'])
        assert body['inputText'] == '12345'

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_create_api_error(self, bedrock_embedder, mock_boto3_client):
        """Test handling of API errors."""
        # Setup mock to raise exception
        mock_boto3_client.invoke_model.side_effect = Exception('API Error')

        # Call method and verify exception is raised
        with pytest.raises(Exception) as exc_info:
            await bedrock_embedder.create('test input')

        assert 'API Error' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_invalid_response_format(self, bedrock_embedder, mock_boto3_client):
        """Test handling of invalid response format."""
        # Setup mock response with missing embedding field
        mock_response = {'body': MagicMock()}
        mock_response_body = {'invalid': 'response'}
        mock_response['body'].read.return_value.decode.return_value = json.dumps(mock_response_body)
        mock_boto3_client.invoke_model.return_value = mock_response

        # Call method and verify KeyError is raised
        with pytest.raises(KeyError):
            await bedrock_embedder.create('test input')


class TestAmazonBedrockEmbedderCreateBatch:
    """Tests for AmazonBedrockEmbedder create_batch method."""

    @pytest.mark.asyncio
    async def test_create_batch_multiple_inputs(
        self, bedrock_embedder, mock_boto3_client, mock_bedrock_response
    ):
        """Test create_batch method with multiple inputs."""
        # Setup
        mock_boto3_client.invoke_model.return_value = mock_bedrock_response
        input_batch = ['First text', 'Second text', 'Third text']

        # Call method
        result = await bedrock_embedder.create_batch(input_batch)

        # Verify API was called for each input
        assert mock_boto3_client.invoke_model.call_count == 3

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(len(embedding) == 1024 for embedding in result)

    @pytest.mark.asyncio
    async def test_create_batch_empty_list(self, bedrock_embedder, mock_boto3_client):
        """Test create_batch method with empty input list."""
        # Call method
        result = await bedrock_embedder.create_batch([])

        # Verify no API calls were made
        mock_boto3_client.invoke_model.assert_not_called()

        # Verify result
        assert result == []

    @pytest.mark.asyncio
    async def test_create_batch_single_input(
        self, bedrock_embedder, mock_boto3_client, mock_bedrock_response
    ):
        """Test create_batch method with single input."""
        # Setup
        mock_boto3_client.invoke_model.return_value = mock_bedrock_response
        input_batch = ['Single text']

        # Call method
        result = await bedrock_embedder.create_batch(input_batch)

        # Verify API was called once
        assert mock_boto3_client.invoke_model.call_count == 1

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 1024


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
