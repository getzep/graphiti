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

# Running tests: pytest -xvs tests/cross_encoder/test_amazon_bedrock_reranker_client.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.amazon_bedrock_reranker_client import (
    DEFAULT_MODEL,
    MODEL_REGIONS,
    AmazonBedrockRerankerClient,
)


@pytest.fixture
def mock_boto3_client():
    """Fixture to mock the boto3 bedrock-agent-runtime client."""
    with patch('boto3.client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_rerank_response():
    """Create a mock Bedrock rerank response."""
    return {
        'results': [
            {'index': 0, 'relevanceScore': 0.85},
            {'index': 1, 'relevanceScore': 0.65},
            {'index': 2, 'relevanceScore': 0.45},
        ]
    }


@pytest.fixture
def reranker_client(mock_boto3_client):
    """Fixture to create an AmazonBedrockRerankerClient with a mocked client."""
    client = AmazonBedrockRerankerClient(model=DEFAULT_MODEL, region='us-east-1', max_results=10)
    client.client = mock_boto3_client
    return client


class TestAmazonBedrockRerankerClientInitialization:
    """Tests for AmazonBedrockRerankerClient initialization."""

    def test_init_with_valid_model_region_combination(self):
        """Test initialization with valid model and region combination."""
        client = AmazonBedrockRerankerClient(
            model='cohere.rerank-v3-5:0', region='us-east-1', max_results=50
        )

        assert client.model == 'cohere.rerank-v3-5:0'
        assert client.region == 'us-east-1'
        assert client.max_results == 50

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        client = AmazonBedrockRerankerClient()

        assert client.model == DEFAULT_MODEL
        assert client.region == 'us-east-1'
        assert client.max_results == 100

    def test_init_with_amazon_model_valid_region(self):
        """Test initialization with Amazon model in valid region."""
        client = AmazonBedrockRerankerClient(model='amazon.rerank-v1:0', region='us-west-2')

        assert client.model == 'amazon.rerank-v1:0'
        assert client.region == 'us-west-2'

    def test_init_with_invalid_model_region_combination(self):
        """Test initialization with invalid model and region combination."""
        with pytest.raises(ValueError) as exc_info:
            AmazonBedrockRerankerClient(
                model='amazon.rerank-v1:0',
                region='us-east-1',  # Amazon model not available in us-east-1
            )

        assert 'Model amazon.rerank-v1:0 is not supported in region us-east-1' in str(
            exc_info.value
        )
        assert 'us-west-2' in str(exc_info.value)  # Should list supported regions

    def test_init_with_cohere_model_invalid_region(self):
        """Test initialization with Cohere model in unsupported region."""
        with pytest.raises(ValueError) as exc_info:
            AmazonBedrockRerankerClient(
                model='cohere.rerank-v3-5:0',
                region='ap-south-1',  # Not in supported regions
            )

        assert 'Model cohere.rerank-v3-5:0 is not supported in region ap-south-1' in str(
            exc_info.value
        )

    def test_model_regions_mapping(self):
        """Test that MODEL_REGIONS mapping is correct."""
        # Verify Amazon model regions
        amazon_regions = MODEL_REGIONS['amazon.rerank-v1:0']
        expected_amazon_regions = ['ap-northeast-1', 'ca-central-1', 'eu-central-1', 'us-west-2']
        assert amazon_regions == expected_amazon_regions

        # Verify Cohere model regions
        cohere_regions = MODEL_REGIONS['cohere.rerank-v3-5:0']
        expected_cohere_regions = [
            'ap-northeast-1',
            'ca-central-1',
            'eu-central-1',
            'us-east-1',
            'us-west-2',
        ]
        assert cohere_regions == expected_cohere_regions


class TestAmazonBedrockRerankerClientRanking:
    """Tests for AmazonBedrockRerankerClient rank method."""

    @pytest.mark.asyncio
    async def test_rank_basic_functionality(
        self, reranker_client, mock_boto3_client, mock_rerank_response
    ):
        """Test basic ranking functionality."""
        # Setup
        query = 'What is machine learning?'
        passages = [
            'Machine learning is a subset of artificial intelligence.',
            'Python is a programming language.',
            'Deep learning uses neural networks for pattern recognition.',
        ]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = mock_rerank_response
            mock_loop.return_value.run_in_executor = mock_executor

            result = await reranker_client.rank(query, passages)

        # Assertions
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(
            isinstance(passage, str) and isinstance(score, float) for passage, score in result
        )

        # Check that results are sorted by relevance score (descending)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

        # Verify specific results based on mock response
        assert result[0][1] == 0.85  # Highest score
        assert result[1][1] == 0.65  # Medium score
        assert result[2][1] == 0.45  # Lowest score

        # Verify executor was called
        mock_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_rank_empty_passages(self, reranker_client):
        """Test ranking with empty passages list."""
        query = 'Test query'
        passages = []

        result = await reranker_client.rank(query, passages)

        assert result == []

    @pytest.mark.asyncio
    async def test_rank_single_passage(self, reranker_client, mock_boto3_client):
        """Test ranking with a single passage."""
        # Setup single result response
        single_response = {'results': [{'index': 0, 'relevanceScore': 0.75}]}

        query = 'Test query'
        passages = ['Single test passage']

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = single_response
            mock_loop.return_value.run_in_executor = mock_executor

            result = await reranker_client.rank(query, passages)

        assert len(result) == 1
        assert result[0][0] == 'Single test passage'
        assert result[0][1] == 0.75

    @pytest.mark.asyncio
    async def test_rank_api_error_handling(self, reranker_client, mock_boto3_client):
        """Test handling of API errors."""
        query = 'Test query'
        passages = ['Passage 1', 'Passage 2']

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.side_effect = Exception('Bedrock API error')
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(Exception) as exc_info:
                await reranker_client.rank(query, passages)

            assert 'Bedrock API error' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rank_empty_results_response(self, reranker_client, mock_boto3_client):
        """Test handling of empty results in response."""
        empty_response = {'results': []}

        query = 'Test query'
        passages = ['Test passage']

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = empty_response
            mock_loop.return_value.run_in_executor = mock_executor

            result = await reranker_client.rank(query, passages)

        assert result == []

    @pytest.mark.asyncio
    async def test_rank_missing_results_key(self, reranker_client, mock_boto3_client):
        """Test handling of response missing results key."""
        invalid_response = {'other_key': 'value'}

        query = 'Test query'
        passages = ['Test passage']

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = invalid_response
            mock_loop.return_value.run_in_executor = mock_executor

            result = await reranker_client.rank(query, passages)

        # Should handle gracefully and return empty list
        assert result == []


class TestAmazonBedrockRerankerClientConfiguration:
    """Tests for AmazonBedrockRerankerClient configuration."""

    def test_different_models_and_regions(self):
        """Test initialization with different valid model/region combinations."""
        # Test Cohere model in different regions
        for region in MODEL_REGIONS['cohere.rerank-v3-5:0']:
            client = AmazonBedrockRerankerClient(model='cohere.rerank-v3-5:0', region=region)
            assert client.model == 'cohere.rerank-v3-5:0'
            assert client.region == region

        # Test Amazon model in different regions
        for region in MODEL_REGIONS['amazon.rerank-v1:0']:
            client = AmazonBedrockRerankerClient(model='amazon.rerank-v1:0', region=region)
            assert client.model == 'amazon.rerank-v1:0'
            assert client.region == region

    def test_max_results_configuration(self):
        """Test different max_results configurations."""
        # Test default
        client = AmazonBedrockRerankerClient()
        assert client.max_results == 100

        # Test custom value
        client = AmazonBedrockRerankerClient(max_results=50)
        assert client.max_results == 50


if __name__ == '__main__':
    pytest.main(['-v', 'test_amazon_bedrock_reranker_client.py'])
