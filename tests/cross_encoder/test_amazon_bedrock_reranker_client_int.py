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

# Running tests: pytest -xvs tests/cross_encoder/test_amazon_bedrock_reranker_client_int.py
# Requires: AWS credentials configured and Bedrock model access

import os
from pathlib import Path

import pytest

from graphiti_core.cross_encoder.amazon_bedrock_reranker_client import (
    AmazonBedrockRerankerClient,
)


def _has_aws_credentials():
    """Check if AWS credentials are available."""
    # Check environment variables
    if os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('AWS_PROFILE'):
        return True

    # Check for default credentials file
    credentials_file = Path.home() / '.aws' / 'credentials'
    return credentials_file.exists()


# Skip all tests if AWS credentials not available
pytestmark = pytest.mark.skipif(not _has_aws_credentials(), reason='AWS credentials not configured')


@pytest.fixture
def cohere_reranker():
    """Create a real AmazonBedrockRerankerClient with Cohere model."""
    return AmazonBedrockRerankerClient(
        model='cohere.rerank-v3-5:0', region='us-east-1', max_results=10
    )


@pytest.fixture
def amazon_reranker():
    """Create a real AmazonBedrockRerankerClient with Amazon model."""
    return AmazonBedrockRerankerClient(
        model='amazon.rerank-v1:0',
        region='us-west-2',  # Amazon model available in us-west-2
        max_results=10,
    )


class TestAmazonBedrockRerankerClientIntegration:
    """Integration tests for AmazonBedrockRerankerClient with real AWS Bedrock."""

    @pytest.mark.asyncio
    async def test_cohere_rerank_basic_functionality(self, cohere_reranker):
        """Test basic reranking functionality with Cohere model."""
        query = 'What is machine learning?'
        passages = [
            'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'Python is a popular programming language used for web development.',
            'Deep learning uses neural networks to learn patterns in data.',
            'JavaScript is commonly used for frontend web development.',
            'Supervised learning requires labeled training data.',
        ]

        result = await cohere_reranker.rank(query, passages)

        # Verify basic structure
        assert isinstance(result, list)
        assert len(result) <= len(passages)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(
            isinstance(passage, str) and isinstance(score, float) for passage, score in result
        )

        # Verify scores are in valid range
        scores = [score for _, score in result]
        assert all(0.0 <= score <= 1.0 for score in scores)

        # Verify results are sorted by relevance (descending)
        assert scores == sorted(scores, reverse=True)

        # ML-related passages should rank higher
        top_passage = result[0][0]
        assert any(
            word in top_passage.lower() for word in ['machine', 'learning', 'neural', 'supervised']
        )

    @pytest.mark.asyncio
    async def test_amazon_rerank_basic_functionality(self, amazon_reranker):
        """Test basic reranking functionality with Amazon model."""
        query = 'Python programming'
        passages = [
            'Python is a high-level programming language known for its simplicity.',
            'The snake is a reptile that moves by slithering.',
            'Programming languages help developers create software applications.',
            'Reptiles are cold-blooded animals that lay eggs.',
        ]

        result = await amazon_reranker.rank(query, passages)

        # Verify basic structure
        assert isinstance(result, list)
        assert len(result) <= len(passages)
        assert all(isinstance(item, tuple) for item in result)

        # Programming-related passages should rank higher
        top_passage = result[0][0]
        assert any(word in top_passage.lower() for word in ['python', 'programming', 'language'])

    @pytest.mark.asyncio
    async def test_rerank_empty_passages(self, cohere_reranker):
        """Test reranking with empty passages list."""
        query = 'Test query'
        passages = []

        result = await cohere_reranker.rank(query, passages)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_single_passage(self, cohere_reranker):
        """Test reranking with single passage."""
        query = 'artificial intelligence'
        passages = ['AI is transforming various industries with automation.']

        result = await cohere_reranker.rank(query, passages)

        assert len(result) == 1
        assert result[0][0] == passages[0]
        assert isinstance(result[0][1], float)
        assert 0.0 <= result[0][1] <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_relevance_ordering(self, cohere_reranker):
        """Test that more relevant passages get higher scores."""
        query = 'climate change effects'
        passages = [
            'Global warming is causing ice caps to melt rapidly.',  # Highly relevant
            'Climate change leads to extreme weather patterns.',  # Highly relevant
            'The recipe for chocolate cake requires flour and eggs.',  # Not relevant
            'Rising sea levels threaten coastal communities.',  # Relevant
            'My favorite color is blue and I like painting.',  # Not relevant
        ]

        result = await cohere_reranker.rank(query, passages)

        # Top results should be climate-related
        top_3_passages = [passage for passage, _ in result[:3]]
        climate_passages = [
            p
            for p in top_3_passages
            if any(word in p.lower() for word in ['climate', 'warming', 'sea', 'weather', 'ice'])
        ]

        # At least 2 of top 3 should be climate-related
        assert len(climate_passages) >= 2

    @pytest.mark.asyncio
    async def test_max_results_parameter(self, cohere_reranker):
        """Test that max_results parameter is respected."""
        # Set max_results to 3
        reranker = AmazonBedrockRerankerClient(
            model='cohere.rerank-v3-5:0', region='us-east-1', max_results=3
        )

        query = 'technology'
        passages = [
            'Artificial intelligence is advancing rapidly.',
            'Smartphones have changed communication.',
            'Electric cars are becoming more popular.',
            'Social media connects people globally.',
            'Renewable energy is the future.',
        ]

        result = await reranker.rank(query, passages)

        # Should return at most 3 results
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_different_query_types(self, cohere_reranker):
        """Test reranking with different types of queries."""
        passages = [
            'The capital of France is Paris.',
            'Machine learning algorithms process data.',
            'Cooking pasta requires boiling water.',
            'Paris is known for the Eiffel Tower.',
        ]

        # Factual query
        factual_result = await cohere_reranker.rank('What is the capital of France?', passages)

        # Technical query
        technical_result = await cohere_reranker.rank('machine learning data processing', passages)

        # Both should return valid results with different rankings
        assert len(factual_result) > 0
        assert len(technical_result) > 0

        # Top results should be different for different queries
        factual_top = factual_result[0][0]
        technical_top = technical_result[0][0]

        assert 'France' in factual_top or 'Paris' in factual_top
        assert 'machine' in technical_top.lower() or 'learning' in technical_top.lower()


if __name__ == '__main__':
    pytest.main(['-v', 'test_amazon_bedrock_reranker_client_int.py'])
