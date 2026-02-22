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

# Running tests: pytest -xvs tests/embedder/test_amazon_bedrock_int.py
# Requires: AWS credentials configured and Bedrock model access

import os
from pathlib import Path

import pytest

from graphiti_core.embedder.amazon_bedrock import (
    DEFAULT_EMBEDDING_MODEL,
    AmazonBedrockEmbedder,
    AmazonBedrockEmbedderConfig,
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
def bedrock_embedder():
    """Create a real AmazonBedrockEmbedder for integration testing."""
    config = AmazonBedrockEmbedderConfig(model=DEFAULT_EMBEDDING_MODEL, region='us-east-1')
    return AmazonBedrockEmbedder(config=config)


class TestAmazonBedrockEmbedderIntegration:
    """Integration tests for AmazonBedrockEmbedder with real AWS Bedrock."""

    @pytest.mark.asyncio
    async def test_create_single_embedding(self, bedrock_embedder):
        """Test creating a single embedding."""
        text = 'This is a test sentence for embedding.'

        result = await bedrock_embedder.create(text)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)
        # Amazon Titan embeddings are typically 1024 or 1536 dimensions
        assert len(result) in [1024, 1536]

    @pytest.mark.asyncio
    async def test_create_batch_embeddings(self, bedrock_embedder):
        """Test creating multiple embeddings in batch."""
        texts = ['First test sentence.', 'Second test sentence.', 'Third test sentence.']

        result = await bedrock_embedder.create_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(len(embedding) > 0 for embedding in result)
        assert all(isinstance(x, float) for embedding in result for x in embedding)

        # All embeddings should have the same dimension
        dimensions = [len(embedding) for embedding in result]
        assert all(dim == dimensions[0] for dim in dimensions)

    @pytest.mark.asyncio
    async def test_embedding_similarity(self, bedrock_embedder):
        """Test that similar texts have similar embeddings."""
        similar_texts = ['The cat sat on the mat.', 'A cat was sitting on a mat.']
        different_text = 'Quantum physics is fascinating.'

        # Get embeddings
        similar_embeddings = await bedrock_embedder.create_batch(similar_texts)
        different_embedding = await bedrock_embedder.create(different_text)

        # Calculate cosine similarity (simplified)
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)

        # Similar texts should have higher similarity than different text
        similar_similarity = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
        different_similarity = cosine_similarity(similar_embeddings[0], different_embedding)

        assert similar_similarity > different_similarity
        assert similar_similarity > 0.7  # Should be quite similar

    @pytest.mark.asyncio
    async def test_different_input_types(self, bedrock_embedder):
        """Test embedding different input types."""
        # String input
        string_result = await bedrock_embedder.create('Test string')

        # List input
        list_result = await bedrock_embedder.create(['Test', 'string', 'list'])

        # Both should return valid embeddings
        assert isinstance(string_result, list)
        assert isinstance(list_result, list)
        assert len(string_result) > 0
        assert len(list_result) > 0
        assert all(isinstance(x, float) for x in string_result)
        assert all(isinstance(x, float) for x in list_result)

    @pytest.mark.asyncio
    async def test_empty_batch(self, bedrock_embedder):
        """Test handling of empty batch."""
        result = await bedrock_embedder.create_batch([])
        assert result == []


if __name__ == '__main__':
    pytest.main(['-v', 'test_amazon_bedrock_int.py'])
