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

# Running tests: pytest -xvs tests/cross_encoder/test_gemini_reranker_client.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.llm_client import LLMConfig, RateLimitError


@pytest.fixture
def mock_gemini_client():
    """Fixture to mock the Google Gemini client."""
    with patch('google.genai.Client') as mock_client:
        # Setup mock instance and its methods
        mock_instance = mock_client.return_value
        mock_instance.aio = MagicMock()
        mock_instance.aio.models = MagicMock()
        mock_instance.aio.models.generate_content = AsyncMock()
        yield mock_instance


@pytest.fixture
def gemini_reranker_client(mock_gemini_client):
    """Fixture to create a GeminiRerankerClient with a mocked client."""
    config = LLMConfig(api_key='test_api_key', model='test-model')
    client = GeminiRerankerClient(config=config)
    # Replace the client's client with our mock to ensure we're using the mock
    client.client = mock_gemini_client
    return client


def create_mock_response(score_text: str) -> MagicMock:
    """Helper function to create a mock Gemini response."""
    mock_response = MagicMock()
    mock_response.text = score_text
    return mock_response


class TestGeminiRerankerClientInitialization:
    """Tests for GeminiRerankerClient initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = LLMConfig(api_key='test_api_key', model='test-model')
        client = GeminiRerankerClient(config=config)

        assert client.config == config

    @patch('google.genai.Client')
    def test_init_without_config(self, mock_client):
        """Test initialization without a config uses defaults."""
        client = GeminiRerankerClient()

        assert client.config is not None

    def test_init_with_custom_client(self):
        """Test initialization with a custom client."""
        mock_client = MagicMock()
        client = GeminiRerankerClient(client=mock_client)

        assert client.client == mock_client


class TestGeminiRerankerClientRanking:
    """Tests for GeminiRerankerClient rank method."""

    @pytest.mark.asyncio
    async def test_rank_basic_functionality(self, gemini_reranker_client, mock_gemini_client):
        """Test basic ranking functionality."""
        # Setup mock responses with different scores
        mock_responses = [
            create_mock_response('85'),  # High relevance
            create_mock_response('45'),  # Medium relevance
            create_mock_response('20'),  # Low relevance
        ]
        mock_gemini_client.aio.models.generate_content.side_effect = mock_responses

        # Test data
        query = 'What is the capital of France?'
        passages = [
            'Paris is the capital and most populous city of France.',
            'London is the capital city of England and the United Kingdom.',
            'Berlin is the capital and largest city of Germany.',
        ]

        # Call method
        result = await gemini_reranker_client.rank(query, passages)

        # Assertions
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(
            isinstance(passage, str) and isinstance(score, float) for passage, score in result
        )

        # Check scores are normalized to [0, 1] and sorted in descending order
        scores = [score for _, score in result]
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert scores == sorted(scores, reverse=True)

        # Check that the highest scoring passage is first
        assert result[0][1] == 0.85  # 85/100
        assert result[1][1] == 0.45  # 45/100
        assert result[2][1] == 0.20  # 20/100

    @pytest.mark.asyncio
    async def test_rank_empty_passages(self, gemini_reranker_client):
        """Test ranking with empty passages list."""
        query = 'Test query'
        passages = []

        result = await gemini_reranker_client.rank(query, passages)

        assert result == []

    @pytest.mark.asyncio
    async def test_rank_single_passage(self, gemini_reranker_client, mock_gemini_client):
        """Test ranking with a single passage."""
        # Setup mock response
        mock_gemini_client.aio.models.generate_content.return_value = create_mock_response('75')

        query = 'Test query'
        passages = ['Single test passage']

        result = await gemini_reranker_client.rank(query, passages)

        assert len(result) == 1
        assert result[0][0] == 'Single test passage'
        assert result[0][1] == 1.0  # Single passage gets full score

    @pytest.mark.asyncio
    async def test_rank_score_extraction_with_regex(
        self, gemini_reranker_client, mock_gemini_client
    ):
        """Test score extraction from various response formats."""
        # Setup mock responses with different formats
        mock_responses = [
            create_mock_response('Score: 90'),  # Contains text before number
            create_mock_response('The relevance is 65 out of 100'),  # Contains text around number
            create_mock_response('8'),  # Just the number
        ]
        mock_gemini_client.aio.models.generate_content.side_effect = mock_responses

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2', 'Passage 3']

        result = await gemini_reranker_client.rank(query, passages)

        # Check that scores were extracted correctly and normalized
        scores = [score for _, score in result]
        assert 0.90 in scores  # 90/100
        assert 0.65 in scores  # 65/100
        assert 0.08 in scores  # 8/100

    @pytest.mark.asyncio
    async def test_rank_invalid_score_handling(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of invalid or non-numeric scores."""
        # Setup mock responses with invalid scores
        mock_responses = [
            create_mock_response('Not a number'),  # Invalid response
            create_mock_response(''),  # Empty response
            create_mock_response('95'),  # Valid response
        ]
        mock_gemini_client.aio.models.generate_content.side_effect = mock_responses

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2', 'Passage 3']

        result = await gemini_reranker_client.rank(query, passages)

        # Check that invalid scores are handled gracefully (assigned 0.0)
        scores = [score for _, score in result]
        assert 0.95 in scores  # Valid score
        assert scores.count(0.0) == 2  # Two invalid scores assigned 0.0

    @pytest.mark.asyncio
    async def test_rank_score_clamping(self, gemini_reranker_client, mock_gemini_client):
        """Test that scores are properly clamped to [0, 1] range."""
        # Setup mock responses with extreme scores
        # Note: regex only matches 1-3 digits, so negative numbers won't match
        mock_responses = [
            create_mock_response('999'),  # Above 100 but within regex range
            create_mock_response('invalid'),  # Invalid response becomes 0.0
            create_mock_response('50'),  # Normal score
        ]
        mock_gemini_client.aio.models.generate_content.side_effect = mock_responses

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2', 'Passage 3']

        result = await gemini_reranker_client.rank(query, passages)

        # Check that scores are normalized and clamped
        scores = [score for _, score in result]
        assert all(0.0 <= score <= 1.0 for score in scores)
        # 999 should be clamped to 1.0 (999/100 = 9.99, clamped to 1.0)
        assert 1.0 in scores
        # Invalid response should be 0.0
        assert 0.0 in scores
        # Normal score should be normalized (50/100 = 0.5)
        assert 0.5 in scores

    @pytest.mark.asyncio
    async def test_rank_rate_limit_error(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of rate limit errors."""
        # Setup mock to raise rate limit error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception(
            'Rate limit exceeded'
        )

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2']

        with pytest.raises(RateLimitError):
            await gemini_reranker_client.rank(query, passages)

    @pytest.mark.asyncio
    async def test_rank_quota_error(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of quota errors."""
        # Setup mock to raise quota error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception('Quota exceeded')

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2']

        with pytest.raises(RateLimitError):
            await gemini_reranker_client.rank(query, passages)

    @pytest.mark.asyncio
    async def test_rank_resource_exhausted_error(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of resource exhausted errors."""
        # Setup mock to raise resource exhausted error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception('resource_exhausted')

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2']

        with pytest.raises(RateLimitError):
            await gemini_reranker_client.rank(query, passages)

    @pytest.mark.asyncio
    async def test_rank_429_error(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of HTTP 429 errors."""
        # Setup mock to raise 429 error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception(
            'HTTP 429 Too Many Requests'
        )

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2']

        with pytest.raises(RateLimitError):
            await gemini_reranker_client.rank(query, passages)

    @pytest.mark.asyncio
    async def test_rank_generic_error(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of generic errors."""
        # Setup mock to raise generic error
        mock_gemini_client.aio.models.generate_content.side_effect = Exception('Generic error')

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2']

        with pytest.raises(Exception) as exc_info:
            await gemini_reranker_client.rank(query, passages)

        assert 'Generic error' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rank_concurrent_requests(self, gemini_reranker_client, mock_gemini_client):
        """Test that multiple passages are scored concurrently."""
        # Setup mock responses
        mock_responses = [
            create_mock_response('80'),
            create_mock_response('60'),
            create_mock_response('40'),
        ]
        mock_gemini_client.aio.models.generate_content.side_effect = mock_responses

        query = 'Test query'
        passages = ['Passage 1', 'Passage 2', 'Passage 3']

        await gemini_reranker_client.rank(query, passages)

        # Verify that generate_content was called for each passage
        assert mock_gemini_client.aio.models.generate_content.call_count == 3

        # Verify that all calls were made with correct parameters
        calls = mock_gemini_client.aio.models.generate_content.call_args_list
        for call in calls:
            args, kwargs = call
            assert kwargs['model'] == gemini_reranker_client.config.model
            assert kwargs['config'].temperature == 0.0
            assert kwargs['config'].max_output_tokens == 3

    @pytest.mark.asyncio
    async def test_rank_response_parsing_error(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of response parsing errors."""
        # Setup mock responses that will trigger ValueError during parsing
        mock_responses = [
            create_mock_response('not a number at all'),  # Will fail regex match
            create_mock_response('also invalid text'),  # Will fail regex match
        ]
        mock_gemini_client.aio.models.generate_content.side_effect = mock_responses

        query = 'Test query'
        # Use multiple passages to avoid the single passage special case
        passages = ['Passage 1', 'Passage 2']

        result = await gemini_reranker_client.rank(query, passages)

        # Should handle the error gracefully and assign 0.0 score to both
        assert len(result) == 2
        assert all(score == 0.0 for _, score in result)

    @pytest.mark.asyncio
    async def test_rank_empty_response_text(self, gemini_reranker_client, mock_gemini_client):
        """Test handling of empty response text."""
        # Setup mock response with empty text
        mock_response = MagicMock()
        mock_response.text = ''  # Empty string instead of None
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        query = 'Test query'
        # Use multiple passages to avoid the single passage special case
        passages = ['Passage 1', 'Passage 2']

        result = await gemini_reranker_client.rank(query, passages)

        # Should handle empty text gracefully and assign 0.0 score to both
        assert len(result) == 2
        assert all(score == 0.0 for _, score in result)


if __name__ == '__main__':
    pytest.main(['-v', 'test_gemini_reranker_client.py'])
