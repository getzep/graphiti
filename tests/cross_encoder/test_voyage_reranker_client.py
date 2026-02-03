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

# Running tests: pytest -xvs tests/cross_encoder/test_voyage_reranker_client.py

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.voyage_reranker_client import (
    DEFAULT_RERANK_MODEL,
    VoyageRateLimitError,
    VoyageRerankerClient,
    VoyageRerankerConfig,
)


class MockRerankingResult:
    """Mock for Voyage RerankingResult object."""

    def __init__(self, index: int, relevance_score: float, document: str = ''):
        self.index = index
        self.relevance_score = relevance_score
        self.document = document


class MockRerankingResponse:
    """Mock for Voyage rerank response."""

    def __init__(self, results: list[MockRerankingResult], total_tokens: int = 100):
        self.results = results
        self.total_tokens = total_tokens


@pytest.fixture
def mock_voyageai_client() -> Generator[Any, Any, None]:
    """Create a mocked VoyageAI async client."""
    with patch('voyageai.AsyncClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.rerank = AsyncMock()
        yield mock_instance


@pytest.fixture
def voyage_reranker_client(mock_voyageai_client: Any) -> VoyageRerankerClient:
    """Create a VoyageRerankerClient with a mocked client."""
    config = VoyageRerankerConfig(api_key='test_api_key')
    client = VoyageRerankerClient(config=config)
    client.client = mock_voyageai_client
    return client


class TestVoyageRerankerClientInitialization:
    """Tests for VoyageRerankerClient initialization."""

    def test_init_with_config(self):
        """Test initialization with a config object."""
        config = VoyageRerankerConfig(api_key='test_api_key', model='rerank-2.5')
        with patch('voyageai.AsyncClient'):
            client = VoyageRerankerClient(config=config)
        assert client.config == config
        assert client.config.model == 'rerank-2.5'

    def test_init_without_config(self):
        """Test initialization without config uses defaults."""
        with patch('voyageai.AsyncClient'):
            client = VoyageRerankerClient()
        assert client.config is not None
        assert client.config.model == DEFAULT_RERANK_MODEL
        assert client.config.truncation is True
        assert client.config.top_k is None

    def test_init_with_custom_top_k(self):
        """Test initialization with custom top_k."""
        config = VoyageRerankerConfig(api_key='test_key', top_k=5)
        with patch('voyageai.AsyncClient'):
            client = VoyageRerankerClient(config=config)
        assert client.config.top_k == 5

    def test_default_model_constant(self):
        """Test that default model constant is set correctly."""
        assert DEFAULT_RERANK_MODEL == 'rerank-2'


class TestVoyageRerankerClientRanking:
    """Tests for VoyageRerankerClient rank method."""

    @pytest.mark.asyncio
    async def test_rank_basic_functionality(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test basic ranking functionality."""
        # Setup mock response with scores in descending order
        mock_response = MockRerankingResponse(
            results=[
                MockRerankingResult(index=0, relevance_score=0.95),
                MockRerankingResult(index=2, relevance_score=0.75),
                MockRerankingResult(index=1, relevance_score=0.30),
            ]
        )
        mock_voyageai_client.rerank.return_value = mock_response

        # Test data
        query = 'What is the capital of France?'
        passages = [
            'Paris is the capital and most populous city of France.',
            'London is the capital city of England and the United Kingdom.',
            'France is a country in Western Europe.',
        ]

        # Call method
        result = await voyage_reranker_client.rank(query, passages)

        # Assertions
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(
            isinstance(passage, str) and isinstance(score, float) for passage, score in result
        )

        # Check results are in correct order (descending by score)
        assert result[0] == (passages[0], 0.95)
        assert result[1] == (passages[2], 0.75)
        assert result[2] == (passages[1], 0.30)

        # Verify API was called correctly
        mock_voyageai_client.rerank.assert_called_once_with(
            query=query,
            documents=passages,
            model=voyage_reranker_client.config.model,
            top_k=voyage_reranker_client.config.top_k,
            truncation=voyage_reranker_client.config.truncation,
        )

    @pytest.mark.asyncio
    async def test_rank_empty_passages(self, voyage_reranker_client: VoyageRerankerClient):
        """Test ranking with empty passages list."""
        query = 'Test query'
        passages: list[str] = []

        result = await voyage_reranker_client.rank(query, passages)

        assert result == []

    @pytest.mark.asyncio
    async def test_rank_single_passage(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test ranking with a single passage."""
        mock_response = MockRerankingResponse(
            results=[MockRerankingResult(index=0, relevance_score=0.85)]
        )
        mock_voyageai_client.rerank.return_value = mock_response

        query = 'Test query'
        passages = ['Single test passage']

        result = await voyage_reranker_client.rank(query, passages)

        assert len(result) == 1
        assert result[0][0] == 'Single test passage'
        assert result[0][1] == 0.85

    @pytest.mark.asyncio
    async def test_rank_with_top_k(self, mock_voyageai_client: Any):
        """Test ranking with top_k parameter."""
        config = VoyageRerankerConfig(api_key='test_key', top_k=2)
        with patch('voyageai.AsyncClient'):
            client = VoyageRerankerClient(config=config)
        client.client = mock_voyageai_client

        # Only return top 2 results
        mock_response = MockRerankingResponse(
            results=[
                MockRerankingResult(index=0, relevance_score=0.95),
                MockRerankingResult(index=2, relevance_score=0.75),
            ]
        )
        mock_voyageai_client.rerank.return_value = mock_response

        passages = ['Passage 1', 'Passage 2', 'Passage 3']
        result = await client.rank('query', passages)

        assert len(result) == 2
        mock_voyageai_client.rerank.assert_called_once()
        _, kwargs = mock_voyageai_client.rerank.call_args
        assert kwargs['top_k'] == 2

    @pytest.mark.asyncio
    async def test_rank_preserves_score_order(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test that results maintain the order from API (descending by score)."""
        # Voyage API returns results sorted by relevance_score descending
        mock_response = MockRerankingResponse(
            results=[
                MockRerankingResult(index=2, relevance_score=0.99),
                MockRerankingResult(index=0, relevance_score=0.50),
                MockRerankingResult(index=1, relevance_score=0.10),
            ]
        )
        mock_voyageai_client.rerank.return_value = mock_response

        passages = ['Low relevance', 'Very low', 'Highest relevance']
        result = await voyage_reranker_client.rank('query', passages)

        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0][0] == 'Highest relevance'


class TestVoyageRerankerClientErrors:
    """Tests for VoyageRerankerClient error handling."""

    @pytest.mark.asyncio
    async def test_rank_rate_limit_error(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test handling of rate limit errors."""
        mock_voyageai_client.rerank.side_effect = Exception('Rate limit exceeded')

        with pytest.raises(VoyageRateLimitError):
            await voyage_reranker_client.rank('query', ['passage'])

    @pytest.mark.asyncio
    async def test_rank_429_error(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test handling of HTTP 429 errors."""
        mock_voyageai_client.rerank.side_effect = Exception('HTTP 429 Too Many Requests')

        with pytest.raises(VoyageRateLimitError):
            await voyage_reranker_client.rank('query', ['passage'])

    @pytest.mark.asyncio
    async def test_rank_quota_error(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test handling of quota errors."""
        mock_voyageai_client.rerank.side_effect = Exception('Quota exceeded for this month')

        with pytest.raises(VoyageRateLimitError):
            await voyage_reranker_client.rank('query', ['passage'])

    @pytest.mark.asyncio
    async def test_rank_generic_error(
        self, voyage_reranker_client: VoyageRerankerClient, mock_voyageai_client: Any
    ):
        """Test handling of generic errors (not rate limit)."""
        mock_voyageai_client.rerank.side_effect = Exception('Network error')

        with pytest.raises(Exception) as exc_info:
            await voyage_reranker_client.rank('query', ['passage'])

        assert 'Network error' in str(exc_info.value)
        # Should NOT be VoyageRateLimitError
        assert not isinstance(exc_info.value, VoyageRateLimitError)


class TestVoyageRerankerConfig:
    """Tests for VoyageRerankerConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = VoyageRerankerConfig()
        assert config.model == DEFAULT_RERANK_MODEL
        assert config.api_key is None
        assert config.top_k is None
        assert config.truncation is True

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = VoyageRerankerConfig(
            model='rerank-2.5-lite',
            api_key='my-api-key',
            top_k=10,
            truncation=False,
        )
        assert config.model == 'rerank-2.5-lite'
        assert config.api_key == 'my-api-key'
        assert config.top_k == 10
        assert config.truncation is False


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
