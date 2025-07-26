"""
Tests for JinaAIRerankerClient
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.jina_reranker_client import (
    JinaAIRerankerClient,
    JinaAIRerankerConfig,
)


@pytest.fixture
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock_client:
        instance = mock_client.return_value
        instance.post = AsyncMock()
        yield instance


@pytest.fixture
def jina_reranker_client(mock_httpx_client):
    config = JinaAIRerankerConfig(api_key='test')
    client = JinaAIRerankerClient(config=config)
    client.client = mock_httpx_client
    return client


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    mock_result = MagicMock()
    mock_result.json.return_value = {
        'results': [
            {'index': 0, 'relevance_score': 0.9},
            {'index': 1, 'relevance_score': 0.7},
        ]
    }
    mock_result.raise_for_status = MagicMock()
    return mock_result


@pytest.mark.asyncio
async def test_rank_basic_functionality(jina_reranker_client, mock_httpx_client, mock_httpx_response):
    mock_httpx_client.post.return_value = mock_httpx_response

    query = 'What is the capital of France?'
    passages = [
        'Paris is the capital of France.',
        'Berlin is the capital of Germany.',
    ]

    result = await jina_reranker_client.rank(query, passages)

    mock_httpx_client.post.assert_called_once()
    _, kwargs = mock_httpx_client.post.call_args
    assert kwargs['json']['query'] == query
    assert kwargs['json']['documents'] == [{'text': passages[0]}, {'text': passages[1]}]

    assert result[0][0] == passages[0]
    assert result[0][1] == 0.9
    assert len(result) == 2


@pytest.mark.asyncio
async def test_rank_empty_input(jina_reranker_client):
    result = await jina_reranker_client.rank('Empty', [])
    assert result == []
