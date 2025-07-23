"""
Tests for JinaAIEmbedder.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.embedder.jina import (
    DEFAULT_EMBEDDING_MODEL,
    JinaAIEmbedder,
    JinaAIEmbedderConfig,
)
from tests.embedder.embedder_fixtures import create_embedding_values


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    mock_result = MagicMock()
    mock_result.json.return_value = {"data": [{"embedding": create_embedding_values()}]}
    mock_result.raise_for_status = MagicMock()
    return mock_result


@pytest.fixture
def mock_httpx_batch_response() -> MagicMock:
    mock_result = MagicMock()
    mock_result.json.return_value = {
        "data": [
            {"embedding": create_embedding_values(0.1)},
            {"embedding": create_embedding_values(0.2)},
            {"embedding": create_embedding_values(0.3)},
        ]
    }
    mock_result.raise_for_status = MagicMock()
    return mock_result


@pytest.fixture
def mock_httpx_client() -> Generator[Any, Any, None]:
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.post = AsyncMock()
        yield mock_instance


@pytest.fixture
def jina_embedder(mock_httpx_client: Any) -> JinaAIEmbedder:
    config = JinaAIEmbedderConfig(api_key="test_api_key")
    embedder = JinaAIEmbedder(config=config)
    embedder.client = mock_httpx_client
    return embedder


@pytest.mark.asyncio
async def test_create_calls_api_correctly(
    jina_embedder: JinaAIEmbedder, mock_httpx_client: Any, mock_httpx_response: MagicMock
) -> None:
    mock_httpx_client.post.return_value = mock_httpx_response

    result = await jina_embedder.create("Test input")

    mock_httpx_client.post.assert_called_once()
    _, kwargs = mock_httpx_client.post.call_args
    assert kwargs["json"]["model"] == DEFAULT_EMBEDDING_MODEL
    assert kwargs["json"]["input"] == [{"text": "Test input"}]

    assert result == mock_httpx_response.json.return_value["data"][0]["embedding"][: jina_embedder.config.embedding_dim]


@pytest.mark.asyncio
async def test_create_batch_processes_multiple_inputs(
    jina_embedder: JinaAIEmbedder, mock_httpx_client: Any, mock_httpx_batch_response: MagicMock
) -> None:
    mock_httpx_client.post.return_value = mock_httpx_batch_response
    input_batch = ["Input 1", "Input 2", "Input 3"]

    result = await jina_embedder.create_batch(input_batch)

    mock_httpx_client.post.assert_called_once()
    _, kwargs = mock_httpx_client.post.call_args
    assert kwargs["json"]["model"] == DEFAULT_EMBEDDING_MODEL
    assert kwargs["json"]["input"] == [{"text": "Input 1"}, {"text": "Input 2"}, {"text": "Input 3"}]

    assert len(result) == 3
    expected = [
        mock_httpx_batch_response.json.return_value["data"][0]["embedding"][: jina_embedder.config.embedding_dim],
        mock_httpx_batch_response.json.return_value["data"][1]["embedding"][: jina_embedder.config.embedding_dim],
        mock_httpx_batch_response.json.return_value["data"][2]["embedding"][: jina_embedder.config.embedding_dim],
    ]
    assert result == expected


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
