"""
Test file for OpenAIRerankerClient, specifically testing compatibility with 
both OpenAIClient and AzureOpenAILLMClient instances.

This test validates the fix for issue #1006 where OpenAIRerankerClient 
failed to properly support AzureOpenAILLMClient.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.openai_client import OpenAIClient


class MockAsyncOpenAI:
    """Mock AsyncOpenAI client for testing"""
    
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()


class MockAsyncAzureOpenAI:
    """Mock AsyncAzureOpenAI client for testing"""
    
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()


@pytest.fixture
def mock_openai_client():
    """Fixture to create a mocked OpenAIClient"""
    client = OpenAIClient(config=LLMConfig(api_key='test-key'))
    # Replace the internal client with our mock
    client.client = MockAsyncOpenAI()
    return client


@pytest.fixture 
def mock_azure_openai_client():
    """Fixture to create a mocked AzureOpenAILLMClient"""
    mock_azure = MockAsyncAzureOpenAI()
    client = AzureOpenAILLMClient(
        azure_client=mock_azure,
        config=LLMConfig(api_key='test-key')
    )
    return client


def test_openai_reranker_accepts_openai_client(mock_openai_client):
    """Test that OpenAIRerankerClient properly unwraps OpenAIClient"""
    # Create reranker with OpenAIClient
    reranker = OpenAIRerankerClient(client=mock_openai_client)
    
    # Verify the internal client is the unwrapped AsyncOpenAI instance
    assert reranker.client == mock_openai_client.client
    assert hasattr(reranker.client, 'chat')


def test_openai_reranker_accepts_azure_client(mock_azure_openai_client):
    """Test that OpenAIRerankerClient properly unwraps AzureOpenAILLMClient
    
    This test validates the fix for issue #1006.
    """
    # Create reranker with AzureOpenAILLMClient - this would fail before the fix
    reranker = OpenAIRerankerClient(client=mock_azure_openai_client)
    
    # Verify the internal client is the unwrapped AsyncAzureOpenAI instance
    assert reranker.client == mock_azure_openai_client.client
    assert hasattr(reranker.client, 'chat')


def test_openai_reranker_accepts_async_openai_directly():
    """Test that OpenAIRerankerClient accepts AsyncOpenAI directly"""
    # Create a mock AsyncOpenAI
    mock_async = MockAsyncOpenAI(api_key='test-key')
    
    # Create reranker with AsyncOpenAI directly
    reranker = OpenAIRerankerClient(client=mock_async)
    
    # Verify the internal client is used as-is
    assert reranker.client == mock_async
    assert hasattr(reranker.client, 'chat')


def test_openai_reranker_creates_default_client():
    """Test that OpenAIRerankerClient creates a default client when none provided"""
    config = LLMConfig(api_key='test-key')
    
    # Create reranker without client
    reranker = OpenAIRerankerClient(config=config)
    
    # Verify a client was created
    assert reranker.client is not None
    # The default should be an AsyncOpenAI instance
    from openai import AsyncOpenAI
    assert isinstance(reranker.client, AsyncOpenAI)


@pytest.mark.asyncio
async def test_rank_method_with_azure_client(mock_azure_openai_client):
    """Test that rank method works correctly with AzureOpenAILLMClient"""
    # Setup mock response for the chat completions
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            top_logprobs=[
                                SimpleNamespace(token='True', logprob=-0.5)
                            ]
                        )
                    ]
                )
            )
        ]
    )
    
    mock_azure_openai_client.client.chat.completions.create.return_value = mock_response
    
    # Create reranker with AzureOpenAILLMClient
    reranker = OpenAIRerankerClient(client=mock_azure_openai_client)
    
    # Test ranking
    query = "test query"
    passages = ["passage 1"]
    
    # This would previously fail with AttributeError before the fix
    results = await reranker.rank(query, passages)
    
    # Verify the method was called
    assert mock_azure_openai_client.client.chat.completions.create.called
    assert len(results) == 1
    assert results[0][0] == "passage 1"
