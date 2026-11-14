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

# Running tests: pytest -xvs tests/llm_client/test_amazon_bedrock_client_int.py
# Requires: AWS credentials configured and Bedrock model access

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.amazon_bedrock_client import AmazonBedrockLLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


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


class ResponseModel(BaseModel):
    """Test model for structured output."""

    answer: str
    confidence: float


@pytest.fixture
def bedrock_client():
    """Create a real AmazonBedrockLLMClient for integration testing."""
    config = LLMConfig(
        model='us.anthropic.claude-sonnet-4-20250514-v1:0', temperature=0.1, max_tokens=100
    )
    return AmazonBedrockLLMClient(config=config, region='us-east-1')


class TestAmazonBedrockLLMClientIntegration:
    """Integration tests for AmazonBedrockLLMClient with real AWS Bedrock."""

    @pytest.mark.asyncio
    async def test_simple_text_generation(self, bedrock_client):
        """Test basic text generation without structured output."""
        messages = [
            Message(role='system', content='You are a helpful assistant.'),
            Message(role='user', content='What is 2+2? Answer briefly.'),
        ]

        result = await bedrock_client._generate_response(messages)

        assert isinstance(result, dict)
        assert 'content' in result
        assert isinstance(result['content'], str)
        assert len(result['content']) > 0
        assert '4' in result['content']

    @pytest.mark.asyncio
    async def test_structured_output_generation(self, bedrock_client):
        """Test structured output generation with Pydantic model."""
        messages = [
            Message(role='system', content='You are a helpful assistant.'),
            Message(
                role='user', content='What is the capital of France? Provide your confidence level.'
            ),
        ]

        result = await bedrock_client._generate_response(messages, response_model=ResponseModel)

        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'confidence' in result
        assert isinstance(result['answer'], str)
        assert isinstance(result['confidence'], float)
        assert 'Paris' in result['answer']
        assert 0.0 <= result['confidence'] <= 1.0

    @pytest.mark.asyncio
    async def test_different_regions(self):
        """Test client works in different supported regions."""
        # Test with a different region
        config = LLMConfig(
            model='us.anthropic.claude-sonnet-4-20250514-v1:0', temperature=0.1, max_tokens=50
        )
        client = AmazonBedrockLLMClient(config=config, region='us-west-2')

        messages = [Message(role='user', content='Say hello in one word.')]

        result = await client._generate_response(messages)

        assert isinstance(result, dict)
        assert 'content' in result
        assert len(result['content']) > 0


if __name__ == '__main__':
    pytest.main(['-v', 'test_amazon_bedrock_client_int.py'])
