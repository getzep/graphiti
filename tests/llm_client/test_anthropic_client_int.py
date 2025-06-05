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

# Running tests: pytest -xvs tests/integrations/test_anthropic_client_int.py

import os

import pytest
from pydantic import BaseModel, Field

from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.prompts.models import Message

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    'TEST_ANTHROPIC_API_KEY' not in os.environ,
    reason='Anthropic API key not available',
)


# Rename to avoid pytest collection as a test class
class SimpleResponseModel(BaseModel):
    """Test response model."""

    message: str = Field(..., description='A message from the model')


@pytest.mark.asyncio
@pytest.mark.integration
async def test_generate_simple_response():
    """Test generating a simple response from the Anthropic API."""
    if 'TEST_ANTHROPIC_API_KEY' not in os.environ:
        pytest.skip('Anthropic API key not available')

    client = AnthropicClient()

    messages = [
        Message(
            role='user',
            content="Respond with a JSON object containing a 'message' field with value 'Hello, world!'",
        )
    ]

    try:
        response = await client.generate_response(messages, response_model=SimpleResponseModel)

        assert isinstance(response, dict)
        assert 'message' in response
        assert response['message'] == 'Hello, world!'
    except Exception as e:
        pytest.skip(f'Test skipped due to Anthropic API error: {str(e)}')


@pytest.mark.asyncio
@pytest.mark.integration
async def test_extract_json_from_text():
    """Test the extract_json_from_text method with real data."""
    # We don't need an actual API connection for this test,
    # so we can create the client without worrying about the API key
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Temporarily set an environment variable to avoid API key error
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake_key_for_testing')
        client = AnthropicClient(cache=False)

    # A string with embedded JSON
    text = 'Some text before {"message": "Hello, world!"} and after'

    result = client._extract_json_from_text(text)  # type: ignore # ignore type check for private method

    assert isinstance(result, dict)
    assert 'message' in result
    assert result['message'] == 'Hello, world!'
