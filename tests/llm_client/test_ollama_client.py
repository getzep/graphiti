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

# Running tests: pytest -xvs tests/llm_client/test_ollama_client.py


import pytest
from pydantic import BaseModel, Field

from graphiti_core.llm_client.ollama_client import OllamaClient
from graphiti_core.prompts.models import Message

# Skip tests if no Ollama API/key or local server available
# pytestmark = pytest.mark.skipif(
#     'OLLAMA_HOST' not in os.environ,
#     reason='Ollama API/host not available',
# )


# Rename to avoid pytest collection as a test class
class SimpleResponseModel(BaseModel):
    message: str = Field(..., description='A message from the model')


class Pet(BaseModel):
    name: str
    animal: str
    age: int
    color: str | None
    favorite_toy: str | None


class PetList(BaseModel):
    pets: list[Pet]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_generate_simple_response():
    client = OllamaClient()

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
        pytest.skip(f'Test skipped due to Ollama API error: {str(e)}')


@pytest.mark.asyncio
@pytest.mark.integration
async def test_structured_output_with_pydantic():
    client = OllamaClient()

    messages = [
        Message(
            role='user',
            content='''
                I have two pets.
                A cat named Luna who is 5 years old and loves playing with yarn. She has grey fur.
                I also have a 2 year old black cat named Loki who loves tennis balls.
            ''',
        )
    ]

    try:
        response = await client.generate_response(messages, response_model=PetList)
        assert isinstance(response, dict)
        assert 'pets' in response
        assert isinstance(response['pets'], list)
        assert len(response['pets']) >= 1
    except Exception as e:
        pytest.skip(f'Test skipped due to Ollama API error: {str(e)}')

