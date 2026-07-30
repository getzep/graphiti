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

import os

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient
from graphiti_core.prompts.models import Message

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', DEFAULT_OLLAMA_BASE_URL)


class Person(BaseModel):
    name: str
    age: int


async def _first_available_model(client: OllamaClient) -> str:
    """Return a model served by the local Ollama, or skip if none is pulled."""
    try:
        models = await client.client.models.list()
    except Exception as e:  # pragma: no cover - depends on a live Ollama
        pytest.skip(f'Ollama not reachable at {OLLAMA_BASE_URL}: {e}')
    ids = [m.id for m in models.data]
    if not ids:
        pytest.skip('Ollama is running but has no models pulled (`ollama pull <model>`)')
    return ids[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_structured_generation_int():
    """Exercise the json_object fallback against a live Ollama and parse the result."""
    client = OllamaClient(config=LLMConfig(base_url=OLLAMA_BASE_URL))
    model = await _first_available_model(client)
    client.model = model

    messages = [
        Message(role='system', content='You extract structured data. Reply with JSON only.'),
        Message(
            role='user',
            content='Extract the person: "Ada Lovelace was 36." '
            'Return keys name (string) and age (integer).',
        ),
    ]

    result = await client.generate_response(messages, response_model=Person)

    assert isinstance(result, dict)
    assert 'name' in result and 'age' in result
    # Validate the model actually produced schema-conforming content.
    person = Person.model_validate(result)
    assert 'ada' in person.name.lower()
