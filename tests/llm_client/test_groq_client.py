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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import DEFAULT_MODEL, GroqClient
from graphiti_core.prompts.models import Message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config_model', 'request_model', 'expected_model'),
    [
        (None, None, DEFAULT_MODEL),
        ('configured-model', None, 'configured-model'),
        ('configured-model', 'request-model', 'request-model'),
    ],
)
async def test_generate_response_selects_expected_model(
    config_model: str | None,
    request_model: str | None,
    expected_model: str,
):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
    )
    with patch('graphiti_core.llm_client.groq_client.AsyncGroq') as mock_groq:
        mock_groq.return_value.chat.completions.create = AsyncMock(return_value=response)
        client = GroqClient(config=LLMConfig(model=config_model))

        result = await client._generate_response(
            [Message(role='user', content='Return JSON.')], model=request_model
        )

    assert result == {'ok': True}
    call_kwargs = mock_groq.return_value.chat.completions.create.call_args.kwargs
    assert call_kwargs['model'] == expected_model
    assert call_kwargs['response_format'] == {'type': 'json_object'}
