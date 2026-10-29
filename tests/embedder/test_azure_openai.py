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

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from tests.embedder.embedder_fixtures import create_embedding_values

DEFAULT_MODEL = 'text-embedding-3-small'


def create_azure_embedding(multiplier: float = 0.1) -> MagicMock:
    """Create a mock Azure OpenAI embedding with the specified value multiplier."""
    mock_embedding = MagicMock()
    mock_embedding.embedding = create_embedding_values(multiplier)
    return mock_embedding


def create_azure_response(*multipliers: float) -> MagicMock:
    """Create a mock Azure OpenAI embeddings response with one entry per multiplier."""
    mock_response = MagicMock()
    mock_response.data = [create_azure_embedding(multiplier) for multiplier in multipliers]
    return mock_response


@pytest.fixture
def mock_azure_client() -> Any:
    """Create a mocked Azure OpenAI client.

    The client is injected through the constructor, so there is no module-level
    symbol to patch.
    """
    mock_client = MagicMock()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock()
    return mock_client


@pytest.fixture
def azure_embedder(mock_azure_client: Any) -> AzureOpenAIEmbedderClient:
    """Create an AzureOpenAIEmbedderClient with a mocked client."""
    return AzureOpenAIEmbedderClient(azure_client=mock_azure_client)


@pytest.mark.asyncio
async def test_create_wraps_a_string_input_in_a_list(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """A single string is sent to the API as a one-element list."""
    mock_azure_client.embeddings.create.return_value = create_azure_response(0.1)

    result = await azure_embedder.create('Test input')

    mock_azure_client.embeddings.create.assert_called_once()
    _, kwargs = mock_azure_client.embeddings.create.call_args
    assert kwargs['model'] == DEFAULT_MODEL
    assert kwargs['input'] == ['Test input']

    assert result == create_embedding_values(0.1)


@pytest.mark.asyncio
async def test_create_passes_a_list_of_strings_through(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """A list of strings is sent unchanged, and the first embedding is returned."""
    mock_azure_client.embeddings.create.return_value = create_azure_response(0.1, 0.2)

    result = await azure_embedder.create(['Input 1', 'Input 2'])

    _, kwargs = mock_azure_client.embeddings.create.call_args
    assert kwargs['input'] == ['Input 1', 'Input 2']

    assert result == create_embedding_values(0.1)


@pytest.mark.asyncio
async def test_create_coerces_a_non_string_input(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """Input that is neither a string nor a list of strings is stringified."""
    mock_azure_client.embeddings.create.return_value = create_azure_response(0.1)

    await azure_embedder.create(42)

    _, kwargs = mock_azure_client.embeddings.create.call_args
    assert kwargs['input'] == ['42']


@pytest.mark.asyncio
async def test_create_batch_returns_one_embedding_per_input(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """A batch is sent unchanged and yields one embedding per input, in order."""
    mock_azure_client.embeddings.create.return_value = create_azure_response(0.1, 0.2, 0.3)
    input_batch = ['Input 1', 'Input 2', 'Input 3']

    result = await azure_embedder.create_batch(input_batch)

    mock_azure_client.embeddings.create.assert_called_once()
    _, kwargs = mock_azure_client.embeddings.create.call_args
    assert kwargs['model'] == DEFAULT_MODEL
    assert kwargs['input'] == input_batch

    assert result == [
        create_embedding_values(0.1),
        create_embedding_values(0.2),
        create_embedding_values(0.3),
    ]


@pytest.mark.asyncio
async def test_create_batch_accepts_an_empty_batch(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """An empty batch yields an empty result rather than an error."""
    mock_azure_client.embeddings.create.return_value = create_azure_response()

    assert await azure_embedder.create_batch([]) == []


@pytest.mark.asyncio
async def test_embeddings_are_returned_at_full_length(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """This client returns whatever dimension the deployment produces.

    Unlike OpenAIEmbedder, it has no embedding_dim config and does not truncate,
    so the caller receives the vector as-is.
    """
    mock_azure_client.embeddings.create.return_value = create_azure_response(0.1)

    result = await azure_embedder.create('Test input')

    assert len(result) == len(create_embedding_values(0.1))


@pytest.mark.asyncio
async def test_a_custom_model_is_used(mock_azure_client: Any) -> None:
    """The configured deployment name is passed on both code paths."""
    embedder = AzureOpenAIEmbedderClient(
        azure_client=mock_azure_client, model='text-embedding-3-large'
    )
    mock_azure_client.embeddings.create.return_value = create_azure_response(0.1)

    await embedder.create('Test input')
    _, kwargs = mock_azure_client.embeddings.create.call_args
    assert kwargs['model'] == 'text-embedding-3-large'

    await embedder.create_batch(['Test input'])
    _, kwargs = mock_azure_client.embeddings.create.call_args
    assert kwargs['model'] == 'text-embedding-3-large'


@pytest.mark.asyncio
async def test_create_propagates_client_errors(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """An API failure is logged and re-raised, not swallowed into a bad vector."""
    mock_azure_client.embeddings.create.side_effect = RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await azure_embedder.create('Test input')


@pytest.mark.asyncio
async def test_create_batch_propagates_client_errors(
    azure_embedder: AzureOpenAIEmbedderClient, mock_azure_client: Any
) -> None:
    """An API failure during a batch is logged and re-raised."""
    mock_azure_client.embeddings.create.side_effect = RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await azure_embedder.create_batch(['Input 1', 'Input 2'])


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
