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

# Running tests: pytest -xvs tests/embedder/test_client.py
# Running tests with coverage: pytest -xvs tests/embedder/test_client.py --cov=graphiti_core.embedder.client --cov-report=term-missing

from typing import Any

import pytest

from graphiti_core.embedder.client import EMBEDDING_DIM, EmbedderClient, EmbedderConfig


def test_embedder_config_defaults() -> None:
    """Test that EmbedderConfig defaults are set correctly"""
    config = EmbedderConfig()
    assert config.embedding_dim == EMBEDDING_DIM


def test_embedder_config_frozen_field() -> None:
    """Test that embedding_dim is frozen and can't be changed after initialization"""
    config = EmbedderConfig(embedding_dim=512)
    assert config.embedding_dim == 512

    # Attempting to change the frozen field should raise an error
    with pytest.raises(ValueError):
        config.embedding_dim = 256


def test_embedder_client_is_abstract() -> None:
    """Test that EmbedderClient cannot be instantiated directly"""
    with pytest.raises(TypeError):
        EmbedderClient()  # Should fail as it's an abstract class # type: ignore[abstract]


class MinimalEmbedderClient(EmbedderClient):
    """Minimal implementation of EmbedderClient for testing inheritance"""

    def __init__(self, mock_values: list[float]) -> None:
        self.mock_values = mock_values

    async def create(self, input_data: Any) -> list[float]:
        return self.mock_values


@pytest.mark.asyncio
async def test_embedder_client_can_be_subclassed(mock_embedding_values: list[float]) -> None:
    """Test that EmbedderClient can be subclassed with a concrete implementation"""
    client = MinimalEmbedderClient(mock_embedding_values)
    result = await client.create('test')

    assert isinstance(result, list)
    assert len(result) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in result)
