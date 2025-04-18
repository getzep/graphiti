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

import pytest

from graphiti_core.embedder.client import EMBEDDING_DIM


@pytest.fixture
def mock_embedding_values() -> list[float]:
    """Returns a list of mock embedding values with the default dimension.

    This can be used across different embedder tests to create consistent mock responses.
    """
    return [0.1] * EMBEDDING_DIM
