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

from graphiti_core.embedder.drevo import DEFAULT_DREVO_EMBEDDER_BASE_URL, DrevoEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig

DREVO_EMBEDDER_BASE_URL = os.getenv('DREVO_EMBEDDER_BASE_URL', DEFAULT_DREVO_EMBEDDER_BASE_URL)
DREVO_EMBEDDER_API_KEY = os.getenv('DREVO_EMBEDDER_API_KEY', 'drevo')
DREVO_EMBEDDER_MODEL = os.getenv('DREVO_EMBEDDER_MODEL', 'text-embedding-3-small')


@pytest.mark.integration
@pytest.mark.asyncio
async def test_drevo_embeddings_int():
    """Embed text via a live drevo `/v1/embeddings` endpoint.

    Skips unless drevo was built with the `embeddings-proxy` feature and reachable
    (a build without it answers /v1/embeddings 404, and a proxy needs a configured
    upstream key).
    """
    embedder = DrevoEmbedder(
        config=OpenAIEmbedderConfig(
            base_url=DREVO_EMBEDDER_BASE_URL,
            api_key=DREVO_EMBEDDER_API_KEY,
            embedding_model=DREVO_EMBEDDER_MODEL,
        )
    )

    try:
        vector = await embedder.create('graphiti drevo embedder integration test')
    except Exception as e:  # pragma: no cover - depends on a live drevo build
        pytest.skip(f'drevo /v1/embeddings unavailable at {DREVO_EMBEDDER_BASE_URL}: {e}')

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)
