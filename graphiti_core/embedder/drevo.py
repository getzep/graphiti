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

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .openai import OpenAIEmbedder, OpenAIEmbedderConfig

# drevo exposes an OpenAI-compatible `/v1/embeddings` route on its HTTP port
# (drevo#217). The embeddings endpoint requires a drevo build compiled with the
# `embeddings-proxy` cargo feature; a build without it answers /v1/embeddings 404.
DEFAULT_DREVO_EMBEDDER_BASE_URL = 'http://localhost:8080/v1'
# drevo ignores/forwards the key depending on how its proxy is configured, but the
# AsyncOpenAI client requires a non-empty one.
DEFAULT_DREVO_EMBEDDER_API_KEY = 'drevo'


class DrevoEmbedder(OpenAIEmbedder):
    """Embedder backed by drevo's OpenAI-compatible ``/v1/embeddings`` endpoint.

    A thin preset over :class:`OpenAIEmbedder` that points at drevo's local HTTP
    endpoint. drevo either proxies an upstream embedder (OpenAI / Voyage
    passthrough) or self-hosts one, so a single drevo instance can back graphiti's
    graph, vector search, and embedding generation.

    * ``config.base_url`` defaults to drevo's local ``/v1`` (override for a remote
      drevo or a non-default port).
    * ``config.api_key`` defaults to a placeholder. When drevo proxies an upstream
      that needs a key, set this to the upstream key — drevo forwards it.
    * ``config.embedding_model`` is whatever drevo's configured backend serves
      (e.g. ``text-embedding-3-small`` for an OpenAI passthrough).

    Note: requires a drevo built with the ``embeddings-proxy`` feature; otherwise
    ``/v1/embeddings`` returns 404.
    """

    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        if config is None:
            config = OpenAIEmbedderConfig()
        if not config.base_url:
            config.base_url = DEFAULT_DREVO_EMBEDDER_BASE_URL
        if not config.api_key:
            config.api_key = DEFAULT_DREVO_EMBEDDER_API_KEY

        super().__init__(config=config, client=client)
