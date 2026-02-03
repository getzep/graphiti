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

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .client import CrossEncoderClient

if TYPE_CHECKING:
    import voyageai
else:
    try:
        import voyageai
    except ImportError:
        raise ImportError(
            'voyageai is required for VoyageRerankerClient. '
            'Install it with: pip install graphiti-core[voyageai]'
        ) from None

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = 'rerank-2'


class VoyageRerankerConfig(BaseModel):
    """Configuration for Voyage Reranker client."""

    model: str = Field(default=DEFAULT_RERANK_MODEL)
    api_key: str | None = None
    top_k: int | None = None  # If set, only return top_k results
    truncation: bool = True  # Truncate inputs that exceed context length


class VoyageRateLimitError(Exception):
    """Raised when Voyage API rate limit is exceeded."""

    pass


class VoyageRerankerClient(CrossEncoderClient):
    """
    Voyage AI Reranker Client.

    Uses Voyage's rerank API to rank passages by relevance to a query.
    Models available:
    - rerank-2.5: Latest, best quality
    - rerank-2.5-lite: Faster, cheaper
    - rerank-2: Good quality ($0.05/1M tokens)
    - rerank-2-lite: Faster, cheaper ($0.02/1M tokens)
    """

    def __init__(self, config: VoyageRerankerConfig | None = None):
        """
        Initialize the VoyageRerankerClient.

        Args:
            config: Configuration including API key and model selection.
                   If not provided, uses defaults and VOYAGE_API_KEY env var.
        """
        if config is None:
            config = VoyageRerankerConfig()
        self.config = config
        self.client = voyageai.AsyncClient(api_key=config.api_key)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages by relevance to the query using Voyage rerank API.

        Args:
            query: The query string to rank against.
            passages: List of passages to rank.

        Returns:
            List of (passage, score) tuples sorted by relevance (descending).
        """
        if not passages:
            return []

        try:
            result = await self.client.rerank(
                query=query,
                documents=passages,
                model=self.config.model,
                top_k=self.config.top_k,
                truncation=self.config.truncation,
            )

            # Build results from rerank response
            # result.results contains RerankingResult objects with index and relevance_score
            ranked_results: list[tuple[str, float]] = []
            for item in result.results:
                passage = passages[item.index]
                score = item.relevance_score
                ranked_results.append((passage, score))

            # Results are already sorted by relevance_score descending
            return ranked_results

        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors
            if any(
                term in error_str
                for term in ['rate limit', 'rate_limit', '429', 'too many requests', 'quota']
            ):
                logger.warning(f'Voyage rate limit exceeded: {e}')
                raise VoyageRateLimitError(f'Rate limit exceeded: {e}') from e
            logger.error(f'Error in Voyage rerank: {e}')
            raise
