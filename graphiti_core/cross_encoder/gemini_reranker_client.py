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
import re
from typing import TYPE_CHECKING

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, RateLimitError
from .client import CrossEncoderClient

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
else:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            'google-genai is required for GeminiRerankerClient. '
            'Install it with: pip install graphiti-core[google-genai]'
        ) from None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-2.5-flash-lite-preview-06-17'


class GeminiRerankerClient(CrossEncoderClient):
    """
    Google Gemini Reranker Client
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: 'genai.Client | None' = None,
    ):
        """
        Initialize the GeminiRerankerClient with the provided configuration and client.

        The Gemini Developer API does not yet support logprobs. Unlike the OpenAI reranker,
        this reranker uses the Gemini API to perform direct relevance scoring of passages.
        Each passage is scored individually on a 0-100 scale.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            client (genai.Client | None): An optional async client instance to use. If not provided, a new genai.Client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            self.client = genai.Client(api_key=config.api_key)
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages based on their relevance to the query using direct scoring.

        Each passage is scored individually on a 0-100 scale, then normalized to [0,1].
        """
        if len(passages) <= 1:
            return [(passage, 1.0) for passage in passages]

        # Generate scoring prompts for each passage
        scoring_prompts = []
        for passage in passages:
            prompt = f"""Rate how well this passage answers or relates to the query. Use a scale from 0 to 100.

Query: {query}

Passage: {passage}

Provide only a number between 0 and 100 (no explanation, just the number):"""

            scoring_prompts.append(
                [
                    types.Content(
                        role='user',
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]
            )

        try:
            # Execute all scoring requests concurrently - O(n) API calls
            responses = await semaphore_gather(
                *[
                    self.client.aio.models.generate_content(
                        model=self.config.model or DEFAULT_MODEL,
                        contents=prompt_messages,  # type: ignore
                        config=types.GenerateContentConfig(
                            system_instruction='You are an expert at rating passage relevance. Respond with only a number from 0-100.',
                            temperature=0.0,
                            max_output_tokens=3,
                        ),
                    )
                    for prompt_messages in scoring_prompts
                ]
            )

            # Extract scores and create results
            results = []
            for passage, response in zip(passages, responses, strict=True):
                try:
                    if hasattr(response, 'text') and response.text:
                        # Extract numeric score from response
                        score_text = response.text.strip()
                        # Handle cases where model might return non-numeric text
                        score_match = re.search(r'\b(\d{1,3})\b', score_text)
                        if score_match:
                            score = float(score_match.group(1))
                            # Normalize to [0, 1] range and clamp to valid range
                            normalized_score = max(0.0, min(1.0, score / 100.0))
                            results.append((passage, normalized_score))
                        else:
                            logger.warning(
                                f'Could not extract numeric score from response: {score_text}'
                            )
                            results.append((passage, 0.0))
                    else:
                        logger.warning('Empty response from Gemini for passage scoring')
                        results.append((passage, 0.0))
                except (ValueError, AttributeError) as e:
                    logger.warning(f'Error parsing score from Gemini response: {e}')
                    results.append((passage, 0.0))

            # Sort by score in descending order (highest relevance first)
            results.sort(reverse=True, key=lambda x: x[1])
            return results

        except Exception as e:
            # Check if it's a rate limit error based on Gemini API error codes
            error_message = str(e).lower()
            if (
                'rate limit' in error_message
                or 'quota' in error_message
                or 'resource_exhausted' in error_message
                or '429' in str(e)
            ):
                raise RateLimitError from e

            logger.error(f'Error in generating LLM response: {e}')
            raise
