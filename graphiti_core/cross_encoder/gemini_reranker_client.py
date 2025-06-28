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
import math
from typing import Any

from google import genai  # type: ignore
from google.genai import types  # type: ignore

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, RateLimitError
from .client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-2.5-flash-lite-preview-06-17'


class GeminiRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: genai.Client | None = None,
    ):
        """
        Initialize the GeminiRerankerClient with the provided configuration and client.

        This reranker uses the Gemini API to run a simple boolean classifier prompt concurrently
        for each passage. Log-probabilities are used to rank the passages, equivalent to the OpenAI approach.

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
        Rank passages based on their relevance to the query using Gemini.

        Uses log probabilities from boolean classification responses, equivalent to the OpenAI approach.
        The model responds with "True" or "False" and we use the log probabilities of these tokens.
        """
        gemini_messages_list: Any = [
            [
                types.Content(
                    role='system',
                    parts=[
                        types.Part.from_text(
                            text='You are an expert tasked with determining whether the passage is relevant to the query'
                        )
                    ],
                ),
                types.Content(
                    role='user',
                    parts=[
                        types.Part.from_text(
                            text=f"""Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
<PASSAGE>
{passage}
</PASSAGE>
<QUERY>
{query}
</QUERY>"""
                        )
                    ],
                ),
            ]
            for passage in passages
        ]

        try:
            responses = await semaphore_gather(
                *[
                    self.client.aio.models.generate_content(
                        model=self.config.model or DEFAULT_MODEL,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=1,
                            response_logprobs=True,
                            logprobs=5,  # Get top 5 candidate tokens for better coverage
                        ),
                    )
                    for gemini_messages in gemini_messages_list
                ]
            )

            scores: list[float] = []
            for response in responses:
                try:
                    # Check if we have logprobs result in the response
                    if (
                        hasattr(response, 'candidates')
                        and response.candidates
                        and len(response.candidates) > 0
                        and hasattr(response.candidates[0], 'logprobs_result')
                        and response.candidates[0].logprobs_result
                    ):
                        logprobs_result = response.candidates[0].logprobs_result

                        # Get the chosen candidates (tokens actually selected by the model)
                        if (
                            hasattr(logprobs_result, 'chosen_candidates')
                            and logprobs_result.chosen_candidates
                            and len(logprobs_result.chosen_candidates) > 0
                        ):
                            # Get the first token's log probability
                            first_token = logprobs_result.chosen_candidates[0]

                            if hasattr(first_token, 'log_probability') and hasattr(
                                first_token, 'token'
                            ):
                                # Convert log probability to probability (similar to OpenAI approach)
                                log_prob = first_token.log_probability
                                probability = math.exp(log_prob)

                                # Check if the token indicates relevance (starts with "True" or similar)
                                token_text = first_token.token.strip().lower()
                                if token_text.startswith(('true', 't')):
                                    scores.append(probability)
                                else:
                                    # For "False" or other tokens, use 1 - probability
                                    scores.append(1.0 - probability)
                            else:
                                # Fallback: try to get from top candidates
                                if (
                                    hasattr(logprobs_result, 'top_candidates')
                                    and logprobs_result.top_candidates
                                    and len(logprobs_result.top_candidates) > 0
                                ):
                                    top_step = logprobs_result.top_candidates[0]
                                    if (
                                        hasattr(top_step, 'candidates')
                                        and top_step.candidates
                                        and len(top_step.candidates) > 0
                                    ):
                                        # Look for "True" or "False" in top candidates
                                        true_prob = 0.0
                                        false_prob = 0.0

                                        for candidate in top_step.candidates:
                                            if hasattr(candidate, 'token') and hasattr(
                                                candidate, 'log_probability'
                                            ):
                                                token_text = candidate.token.strip().lower()
                                                prob = math.exp(candidate.log_probability)

                                                if token_text.startswith(('true', 't')):
                                                    true_prob = max(true_prob, prob)
                                                elif token_text.startswith(('false', 'f')):
                                                    false_prob = max(false_prob, prob)

                                        # Use the probability of "True" as the relevance score
                                        scores.append(true_prob)
                                    else:
                                        scores.append(0.0)
                                else:
                                    scores.append(0.0)
                        else:
                            scores.append(0.0)
                    else:
                        # Fallback: parse the response text if no logprobs available
                        if hasattr(response, 'text') and response.text:
                            response_text = response.text.strip().lower()
                            if response_text.startswith(('true', 't')):
                                scores.append(0.9)  # High confidence for "True"
                            elif response_text.startswith(('false', 'f')):
                                scores.append(0.1)  # Low confidence for "False"
                            else:
                                scores.append(0.0)
                        else:
                            scores.append(0.0)

                except (ValueError, AttributeError) as e:
                    logger.warning(f'Error parsing log probabilities from Gemini response: {e}')
                    scores.append(0.0)

            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
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
