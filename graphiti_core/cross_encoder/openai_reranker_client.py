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
import os
import re
from typing import Any

import numpy as np
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, OpenAIClient, RateLimitError
from ..prompts import Message
from .client import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-nano'


class OpenAIRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | OpenAIClient | None = None,
    ):
        """
        Initialize the OpenAIRerankerClient with the provided configuration and client.

        This reranker uses the OpenAI API to run a simple boolean classifier prompt concurrently
        for each passage. Log-probabilities are used to rank the passages.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            client (AsyncOpenAI | AsyncAzureOpenAI | OpenAIClient | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        elif isinstance(client, OpenAIClient):
            self.client = client.client
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        # RERANKER_MODE=listwise scores all passages in ONE completion
        # instead of one boolean-classifier call per passage. Read at call
        # time (not import time) so a container env flip needs no reload.
        if os.getenv('RERANKER_MODE', 'pointwise') == 'listwise' and len(passages) > 1:
            return await self._rank_listwise(query, passages)
        openai_messages_list: Any = [
            [
                Message(
                    role='system',
                    content='You are an expert tasked with determining whether the passage is relevant to the query',
                ),
                Message(
                    role='user',
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]
        try:
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=self.config.model or DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=0,
                        max_tokens=1,
                        logit_bias={'6432': 1, '7983': 1},
                        logprobs=True,
                        top_logprobs=2,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            responses_top_logprobs = [
                response.choices[0].logprobs.content[0].top_logprobs
                if response.choices[0].logprobs is not None
                and response.choices[0].logprobs.content is not None
                else []
                for response in responses
            ]
            scores: list[float] = []
            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    continue
                norm_logprobs = np.exp(top_logprobs[0].logprob)
                if top_logprobs[0].token.strip().split(' ')[0].lower() == 'true':
                    scores.append(norm_logprobs)
                else:
                    scores.append(1 - norm_logprobs)

            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
            results.sort(reverse=True, key=lambda x: x[1])
            return results
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def _rank_listwise(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Rank all passages with a single completion.

        The model returns passage numbers in relevance order. Malformed
        output degrades gracefully: out-of-range and duplicate numbers are
        dropped, omitted passages keep their original relative order at the
        tail. Scores are rank-based pseudo-scores in (0, 1] (descending),
        compatible with the callers' `score >= reranker_min_score` filter
        (default 0) but NOT calibrated probabilities like pointwise mode.
        """
        max_chars = int(os.getenv('RERANKER_LISTWISE_PASSAGE_CHARS', 4000))
        numbered = '\n\n'.join(
            f'[{i + 1}] {passage[:max_chars]}' for i, passage in enumerate(passages)
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model or DEFAULT_MODEL,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert reranker: order passages by relevance to a query.',
                    },
                    {
                        'role': 'user',
                        'content': f"""
                               Rank the passages below by relevance to QUERY, most relevant first.
                               Respond with ONLY the passage numbers in ranked order, comma-separated
                               (example: 3,1,2). Include every number from 1 to {len(passages)} exactly once.
                               <QUERY>
                               {query}
                               </QUERY>
                               <PASSAGES>
                               {numbered}
                               </PASSAGES>
                               """,
                    },
                ],
                temperature=0,
                max_tokens=max(200, 5 * len(passages)),
            )
        except openai.RateLimitError as e:
            raise RateLimitError from e

        content = response.choices[0].message.content or ''
        order: list[int] = []
        seen: set[int] = set()
        for token in re.findall(r'\d+', content):
            idx = int(token) - 1
            if 0 <= idx < len(passages) and idx not in seen:
                seen.add(idx)
                order.append(idx)
        for idx in range(len(passages)):
            if idx not in seen:
                order.append(idx)
        n = len(passages)
        return [(passages[idx], (n - rank) / n) for rank, idx in enumerate(order)]
