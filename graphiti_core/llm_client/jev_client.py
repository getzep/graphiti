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

import asyncio
import json
import os
from typing import Any

import httpx
from pydantic import BaseModel

from graphiti_core.prompts.models import Message

from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .decision_prompts import (
    CON_RULES,
    DUP_RULES,
    NODE_NONE,
    NODE_RULES,
    RULES_HEADER,
    _contra_strict,
    _encode_payload,
    _jev_edge_builder,
    _jev_node_builder,
    _user_text,
    assemble_edge_answers,
    assemble_node_answers,
    edge_e2,
    jev_prompt_overrides,
    node_n6,
    parse_list,
    shrink_state,
    tag,
)
from .errors import RateLimitError

_sleep = asyncio.sleep

__all__ = [
    'CON_RULES',
    'DUP_RULES',
    'JevClient',
    'JevInputTooLongError',
    'NODE_NONE',
    'NODE_RULES',
    'RULES_HEADER',
    '_contra_strict',
    '_encode_payload',
    '_jev_edge_builder',
    '_jev_node_builder',
    '_user_text',
    'assemble_edge_answers',
    'assemble_node_answers',
    'edge_e2',
    'jev_prompt_overrides',
    'node_n6',
    'parse_list',
    'shrink_state',
    'tag',
]


class JevInputTooLongError(Exception):
    """Raised when Jev cannot fit a single question after state shrinking."""


class _JevInputTooLongSignal(Exception):
    pass


class JevClient(LLMClient):
    """LLM client for Jev node dedupe and edge resolution."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        base_url: str = 'https://api.typesafe.ai/v1/systemone',
        duplicate_threshold: float = 0.7,
        contradiction_threshold: float = 0.8,
        timeout_seconds: float = 120.0,
        max_retries: int = 5,
        http_client: httpx.AsyncClient | None = None,
    ):
        source_config = config or LLMConfig()
        api_key = source_config.api_key or os.getenv('JEV_API_KEY')
        if not api_key:
            raise ValueError('JEV_API_KEY must be set or provided in LLMConfig.api_key')
        resolved_config = LLMConfig(
            api_key=api_key,
            model=source_config.model or 'jev-latest',
            base_url=source_config.base_url,
            temperature=source_config.temperature,
            max_tokens=source_config.max_tokens,
            small_model=source_config.small_model,
        )
        super().__init__(resolved_config, cache=False)
        self.api_key = api_key
        self.base_url = base_url
        self.duplicate_threshold = duplicate_threshold
        self.contradiction_threshold = contradiction_threshold
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.http_client = (
            http_client if http_client is not None else httpx.AsyncClient(timeout=timeout_seconds)
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError('JevClient requires a dedupe prompt_name through LLMRuntime')

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        node_prompt = 'dedupe_nodes.nodes'
        edge_prompt = 'dedupe_edges.resolve_edge'
        if prompt_name not in (node_prompt, edge_prompt):
            raise ValueError(
                f'JevClient cannot serve prompt {prompt_name!r}; it serves only '
                'dedupe_nodes.nodes and dedupe_edges.resolve_edge via jev_prompt_overrides().'
            )

        payload = json.loads(_user_text(messages))
        state = payload['state']
        questions = payload['questions']
        model_id = model or self.model
        if model_id is None:
            raise ValueError('Jev model must be set')
        entities = parse_list(tag(state, 'ENTITIES')) if prompt_name == node_prompt else []
        existing_indices = (
            sorted(int(name[3:]) for name in questions if name.startswith('dup'))
            if prompt_name == edge_prompt
            else []
        )
        contradiction_indices = (
            sorted(int(name[3:]) for name in questions if name.startswith('con'))
            if prompt_name == edge_prompt
            else []
        )
        usage = {'input_tokens': 0, 'output_tokens': 0}

        if max_tokens is None:
            max_tokens = self.max_tokens

        with self.tracer.start_span('llm.generate') as span:
            span.add_attributes(
                {
                    'llm.provider': 'jev',
                    'model.size': model_size.value,
                    'max_tokens': max_tokens,
                    'cache.enabled': self.cache_enabled,
                    'prompt.name': prompt_name,
                    'cache.hit': False,
                }
            )
            try:
                answers = (
                    await self._call_jev_fit(state, questions, model_id, usage) if questions else {}
                )
                if prompt_name == node_prompt:
                    return assemble_node_answers(answers, entities)
                return assemble_edge_answers(
                    answers,
                    existing_indices,
                    contradiction_indices,
                    self.duplicate_threshold,
                    self.contradiction_threshold,
                )
            except Exception as error:
                span.set_status('error', str(error))
                span.record_exception(error)
                raise
            finally:
                self.token_tracker.record(
                    prompt_name, usage['input_tokens'], usage['output_tokens']
                )

    async def _post(
        self, state: str, questions: dict[str, dict[str, Any]], model: str
    ) -> dict[str, Any]:
        request_body = {'model': model, 'state': state, 'questions': questions}
        last_error: Exception | None = None
        retry_statuses = (429, 500, 502, 503, 504)

        for attempt in range(self.max_retries):
            try:
                response = await self.http_client.post(
                    self.base_url,
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json',
                    },
                    json=request_body,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 400 and 'max_tokens_exceeded' in response.text:
                    raise _JevInputTooLongSignal(response.text)
                if response.status_code in retry_statuses or not 200 <= response.status_code < 300:
                    error = httpx.HTTPStatusError(
                        f'HTTP {response.status_code}: {response.text[:500]}',
                        request=response.request,
                        response=response,
                    )
                    raise error
                return response.json()
            except _JevInputTooLongSignal:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = error
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in retry_statuses:
                    raise
                last_error = error

            if attempt < self.max_retries - 1:
                await _sleep(2**attempt)

        if isinstance(last_error, httpx.HTTPStatusError) and last_error.response.status_code == 429:
            raise RateLimitError(str(last_error)) from last_error
        if last_error is not None:
            raise last_error
        raise RuntimeError('Jev request did not run; max_retries must be greater than zero')

    async def _call_jev_fit(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str,
        usage: dict[str, int],
    ) -> dict[str, Any]:
        try:
            response = await self._post(state, questions, model)
        except _JevInputTooLongSignal as error:
            keys = list(questions)
            if len(keys) > 1:
                middle = len(keys) // 2
                first = await self._call_jev_fit(
                    state, {key: questions[key] for key in keys[:middle]}, model, usage
                )
                second = await self._call_jev_fit(
                    state, {key: questions[key] for key in keys[middle:]}, model, usage
                )
                return {**first, **second}
            smaller = shrink_state(state)
            if smaller is None:
                raise JevInputTooLongError(
                    'Jev could not fit one question after shrinking the state.'
                ) from error
            return await self._call_jev_fit(smaller, questions, model, usage)

        response_usage = response.get('usage') or {}
        usage['input_tokens'] += int(response_usage.get('input_tokens', 0))
        usage['output_tokens'] += int(response_usage.get('output_tokens', 0))
        return response['answers']
