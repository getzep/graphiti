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
from typing import Any, TypeGuard

import httpx
from pydantic import BaseModel

from graphiti_core.prompts.lib import ChatPromptLibrary
from graphiti_core.prompts.models import Message

from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .decision_prompts import (
    _user_text,
    assemble_edge_answers,
    assemble_node_answers,
    jev_prompt_overrides,
    parse_list,
    shrink_state,
    tag,
)
from .errors import RateLimitError
from .prompt_config import LLMPromptOverrides

_sleep = asyncio.sleep


class OpenAIDecisionsInputTooLongError(Exception):
    """Raised when one Decisions question cannot fit after state shrinking."""


class _OpenAIInputTooLongSignal(Exception):
    pass


class OpenAIDecisionsClient(LLMClient):
    """LLM client for OpenAI Decisions node dedupe and edge resolution."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        base_url: str | None = None,
        duplicate_threshold: float = 0.7,
        contradiction_threshold: float = 0.8,
        timeout_seconds: float = 120.0,
        max_retries: int = 5,
        http_client: httpx.AsyncClient | None = None,
    ):
        source_config = config or LLMConfig()
        api_key = source_config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY must be set or provided in LLMConfig.api_key')
        resolved_config = LLMConfig(
            api_key=api_key,
            model=source_config.model or 'gpt-6-luna',
            base_url=source_config.base_url,
            temperature=source_config.temperature,
            max_tokens=source_config.max_tokens,
            small_model=source_config.small_model,
        )
        super().__init__(resolved_config, cache=False)
        self.api_key = api_key
        self.base_url = (
            base_url
            or source_config.base_url
            or os.getenv('OPENAI_BASE_URL')
            or 'https://api.openai.com/v1'
        ).rstrip('/')
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
        raise NotImplementedError(
            'OpenAIDecisionsClient requires a dedupe prompt_name through LLMRuntime'
        )

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
                f'OpenAIDecisionsClient cannot serve prompt {prompt_name!r}; it serves only '
                'dedupe_nodes.nodes and dedupe_edges.resolve_edge via '
                'openai_decisions_prompt_overrides().'
            )

        payload = json.loads(_user_text(messages))
        instructions = payload.get('instructions', '')
        state = payload['state']
        questions = payload['questions']
        if not isinstance(instructions, str) or not isinstance(state, str):
            raise ValueError('Decision prompt instructions and state must be strings')
        if not isinstance(questions, dict):
            raise ValueError('Decision prompt questions must be an object')
        model_id = model or self.model
        if model_id is None:
            raise ValueError('OpenAI Decisions model must be set')

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
                    'llm.provider': 'openai',
                    'model.size': model_size.value,
                    'max_tokens': max_tokens,
                    'cache.enabled': self.cache_enabled,
                    'prompt.name': prompt_name,
                    'cache.hit': False,
                }
            )
            try:
                answers = (
                    await self._call_fit(instructions, state, questions, model_id, usage)
                    if questions
                    else {}
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
        self,
        instructions: str,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str,
    ) -> dict[str, Any]:
        input_text = '\n\n'.join(
            part.rstrip('\n') for part in (instructions, state) if part.strip()
        )
        wire_questions = []
        for name, question in questions.items():
            question_type = question.get('type')
            if question_type == 'choice':
                criteria = question.get('criteria')
                if not isinstance(criteria, dict):
                    raise ValueError(f'Choice question {name!r} must have criteria')
                wire_questions.append(
                    {
                        'name': name,
                        'type': 'choice',
                        'instructions': question.get('instructions') or '',
                        'choices': [
                            {'value': str(value), 'description': str(description)}
                            for value, description in criteria.items()
                        ],
                    }
                )
            elif question_type in ('predicate', 'noul'):
                wire_questions.append(
                    {
                        'name': name,
                        'type': 'predicate',
                        'instructions': question.get('instructions') or '',
                    }
                )
            else:
                raise ValueError(f'Unsupported decision question type {question_type!r}')

        request_body = {'model': model, 'input': input_text, 'questions': wire_questions}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self.http_client.post(
                    f'{self.base_url}/decisions',
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json',
                    },
                    json=request_body,
                    timeout=self.timeout_seconds,
                )
                if self._is_input_too_long(response):
                    raise _OpenAIInputTooLongSignal(response.text)
                if not 200 <= response.status_code < 300:
                    raise httpx.HTTPStatusError(
                        f'HTTP {response.status_code}: {response.text[:500]}',
                        request=response.request,
                        response=response,
                    )
                result = response.json()
                answers = self._map_answers(questions, result.get('answers'))
                usage = result.get('usage') or {}
                return {
                    'answers': answers,
                    'usage': {
                        'input_tokens': int(usage.get('input_tokens', 0)),
                        'output_tokens': int(usage.get('output_tokens', 0)),
                    },
                }
            except _OpenAIInputTooLongSignal:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = error
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 429 and error.response.status_code < 500:
                    raise
                last_error = error

            if attempt < self.max_retries - 1:
                await _sleep(2**attempt)

        if isinstance(last_error, httpx.HTTPStatusError) and last_error.response.status_code == 429:
            raise RateLimitError(str(last_error)) from last_error
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            'OpenAI Decisions request did not run; max_retries must be greater than zero'
        )

    @staticmethod
    def _is_input_too_long(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        try:
            error = response.json().get('error', {})
        except (ValueError, AttributeError):
            return False
        if not isinstance(error, dict):
            return False
        if error.get('code') in {'context_length_exceeded', 'string_above_max_length'}:
            return True
        message = error.get('message')
        return (
            error.get('type') == 'invalid_request_error'
            and isinstance(message, str)
            and 'exceeds the token limit' in message.lower()
        )

    @classmethod
    def _map_answers(cls, questions: dict[str, dict[str, Any]], raw_answers: Any) -> dict[str, Any]:
        if not isinstance(raw_answers, list):
            raise ValueError('OpenAI Decisions response must contain an answers list')
        mapped: dict[str, Any] = {}
        for answer in raw_answers:
            if not isinstance(answer, dict):
                raise ValueError('OpenAI Decisions answer must be an object')
            name = answer.get('name')
            if not isinstance(name, str) or name not in questions:
                raise ValueError(f'OpenAI Decisions returned an unknown answer name {name!r}')
            if name in mapped:
                raise ValueError(f'OpenAI Decisions returned duplicate answer {name!r}')
            question = questions[name]
            expected_type = 'choice' if question.get('type') == 'choice' else 'predicate'
            if answer.get('type') != expected_type:
                raise ValueError(
                    f'OpenAI Decisions answer {name!r} has type {answer.get("type")!r}; '
                    f'expected {expected_type!r}'
                )
            if expected_type == 'predicate':
                probability = answer.get('probability')
                if not cls._valid_probability(probability):
                    raise ValueError(
                        f'OpenAI Decisions predicate answer {name!r} needs a valid probability'
                    )
                mapped[name] = {'noul': float(probability)}
                continue

            criteria = question.get('criteria')
            if not isinstance(criteria, dict):
                raise ValueError(f'Choice question {name!r} must have criteria')
            allowed = {str(value) for value in criteria}
            choice = answer.get('choice')
            if choice is not None and (not isinstance(choice, str) or choice not in allowed):
                raise ValueError(
                    f'OpenAI Decisions choice {choice!r} is not allowed for answer {name!r}'
                )
            probabilities = answer.get('probabilities')
            if isinstance(probabilities, list) and probabilities:
                mapped_probabilities: dict[str, float] = {}
                for item in probabilities:
                    if not isinstance(item, dict):
                        raise ValueError(
                            f'OpenAI Decisions probability for {name!r} must be an object'
                        )
                    value = item.get('value')
                    probability = item.get('probability')
                    if not isinstance(value, str) or value not in allowed:
                        raise ValueError(
                            f'OpenAI Decisions choice {value!r} is not allowed for answer {name!r}'
                        )
                    if value in mapped_probabilities or not cls._valid_probability(probability):
                        raise ValueError(
                            f'OpenAI Decisions probability for choice {value!r} is not valid'
                        )
                    mapped_probabilities[value] = float(probability)
                mapped[name] = {'probabilities': mapped_probabilities}
            else:
                confidence = answer.get('confidence')
                if choice is None or not cls._valid_probability(confidence):
                    raise ValueError(
                        f'OpenAI Decisions choice answer {name!r} needs probabilities or '
                        'a choice with confidence'
                    )
                mapped[name] = {'choice': choice, 'probabilities': {choice: float(confidence)}}

        missing = set(questions) - set(mapped)
        if missing:
            raise ValueError(f'OpenAI Decisions response is missing answers for {sorted(missing)}')
        return mapped

    @staticmethod
    def _valid_probability(value: Any) -> TypeGuard[int | float]:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= float(value) <= 1.0
        )

    async def _call_fit(
        self,
        instructions: str,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str,
        usage: dict[str, int],
    ) -> dict[str, Any]:
        try:
            response = await self._post(instructions, state, questions, model)
        except _OpenAIInputTooLongSignal as error:
            keys = list(questions)
            if len(keys) > 1:
                middle = len(keys) // 2
                first = await self._call_fit(
                    instructions,
                    state,
                    {key: questions[key] for key in keys[:middle]},
                    model,
                    usage,
                )
                second = await self._call_fit(
                    instructions,
                    state,
                    {key: questions[key] for key in keys[middle:]},
                    model,
                    usage,
                )
                return {**first, **second}
            smaller = shrink_state(state)
            if smaller is None:
                raise OpenAIDecisionsInputTooLongError(
                    'OpenAI Decisions could not fit one question after shrinking the state.'
                ) from error
            return await self._call_fit(instructions, smaller, questions, model, usage)

        response_usage = response.get('usage') or {}
        usage['input_tokens'] += int(response_usage.get('input_tokens', 0))
        usage['output_tokens'] += int(response_usage.get('output_tokens', 0))
        return response['answers']


def openai_decisions_prompt_overrides(
    library: ChatPromptLibrary | None = None,
) -> LLMPromptOverrides:
    """Return the shared n6 and e2 prompt overrides for OpenAI Decisions."""
    return jev_prompt_overrides(library)
