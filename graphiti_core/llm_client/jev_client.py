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
import re
from typing import Any

import httpx
from pydantic import BaseModel

from graphiti_core.prompts.lib import (
    ChatPromptLibrary,
    default_chat_prompt_library,
    get_prompt_builder,
)
from graphiti_core.prompts.models import ChatPrompt, Message, SystemMessage, UserMessage

from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError
from .prompt_config import LLMPromptOverrides

_sleep = asyncio.sleep

RULES_HEADER = '\n\nRULES FOR THE QUESTIONS:\n'

NODE_RULES = (
    'Entities are duplicates only if they refer to the same real-world object or concept. '
    'They are NOT duplicates if they are related but distinct, or if they have similar names or '
    'purposes but refer to separate instances or concepts. The same name for different real-world '
    'things (for example Java the programming language and Java the island) is not a duplicate. '
    'An abbreviation, a synonym, or a more complete name for the same thing with the same '
    "possessor is a duplicate (for example NYC and New York City; Marco's car and Marco's vehicle)."
)

NODE_NONE = (
    'No EXISTING ENTITY refers to the same real-world object or concept as this entity. '
    'Related but distinct entities, separate instances, and entities that only share a name or '
    'purpose are not duplicates.'
)

DUP_RULES = (
    'A duplicate states identical factual information: the same assertion about the same '
    'entities, also when the wording or the language is different. It is NOT a duplicate when '
    'there are key differences, particularly in numeric values, dates, identifiers, or key '
    'qualifiers, or when the facts describe different events.'
)

CON_RULES = (
    'A contradiction means that the NEW FACT and the old fact cannot both be true. It needs '
    'explicit evidence that the old fact stopped being true or was replaced, such as termination, '
    'replacement, transfer, or mutually exclusive current-state language (for example a changed '
    'job title for the same job). These are NOT contradictions: a restatement, paraphrase, or '
    'translation of the same fact; a repeated event; different events on different dates (Bob ran '
    '5 miles on Tuesday and Bob ran 3 miles on Wednesday); recency alone; new activity, progress, '
    'status, updates, blockers, or collaboration about the same person or project, which do not '
    'contradict a role, ownership, responsibility, membership, or assignment fact; and a second '
    'employer, role, or location that can be true at the same time.'
)


def tag(text, name):
    m = re.search(rf'<{name}>\s*(.*?)\s*</{name}>', text, re.S)
    return m.group(1) if m else None


def parse_list(text: str | None) -> list[dict[str, Any]]:
    if text is None or not text.strip():
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import ast

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"\{.*?\}(?=\s*'?,|\s*'?\]$|\s*'?\]\s*$)", text, re.S):
        span = m.group(0)
        obj: Any = None
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(span)
                break
            except Exception:
                continue
        if not isinstance(obj, dict):
            obj = {'raw': span}
            for key in ('candidate_id', 'idx', 'id'):
                km = re.search(rf'["\']{key}["\']\s*:\s*(-?\d+)', span)
                if km:
                    obj[key] = int(km.group(1))
            for key in ('name', 'fact'):
                km = re.search(rf'["\']{key}["\']\s*:\s*["\'](.*?)["\']\s*[,}}]', span, re.S)
                if km:
                    obj[key] = km.group(1)
        out.append(obj)
    return out


def shrink_state(state):
    """Drop PREVIOUS MESSAGES first, then halve the CURRENT MESSAGE from the front."""
    prev = tag(state, 'PREVIOUS MESSAGES')
    if prev and prev.strip() not in ('[]', ''):
        return state.replace(prev, '[]', 1)
    cur = tag(state, 'CURRENT MESSAGE')
    if cur and len(cur) > 2000:
        return state.replace(cur, '...' + cur[len(cur) // 2 :], 1)
    return None


def node_n6(user):
    """Decompose a rendered dedupe_nodes.nodes user message into state + questions."""
    entities = parse_list(tag(user, 'ENTITIES'))
    existing = parse_list(tag(user, 'EXISTING ENTITIES'))
    state = user.split('</EXISTING ENTITIES>')[0] + '</EXISTING ENTITIES>'
    criteria = {
        str(c['candidate_id']): f'candidate_id {c["candidate_id"]}: {c.get("name")}'
        for c in existing
        if 'candidate_id' in c
    }
    criteria['none'] = NODE_NONE
    questions = {
        f'e{e["id"]}': {
            'type': 'choice',
            'instructions': (
                f'The entity with id {e["id"]} in ENTITIES is "{e.get("name")}" '
                f'({", ".join(e.get("entity_type") or [])}). It was extracted from the CURRENT MESSAGE. '
                'Which EXISTING ENTITY (by candidate_id) refers to the same real-world object or concept '
                'as this entity? Obey the RULES. Choose none if there is no duplicate.'
            ),
            'criteria': criteria,
        }
        for e in entities
    }
    return state + RULES_HEADER + NODE_RULES, questions, entities, existing


def _contra_strict(new_fact, f):
    return {
        'type': 'noul',
        'instructions': (
            f'Does the NEW FACT "{new_fact}" directly contradict the fact idx={f["idx"]} '
            f'"{f.get("fact", f.get("raw", ""))}", so that both cannot be true at the same time? A contradiction '
            'needs explicit evidence that the old fact stopped being true or was replaced '
            '(termination, replacement, transfer, or mutually exclusive current-state '
            'language). New activity, progress, or status about the same entities does not '
            'contradict a role, ownership, or membership fact.'
        ),
        'criteria': {
            'true': 'The old fact and the NEW FACT cannot both be true.',
            'false': 'Both facts can be true, or the NEW FACT is a different event or a duplicate.',
        },
    }


def edge_e2(user):
    """Decompose a rendered dedupe_edges.resolve_edge user message (strict contradiction)."""
    existing = parse_list(tag(user, 'EXISTING FACTS'))
    candidates = parse_list(tag(user, 'FACT INVALIDATION CANDIDATES'))
    new_fact = (tag(user, 'NEW FACT') or '').strip()
    lines = [f'NEW FACT: {new_fact}', '', 'EXISTING FACTS:']
    lines += [f'  idx={f.get("idx")}: {f.get("fact", f.get("raw", ""))}' for f in existing] or [
        '  (none)'
    ]
    lines += ['', 'FACT INVALIDATION CANDIDATES:']
    lines += [f'  idx={f.get("idx")}: {f.get("fact", f.get("raw", ""))}' for f in candidates] or [
        '  (none)'
    ]
    state = '\n'.join(lines)
    questions = {}
    for f in existing:
        questions[f'dup{f["idx"]}'] = {
            'type': 'noul',
            'instructions': (
                f'Does the NEW FACT "{new_fact}" state identical factual information to '
                f'the existing fact idx={f["idx"]} "{f.get("fact", f.get("raw", ""))}"? Facts with key differences '
                'in numeric values, dates, or qualifiers are not duplicates.'
            ),
            'criteria': {
                'true': 'The two facts assert the same information about the same entities.',
                'false': 'The facts differ in some detail, or describe different events or relationships.',
            },
        }
    for f in existing + candidates:
        questions[f'con{f["idx"]}'] = _contra_strict(new_fact, f)
    return (
        state + RULES_HEADER + 'DUPLICATE: ' + DUP_RULES + '\nCONTRADICTION: ' + CON_RULES,
        questions,
        existing,
        candidates,
    )


def _user_text(messages: list[Message]) -> str:
    for message in messages:
        if message.role == 'user':
            return message.content
    raise ValueError('rendered prompt has no user message')


def _encode_payload(state: str, questions: dict[str, dict[str, Any]]) -> str:
    return json.dumps({'state': state, 'questions': questions}, ensure_ascii=False)


def clef_node_builder(default_builder):
    """Render the default prompt and convert it to n6 questions."""

    def _builder(context: dict[str, Any]) -> ChatPrompt:
        messages = default_builder(context).as_messages()
        state, questions, _, _ = node_n6(_user_text(messages))
        return ChatPrompt(
            system=SystemMessage(
                content=messages[0].content if messages[0].role == 'system' else ''
            ),
            user=UserMessage(content=_encode_payload(state, questions)),
        )

    return _builder


def clef_edge_builder(default_builder):
    """Render the default prompt and convert it to e2 questions."""

    def _builder(context: dict[str, Any]) -> ChatPrompt:
        messages = default_builder(context).as_messages()
        state, questions, _, _ = edge_e2(_user_text(messages))
        return ChatPrompt(
            system=SystemMessage(
                content=messages[0].content if messages[0].role == 'system' else ''
            ),
            user=UserMessage(content=_encode_payload(state, questions)),
        )

    return _builder


def assemble_node_answers(
    answers: dict[str, Any], entities: list[dict[str, Any]]
) -> dict[str, Any]:
    """Map Jev choice answers onto NodeResolutions."""
    resolutions = []
    for entity in entities:
        answer = answers.get(f'e{entity["id"]}') or {}
        probabilities = answer.get('probabilities') or {}
        choice = (
            max(probabilities.items(), key=lambda item: item[1])[0]
            if probabilities
            else answer.get('choice')
        )
        candidate_id = -1 if choice in (None, 'none') else int(choice)
        resolutions.append(
            {
                'id': entity['id'],
                'name': entity.get('name'),
                'duplicate_candidate_id': candidate_id,
            }
        )
    return {'entity_resolutions': resolutions}


def assemble_edge_answers(
    answers: dict[str, Any],
    existing_indices: list[int],
    contradiction_indices: list[int],
    duplicate_threshold: float,
    contradiction_threshold: float,
) -> dict[str, Any]:
    """Map Jev noul answers onto EdgeDuplicate."""
    return {
        'duplicate_facts': [
            idx
            for idx in existing_indices
            if (answers.get(f'dup{idx}') or {}).get('noul', 0.0) >= duplicate_threshold
        ],
        'contradicted_facts': [
            idx
            for idx in contradiction_indices
            if (answers.get(f'con{idx}') or {}).get('noul', 0.0) >= contradiction_threshold
        ],
    }


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
        duplicate_threshold: float = 0.8,
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


def jev_prompt_overrides(
    library: ChatPromptLibrary | None = None,
) -> LLMPromptOverrides:
    """Return prompt overrides for the Jev n6 and e2 dedupe questions."""
    prompt_library = library if library is not None else default_chat_prompt_library
    return LLMPromptOverrides(
        dedupe_nodes=LLMPromptOverrides.DedupeNodes(
            nodes=clef_node_builder(get_prompt_builder(prompt_library, 'dedupe_nodes.nodes'))
        ),
        dedupe_edges=LLMPromptOverrides.DedupeEdges(
            resolve_edge=clef_edge_builder(
                get_prompt_builder(prompt_library, 'dedupe_edges.resolve_edge')
            )
        ),
    )
