import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from graphiti_core.llm_client import (
    JevClient,
    OpenAIDecisionsClient,
    OpenAIDecisionsInputTooLongError,
    jev_prompt_overrides,
    openai_decisions_prompt_overrides,
)
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message

FIXTURE = Path(__file__).with_name('decision_prompt_fixture.json')


def _payload(
    state: str = 'current state', instructions: str = 'caller instructions'
) -> dict[str, Any]:
    return {
        'instructions': instructions,
        'state': state,
        'questions': {
            'dup0': {
                'type': 'noul',
                'instructions': 'Is this a duplicate?',
                'criteria': {'true': 'yes', 'false': 'no'},
            },
            'con0': {
                'type': 'predicate',
                'instructions': 'Is this a contradiction?',
            },
            'pick': {
                'type': 'choice',
                'instructions': 'Choose one.',
                'criteria': {'none': 'No match', 'alice': 'Alice Smith'},
            },
        },
    }


def _answer_set(
    *,
    duplicate: float = 0.69,
    contradiction: float = 0.79,
    choice: str = 'alice',
) -> list[dict[str, Any]]:
    return [
        {'type': 'predicate', 'name': 'dup0', 'probability': duplicate},
        {'type': 'predicate', 'name': 'con0', 'probability': contradiction},
        {
            'type': 'choice',
            'name': 'pick',
            'choice': choice,
            'probabilities': [
                {'value': 'none', 'probability': 0.0},
                {'value': 'alice', 'probability': 1.0},
            ],
            'confidence': 1.0,
        },
    ]


def _client(handler, **kwargs) -> OpenAIDecisionsClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return OpenAIDecisionsClient(
        config=LLMConfig(api_key='test-key'),
        base_url='https://decisions.test/v1',
        http_client=http_client,
        **kwargs,
    )


async def _call(client: OpenAIDecisionsClient, payload=None) -> dict[str, Any]:
    return await client.generate_response(
        [Message(role='user', content=json.dumps(payload or _payload()))],
        prompt_name='dedupe_edges.resolve_edge',
    )


@pytest.mark.asyncio
async def test_request_mapping_and_response_mapping() -> None:
    request_data: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_data['url'] = str(request.url)
        request_data['body'] = json.loads(request.content)
        request_data['authorization'] = request.headers['Authorization']
        return httpx.Response(
            200,
            json={
                'model': 'gpt-6-luna',
                'answers': _answer_set(),
                'usage': {'input_tokens': 31, 'output_tokens': 7, 'total_tokens': 38},
            },
        )

    client = _client(handler)
    result = await _call(client)

    assert request_data['url'] == 'https://decisions.test/v1/decisions'
    assert request_data['authorization'] == 'Bearer test-key'
    assert request_data['body'] == {
        'model': 'gpt-6-luna',
        'input': 'caller instructions\n\ncurrent state',
        'questions': [
            {
                'name': 'dup0',
                'type': 'predicate',
                'instructions': 'Is this a duplicate?',
            },
            {
                'name': 'con0',
                'type': 'predicate',
                'instructions': 'Is this a contradiction?',
            },
            {
                'name': 'pick',
                'type': 'choice',
                'instructions': 'Choose one.',
                'choices': [
                    {'value': 'none', 'description': 'No match'},
                    {'value': 'alice', 'description': 'Alice Smith'},
                ],
            },
        ],
    }
    assert result == {'duplicate_facts': [], 'contradicted_facts': []}
    usage = client.token_tracker.get_total_usage()
    assert usage.input_tokens == 31
    assert usage.output_tokens == 7
    await client.http_client.aclose()


@pytest.mark.parametrize(
    ('instructions', 'expected_input'),
    [
        ('caller instructions', 'caller instructions\n\ncurrent state'),
        ('', 'current state'),
    ],
)
@pytest.mark.asyncio
async def test_input_joins_only_nonempty_instructions_and_state(
    instructions: str, expected_input: str
) -> None:
    inputs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs.append(json.loads(request.content)['input'])
        return httpx.Response(200, json={'answers': _answer_set()})

    client = _client(handler)
    await _call(client, _payload(instructions=instructions))

    assert inputs == [expected_input]
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_answer_mapping_supports_choice_probabilities_and_confidence() -> None:
    questions = _payload()['questions']
    probabilities = OpenAIDecisionsClient._map_answers(
        questions,
        [
            {'type': 'predicate', 'name': 'dup0', 'probability': 0.75},
            {'type': 'predicate', 'name': 'con0', 'probability': 0.25},
            {
                'type': 'choice',
                'name': 'pick',
                'probabilities': [
                    {'value': 'none', 'probability': 0.2},
                    {'value': 'alice', 'probability': 0.8},
                ],
            },
        ],
    )
    confidence = OpenAIDecisionsClient._map_answers(
        questions,
        [
            {'type': 'predicate', 'name': 'dup0', 'probability': 0.75},
            {'type': 'predicate', 'name': 'con0', 'probability': 0.25},
            {'type': 'choice', 'name': 'pick', 'choice': 'alice', 'confidence': 0.8},
        ],
    )

    assert probabilities['pick'] == {'probabilities': {'none': 0.2, 'alice': 0.8}}
    assert confidence['pick'] == {'choice': 'alice', 'probabilities': {'alice': 0.8}}


@pytest.mark.parametrize(
    ('duplicate', 'contradiction', 'expected_duplicates', 'expected_contradictions'),
    [
        (0.69, 0.79, [], []),
        (0.70, 0.80, [0], [0]),
    ],
)
@pytest.mark.asyncio
async def test_default_threshold_boundaries(
    duplicate: float,
    contradiction: float,
    expected_duplicates: list[int],
    expected_contradictions: list[int],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'answers': _answer_set(duplicate=duplicate, contradiction=contradiction)},
        )

    client = _client(handler)
    assert (client.duplicate_threshold, client.contradiction_threshold) == (0.7, 0.8)
    assert await _call(client) == {
        'duplicate_facts': expected_duplicates,
        'contradicted_facts': expected_contradictions,
    }
    await client.http_client.aclose()


@pytest.mark.parametrize(
    ('answers', 'error'),
    [
        (
            [
                {'type': 'predicate', 'name': 'unknown', 'probability': 0.9},
                *_answer_set()[1:],
            ],
            'unknown answer name',
        ),
        (
            [
                *_answer_set()[:2],
                {'type': 'choice', 'name': 'pick', 'choice': 'other', 'confidence': 1.0},
            ],
            'not allowed',
        ),
        ([], 'missing answers'),
        (
            [
                {'type': 'choice', 'name': 'dup0', 'choice': 'true', 'confidence': 0.9},
                *_answer_set()[1:],
            ],
            'expected.*predicate',
        ),
        (
            [
                {'type': 'predicate', 'name': 'dup0'},
                *_answer_set()[1:],
            ],
            'needs a valid probability',
        ),
        (
            [
                *_answer_set()[:2],
                {'type': 'choice', 'name': 'pick', 'choice': 'alice'},
            ],
            'needs probabilities or a choice with confidence',
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_answers_raise(answers: list[dict[str, Any]], error: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'answers': answers})

    client = _client(handler)
    with pytest.raises(ValueError, match=error):
        await _call(client)
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_http_error_is_not_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={'error': {'message': 'unauthorized'}})

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError, match='HTTP 401'):
        await _call(client)
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_retries_timeout_429_and_5xx_with_exponential_backoff(monkeypatch) -> None:
    from graphiti_core.llm_client import openai_decisions_client

    calls = 0
    delays: list[int] = []

    async def no_wait(delay: int) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout('request timed out')
        if calls == 2:
            return httpx.Response(500)
        if calls == 3:
            return httpx.Response(429)
        return httpx.Response(200, json={'answers': _answer_set()})

    monkeypatch.setattr(openai_decisions_client, '_sleep', no_wait)
    client = _client(handler)
    await _call(client)

    assert calls == 4
    assert delays == [1, 2, 4]
    await client.http_client.aclose()


@pytest.mark.parametrize(
    'error_code',
    ['context_length_exceeded', 'string_above_max_length'],
)
@pytest.mark.asyncio
async def test_too_long_input_removes_previous_then_halves_current(
    error_code: str,
) -> None:
    states: list[str] = []
    question_counts: list[int] = []
    answers_by_name = {answer['name']: answer for answer in _answer_set()}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state = body['input']
        states.append(state)
        question_counts.append(len(body['questions']))
        if len(states) < 4:
            return httpx.Response(
                400,
                json={'error': {'code': error_code, 'type': 'invalid_request_error'}},
            )
        return httpx.Response(
            200,
            json={'answers': [answers_by_name[question['name']] for question in body['questions']]},
        )

    payload = _payload(
        '<PREVIOUS MESSAGES>["previous"]</PREVIOUS MESSAGES>\n'
        '<CURRENT MESSAGE>' + ('current ' * 400) + '</CURRENT MESSAGE>'
    )
    client = _client(handler)
    await _call(client, payload)

    assert question_counts == [3, 1, 1, 1, 2]
    assert '<PREVIOUS MESSAGES>[]</PREVIOUS MESSAGES>' in states[2]
    assert len(states[3]) < len(states[2])
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_too_long_single_question_raises_after_state_shrinking() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={'error': {'code': 'context_length_exceeded', 'type': 'invalid_request_error'}},
        )

    payload = _payload(
        '<PREVIOUS MESSAGES>["previous"]</PREVIOUS MESSAGES>\n'
        '<CURRENT MESSAGE>' + ('current ' * 400) + '</CURRENT MESSAGE>'
    )
    client = _client(handler)
    with pytest.raises(OpenAIDecisionsInputTooLongError):
        await _call(client, payload)
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_observed_token_limit_error_splits_questions_then_raises() -> None:
    observed_probe_body = {
        'error': {
            'message': 'Decision input exceeds the token limit.',
            'type': 'invalid_request_error',
            'param': None,
            'code': None,
        }
    }
    question_counts: list[int] = []
    input_lengths: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        question_counts.append(len(body['questions']))
        input_lengths.append(len(body['input']))
        return httpx.Response(400, json=observed_probe_body)

    payload = _payload(
        '<PREVIOUS MESSAGES>["previous"]</PREVIOUS MESSAGES>\n'
        '<CURRENT MESSAGE>' + ('current ' * 400) + '</CURRENT MESSAGE>'
    )
    client = _client(handler)
    with pytest.raises(OpenAIDecisionsInputTooLongError):
        await _call(client, payload)

    assert question_counts[0] == 3
    assert question_counts[1] == 1
    assert all(count == 1 for count in question_counts[1:])
    assert len(input_lengths) > 2
    assert input_lengths[-1] < input_lengths[1]
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_other_invalid_request_error_is_not_retried_or_split() -> None:
    calls = 0
    question_counts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        question_counts.append(len(json.loads(request.content)['questions']))
        return httpx.Response(
            400,
            json={
                'error': {
                    'message': 'The request has an invalid parameter.',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                }
            },
        )

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError, match='HTTP 400'):
        await _call(client)

    assert calls == 1
    assert question_counts == [3]
    await client.http_client.aclose()


def test_shared_prompt_renders_match_zep_fixture_for_both_clients() -> None:
    expected = json.loads(FIXTURE.read_text())
    node_context = {
        'previous_episodes': ['Alice joined Acme.'],
        'episode_content': 'Alice joined Acme as an engineer.',
        'extracted_nodes': [
            {'id': 0, 'name': 'Alice', 'entity_type': ['Person']},
            {'id': 1, 'name': 'Acme', 'entity_type': ['Organization']},
        ],
        'existing_nodes': [{'candidate_id': 7, 'name': 'Alice Smith'}],
    }
    edge_context = {
        'existing_edges': '[{"idx": 0, "fact": "Alice works at Acme"}]',
        'edge_invalidation_candidates': '[{"idx": 1, "fact": "Alice was a manager at Acme"}]',
        'new_edge': 'Alice works at Acme as an engineer',
    }

    for overrides in (jev_prompt_overrides(), openai_decisions_prompt_overrides()):
        node = overrides.dedupe_nodes.nodes(node_context)
        edge = overrides.dedupe_edges.resolve_edge(edge_context)
        assert node.user.content == expected['node']
        assert edge.user.content == expected['edge']


async def test_jev_client_threshold_defaults_match_decisions_client() -> None:
    jev = JevClient(
        config=LLMConfig(api_key='test-key'),
        http_client=httpx.AsyncClient(),
    )
    assert jev.duplicate_threshold == 0.7
    assert jev.contradiction_threshold == 0.8
    await jev.http_client.aclose()


@pytest.mark.parametrize(
    ('argument_base_url', 'config_base_url', 'environment_base_url', 'expected_base_url'),
    [
        (
            'https://argument.test/v1/',
            'https://config.test/v1/',
            'https://environment.test/v1/',
            'https://argument.test/v1',
        ),
        (
            None,
            'https://config.test/v1/',
            'https://environment.test/v1/',
            'https://config.test/v1',
        ),
        (None, None, 'https://environment.test/v1/', 'https://environment.test/v1'),
        (None, None, None, 'https://api.openai.com/v1'),
    ],
)
@pytest.mark.asyncio
async def test_base_url_precedence(
    argument_base_url: str | None,
    config_base_url: str | None,
    environment_base_url: str | None,
    expected_base_url: str,
    monkeypatch,
) -> None:
    if environment_base_url is None:
        monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    else:
        monkeypatch.setenv('OPENAI_BASE_URL', environment_base_url)

    client = OpenAIDecisionsClient(
        config=LLMConfig(api_key='test-key', base_url=config_base_url),
        base_url=argument_base_url,
    )

    assert client.base_url == expected_base_url
    await client.http_client.aclose()
