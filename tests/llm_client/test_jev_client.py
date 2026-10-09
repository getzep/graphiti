import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import pytest_asyncio

import graphiti_core.llm_client.jev_client as jev_module
from graphiti_core.llm_client import (
    JevClient,
    JevInputTooLongError,
    LLMRuntime,
    LLMTransport,
    PromptRoutes,
    RateLimitError,
    jev_prompt_overrides,
)
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


@pytest_asyncio.fixture
async def client_factory():
    clients: list[httpx.AsyncClient] = []

    def create(
        handler: Callable[[httpx.Request], httpx.Response],
        **kwargs: Any,
    ) -> JevClient:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(http_client)
        return JevClient(
            config=LLMConfig(api_key='test-key', model='jev-latest'),
            base_url='https://jev.test/v1/systemone',
            http_client=http_client,
            **kwargs,
        )

    yield create
    await asyncio.gather(*(client.aclose() for client in clients))


def _messages(state: str, questions: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='user',
            content=json.dumps({'state': state, 'questions': questions}, ensure_ascii=False),
        )
    ]


def _node_state(entities: list[dict[str, Any]]) -> str:
    return f'<ENTITIES>{json.dumps(entities)}</ENTITIES>\n<EXISTING ENTITIES>[]</EXISTING ENTITIES>'


def _questions(*names: str) -> dict[str, dict[str, Any]]:
    return {name: {'type': 'noul', 'instructions': 'decide', 'criteria': {}} for name in names}


def _node_context() -> dict[str, Any]:
    return {
        'previous_episodes': ['Alice joined Acme.'],
        'episode_content': 'Alice joined Acme as an engineer.',
        'extracted_nodes': [
            {'id': 0, 'name': 'Alice', 'entity_type': ['Person']},
            {'id': 1, 'name': 'Acme', 'entity_type': ['Organization']},
        ],
        'existing_nodes': [{'candidate_id': 7, 'name': 'Alice Smith'}],
    }


def _edge_context() -> dict[str, Any]:
    return {
        'existing_edges': '[{"idx": 0, "fact": "Alice works at Acme"}]',
        'edge_invalidation_candidates': '[{"idx": 1, "fact": "Alice was a manager at Acme"}]',
        'new_edge': 'Alice works at Acme as an engineer',
    }


def test_jev_prompt_overrides_build_n6_and_e2_questions():
    overrides = jev_prompt_overrides()
    node_prompt = overrides.dedupe_nodes.nodes(_node_context())
    node_payload = json.loads(node_prompt.user.content)
    assert node_payload['state'].endswith(
        '</EXISTING ENTITIES>\n\nRULES FOR THE QUESTIONS:\n' + jev_module.NODE_RULES
    )
    assert list(node_payload['questions']) == ['e0', 'e1']
    assert node_payload['questions']['e0']['type'] == 'choice'
    assert set(node_payload['questions']['e0']['criteria']) == {'7', 'none'}
    assert node_payload['questions']['e0']['criteria']['7'] == 'candidate_id 7: Alice Smith'
    assert node_payload['questions']['e0']['criteria']['none'] == jev_module.NODE_NONE

    edge_prompt = overrides.dedupe_edges.resolve_edge(_edge_context())
    edge_payload = json.loads(edge_prompt.user.content)
    assert edge_payload['state'].startswith(
        'NEW FACT: Alice works at Acme as an engineer\n\nEXISTING FACTS:\n'
    )
    assert edge_payload['state'].endswith(
        'RULES FOR THE QUESTIONS:\n'
        + 'DUPLICATE: '
        + jev_module.DUP_RULES
        + '\nCONTRADICTION: '
        + jev_module.CON_RULES
    )
    assert list(edge_payload['questions']) == ['dup0', 'con0', 'con1']
    assert all(question['type'] == 'noul' for question in edge_payload['questions'].values())
    assert all(
        set(question['criteria']) == {'true', 'false'}
        for question in edge_payload['questions'].values()
    )
    assert 'identical factual information' in edge_payload['questions']['dup0']['instructions']
    assert 'directly contradict' in edge_payload['questions']['con1']['instructions']


@pytest.mark.asyncio
async def test_node_answer_mapping_uses_argmax_none_missing_and_choice_fallback(client_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'answers': {
                    'e0': {'probabilities': {'4': 0.3, '7': 0.7}},
                    'e1': {'probabilities': {'none': 1.0, '7': 0.0}},
                    'e3': {'choice': '8'},
                },
                'usage': {'input_tokens': 4, 'output_tokens': 2},
            },
        )

    client = client_factory(handler)
    entities = [
        {'id': 0, 'name': 'Alice'},
        {'id': 1, 'name': 'Java'},
        {'id': 2, 'name': 'Missing'},
        {'id': 3, 'name': 'Fallback'},
    ]
    result = await client.generate_response(
        _messages(_node_state(entities), _questions('e0', 'e1', 'e2', 'e3')),
        prompt_name='dedupe_nodes.nodes',
    )

    assert result == {
        'entity_resolutions': [
            {'id': 0, 'name': 'Alice', 'duplicate_candidate_id': 7},
            {'id': 1, 'name': 'Java', 'duplicate_candidate_id': -1},
            {'id': 2, 'name': 'Missing', 'duplicate_candidate_id': -1},
            {'id': 3, 'name': 'Fallback', 'duplicate_candidate_id': 8},
        ]
    }


@pytest.mark.asyncio
async def test_edge_thresholds_include_values_at_the_boundary(client_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'answers': {
                    'dup0': {'noul': 0.79},
                    'dup1': {'noul': 0.8},
                    'con0': {'noul': 0.79},
                    'con1': {'noul': 0.8},
                }
            },
        )

    client = client_factory(handler)
    result = await client.generate_response(
        _messages('edge state', _questions('dup0', 'dup1', 'con0', 'con1')),
        prompt_name='dedupe_edges.resolve_edge',
    )

    assert result == {'duplicate_facts': [1], 'contradicted_facts': [1]}


@pytest.mark.asyncio
async def test_request_uses_systemone_url_headers_body_and_selected_model(client_factory):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'answers': {'e0': {'choice': 'none'}}})

    client = client_factory(handler)
    questions = _questions('e0')
    await client.generate_response(
        _messages(_node_state([{'id': 0, 'name': 'Alice'}]), questions),
        prompt_name='dedupe_nodes.nodes',
        model='jev-test-model',
    )

    assert len(requests) == 1
    assert str(requests[0].url) == 'https://jev.test/v1/systemone'
    assert requests[0].headers['Authorization'] == 'Bearer test-key'
    assert requests[0].headers['Content-Type'] == 'application/json'
    assert json.loads(requests[0].content) == {
        'model': 'jev-test-model',
        'state': _node_state([{'id': 0, 'name': 'Alice'}]),
        'questions': questions,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [429, 503])
async def test_retries_transient_http_status_then_succeeds(client_factory, monkeypatch, status):
    attempts = 0
    sleeps: list[float] = []

    async def no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(jev_module, '_sleep', no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status, text='try again')
        return httpx.Response(200, json={'answers': {'e0': {'choice': 'none'}}})

    client = client_factory(handler)
    result = await client.generate_response(
        _messages(_node_state([{'id': 0, 'name': 'Alice'}]), _questions('e0')),
        prompt_name='dedupe_nodes.nodes',
    )

    assert result['entity_resolutions'][0]['duplicate_candidate_id'] == -1
    assert attempts == 2
    assert sleeps == [1]


@pytest.mark.asyncio
async def test_exhausted_429_raises_rate_limit_error(client_factory, monkeypatch):
    attempts = 0

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(jev_module, '_sleep', no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, text='rate limited')

    client = client_factory(handler, max_retries=3)
    with pytest.raises(RateLimitError, match='429'):
        await client.generate_response(
            _messages(_node_state([{'id': 0, 'name': 'Alice'}]), _questions('e0')),
            prompt_name='dedupe_nodes.nodes',
        )
    assert attempts == 3


@pytest.mark.asyncio
async def test_retries_timeout_and_transport_errors(client_factory, monkeypatch):
    attempts = 0
    sleeps: list[float] = []

    async def no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(jev_module, '_sleep', no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout('timed out', request=request)
        if attempts == 2:
            raise httpx.ConnectError('connection failed', request=request)
        return httpx.Response(200, json={'answers': {'e0': {'choice': 'none'}}})

    client = client_factory(handler)
    result = await client.generate_response(
        _messages(_node_state([{'id': 0, 'name': 'Alice'}]), _questions('e0')),
        prompt_name='dedupe_nodes.nodes',
    )

    assert result['entity_resolutions'][0]['duplicate_candidate_id'] == -1
    assert attempts == 3
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_non_retryable_http_error_includes_response_body(client_factory):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text='invalid Jev request')

    client = client_factory(handler)
    with pytest.raises(httpx.HTTPStatusError, match='HTTP 400: invalid Jev request'):
        await client.generate_response(
            _messages(_node_state([{'id': 0, 'name': 'Alice'}]), _questions('e0')),
            prompt_name='dedupe_nodes.nodes',
        )
    assert attempts == 1


@pytest.mark.asyncio
async def test_splits_questions_after_max_tokens_exceeded_and_merges_answers(
    client_factory, monkeypatch
):
    requests: list[dict[str, Any]] = []
    attempts = 0

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(jev_module, '_sleep', no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        body = json.loads(request.content)
        requests.append(body)
        if attempts == 1:
            return httpx.Response(400, text='max_tokens_exceeded')
        question = next(iter(body['questions']))
        candidate = '7' if question == 'e0' else '8'
        return httpx.Response(
            200,
            json={
                'answers': {question: {'choice': candidate}},
                'usage': {'input_tokens': 10, 'output_tokens': 2},
            },
        )

    client = client_factory(handler)
    entities = [{'id': 0, 'name': 'Alice'}, {'id': 1, 'name': 'Bob'}]
    result = await client.generate_response(
        _messages(_node_state(entities), _questions('e0', 'e1')),
        prompt_name='dedupe_nodes.nodes',
    )

    assert len(requests) == 3
    assert [list(body['questions']) for body in requests] == [['e0', 'e1'], ['e0'], ['e1']]
    assert [item['duplicate_candidate_id'] for item in result['entity_resolutions']] == [7, 8]


@pytest.mark.asyncio
async def test_single_question_that_stays_too_long_shrinks_then_raises(client_factory, monkeypatch):
    states: list[str] = []

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(jev_module, '_sleep', no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        states.append(json.loads(request.content)['state'])
        return httpx.Response(400, text='max_tokens_exceeded')

    client = client_factory(handler)
    state = (
        '<PREVIOUS MESSAGES>["earlier"]</PREVIOUS MESSAGES>\n'
        f'<CURRENT MESSAGE>{"A" * 5000}</CURRENT MESSAGE>'
    )
    with pytest.raises(JevInputTooLongError):
        await client.generate_response(
            _messages(state, _questions('e0')),
            prompt_name='dedupe_nodes.nodes',
        )

    assert len(states) > 2
    assert '["earlier"]' not in states[1]
    assert states[1].startswith('<PREVIOUS MESSAGES>[]</PREVIOUS MESSAGES>')
    assert len(states[-1]) < len(states[1])


@pytest.mark.asyncio
async def test_empty_questions_return_empty_results_without_http_call(client_factory):
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        pytest.fail('Jev must not receive an empty question set')

    client = client_factory(fail_if_called)
    node_result = await client.generate_response(
        _messages(_node_state([{'id': 4, 'name': 'No match'}]), {}),
        prompt_name='dedupe_nodes.nodes',
    )
    edge_result = await client.generate_response(
        _messages('edge state', {}),
        prompt_name='dedupe_edges.resolve_edge',
    )

    assert node_result == {
        'entity_resolutions': [{'id': 4, 'name': 'No match', 'duplicate_candidate_id': -1}]
    }
    assert edge_result == {'duplicate_facts': [], 'contradicted_facts': []}


@pytest.mark.asyncio
async def test_rejects_prompts_outside_the_two_dedupe_prompts(client_factory):
    client = client_factory(lambda request: httpx.Response(200, json={'answers': {}}))
    with pytest.raises(ValueError, match='extract_nodes.extract_message.*jev_prompt_overrides'):
        await client.generate_response(
            _messages('state', {}),
            prompt_name='extract_nodes.extract_message',
        )


def test_requires_api_key_when_config_and_environment_are_empty(monkeypatch):
    monkeypatch.delenv('JEV_API_KEY', raising=False)
    with pytest.raises(ValueError, match='JEV_API_KEY'):
        JevClient(config=LLMConfig(api_key=None))


@pytest.mark.asyncio
async def test_token_tracker_sums_usage_across_split_requests(client_factory, monkeypatch):
    attempts = 0

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(jev_module, '_sleep', no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        body = json.loads(request.content)
        if attempts == 1:
            return httpx.Response(400, text='max_tokens_exceeded')
        name = next(iter(body['questions']))
        return httpx.Response(
            200,
            json={
                'answers': {name: {'choice': 'none'}},
                'usage': {'input_tokens': 3 + attempts, 'output_tokens': attempts},
            },
        )

    client = client_factory(handler)
    await client.generate_response(
        _messages(
            _node_state([{'id': 0, 'name': 'A'}, {'id': 1, 'name': 'B'}]), _questions('e0', 'e1')
        ),
        prompt_name='dedupe_nodes.nodes',
    )

    usage = client.token_tracker.get_usage()['dedupe_nodes.nodes']
    assert usage.call_count == 1
    assert usage.total_input_tokens == 11
    assert usage.total_output_tokens == 5


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model='default-model'), cache=False)
        self.calls: list[dict[str, Any]] = []

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {'default': True}

    async def generate_response(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({'messages': messages, **kwargs})
        return {'default': True}


@pytest.mark.asyncio
async def test_runtime_routes_only_dedupe_prompts_to_jev(client_factory):
    jev_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        jev_requests.append(body)
        answers = {}
        for name in body['questions']:
            if name.startswith('e'):
                answers[name] = {'probabilities': {'7': 1.0, 'none': 0.0}}
            elif name.startswith('dup'):
                answers[name] = {'noul': 0.8}
            else:
                answers[name] = {'noul': 0.0}
        return httpx.Response(200, json={'answers': answers})

    default_client = FakeLLM()
    jev_client = client_factory(handler)
    default_transport = LLMTransport(default_client, models=['default-model'])
    jev_transport = LLMTransport(jev_client, models=['jev-latest'])
    jev_model = jev_transport.model('jev-latest', prompt_overrides=jev_prompt_overrides())
    runtime = LLMRuntime(
        model=default_transport.model('default-model'),
        routes=PromptRoutes(
            dedupe_nodes=PromptRoutes.DedupeNodes(nodes=jev_model),
            dedupe_edges=PromptRoutes.DedupeEdges(resolve_edge=jev_model),
        ),
    )

    node_result = await runtime.complete('dedupe_nodes.nodes', _node_context())
    edge_result = await runtime.complete('dedupe_edges.resolve_edge', _edge_context())
    await runtime.complete(
        'extract_nodes.extract_message',
        {
            'episode_content': 'Alice works at Acme.',
            'previous_episodes': [],
            'custom_extraction_instructions': '',
            'entity_types': [],
            'source_description': 'test',
        },
    )

    assert node_result['entity_resolutions'][0]['duplicate_candidate_id'] == 7
    assert edge_result == {'duplicate_facts': [0], 'contradicted_facts': []}
    assert len(jev_requests) == 2
    assert [call['prompt_name'] for call in default_client.calls] == [
        'extract_nodes.extract_message'
    ]
