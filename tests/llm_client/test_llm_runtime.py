"""Tests for LLMRuntime routing and override layering."""

import asyncio

import pytest

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.llm_runtime import LLMRuntime
from graphiti_core.llm_client.prompt_config import (
    LLMModel,
    LLMPromptOverrides,
    LLMTransport,
    PromptRoutes,
)
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage


class FakeLLM(LLMClient):
    """Minimal LLMClient that records the pinned model id per call."""

    def __init__(self, model: str = 'gpt-4.1') -> None:
        super().__init__(LLMConfig(model=model), cache=False)
        self.calls: list[dict] = []
        self.tracers: list[object] = []

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        *,
        model=None,
    ):
        return {'extracted_entities': []}

    async def generate_response(self, messages, **kwargs):
        self.calls.append(
            {
                'messages': messages,
                'transport_model': self.model,
                'client_id': id(self),
                **kwargs,
            }
        )
        return {'extracted_entities': []}

    def set_tracer(self, tracer: object) -> None:
        self.tracers.append(tracer)


def _chat(marker: str) -> ChatPrompt:
    return ChatPrompt(
        system=SystemMessage(content=marker),
        user=UserMessage(content=marker),
    )


MAIN_ID = 'gpt-4.1'
NANO_ID = 'gpt-4.1-nano'


def _models(client: FakeLLM):
    transport = LLMTransport(client)
    return transport, transport.model(MAIN_ID), transport.model(NANO_ID)


def _extract_ctx() -> dict:
    return {
        'episode_content': 'hi',
        'previous_episodes': [],
        'custom_extraction_instructions': '',
        'entity_types': [],
        'source_description': 't',
    }


@pytest.mark.asyncio
async def test_llm_runtime_routes_to_mapped_model():
    client = FakeLLM()
    _, main, nano = _models(client)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
        ),
    )

    await runtime.complete('extract_nodes.extract_message', _extract_ctx())

    assert client.model == 'gpt-4.1'
    assert len(client.calls) == 1
    assert client.calls[0]['model'] == 'gpt-4.1-nano'
    assert client.calls[0]['prompt_name'] == 'extract_nodes.extract_message'
    assert client.calls[0]['response_model'] is ExtractedEntities


@pytest.mark.asyncio
async def test_model_prompt_overrides_beat_general_overrides():
    client = FakeLLM()
    transport, main, _ = _models(client)
    nano = transport.model(
        id='gpt-4.1-nano',
        prompt_overrides=LLMPromptOverrides(
            extract_nodes=LLMPromptOverrides.ExtractNodes(
                extract_message=lambda ctx: _chat('MODEL'),
            ),
        ),
    )
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
        ),
        prompt_overrides=LLMPromptOverrides(
            extract_nodes=LLMPromptOverrides.ExtractNodes(
                extract_message=lambda ctx: _chat('GENERAL'),
            ),
        ),
    )

    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    messages = client.calls[0]['messages']
    assert messages[0].content.startswith('MODEL')


@pytest.mark.asyncio
async def test_schema_override_rejected_on_runtime_complete():
    from pydantic import BaseModel

    class Other(BaseModel):
        x: int = 1

    _, main, _ = _models(FakeLLM())
    runtime = LLMRuntime(model=main)
    with pytest.raises(ValueError, match='schema overrides are not allowed'):
        await runtime.complete(
            'extract_nodes.extract_message',
            _extract_ctx(),
            response_model=Other,
        )


@pytest.mark.asyncio
async def test_same_response_model_identity_allowed():
    client = FakeLLM()
    _, main, _ = _models(client)
    runtime = LLMRuntime(model=main)
    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    assert len(client.calls) == 1


def test_runtime_rejects_unknown_route_field():
    with pytest.raises(TypeError):
        PromptRoutes(not_a_prompt=None)  # type: ignore[call-arg]


def test_runtime_rejects_unknown_prompt_override_field():
    with pytest.raises(TypeError):
        LLMPromptOverrides.ExtractNodes(not_a_prompt=lambda ctx: _chat('x'))  # type: ignore[call-arg]


def test_runtime_rejects_non_prompt_routes_object():
    _, main, nano = _models(FakeLLM())
    with pytest.raises(TypeError, match='PromptRoutes'):
        LLMRuntime(
            model=main,
            routes={'extract_nodes.extract_message': nano},  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_unmapped_prompt_falls_back_to_default():
    client = FakeLLM()
    _, main, nano = _models(client)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_attributes=nano),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['model'] == 'gpt-4.1'


@pytest.mark.asyncio
async def test_group_route_applies_to_all_prompts_in_group():
    client = FakeLLM()
    _, main, nano = _models(client)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(extract_nodes=nano),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['model'] == 'gpt-4.1-nano'


@pytest.mark.asyncio
async def test_prompt_route_beats_group_route():
    client = FakeLLM()
    transport, main, nano = _models(client)
    other = transport.model('gpt-4.1-mini')
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(default=nano, extract_message=other),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['model'] == 'gpt-4.1-mini'


@pytest.mark.asyncio
async def test_dynamic_schema_requires_response_model():
    _, main, _ = _models(FakeLLM())
    runtime = LLMRuntime(model=main)
    with pytest.raises(ValueError, match='dynamic_schema'):
        await runtime.complete('extract_nodes.extract_attributes', {'x': 1})


@pytest.mark.asyncio
async def test_dynamic_schema_accepts_call_site_model():
    from pydantic import BaseModel, Field

    class PersonAttrs(BaseModel):
        role: str = Field(default='')

    client = FakeLLM()
    _, main, _ = _models(client)
    runtime = LLMRuntime(model=main)
    await runtime.complete(
        'extract_nodes.extract_attributes',
        {
            'node': {'name': 'Alice', 'entity_types': ['Entity'], 'attributes': {}},
            'episode_content': 'hi',
            'previous_episodes': [],
        },
        response_model=PersonAttrs,
        attribute_extraction=True,
    )
    kwargs = client.calls[0]
    assert kwargs['response_model'] is PersonAttrs
    assert kwargs['attribute_extraction'] is True


@pytest.mark.asyncio
async def test_runtime_never_clones_transport():
    client = FakeLLM()
    _, main, nano = _models(client)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    assert {call['client_id'] for call in client.calls} == {id(client)}
    assert not hasattr(runtime, '_clients')


@pytest.mark.asyncio
async def test_concurrent_completes_do_not_serialize_or_clobber_model():
    client = FakeLLM()
    _, main, nano = _models(client)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
        ),
    )
    await asyncio.gather(
        runtime.complete('extract_nodes.extract_message', _extract_ctx()),
        runtime.complete(
            'dedupe_edges.resolve_edge',
            {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
        ),
    )
    models = {call['model'] for call in client.calls}
    assert models == {'gpt-4.1', 'gpt-4.1-nano'}
    assert client.model == 'gpt-4.1'


def test_missing_model_is_a_type_error():
    with pytest.raises(TypeError):
        LLMRuntime()  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_runtime_preserves_client_model_configuration():
    client = FakeLLM(model=MAIN_ID)
    client.small_model = 'kept-small'
    _, main, nano = _models(client)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    assert client.model == MAIN_ID
    assert client.small_model == 'kept-small'
    assert all('small_model' not in call for call in client.calls)


def test_transport_rejects_undeclared_model_ids():
    transport = LLMTransport(FakeLLM(), models=['a'])
    with pytest.raises(ValueError, match="'x' is not declared"):
        transport.model('x')
    with pytest.raises(ValueError, match="'x' is not declared"):
        LLMModel(id='x', transport=transport)

    unrestricted = LLMTransport(FakeLLM())
    assert unrestricted.model('any-id').id == 'any-id'


def test_transport_and_model_validate_types_and_model_ids():
    with pytest.raises(TypeError, match='LLMTransport'):
        LLMModel(id='x', transport=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match='LLMClient'):
        LLMTransport(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='non-empty model ids'):
        LLMTransport(FakeLLM(), models=[])
    with pytest.raises(ValueError, match='non-empty model ids'):
        LLMTransport(FakeLLM(), models=[''])
    with pytest.raises(ValueError, match='non-empty model ids'):
        LLMTransport(FakeLLM(), models=['  '])


@pytest.mark.asyncio
async def test_runtime_routes_across_transport_clients():
    client_a = FakeLLM()
    client_b = FakeLLM()
    transport_a, _, _ = _models(client_a)
    transport_b = LLMTransport(client_b)
    runtime = LLMRuntime(
        model=transport_a.model('a-id'),
        routes=PromptRoutes(dedupe_edges=transport_b.model('b-id')),
    )

    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())

    assert [call['model'] for call in client_b.calls] == ['b-id']
    assert [call['model'] for call in client_a.calls] == ['a-id']
    assert all('small_model' not in call for call in client_a.calls + client_b.calls)


@pytest.mark.asyncio
async def test_routed_model_keeps_model_size_without_selecting_small_model():
    client_a = FakeLLM()
    client_b = FakeLLM()
    transport_a, main, _ = _models(client_a)
    transport_b = LLMTransport(client_b)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(dedupe_edges=transport_b.model('routed-id')),
    )

    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
        model_size=ModelSize.small,
    )

    assert client_b.calls[0]['model'] == 'routed-id'
    assert client_b.calls[0]['model_size'] is ModelSize.small
    assert 'small_model' not in client_b.calls[0]


def test_runtime_transports_are_unique_and_default_first():
    client_a = FakeLLM()
    client_b = FakeLLM()
    transport_a, _, _ = _models(client_a)
    transport_b = LLMTransport(client_b)
    runtime = LLMRuntime(
        model=transport_a.model('default'),
        routes=PromptRoutes(
            extract_nodes=transport_a.model('extract'),
            dedupe_edges=transport_b.model('dedupe'),
            extract_edges=transport_a.model('edges'),
        ),
    )

    assert runtime.transports == (transport_a, transport_b)


def test_set_tracer_reaches_each_transport_client():
    client_a = FakeLLM()
    client_b = FakeLLM()
    transport_a, main, _ = _models(client_a)
    transport_b = LLMTransport(client_b)
    runtime = LLMRuntime(
        model=main,
        routes=PromptRoutes(dedupe_edges=transport_b.model('b-id')),
    )
    tracer = object()

    runtime.set_tracer(tracer)  # type: ignore[arg-type]

    assert client_a.tracers == [tracer]
    assert client_b.tracers == [tracer]


def test_runtime_client_is_the_default_model_client():
    client = FakeLLM()
    _, main, _ = _models(client)
    runtime = LLMRuntime(model=main)

    assert runtime.client is main.transport.client


def test_models_compare_and_hash_by_transport_identity():
    client = FakeLLM()
    transport_a = LLMTransport(client)
    transport_b = LLMTransport(client)
    first = transport_a.model('same-id')
    same_transport = transport_a.model('same-id')
    override_model = transport_a.model(
        'same-id',
        prompt_overrides=LLMPromptOverrides(
            extract_nodes=LLMPromptOverrides.ExtractNodes(extract_message=lambda ctx: _chat('x'))
        ),
    )
    other_transport = transport_b.model('same-id')

    assert first == same_transport
    assert hash(first) == hash(same_transport)
    assert first == override_model
    assert hash(first) == hash(override_model)
    assert first != other_transport


def test_llm_model_rejects_non_callable_override():
    transport = LLMTransport(FakeLLM())
    with pytest.raises(TypeError, match='must be callable'):
        transport.model(
            MAIN_ID,
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(
                    extract_message=transport.model(MAIN_ID)  # type: ignore[arg-type]
                )
            ),
        )


def test_flatten_overrides_rejects_wrong_group_class():
    from graphiti_core.llm_client.prompt_config import flatten_overrides

    transport = LLMTransport(FakeLLM())
    with pytest.raises(TypeError, match='LLMPromptOverrides.ExtractNodes'):
        flatten_overrides(
            LLMPromptOverrides(
                extract_nodes=PromptRoutes.ExtractNodes(  # type: ignore[arg-type]
                    extract_message=transport.model(MAIN_ID)
                )
            )
        )


def test_llm_model_rejects_empty_id():
    transport = LLMTransport(FakeLLM())
    with pytest.raises(ValueError, match='non-empty'):
        transport.model('  ')


def test_llm_model_rejects_non_positive_max_tokens():
    transport = LLMTransport(FakeLLM())
    with pytest.raises(ValueError, match='positive'):
        transport.model(MAIN_ID, max_tokens=0)


def test_llm_model_is_hashable():
    transport = LLMTransport(FakeLLM())
    hash(transport.model(MAIN_ID))
    hash(
        transport.model(
            MAIN_ID,
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(
                    extract_message=lambda ctx: _chat('x'),
                )
            ),
        )
    )
