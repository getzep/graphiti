"""Tests for LLMRuntime routing and override layering."""

import asyncio

import pytest

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.llm_runtime import LLMRuntime
from graphiti_core.llm_client.prompt_config import LLMModel, LLMPromptOverrides, PromptRoutes
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage


class FakeLLM(LLMClient):
    """Minimal LLMClient that records the pinned model id per call."""

    def __init__(self, model: str = 'gpt-4.1') -> None:
        super().__init__(LLMConfig(model=model), cache=False)
        self.calls: list[dict] = []

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        *,
        model=None,
        small_model=None,
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


def _chat(marker: str) -> ChatPrompt:
    return ChatPrompt(
        system=SystemMessage(content=marker),
        user=UserMessage(content=marker),
    )


MAIN = LLMModel(id='gpt-4.1')
NANO = LLMModel(id='gpt-4.1-nano')


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
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=NANO),
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
    nano = LLMModel(
        id='gpt-4.1-nano',
        prompt_overrides=LLMPromptOverrides(
            extract_nodes=LLMPromptOverrides.ExtractNodes(
                extract_message=lambda ctx: _chat('MODEL'),
            ),
        ),
    )
    runtime = LLMRuntime(
        client,
        model=MAIN,
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

    runtime = LLMRuntime(FakeLLM(), model=MAIN)
    with pytest.raises(ValueError, match='schema overrides are not allowed'):
        await runtime.complete(
            'extract_nodes.extract_message',
            _extract_ctx(),
            response_model=Other,
        )


@pytest.mark.asyncio
async def test_same_response_model_identity_allowed():
    client = FakeLLM()
    runtime = LLMRuntime(client, model=MAIN)
    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    assert len(client.calls) == 1


def test_runtime_rejects_unknown_route_field():
    with pytest.raises(TypeError):
        PromptRoutes(not_a_prompt=NANO)  # type: ignore[call-arg]


def test_runtime_rejects_unknown_prompt_override_field():
    with pytest.raises(TypeError):
        LLMPromptOverrides.ExtractNodes(not_a_prompt=lambda ctx: _chat('x'))  # type: ignore[call-arg]


def test_runtime_rejects_non_prompt_routes_object():
    with pytest.raises(TypeError, match='PromptRoutes'):
        LLMRuntime(
            FakeLLM(),
            model=MAIN,
            routes={'extract_nodes.extract_message': NANO},  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_unmapped_prompt_falls_back_to_default():
    client = FakeLLM()
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_attributes=NANO),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['model'] == 'gpt-4.1'


@pytest.mark.asyncio
async def test_group_route_applies_to_all_prompts_in_group():
    client = FakeLLM()
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(extract_nodes=NANO),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['model'] == 'gpt-4.1-nano'


@pytest.mark.asyncio
async def test_prompt_route_beats_group_route():
    client = FakeLLM()
    other = LLMModel(id='gpt-4.1-mini')
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(default=NANO, extract_message=other),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['model'] == 'gpt-4.1-mini'


@pytest.mark.asyncio
async def test_dynamic_schema_requires_response_model():
    runtime = LLMRuntime(FakeLLM(), model=MAIN)
    with pytest.raises(ValueError, match='dynamic_schema'):
        await runtime.complete('extract_nodes.extract_attributes', {'x': 1})


@pytest.mark.asyncio
async def test_dynamic_schema_accepts_call_site_model():
    from pydantic import BaseModel, Field

    class PersonAttrs(BaseModel):
        role: str = Field(default='')

    client = FakeLLM()
    runtime = LLMRuntime(client, model=MAIN)
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
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=NANO),
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
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=NANO),
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
        LLMRuntime(FakeLLM())  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_complete_forwards_model_size_and_preserves_transport_small_model():
    from graphiti_core.llm_client.config import ModelSize

    client = FakeLLM(model='gpt-4.1')
    client.small_model = 'gpt-4.1-nano'
    runtime = LLMRuntime(client, model=LLMModel(id='gpt-4.1'))
    await runtime.complete(
        'extract_nodes.extract_message',
        _extract_ctx(),
        model_size=ModelSize.small,
    )
    assert client.small_model == 'gpt-4.1-nano'
    assert client.calls[0]['model_size'] is ModelSize.small
    assert client.calls[0]['small_model'] is None
    assert client.calls[0]['model'] == 'gpt-4.1'


@pytest.mark.asyncio
async def test_routed_model_without_small_id_pins_small_to_own_id():
    client = FakeLLM(model='gpt-4.1')
    client.small_model = 'gpt-4.1-nano'
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=NANO),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['small_model'] == 'gpt-4.1-nano'
    assert client.small_model == 'gpt-4.1-nano'


@pytest.mark.asyncio
async def test_explicit_small_id_is_passed_through():
    client = FakeLLM()
    runtime = LLMRuntime(client, model=LLMModel(id='gpt-4.1', small_id='gpt-4.1-mini'))
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    assert client.calls[0]['small_model'] == 'gpt-4.1-mini'
    assert client.small_model != 'gpt-4.1-mini'


@pytest.mark.asyncio
async def test_transport_attributes_never_mutated():
    client = FakeLLM(model='gpt-4.1')
    client.small_model = 'kept-small'
    runtime = LLMRuntime(
        client,
        model=MAIN,
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=NANO),
        ),
    )
    await runtime.complete('extract_nodes.extract_message', _extract_ctx())
    await runtime.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    assert client.model == 'gpt-4.1'
    assert client.small_model == 'kept-small'


def test_llm_model_rejects_non_callable_override():
    with pytest.raises(TypeError, match='must be callable'):
        LLMModel(
            id='gpt-4.1',
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(extract_message=MAIN)  # type: ignore[arg-type]
            ),
        )


def test_flatten_overrides_rejects_wrong_group_class():
    from graphiti_core.llm_client.prompt_config import flatten_overrides

    with pytest.raises(TypeError, match='LLMPromptOverrides.ExtractNodes'):
        flatten_overrides(
            LLMPromptOverrides(
                extract_nodes=PromptRoutes.ExtractNodes(extract_message=MAIN)  # type: ignore[arg-type]
            )
        )


def test_llm_model_rejects_empty_id():
    with pytest.raises(ValueError, match='non-empty'):
        LLMModel(id='  ')


def test_llm_model_rejects_non_positive_max_tokens():
    with pytest.raises(ValueError, match='positive'):
        LLMModel(id='gpt-4.1', max_tokens=0)


def test_llm_model_is_hashable():
    hash(LLMModel(id='gpt-4.1'))
    hash(
        LLMModel(
            id='gpt-4.1',
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(
                    extract_message=lambda ctx: _chat('x'),
                )
            ),
        )
    )
