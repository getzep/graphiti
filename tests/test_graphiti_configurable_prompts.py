"""Tests for Graphiti constructor prompt_library / llm_runtime wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.prompts import create_prompt_library, prompt_library
from graphiti_core.prompts.lib import (
    PROMPT_GROUPS,
    ensure_prompt_library_wrapped,
    validate_prompt_library,
)
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage
from graphiti_core.prompts.prompt_helpers import DO_NOT_ESCAPE_UNICODE


def _custom_extract_message(context: dict) -> ChatPrompt:
    return ChatPrompt(
        system=SystemMessage(content='custom system'),
        user=UserMessage(content='custom user'),
    )


def _make_graphiti(**kwargs) -> Graphiti:
    if 'llm_runtime' not in kwargs and 'llm_client' not in kwargs:
        llm_client = MagicMock()
        llm_client.set_tracer = MagicMock()
        kwargs['llm_client'] = llm_client
    elif kwargs.get('llm_client') is not None:
        kwargs['llm_client'].set_tracer = MagicMock()
    with patch(
        'graphiti_core.graphiti.GraphitiClients',
        side_effect=lambda **kw: GraphitiClients.model_construct(**kw),
    ):
        return Graphiti(
            graph_driver=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
            **kwargs,
        )


def test_graphiti_stores_default_prompt_library_when_unconfigured():
    graphiti = _make_graphiti()
    assert graphiti.prompt_library is prompt_library
    assert graphiti.clients.prompt_library is prompt_library


def test_graphiti_stores_custom_prompt_library():
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    graphiti = _make_graphiti(prompt_library=custom)
    assert graphiti.prompt_library is custom
    assert graphiti.clients.prompt_library is custom


def test_graphiti_clients_include_instance_prompt_library():
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    graphiti = _make_graphiti(prompt_library=custom)
    assert graphiti.clients.prompt_library is graphiti.prompt_library


def test_graphiti_instances_isolate_prompt_libraries():
    custom_a = create_prompt_library(
        {'extract_nodes': {'extract_message': _custom_extract_message}}
    )

    def other_extract(context: dict) -> ChatPrompt:
        return ChatPrompt(
            system=SystemMessage(content='other'),
            user=UserMessage(content='other'),
        )

    custom_b = create_prompt_library({'extract_nodes': {'extract_message': other_extract}})
    a = _make_graphiti(prompt_library=custom_a)
    b = _make_graphiti(prompt_library=custom_b)
    assert a.prompt_library is not b.prompt_library
    assert a.clients.prompt_library is not b.clients.prompt_library
    assert a.prompt_library.extract_nodes.extract_message({}).system.content == 'custom system'
    assert b.prompt_library.extract_nodes.extract_message({}).system.content == 'other'


def test_graphiti_accepts_prompt_library_created_from_overrides():
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    graphiti = _make_graphiti(prompt_library=custom)
    prompt = graphiti.prompt_library.extract_nodes.extract_message({})
    assert prompt.system.content == 'custom system'


def test_graphiti_accepts_complete_prompt_library():
    complete = create_prompt_library()
    graphiti = _make_graphiti(prompt_library=complete)
    assert graphiti.prompt_library is complete


def _hand_built_complete_library():
    def raw(_context: dict) -> ChatPrompt:
        return ChatPrompt(
            system=SystemMessage(content='hand-built system'),
            user=UserMessage(content='hand-built user'),
        )

    return SimpleNamespace(
        **{
            group_name: SimpleNamespace(**{fn_name: raw for fn_name in fn_names})
            for group_name, fn_names in PROMPT_GROUPS.items()
        }
    )


def test_graphiti_wraps_hand_built_complete_prompt_library():
    hand_built = _hand_built_complete_library()
    graphiti = _make_graphiti(prompt_library=hand_built)  # type: ignore[arg-type]
    messages = graphiti.prompt_library.extract_nodes.extract_message({}).as_messages()
    assert messages[0].content.startswith('hand-built system')
    assert messages[0].content.endswith(DO_NOT_ESCAPE_UNICODE)
    assert not messages[1].content.endswith(DO_NOT_ESCAPE_UNICODE)
    assert hasattr(graphiti.prompt_library, 'specs')


def test_ensure_prompt_library_wrapped_preserves_library_with_specs():
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    assert ensure_prompt_library_wrapped(custom) is custom


def test_graphiti_validates_complete_prompt_library():
    complete = create_prompt_library()
    validate_prompt_library(complete)
    _make_graphiti(prompt_library=complete)


def test_graphiti_rejects_complete_library_missing_group():
    incomplete = SimpleNamespace()
    with pytest.raises(ValueError, match='Prompt library missing group: extract_nodes'):
        _make_graphiti(prompt_library=incomplete)  # type: ignore[arg-type]


def test_graphiti_rejects_complete_library_missing_function():
    group = SimpleNamespace()
    incomplete = SimpleNamespace(**{name: group for name in PROMPT_GROUPS})
    with pytest.raises(ValueError, match='Prompt library missing function: extract_nodes.'):
        _make_graphiti(prompt_library=incomplete)  # type: ignore[arg-type]


def test_graphiti_rejects_complete_library_non_callable_function():
    def make_group(function_names):
        return SimpleNamespace(**{name: 'not-callable' for name in function_names})

    incomplete = SimpleNamespace(**{group: make_group(fns) for group, fns in PROMPT_GROUPS.items()})
    with pytest.raises(ValueError, match='Prompt library function must be callable:'):
        _make_graphiti(prompt_library=incomplete)  # type: ignore[arg-type]


def test_create_prompt_library_does_not_mutate_default_prompt_library():
    context = {
        'episode_content': 'hi',
        'previous_episodes': [],
        'custom_extraction_instructions': '',
        'entity_types': [],
        'source_description': 'test',
    }
    before = prompt_library.extract_nodes.extract_message(context).system.content
    _ = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    graphiti = _make_graphiti()
    assert graphiti.prompt_library is prompt_library
    assert graphiti.prompt_library.extract_nodes.extract_message(context).system.content == before
    assert prompt_library.extract_nodes.extract_message(context).system.content == before
    assert before != 'custom system'


def test_graphiti_rejects_prompt_library_and_llm_runtime_together():
    from graphiti_core.llm_client.llm_runtime import LLMModel, LLMRuntime

    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    llm = MagicMock()
    llm.set_tracer = MagicMock()
    runtime = LLMRuntime(llm, model=LLMModel(id='gpt-4.1-mini'))
    with pytest.raises(ValueError, match='cannot be combined with: prompt_library'):
        _make_graphiti(prompt_library=custom, llm_runtime=runtime)


def test_graphiti_rejects_llm_client_and_llm_runtime_together():
    from graphiti_core.llm_client.llm_runtime import LLMModel, LLMRuntime

    llm = MagicMock()
    llm.set_tracer = MagicMock()
    runtime = LLMRuntime(llm, model=LLMModel(id='gpt-4.1-mini'))
    with pytest.raises(ValueError, match='cannot be combined with: llm_client'):
        _make_graphiti(llm_client=MagicMock(), llm_runtime=runtime)


def test_graphiti_llm_runtime_populates_prompt_library_from_bundle():
    from graphiti_core.llm_client.llm_runtime import LLMModel, LLMRuntime

    llm = MagicMock()
    llm.set_tracer = MagicMock()
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    runtime = LLMRuntime(
        llm,
        model=LLMModel(id='gpt-4.1-mini'),
        library=custom,
    )
    graphiti = _make_graphiti(llm_runtime=runtime)
    assert graphiti.llm_runtime is runtime
    assert graphiti.clients.llm_runtime is runtime
    assert graphiti.llm_client is llm
    assert (
        graphiti.prompt_library.extract_nodes.extract_message({}).system.content == 'custom system'
    )
