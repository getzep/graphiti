"""Tests for Graphiti constructor prompt_library wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.prompts import create_prompt_library, prompt_library
from graphiti_core.prompts.lib import (
    PROMPT_LIBRARY_IMPL,
    ensure_prompt_library_wrapped,
    validate_prompt_library,
)
from graphiti_core.prompts.models import Message
from graphiti_core.prompts.prompt_helpers import DO_NOT_ESCAPE_UNICODE


def _custom_extract_message(context: dict) -> list[Message]:
    return [
        Message(role='system', content='custom system'),
        Message(role='user', content='custom user'),
    ]


def _make_graphiti(**kwargs) -> Graphiti:
    llm_client = MagicMock()
    llm_client.set_tracer = MagicMock()
    # Bypass Pydantic isinstance checks for test doubles.
    with patch(
        'graphiti_core.graphiti.GraphitiClients',
        side_effect=lambda **kw: GraphitiClients.model_construct(**kw),
    ):
        return Graphiti(
            graph_driver=MagicMock(),
            llm_client=llm_client,
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

    def other_extract(context: dict) -> list[Message]:
        return [Message(role='system', content='other'), Message(role='user', content='other')]

    custom_b = create_prompt_library({'extract_nodes': {'extract_message': other_extract}})
    a = _make_graphiti(prompt_library=custom_a)
    b = _make_graphiti(prompt_library=custom_b)
    assert a.prompt_library is not b.prompt_library
    assert a.clients.prompt_library is not b.clients.prompt_library
    assert a.prompt_library.extract_nodes.extract_message({})[0].content.startswith('custom system')
    assert b.prompt_library.extract_nodes.extract_message({})[0].content.startswith('other')


def test_graphiti_accepts_prompt_library_created_from_overrides():
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    graphiti = _make_graphiti(prompt_library=custom)
    messages = graphiti.prompt_library.extract_nodes.extract_message({})
    assert messages[0].content.startswith('custom system')


def test_graphiti_accepts_complete_prompt_library():
    complete = create_prompt_library()
    graphiti = _make_graphiti(prompt_library=complete)
    assert graphiti.prompt_library is complete


def _hand_built_complete_library():
    def raw(_context: dict) -> list[Message]:
        return [
            Message(role='system', content='hand-built system'),
            Message(role='user', content='hand-built user'),
        ]

    return SimpleNamespace(
        **{
            group_name: SimpleNamespace(**{fn_name: raw for fn_name in fn_names})
            for group_name, fn_names in PROMPT_LIBRARY_IMPL.items()
        }
    )


def test_graphiti_wraps_hand_built_complete_prompt_library():
    hand_built = _hand_built_complete_library()
    graphiti = _make_graphiti(prompt_library=hand_built)  # type: ignore[arg-type]
    messages = graphiti.prompt_library.extract_nodes.extract_message({})
    assert messages[0].content.startswith('hand-built system')
    assert messages[0].content.endswith(DO_NOT_ESCAPE_UNICODE)
    assert not messages[1].content.endswith(DO_NOT_ESCAPE_UNICODE)
    # Hand-built libraries are re-wrapped; wrapper identity is not preserved.
    assert graphiti.prompt_library is not hand_built


def test_ensure_prompt_library_wrapped_preserves_wrapper_identity():
    custom = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    assert ensure_prompt_library_wrapped(custom) is custom


def test_graphiti_validates_complete_prompt_library():
    complete = create_prompt_library()
    validate_prompt_library(complete)  # does not raise
    _make_graphiti(prompt_library=complete)


def test_graphiti_rejects_complete_library_missing_group():
    incomplete = SimpleNamespace()
    with pytest.raises(ValueError, match='Prompt library missing group: extract_nodes'):
        _make_graphiti(prompt_library=incomplete)  # type: ignore[arg-type]


def test_graphiti_rejects_complete_library_missing_function():
    group = SimpleNamespace()  # missing extract_message etc.
    incomplete = SimpleNamespace(**{name: group for name in PROMPT_LIBRARY_IMPL})
    with pytest.raises(ValueError, match='Prompt library missing function: extract_nodes.'):
        _make_graphiti(prompt_library=incomplete)  # type: ignore[arg-type]


def test_graphiti_rejects_complete_library_non_callable_function():
    def make_group(function_names: dict):
        return SimpleNamespace(**{name: 'not-callable' for name in function_names})

    incomplete = SimpleNamespace(
        **{group: make_group(fns) for group, fns in PROMPT_LIBRARY_IMPL.items()}
    )
    with pytest.raises(ValueError, match='Prompt library function must be callable:'):
        _make_graphiti(prompt_library=incomplete)  # type: ignore[arg-type]


def test_create_prompt_library_does_not_mutate_default_prompt_library():
    before = prompt_library.extract_nodes.extract_message
    _ = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    graphiti = _make_graphiti()
    assert graphiti.prompt_library.extract_nodes.extract_message is before
    assert prompt_library.extract_nodes.extract_message is before


def test_graphiti_existing_constructor_signature_still_works():
    graphiti = _make_graphiti()
    assert graphiti.prompt_library is prompt_library


def test_graphiti_subclass_without_prompt_configuration_uses_defaults():
    class ZepGraphiti(Graphiti):
        pass

    llm_client = MagicMock()
    llm_client.set_tracer = MagicMock()
    with patch(
        'graphiti_core.graphiti.GraphitiClients',
        side_effect=lambda **kw: GraphitiClients.model_construct(**kw),
    ):
        graphiti = ZepGraphiti(
            graph_driver=MagicMock(),
            llm_client=llm_client,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
        )
    assert graphiti.prompt_library is prompt_library
