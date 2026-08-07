"""Tests for create_prompt_library and default prompt library shape."""

from types import SimpleNamespace

import pytest

from graphiti_core.prompts import create_prompt_library, prompt_library
from graphiti_core.prompts.lib import PROMPT_GROUPS, ensure_prompt_library_wrapped
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage
from graphiti_core.prompts.prompt_helpers import DO_NOT_ESCAPE_UNICODE


def _default_extract_context() -> dict:
    return {
        'episode_content': 'hi',
        'previous_episodes': [],
        'custom_extraction_instructions': '',
        'entity_types': [],
        'source_description': 'test',
    }


REQUIRED_GROUPS = {group: set(methods) for group, methods in PROMPT_GROUPS.items()}


def _custom_extract_message(context: dict) -> ChatPrompt:
    return ChatPrompt(
        system=SystemMessage(content='custom system'),
        user=UserMessage(content='custom user'),
    )


def test_default_prompt_library_remains_importable():
    assert prompt_library is not None
    assert hasattr(prompt_library, 'extract_nodes')
    assert hasattr(prompt_library, 'specs')


def test_create_prompt_library_returns_default_equivalent_when_no_overrides():
    lib = create_prompt_library()
    context = _default_extract_context()
    default_prompt = prompt_library.extract_nodes.extract_message(context)
    created_prompt = lib.extract_nodes.extract_message(context)
    assert isinstance(default_prompt, ChatPrompt)
    assert isinstance(created_prompt, ChatPrompt)
    assert default_prompt.system.content == created_prompt.system.content
    assert default_prompt.user.content == created_prompt.user.content


def test_create_prompt_library_applies_single_override():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    prompt = lib.extract_nodes.extract_message({})
    assert isinstance(prompt, ChatPrompt)
    assert prompt.system.content == 'custom system'
    assert prompt.user.content == 'custom user'


def test_create_prompt_library_preserves_unspecified_defaults():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    assert callable(lib.extract_nodes.extract_text)
    assert callable(lib.extract_edges.edge)


def test_as_messages_appends_unicode_note_for_overrides():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    messages = lib.extract_nodes.extract_message({}).as_messages()
    assert messages[0].content.endswith(DO_NOT_ESCAPE_UNICODE)
    assert not messages[1].content.endswith(DO_NOT_ESCAPE_UNICODE)


def test_list_message_override_raises_type_error():
    def bad(_context: dict):
        return [{'role': 'system', 'content': 'x'}]

    lib = create_prompt_library({'extract_nodes': {'extract_message': bad}})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match='ChatPrompt'):
        lib.extract_nodes.extract_message({})


def test_unknown_override_group_rejected():
    with pytest.raises(ValueError, match='Unknown prompt group'):
        create_prompt_library({'nope': {'extract_message': _custom_extract_message}})


def test_unknown_override_function_rejected():
    with pytest.raises(ValueError, match='Unknown prompt function'):
        create_prompt_library({'extract_nodes': {'nope': _custom_extract_message}})


def test_default_library_exposes_required_groups_and_functions():
    for group_name, functions in REQUIRED_GROUPS.items():
        assert hasattr(prompt_library, group_name)
        group = getattr(prompt_library, group_name)
        for function_name in functions:
            assert hasattr(group, function_name)
            assert callable(getattr(group, function_name))


def test_prompt_specs_are_fixed():
    from graphiti_core.prompts.extract_nodes import ExtractedEntities

    spec = prompt_library.specs['extract_nodes.extract_message']
    assert spec.response_model is ExtractedEntities
    assert spec.dynamic_schema is False
    assert prompt_library.specs['extract_nodes.extract_attributes'].dynamic_schema is True


def test_ensure_prompt_library_wrapped_attaches_specs_to_hand_built_library():
    def raw(_context: dict) -> ChatPrompt:
        return ChatPrompt(
            system=SystemMessage(content='hand-built system'),
            user=UserMessage(content='hand-built user'),
        )

    hand_built = SimpleNamespace(
        **{
            group_name: SimpleNamespace(**{fn_name: raw for fn_name in fn_names})
            for group_name, fn_names in PROMPT_GROUPS.items()
        }
    )
    wrapped = ensure_prompt_library_wrapped(hand_built)
    assert 'extract_nodes.extract_message' in wrapped.specs
    messages = wrapped.extract_nodes.extract_message({}).as_messages()
    assert messages[0].content.startswith('hand-built system')
    assert messages[0].content.endswith(DO_NOT_ESCAPE_UNICODE)
