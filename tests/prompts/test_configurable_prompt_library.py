"""Tests for create_prompt_library and default prompt library shape."""

from types import SimpleNamespace

import pytest

from graphiti_core.prompts import create_prompt_library, prompt_library
from graphiti_core.prompts.lib import PROMPT_LIBRARY_IMPL, ensure_prompt_library_wrapped
from graphiti_core.prompts.models import Message
from graphiti_core.prompts.prompt_helpers import DO_NOT_ESCAPE_UNICODE


def _default_extract_context() -> dict:
    return {
        'episode_content': 'hi',
        'previous_episodes': [],
        'custom_extraction_instructions': '',
        'entity_types': [],
        'source_description': 'test',
    }


REQUIRED_GROUPS = {
    'extract_nodes': {
        'extract_message',
        'extract_json',
        'extract_text',
        'classify_nodes',
        'extract_attributes',
        'extract_summary',
        'extract_summaries_batch',
        'extract_entity_summaries_from_episodes',
    },
    'dedupe_nodes': {'node', 'node_list', 'nodes'},
    'extract_edges': {
        'edge',
        'extract_attributes',
        'extract_timestamps',
        'extract_timestamps_batch',
    },
    'extract_nodes_and_edges': {'extract_message'},
    'dedupe_edges': {'resolve_edge'},
    'summarize_nodes': {'summarize_pair', 'summarize_context', 'summary_description'},
    'summarize_sagas': {'summarize_saga'},
    'eval': {
        'query_expansion',
        'qa_prompt',
        'eval_prompt',
        'eval_add_episode_results',
    },
}


def _custom_extract_message(context: dict) -> list[Message]:
    return [
        Message(role='system', content='custom system'),
        Message(role='user', content='custom user'),
    ]


def test_default_prompt_library_remains_importable():
    assert prompt_library is not None
    assert hasattr(prompt_library, 'extract_nodes')


def test_create_prompt_library_returns_default_equivalent_when_no_overrides():
    lib = create_prompt_library()
    context = _default_extract_context()
    default_msgs = prompt_library.extract_nodes.extract_message(context)
    created_msgs = lib.extract_nodes.extract_message(context)
    assert [m.role for m in default_msgs] == [m.role for m in created_msgs]
    assert [m.content for m in default_msgs] == [m.content for m in created_msgs]


def test_create_prompt_library_applies_single_override():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    messages = lib.extract_nodes.extract_message({})
    assert messages[0].role == 'system'
    assert messages[0].content.startswith('custom system')
    assert messages[1].content == 'custom user'


def test_create_prompt_library_preserves_unspecified_defaults():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    # Unspecified function still works via default implementation
    assert callable(lib.extract_nodes.extract_text)
    assert callable(lib.extract_edges.edge)


def test_system_messages_receive_do_not_escape_unicode_for_overrides():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    messages = lib.extract_nodes.extract_message({})
    assert messages[0].content.endswith(DO_NOT_ESCAPE_UNICODE)


def test_user_messages_do_not_receive_do_not_escape_unicode_for_overrides():
    lib = create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    messages = lib.extract_nodes.extract_message({})
    assert not messages[1].content.endswith(DO_NOT_ESCAPE_UNICODE)


def test_create_prompt_library_rejects_unknown_group():
    with pytest.raises(ValueError, match='Unknown prompt group: not_a_group'):
        create_prompt_library({'not_a_group': {'extract_message': _custom_extract_message}})


def test_create_prompt_library_rejects_unknown_function():
    with pytest.raises(
        ValueError, match='Unknown prompt function for group extract_nodes: not_a_fn'
    ):
        create_prompt_library({'extract_nodes': {'not_a_fn': _custom_extract_message}})


def test_create_prompt_library_rejects_non_callable_override():
    with pytest.raises(
        ValueError, match='Prompt override must be callable: extract_nodes.extract_message'
    ):
        create_prompt_library({'extract_nodes': {'extract_message': 'not-callable'}})  # type: ignore[dict-item]


def test_default_prompt_library_contains_required_groups():
    for group_name in REQUIRED_GROUPS:
        assert hasattr(prompt_library, group_name)


def test_default_prompt_library_contains_required_functions():
    for group_name, functions in REQUIRED_GROUPS.items():
        group = getattr(prompt_library, group_name)
        for function_name in functions:
            assert hasattr(group, function_name), f'{group_name}.{function_name}'
            assert callable(getattr(group, function_name))


def test_default_prompt_library_impl_contains_required_groups():
    for group_name in REQUIRED_GROUPS:
        assert group_name in PROMPT_LIBRARY_IMPL


def test_default_prompt_library_impl_contains_required_functions():
    for group_name, functions in REQUIRED_GROUPS.items():
        assert set(PROMPT_LIBRARY_IMPL[group_name].keys()) >= functions


def test_create_prompt_library_does_not_mutate_default_via_overrides_map():
    # Ensure overriding a composed library does not change the module default callable
    before = prompt_library.extract_nodes.extract_message
    create_prompt_library({'extract_nodes': {'extract_message': _custom_extract_message}})
    assert prompt_library.extract_nodes.extract_message is before


def test_ensure_prompt_library_wrapped_applies_unicode_to_hand_built_library():
    def raw(_context: dict) -> list[Message]:
        return [
            Message(role='system', content='raw system'),
            Message(role='user', content='raw user'),
        ]

    hand_built = SimpleNamespace(
        **{
            group_name: SimpleNamespace(**{fn_name: raw for fn_name in fn_names})
            for group_name, fn_names in PROMPT_LIBRARY_IMPL.items()
        }
    )
    wrapped = ensure_prompt_library_wrapped(hand_built)  # type: ignore[arg-type]
    messages = wrapped.extract_nodes.extract_message({})
    assert messages[0].content.endswith(DO_NOT_ESCAPE_UNICODE)
    assert not messages[1].content.endswith(DO_NOT_ESCAPE_UNICODE)
