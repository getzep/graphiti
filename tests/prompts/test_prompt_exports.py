"""Tests for public prompt package exports and backward-compatible imports."""

from graphiti_core.prompts import (
    ChatPrompt,
    DefaultPromptLibrary,
    Message,
    PromptFunction,
    PromptLibrary,
    PromptOverrides,
    PromptSpec,
    SystemMessage,
    UserMessage,
    create_prompt_library,
    prompt_library,
)
from graphiti_core.prompts.lib import prompt_library as prompt_library_from_lib


def test_prompt_library_import_from_prompts_package():
    assert prompt_library is not None
    assert hasattr(prompt_library, 'extract_nodes')
    assert hasattr(prompt_library, 'specs')


def test_prompt_library_import_from_prompts_lib():
    assert prompt_library_from_lib is prompt_library


def test_prompt_customization_exports_available():
    assert Message is not None
    assert ChatPrompt is not None
    assert SystemMessage is not None
    assert UserMessage is not None
    assert PromptFunction is not None
    assert PromptLibrary is not None
    assert DefaultPromptLibrary is not None
    assert PromptOverrides is not None
    assert PromptSpec is not None
    assert callable(create_prompt_library)
    assert prompt_library is not None


def test_eval_prompts_remain_available_from_default_prompt_library():
    assert callable(prompt_library.eval.query_expansion)
    assert callable(prompt_library.eval.qa_prompt)
    assert callable(prompt_library.eval.eval_prompt)
    assert callable(prompt_library.eval.eval_add_episode_results)
