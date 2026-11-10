"""Tests that PromptName / PromptGroup stay in sync with the builtin registry."""

from dataclasses import fields
from typing import get_args

from graphiti_core.llm_client.prompt_config import LLMPromptOverrides, PromptRoutes
from graphiti_core.prompts.lib import BUILTIN_PROMPT_SPECS, PROMPT_GROUPS
from graphiti_core.prompts.names import PromptGroup, PromptName


def test_prompt_name_literal_matches_builtin_specs():
    assert set(get_args(PromptName)) == set(BUILTIN_PROMPT_SPECS)


def test_prompt_group_literal_matches_prompt_groups():
    assert set(get_args(PromptGroup)) == set(PROMPT_GROUPS)


def test_prompt_routes_groups_match_registry():
    assert {item.name for item in fields(PromptRoutes)} == set(PROMPT_GROUPS)


def test_prompt_override_groups_match_registry():
    assert {item.name for item in fields(LLMPromptOverrides)} == set(PROMPT_GROUPS)


def test_nested_route_and_override_fields_match_prompt_groups():
    group_classes = {
        'extract_nodes': (PromptRoutes.ExtractNodes, LLMPromptOverrides.ExtractNodes),
        'dedupe_nodes': (PromptRoutes.DedupeNodes, LLMPromptOverrides.DedupeNodes),
        'extract_edges': (PromptRoutes.ExtractEdges, LLMPromptOverrides.ExtractEdges),
        'extract_nodes_and_edges': (
            PromptRoutes.ExtractNodesAndEdges,
            LLMPromptOverrides.ExtractNodesAndEdges,
        ),
        'dedupe_edges': (PromptRoutes.DedupeEdges, LLMPromptOverrides.DedupeEdges),
        'summarize_nodes': (PromptRoutes.SummarizeNodes, LLMPromptOverrides.SummarizeNodes),
        'summarize_sagas': (PromptRoutes.SummarizeSagas, LLMPromptOverrides.SummarizeSagas),
        'eval': (PromptRoutes.Eval, LLMPromptOverrides.Eval),
    }
    assert set(group_classes) == set(PROMPT_GROUPS)
    for group_name, (route_cls, override_cls) in group_classes.items():
        expected = set(PROMPT_GROUPS[group_name])
        route_fields = {item.name for item in fields(route_cls) if item.name != 'default'}
        override_fields = {item.name for item in fields(override_cls)}
        assert route_fields == expected
        assert override_fields == expected
