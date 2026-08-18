"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import cast

from graphiti_core.prompts.lib import PROMPT_GROUPS
from graphiti_core.prompts.models import PromptFunction


@dataclass(frozen=True)
class LLMPromptOverrides:
    """Per-prompt builder overrides. Unknown field names are a type / constructor error."""

    @dataclass(frozen=True)
    class ExtractNodes:
        extract_message: PromptFunction | None = None
        extract_json: PromptFunction | None = None
        extract_text: PromptFunction | None = None
        classify_nodes: PromptFunction | None = None
        extract_attributes: PromptFunction | None = None
        extract_summary: PromptFunction | None = None
        extract_summaries_batch: PromptFunction | None = None
        extract_entity_summaries_from_episodes: PromptFunction | None = None

    @dataclass(frozen=True)
    class DedupeNodes:
        node: PromptFunction | None = None
        node_list: PromptFunction | None = None
        nodes: PromptFunction | None = None

    @dataclass(frozen=True)
    class ExtractEdges:
        edge: PromptFunction | None = None
        extract_attributes: PromptFunction | None = None
        extract_timestamps: PromptFunction | None = None
        extract_timestamps_batch: PromptFunction | None = None

    @dataclass(frozen=True)
    class ExtractNodesAndEdges:
        extract_message: PromptFunction | None = None

    @dataclass(frozen=True)
    class DedupeEdges:
        resolve_edge: PromptFunction | None = None

    @dataclass(frozen=True)
    class SummarizeNodes:
        summarize_pair: PromptFunction | None = None
        summarize_context: PromptFunction | None = None
        summary_description: PromptFunction | None = None

    @dataclass(frozen=True)
    class SummarizeSagas:
        summarize_saga: PromptFunction | None = None

    @dataclass(frozen=True)
    class Eval:
        query_expansion: PromptFunction | None = None
        qa_prompt: PromptFunction | None = None
        eval_prompt: PromptFunction | None = None
        eval_add_episode_results: PromptFunction | None = None

    extract_nodes: ExtractNodes | None = None
    dedupe_nodes: DedupeNodes | None = None
    extract_edges: ExtractEdges | None = None
    extract_nodes_and_edges: ExtractNodesAndEdges | None = None
    dedupe_edges: DedupeEdges | None = None
    summarize_nodes: SummarizeNodes | None = None
    summarize_sagas: SummarizeSagas | None = None
    eval: Eval | None = None


def _nested_group_class(parent: type, group_name: str) -> type:
    nested_name = ''.join(part.capitalize() for part in group_name.split('_'))
    return getattr(parent, nested_name)


def flatten_overrides(overrides: LLMPromptOverrides | None) -> dict[str, PromptFunction]:
    """Turn nested override classes into ``group.function`` → builder."""
    if overrides is None:
        return {}
    flat: dict[str, PromptFunction] = {}
    for group_field in fields(overrides):
        group_name = group_field.name
        if group_name not in PROMPT_GROUPS:
            continue
        group = getattr(overrides, group_name)
        if group is None:
            continue
        expected = _nested_group_class(LLMPromptOverrides, group_name)
        if not isinstance(group, expected):
            raise TypeError(
                f'prompt_overrides.{group_name} must be {expected.__qualname__}, '
                f'got {type(group).__name__}'
            )
        for method_name in PROMPT_GROUPS[group_name]:
            builder = getattr(group, method_name)
            if builder is None:
                continue
            if not callable(builder):
                raise TypeError(
                    f'prompt_overrides.{group_name}.{method_name} must be callable, '
                    f'got {type(builder).__name__}'
                )
            flat[f'{group_name}.{method_name}'] = cast(PromptFunction, builder)
    return flat


@dataclass(frozen=True)
class LLMModel:
    """A provider model id, plus optional per-model prompt text overrides.

    Graphiti does not assign nicknames. Bind this object to a local Python
    variable and pass that variable into ``PromptRoutes``.
    """

    id: str
    small_id: str | None = None
    prompt_overrides: LLMPromptOverrides | None = field(default=None, compare=False, hash=False)
    max_tokens: int | None = None
    _flat_overrides: Mapping[str, PromptFunction] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError('LLMModel.id must be a non-empty string')
        if self.small_id is not None and not self.small_id.strip():
            raise ValueError('LLMModel.small_id must be a non-empty string')
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError('LLMModel.max_tokens must be positive')
        object.__setattr__(
            self, '_flat_overrides', MappingProxyType(flatten_overrides(self.prompt_overrides))
        )

    @property
    def flat_overrides(self) -> Mapping[str, PromptFunction]:
        return self._flat_overrides


@dataclass(frozen=True)
class PromptRoutes:
    """Per-prompt or per-group model routing. Unknown field names are a type error.

    Pass an ``LLMModel`` to route a whole group, or a nested group class to route
    individual prompts. Nested ``default`` is the group fallback when some methods
    are also set.
    """

    @dataclass(frozen=True)
    class ExtractNodes:
        default: LLMModel | None = None
        extract_message: LLMModel | None = None
        extract_json: LLMModel | None = None
        extract_text: LLMModel | None = None
        classify_nodes: LLMModel | None = None
        extract_attributes: LLMModel | None = None
        extract_summary: LLMModel | None = None
        extract_summaries_batch: LLMModel | None = None
        extract_entity_summaries_from_episodes: LLMModel | None = None

    @dataclass(frozen=True)
    class DedupeNodes:
        default: LLMModel | None = None
        node: LLMModel | None = None
        node_list: LLMModel | None = None
        nodes: LLMModel | None = None

    @dataclass(frozen=True)
    class ExtractEdges:
        default: LLMModel | None = None
        edge: LLMModel | None = None
        extract_attributes: LLMModel | None = None
        extract_timestamps: LLMModel | None = None
        extract_timestamps_batch: LLMModel | None = None

    @dataclass(frozen=True)
    class ExtractNodesAndEdges:
        default: LLMModel | None = None
        extract_message: LLMModel | None = None

    @dataclass(frozen=True)
    class DedupeEdges:
        default: LLMModel | None = None
        resolve_edge: LLMModel | None = None

    @dataclass(frozen=True)
    class SummarizeNodes:
        default: LLMModel | None = None
        summarize_pair: LLMModel | None = None
        summarize_context: LLMModel | None = None
        summary_description: LLMModel | None = None

    @dataclass(frozen=True)
    class SummarizeSagas:
        default: LLMModel | None = None
        summarize_saga: LLMModel | None = None

    @dataclass(frozen=True)
    class Eval:
        default: LLMModel | None = None
        query_expansion: LLMModel | None = None
        qa_prompt: LLMModel | None = None
        eval_prompt: LLMModel | None = None
        eval_add_episode_results: LLMModel | None = None

    extract_nodes: LLMModel | ExtractNodes | None = None
    dedupe_nodes: LLMModel | DedupeNodes | None = None
    extract_edges: LLMModel | ExtractEdges | None = None
    extract_nodes_and_edges: LLMModel | ExtractNodesAndEdges | None = None
    dedupe_edges: LLMModel | DedupeEdges | None = None
    summarize_nodes: LLMModel | SummarizeNodes | None = None
    summarize_sagas: LLMModel | SummarizeSagas | None = None
    eval: LLMModel | Eval | None = None


def flatten_routes(routes: PromptRoutes | None) -> dict[str, LLMModel]:
    """Turn ``PromptRoutes`` into group and ``group.function`` → ``LLMModel``."""
    if routes is None:
        return {}
    flat: dict[str, LLMModel] = {}
    for group_field in fields(routes):
        group_name = group_field.name
        if group_name not in PROMPT_GROUPS:
            continue
        value = getattr(routes, group_name)
        if value is None:
            continue
        if isinstance(value, LLMModel):
            flat[group_name] = value
            continue
        expected = _nested_group_class(PromptRoutes, group_name)
        if not isinstance(value, expected):
            raise TypeError(
                f'routes.{group_name} must be LLMModel or {expected.__qualname__}, '
                f'got {type(value).__name__}'
            )
        if value.default is not None:
            if not isinstance(value.default, LLMModel):
                raise TypeError(
                    f'routes.{group_name}.default must be LLMModel, '
                    f'got {type(value.default).__name__}'
                )
            flat[group_name] = value.default
        for method_name in PROMPT_GROUPS[group_name]:
            model = getattr(value, method_name)
            if model is None:
                continue
            if not isinstance(model, LLMModel):
                raise TypeError(
                    f'routes.{group_name}.{method_name} must be LLMModel, '
                    f'got {type(model).__name__}'
                )
            flat[f'{group_name}.{method_name}'] = model
    return flat
