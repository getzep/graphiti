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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from .dedupe_edges import DedupeEdgesPrompts, DefaultDedupeEdgesPrompts, EdgeDuplicate
from .dedupe_nodes import (
    DedupeNodesPrompts,
    DefaultDedupeNodesPrompts,
    NodeDuplicate,
    NodeResolutions,
)
from .eval import (
    DefaultEvalPrompts,
    EvalAddEpisodeResults,
    EvalPrompts,
    EvalResponse,
    QAResponse,
    QueryExpansion,
)
from .extract_edges import (
    BatchEdgeTimestamps,
    DefaultExtractEdgesPrompts,
    EdgeTimestamps,
    ExtractedEdges,
    ExtractEdgesPrompts,
)
from .extract_nodes import (
    DefaultExtractNodesPrompts,
    ExtractedEntities,
    ExtractNodesPrompts,
    SummarizedEntities,
)
from .extract_nodes_and_edges import (
    CombinedExtraction,
    DefaultExtractNodesAndEdgesPrompts,
    ExtractNodesAndEdgesPrompts,
)
from .models import ChatPrompt, PromptFunction, PromptSpec
from .summarize_nodes import (
    DefaultSummarizeNodesPrompts,
    SummarizeNodesPrompts,
    Summary,
    SummaryDescription,
)
from .summarize_sagas import DefaultSummarizeSagasPrompts, SagaSummary, SummarizeSagasPrompts

# Canonical group -> method names for structural validation and overrides.
PROMPT_GROUPS: dict[str, tuple[str, ...]] = {
    'extract_nodes': (
        'extract_message',
        'extract_json',
        'extract_text',
        'classify_nodes',
        'extract_attributes',
        'extract_summary',
        'extract_summaries_batch',
        'extract_entity_summaries_from_episodes',
    ),
    'dedupe_nodes': ('node', 'node_list', 'nodes'),
    'extract_edges': (
        'edge',
        'extract_attributes',
        'extract_timestamps',
        'extract_timestamps_batch',
    ),
    'extract_nodes_and_edges': ('extract_message',),
    'dedupe_edges': ('resolve_edge',),
    'summarize_nodes': ('summarize_pair', 'summarize_context', 'summary_description'),
    'summarize_sagas': ('summarize_saga',),
    'eval': (
        'query_expansion',
        'qa_prompt',
        'eval_prompt',
        'eval_add_episode_results',
    ),
}

# Backward-compatible alias used by older tests.
PROMPT_LIBRARY_IMPL: dict[str, dict[str, None]] = {
    group: {name: None for name in methods} for group, methods in PROMPT_GROUPS.items()
}

CHAT_PROMPT_MIGRATION_NOTE = (
    'Prompt overrides must return ChatPrompt(system=..., user=...). '
    'Returning list[Message] is no longer supported.'
)


def _builtin_specs() -> dict[str, PromptSpec]:
    """Fixed prompt_name → response_model registry. Schemas are not overridable."""
    return {
        'extract_nodes.extract_message': PromptSpec(
            name='extract_nodes.extract_message',
            response_model=ExtractedEntities,
        ),
        'extract_nodes.extract_text': PromptSpec(
            name='extract_nodes.extract_text',
            response_model=ExtractedEntities,
        ),
        'extract_nodes.extract_json': PromptSpec(
            name='extract_nodes.extract_json',
            response_model=ExtractedEntities,
        ),
        'extract_nodes.classify_nodes': PromptSpec(
            name='extract_nodes.classify_nodes',
            response_model=None,
        ),
        'extract_nodes.extract_attributes': PromptSpec(
            name='extract_nodes.extract_attributes',
            response_model=None,
            dynamic_schema=True,
        ),
        'extract_nodes.extract_summary': PromptSpec(
            name='extract_nodes.extract_summary',
            response_model=None,
        ),
        'extract_nodes.extract_summaries_batch': PromptSpec(
            name='extract_nodes.extract_summaries_batch',
            response_model=SummarizedEntities,
        ),
        'extract_nodes.extract_entity_summaries_from_episodes': PromptSpec(
            name='extract_nodes.extract_entity_summaries_from_episodes',
            response_model=SummarizedEntities,
        ),
        'dedupe_nodes.node': PromptSpec(name='dedupe_nodes.node', response_model=NodeDuplicate),
        'dedupe_nodes.node_list': PromptSpec(name='dedupe_nodes.node_list', response_model=None),
        'dedupe_nodes.nodes': PromptSpec(name='dedupe_nodes.nodes', response_model=NodeResolutions),
        'extract_edges.edge': PromptSpec(name='extract_edges.edge', response_model=ExtractedEdges),
        'extract_edges.extract_attributes': PromptSpec(
            name='extract_edges.extract_attributes',
            response_model=None,
            dynamic_schema=True,
        ),
        'extract_edges.extract_timestamps': PromptSpec(
            name='extract_edges.extract_timestamps',
            response_model=EdgeTimestamps,
        ),
        'extract_edges.extract_timestamps_batch': PromptSpec(
            name='extract_edges.extract_timestamps_batch',
            response_model=BatchEdgeTimestamps,
        ),
        'extract_nodes_and_edges.extract_message': PromptSpec(
            name='extract_nodes_and_edges.extract_message',
            response_model=CombinedExtraction,
        ),
        'dedupe_edges.resolve_edge': PromptSpec(
            name='dedupe_edges.resolve_edge',
            response_model=EdgeDuplicate,
        ),
        'summarize_nodes.summarize_pair': PromptSpec(
            name='summarize_nodes.summarize_pair',
            response_model=Summary,
        ),
        'summarize_nodes.summarize_context': PromptSpec(
            name='summarize_nodes.summarize_context',
            response_model=Summary,
        ),
        'summarize_nodes.summary_description': PromptSpec(
            name='summarize_nodes.summary_description',
            response_model=SummaryDescription,
        ),
        'summarize_sagas.summarize_saga': PromptSpec(
            name='summarize_sagas.summarize_saga',
            response_model=SagaSummary,
        ),
        'eval.query_expansion': PromptSpec(
            name='eval.query_expansion', response_model=QueryExpansion
        ),
        'eval.qa_prompt': PromptSpec(name='eval.qa_prompt', response_model=QAResponse),
        'eval.eval_prompt': PromptSpec(name='eval.eval_prompt', response_model=EvalResponse),
        'eval.eval_add_episode_results': PromptSpec(
            name='eval.eval_add_episode_results',
            response_model=EvalAddEpisodeResults,
        ),
    }


BUILTIN_PROMPT_SPECS: Mapping[str, PromptSpec] = MappingProxyType(_builtin_specs())


def resolve_response_model(
    prompt_name: str,
    response_model: type[BaseModel] | None = None,
) -> type[BaseModel] | None:
    """Return the fixed schema for ``prompt_name``, or the caller schema if dynamic.

    Raises ``ValueError`` for unknown names, missing dynamic schemas, and attempts
    to replace a fixed schema.
    """
    spec = BUILTIN_PROMPT_SPECS.get(prompt_name)
    if spec is None:
        raise ValueError(f'Unknown prompt_name: {prompt_name}')

    if spec.dynamic_schema:
        if response_model is None:
            raise ValueError(
                f'prompt {prompt_name} has dynamic_schema=True; response_model is required'
            )
        return response_model

    if response_model is not None and response_model is not spec.response_model:
        raise ValueError(
            f'Prompt schema overrides are not allowed for {prompt_name}: '
            f'got {response_model.__name__}, expected '
            f'{spec.response_model.__name__ if spec.response_model else None}'
        )
    return spec.response_model


class PromptLibrary(ABC):
    """Top-level prompt library ABC. Prefer subclassing; duck-typed libraries also work."""

    @property
    @abstractmethod
    def extract_nodes(self) -> ExtractNodesPrompts: ...

    @property
    @abstractmethod
    def dedupe_nodes(self) -> DedupeNodesPrompts: ...

    @property
    @abstractmethod
    def extract_edges(self) -> ExtractEdgesPrompts: ...

    @property
    @abstractmethod
    def extract_nodes_and_edges(self) -> ExtractNodesAndEdgesPrompts: ...

    @property
    @abstractmethod
    def dedupe_edges(self) -> DedupeEdgesPrompts: ...

    @property
    @abstractmethod
    def summarize_nodes(self) -> SummarizeNodesPrompts: ...

    @property
    @abstractmethod
    def summarize_sagas(self) -> SummarizeSagasPrompts: ...

    @property
    @abstractmethod
    def eval(self) -> EvalPrompts: ...

    @property
    @abstractmethod
    def specs(self) -> Mapping[str, PromptSpec]: ...


class DefaultPromptLibrary(PromptLibrary):
    """Built-in Graphiti prompt library."""

    def __init__(self) -> None:
        self._extract_nodes = DefaultExtractNodesPrompts()
        self._dedupe_nodes = DefaultDedupeNodesPrompts()
        self._extract_edges = DefaultExtractEdgesPrompts()
        self._extract_nodes_and_edges = DefaultExtractNodesAndEdgesPrompts()
        self._dedupe_edges = DefaultDedupeEdgesPrompts()
        self._summarize_nodes = DefaultSummarizeNodesPrompts()
        self._summarize_sagas = DefaultSummarizeSagasPrompts()
        self._eval = DefaultEvalPrompts()
        self._specs = BUILTIN_PROMPT_SPECS

    @property
    def extract_nodes(self) -> ExtractNodesPrompts:
        return self._extract_nodes

    @property
    def dedupe_nodes(self) -> DedupeNodesPrompts:
        return self._dedupe_nodes

    @property
    def extract_edges(self) -> ExtractEdgesPrompts:
        return self._extract_edges

    @property
    def extract_nodes_and_edges(self) -> ExtractNodesAndEdgesPrompts:
        return self._extract_nodes_and_edges

    @property
    def dedupe_edges(self) -> DedupeEdgesPrompts:
        return self._dedupe_edges

    @property
    def summarize_nodes(self) -> SummarizeNodesPrompts:
        return self._summarize_nodes

    @property
    def summarize_sagas(self) -> SummarizeSagasPrompts:
        return self._summarize_sagas

    @property
    def eval(self) -> EvalPrompts:
        return self._eval

    @property
    def specs(self) -> Mapping[str, PromptSpec]:
        return self._specs


PromptOverrides = dict[str, dict[str, PromptFunction]]


class _OverrideGroup:
    """Group proxy that prefers override callables, else delegates to the base group."""

    def __init__(self, base: Any, overrides: dict[str, PromptFunction]) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            override = self._overrides[name]

            def _bound(context: dict[str, Any]) -> ChatPrompt:
                result = override(context)
                return _ensure_chat_prompt(result, f'override.{name}')

            return _bound
        return getattr(self._base, name)


class _ComposedPromptLibrary:
    """Duck-typed library composed from defaults + partial overrides."""

    def __init__(self, base: PromptLibrary, overrides: PromptOverrides) -> None:
        self._base = base
        self._groups: dict[str, Any] = {}
        for group_name in PROMPT_GROUPS:
            group_overrides = overrides.get(group_name, {})
            base_group = getattr(base, group_name)
            if group_overrides:
                self._groups[group_name] = _OverrideGroup(base_group, group_overrides)
            else:
                self._groups[group_name] = base_group

    def __getattr__(self, name: str) -> Any:
        if name == 'specs':
            return self._base.specs
        if name in self._groups:
            return self._groups[name]
        raise AttributeError(name)


def ensure_chat_prompt(result: Any, label: str) -> ChatPrompt:
    if isinstance(result, ChatPrompt):
        return result
    if isinstance(result, list):
        raise TypeError(f'{label} returned list[Message]; {CHAT_PROMPT_MIGRATION_NOTE}')
    raise TypeError(
        f'{label} must return ChatPrompt, got {type(result).__name__}. {CHAT_PROMPT_MIGRATION_NOTE}'
    )


def _ensure_chat_prompt(result: Any, label: str) -> ChatPrompt:
    return ensure_chat_prompt(result, label)


def get_prompt_builder(library: Any, prompt_name: str) -> PromptFunction:
    """Resolve ``group.method`` from a prompt library to a callable builder."""
    if '.' not in prompt_name:
        raise ValueError(f'Invalid prompt_name (expected group.method): {prompt_name}')
    group_name, method_name = prompt_name.split('.', 1)
    if group_name not in PROMPT_GROUPS:
        raise ValueError(f'Unknown prompt group: {group_name}')
    if method_name not in PROMPT_GROUPS[group_name]:
        raise ValueError(f'Unknown prompt function for group {group_name}: {method_name}')
    group = getattr(library, group_name)
    builder = getattr(group, method_name)
    if not callable(builder):
        raise ValueError(f'Prompt library function must be callable: {prompt_name}')

    def _call(context: dict[str, Any]) -> ChatPrompt:
        return ensure_chat_prompt(builder(context), prompt_name)

    return _call


prompt_library: PromptLibrary = DefaultPromptLibrary()


def create_prompt_library(overrides: PromptOverrides | None = None) -> PromptLibrary:
    """Create a prompt library, optionally applying partial overrides to the defaults.

    Override callables must return ``ChatPrompt``. Returning ``list[Message]`` raises
    ``TypeError`` with a migration note.
    """
    if not overrides:
        return DefaultPromptLibrary()

    for group_name, group_overrides in overrides.items():
        if group_name not in PROMPT_GROUPS:
            raise ValueError(f'Unknown prompt group: {group_name}')
        for function_name, function in group_overrides.items():
            if function_name not in PROMPT_GROUPS[group_name]:
                raise ValueError(f'Unknown prompt function for group {group_name}: {function_name}')
            if not callable(function):
                raise ValueError(f'Prompt override must be callable: {group_name}.{function_name}')

    return _ComposedPromptLibrary(DefaultPromptLibrary(), overrides)  # type: ignore[return-value]


class _LibraryWithBuiltinSpecs:
    """Attach fixed builtin specs to a duck-typed library that lacks them."""

    def __init__(self, library: Any) -> None:
        self._library = library
        self._specs = BUILTIN_PROMPT_SPECS

    @property
    def specs(self) -> Mapping[str, PromptSpec]:
        return self._specs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._library, name)


def validate_prompt_library(library: Any) -> None:
    """Structurally validate a complete prompt library (duck-typed; ABC not required)."""
    for group_name, function_names in PROMPT_GROUPS.items():
        if not hasattr(library, group_name):
            raise ValueError(f'Prompt library missing group: {group_name}')

        group = getattr(library, group_name)
        for function_name in function_names:
            if not hasattr(group, function_name):
                raise ValueError(f'Prompt library missing function: {group_name}.{function_name}')

            function = getattr(group, function_name)
            if not callable(function):
                raise ValueError(
                    f'Prompt library function must be callable: {group_name}.{function_name}'
                )

    if hasattr(library, 'specs'):
        specs = library.specs
        if not isinstance(specs, Mapping):
            raise ValueError('Prompt library specs must be a mapping')
        for name, spec in BUILTIN_PROMPT_SPECS.items():
            if name not in specs:
                raise ValueError(f'Prompt library specs missing entry: {name}')
            lib_spec = specs[name]
            if not isinstance(lib_spec, PromptSpec):
                raise ValueError(f'Prompt library specs[{name!r}] must be PromptSpec')
            if lib_spec.dynamic_schema != spec.dynamic_schema:
                raise ValueError(
                    f'Prompt schema overrides are not allowed: {name} dynamic_schema mismatch'
                )
            if lib_spec.response_model is not spec.response_model:
                raise ValueError(
                    f'Prompt schema overrides are not allowed: {name} response_model mismatch'
                )


def ensure_prompt_library_wrapped(library: Any) -> Any:
    """Validate and return the library, attaching builtin specs when missing.

    Unicode post-processing is handled by ``ChatPrompt.as_messages``; no VersionWrapper.
    """
    validate_prompt_library(library)
    if not hasattr(library, 'specs'):
        return _LibraryWithBuiltinSpecs(library)
    return library
