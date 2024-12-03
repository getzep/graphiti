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

from typing import Any, Protocol, TypedDict

from .dedupe_edges import (
    Prompt as DedupeEdgesPrompt,
)
from .dedupe_edges import (
    Versions as DedupeEdgesVersions,
)
from .dedupe_edges import (
    versions as dedupe_edges_versions,
)
from .dedupe_nodes import (
    Prompt as DedupeNodesPrompt,
)
from .dedupe_nodes import (
    Versions as DedupeNodesVersions,
)
from .dedupe_nodes import (
    versions as dedupe_nodes_versions,
)
from .eval import Prompt as EvalPrompt
from .eval import Versions as EvalVersions
from .eval import versions as eval_versions
from .extract_edge_dates import (
    Prompt as ExtractEdgeDatesPrompt,
)
from .extract_edge_dates import (
    Versions as ExtractEdgeDatesVersions,
)
from .extract_edge_dates import (
    versions as extract_edge_dates_versions,
)
from .extract_edges import (
    Prompt as ExtractEdgesPrompt,
)
from .extract_edges import (
    Versions as ExtractEdgesVersions,
)
from .extract_edges import (
    versions as extract_edges_versions,
)
from .extract_nodes import (
    Prompt as ExtractNodesPrompt,
)
from .extract_nodes import (
    Versions as ExtractNodesVersions,
)
from .extract_nodes import (
    versions as extract_nodes_versions,
)
from .invalidate_edges import (
    Prompt as InvalidateEdgesPrompt,
)
from .invalidate_edges import (
    Versions as InvalidateEdgesVersions,
)
from .invalidate_edges import (
    versions as invalidate_edges_versions,
)
from .models import Message, PromptFunction
from .prompt_helpers import DO_NOT_ESCAPE_UNICODE
from .summarize_nodes import Prompt as SummarizeNodesPrompt
from .summarize_nodes import Versions as SummarizeNodesVersions
from .summarize_nodes import versions as summarize_nodes_versions


class PromptLibrary(Protocol):
    extract_nodes: ExtractNodesPrompt
    dedupe_nodes: DedupeNodesPrompt
    extract_edges: ExtractEdgesPrompt
    dedupe_edges: DedupeEdgesPrompt
    invalidate_edges: InvalidateEdgesPrompt
    extract_edge_dates: ExtractEdgeDatesPrompt
    summarize_nodes: SummarizeNodesPrompt
    eval: EvalPrompt


class PromptLibraryImpl(TypedDict):
    extract_nodes: ExtractNodesVersions
    dedupe_nodes: DedupeNodesVersions
    extract_edges: ExtractEdgesVersions
    dedupe_edges: DedupeEdgesVersions
    invalidate_edges: InvalidateEdgesVersions
    extract_edge_dates: ExtractEdgeDatesVersions
    summarize_nodes: SummarizeNodesVersions
    eval: EvalVersions


class VersionWrapper:
    def __init__(self, func: PromptFunction):
        self.func = func

    def __call__(self, context: dict[str, Any]) -> list[Message]:
        messages = self.func(context)
        for message in messages:
            message.content += DO_NOT_ESCAPE_UNICODE if message.role == 'system' else ''
        return messages


class PromptTypeWrapper:
    def __init__(self, versions: dict[str, PromptFunction]):
        for version, func in versions.items():
            setattr(self, version, VersionWrapper(func))


class PromptLibraryWrapper:
    def __init__(self, library: PromptLibraryImpl):
        for prompt_type, versions in library.items():
            setattr(self, prompt_type, PromptTypeWrapper(versions))  # type: ignore[arg-type]


PROMPT_LIBRARY_IMPL: PromptLibraryImpl = {
    'extract_nodes': extract_nodes_versions,
    'dedupe_nodes': dedupe_nodes_versions,
    'extract_edges': extract_edges_versions,
    'dedupe_edges': dedupe_edges_versions,
    'invalidate_edges': invalidate_edges_versions,
    'extract_edge_dates': extract_edge_dates_versions,
    'summarize_nodes': summarize_nodes_versions,
    'eval': eval_versions,
}
prompt_library: PromptLibrary = PromptLibraryWrapper(PROMPT_LIBRARY_IMPL)  # type: ignore[assignment]
