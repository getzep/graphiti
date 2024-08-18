from typing import TypedDict, Protocol
from .models import Message, PromptFunction
from typing import TypedDict, Protocol
from .models import Message, PromptFunction
from .extract_nodes import (
    Prompt as ExtractNodesPrompt,
    Versions as ExtractNodesVersions,
    versions as extract_nodes_versions,
)

from .dedupe_nodes import (
    Prompt as DedupeNodesPrompt,
    Versions as DedupeNodesVersions,
    versions as dedupe_nodes_versions,
)

from .extract_edges import (
    Prompt as ExtractEdgesPrompt,
    Versions as ExtractEdgesVersions,
    versions as extract_edges_versions,
)

from .dedupe_edges import (
    Prompt as DedupeEdgesPrompt,
    Versions as DedupeEdgesVersions,
    versions as dedupe_edges_versions,
)


class PromptLibrary(Protocol):
    extract_nodes: ExtractNodesPrompt
    dedupe_nodes: DedupeNodesPrompt
    extract_edges: ExtractEdgesPrompt
    dedupe_edges: DedupeEdgesPrompt


class PromptLibraryImpl(TypedDict):
    extract_nodes: ExtractNodesVersions
    dedupe_nodes: DedupeNodesVersions
    extract_edges: ExtractEdgesVersions
    dedupe_edges: DedupeEdgesVersions


class VersionWrapper:
    def __init__(self, func: PromptFunction):
        self.func = func

    def __call__(self, context: dict[str, any]) -> list[Message]:
        return self.func(context)


class PromptTypeWrapper:
    def __init__(self, versions: dict[str, PromptFunction]):
        for version, func in versions.items():
            setattr(self, version, VersionWrapper(func))


class PromptLibraryWrapper:
    def __init__(self, library: PromptLibraryImpl):
        for prompt_type, versions in library.items():
            setattr(self, prompt_type, PromptTypeWrapper(versions))


PROMPT_LIBRARY_IMPL: PromptLibraryImpl = {
    "extract_nodes": extract_nodes_versions,
    "dedupe_nodes": dedupe_nodes_versions,
    "extract_edges": extract_edges_versions,
    "dedupe_edges": dedupe_edges_versions,
}

prompt_library: PromptLibrary = PromptLibraryWrapper(PROMPT_LIBRARY_IMPL)
