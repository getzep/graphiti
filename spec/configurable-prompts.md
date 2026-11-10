# Configurable Prompt Library

Specification	Configurable Prompt Library
Category	Graphiti Core
Drafted At	2026-05-12
Authors
Paul Paliychuk

## 1. Overview

Graphiti must keep its existing module-level prompt API and support opt-in prompt
customization through a separate chat prompt library. Callers can provide a chat prompt
library when they create a Graphiti client. Prompt selection must be deterministic and
compatible with the existing default prompts.

Legacy call sites use module-level prompt functions. The `generate_prompt_response` helper
uses a configured chat prompt library or runtime when a real `GraphitiClients` bundle has
one. Otherwise, it calls the legacy prompt function at call time.

This feature does not change the response models, graph schema, LLM client API, embedding
behavior, database drivers, search behavior, or prompt output contracts. It only changes how
prompt message builders are selected.

## 2. Terminology

### 2.1. Prompt Function

A legacy prompt function accepts a context map and returns ``list[Message]``. The
``PromptFunction`` alias keeps this meaning.

A chat prompt builder accepts a context map and returns a ``ChatPrompt`` (typed system + user
messages). Call ``ChatPrompt.as_messages()`` to render transport ``Message`` objects,
including the do-not-escape-unicode note on the system message. The ``ChatPromptFunction``
alias names this builder type.

Pseudocode:

```text
PromptFunction(context: Map<String, Any>) -> list[Message]
ChatPromptFunction(context: Map<String, Any>) -> ChatPrompt
```

Chat prompt overrides must return ``ChatPrompt``. Returning ``list[Message]`` from a chat
override raises ``TypeError``.

### 2.2. Prompt Group

A prompt group is a named collection of prompt functions for one domain. Examples include
node extraction, edge extraction, node deduplication, edge deduplication, node summaries,
saga summaries, combined node and edge extraction, and eval prompts.

### 2.3. Prompt Library

A prompt library is an object with one attribute per prompt group. Each group exposes the
same prompt function names as the default Graphiti prompt library.

### 2.4. Default Prompt Library

The package-level ``prompt_library`` is the built-in legacy library. It returns
``list[Message]`` and must remain available for backward compatibility. The opt-in chat API
uses ``ChatPromptLibrary`` and ``DefaultChatPromptLibrary``.

### 2.5. Prompt Overrides

Prompt overrides are a partial prompt definition supplied by a caller. Overrides replace only
the named prompt functions they contain. All unspecified prompt functions are inherited from
the default prompt library.

### 2.6. Inversion of Control

Inversion of control means Graphiti does not hard-code or globally select the prompt library
inside prompt-consuming code. Instead, the caller supplies a prompt library or prompt overrides
when constructing Graphiti, and Graphiti passes that dependency to the code that needs it.

## 3. Requirements

### 3.1. Instance-Scoped Configuration

Each Graphiti instance can have one configured chat prompt library. When no library is
configured, ``self.prompt_library`` and ``self.clients.prompt_library`` stay ``None``.
Different Graphiti instances may use different prompt libraries without affecting each other.

### 3.2. Default Compatibility

When no prompt library is provided, Graphiti must use the same built-in prompt functions and
produce the same prompt messages as it does before this feature.

### 3.3. Partial Override Support

Callers must be able to override one prompt function without redefining every prompt function.
Unspecified prompts must fall back to the built-in default implementation.

### 3.4. Full Library Support

Callers must be able to provide a complete prompt library object. A complete prompt library
must be used directly after validation and must not be merged with defaults.

### 3.5. Prompt Message Wrapping

All configured prompt functions must receive the same prompt message post-processing behavior
as default prompt functions. In particular, system messages must receive Graphiti's
do-not-escape-unicode suffix.

### 3.6. Response Schema Stability

Configurable prompts must not change response schemas. Every custom prompt must still produce
LLM responses compatible with the response model used at that call site.

### 3.7. Prompt Name Stability

The `prompt_name` passed to LLM clients must remain stable. Prompt customization must not
rename telemetry or token-tracking keys.

### 3.8. Backward-Compatible Imports

The existing module-level default prompt library import must remain available. Existing code
that imports the default prompt library for direct prompt rendering must continue to work.

### 3.9. No Global Runtime Mutation

The implementation must not use a global setter or mutate the process-wide default prompt
library to customize a Graphiti instance.

### 3.10. Public API Documentation

The public Graphiti constructor documentation must describe how to pass a prompt library and
how to compose partial prompt overrides with the prompt library helper.

## 4. Public API

### 4.1. Graphiti Constructor

The Graphiti constructor must accept one new optional parameter:

```text
Graphiti(
    ...existing parameters...,
    prompt_library: ChatPromptLibrary | None = None,
)
```

Rules:

1. If `prompt_library` is not provided, Graphiti keeps the prompt library fields unset and
   uses the legacy module-level prompt functions.
2. If `prompt_library` is provided, Graphiti uses it as the instance prompt library after
   validating its shape.
3. Partial customization is performed by calling `create_prompt_library(overrides)` before
   constructing Graphiti and passing the returned library as `prompt_library`.
4. The configured library is stored as `self.prompt_library` and
   `self.clients.prompt_library`.

### 4.2. Prompt Library Creation Helper

Graphiti must expose a helper that creates a prompt library from partial overrides:

```text
create_prompt_library(overrides: PromptOverrides | None = None) -> ChatPromptLibrary
```

Rules:

1. When `overrides` is omitted or empty, the helper returns a prompt library equivalent to
   the default prompt library.
2. When overrides are provided, each override replaces the corresponding default prompt
   function.
3. Unknown prompt group names are rejected.
4. Unknown prompt function names inside known groups are rejected.
5. Non-callable override values are rejected.
6. Returned prompt libraries wrap prompt functions with the standard prompt wrapper.

### 4.3. Public Exports

The prompt package must export:

```text
Message
PromptFunction
PromptLibrary (graphiti_core.prompts.lib)
ChatPromptFunction
ChatPromptLibrary
DefaultChatPromptLibrary
PromptOverrides
create_prompt_library
prompt_library
```

`PromptLibrary` is the legacy protocol for version maps. The package-level
`prompt_library` remains the legacy default wrapper and returns `list[Message]`.
`ChatPromptLibrary` names the opt-in prompt customization API.

### 4.4. Complete Prompt Library Objects

A complete prompt library object must expose every default prompt group and every prompt
function in each group. Implementations may be class instances, simple objects, or objects
created by `create_prompt_library`.

### 4.5. Partial Override Shape

Prompt overrides are a nested map:

```text
PromptOverrides = Map<PromptGroupName, Map<PromptFunctionName, ChatPromptFunction>>
```

Example:

```text
custom_prompts = create_prompt_library({
    "extract_nodes": {
        "extract_message": custom_extract_message,
    },
})

graphiti = Graphiti(..., prompt_library=custom_prompts)
```

## 5. Prompt Library Model

### 5.1. Default Prompt Groups

The prompt library must contain these prompt groups:

1. `extract_nodes`
2. `dedupe_nodes`
3. `extract_edges`
4. `extract_nodes_and_edges`
5. `dedupe_edges`
6. `summarize_nodes`
7. `summarize_sagas`
8. `eval`

### 5.2. Prompt Functions By Group

The prompt library must contain these prompt functions:

```text
extract_nodes:
  extract_message
  extract_json
  extract_text
  classify_nodes
  extract_attributes
  extract_summary
  extract_summaries_batch
  extract_entity_summaries_from_episodes

dedupe_nodes:
  node
  node_list
  nodes

extract_edges:
  edge
  extract_attributes
  extract_timestamps
  extract_timestamps_batch

extract_nodes_and_edges:
  extract_message

dedupe_edges:
  resolve_edge

summarize_nodes:
  summarize_pair
  summarize_context
  summary_description

summarize_sagas:
  summarize_saga

eval:
  query_expansion
  qa_prompt
  eval_prompt
  eval_add_episode_results
```

### 5.3. Prompt Library Implementation Map

Graphiti must keep an implementation map that contains raw prompt functions before wrapping.
This map is the canonical source for defaults and merging.

Pseudocode:

```text
DEFAULT_PROMPT_LIBRARY_IMPL = {
    "extract_nodes": extract_nodes_versions,
    "dedupe_nodes": dedupe_nodes_versions,
    "extract_edges": extract_edges_versions,
    "extract_nodes_and_edges": extract_nodes_and_edges_versions,
    "dedupe_edges": dedupe_edges_versions,
    "summarize_nodes": summarize_nodes_versions,
    "summarize_sagas": summarize_sagas_versions,
    "eval": eval_versions,
}
```

### 5.4. Prompt Function Wrapping

Every prompt function in a library created by Graphiti must be wrapped in a callable wrapper.
The wrapper must call the raw prompt function, then append the do-not-escape-unicode suffix to
every system message.

Pseudocode:

```text
function wrapped_prompt(context):
    messages = raw_prompt_function(context)
    for message in messages:
        if message.role == "system":
            message.content = message.content + DO_NOT_ESCAPE_UNICODE
    return messages
```

### 5.5. Validation

Graphiti must validate complete prompt libraries before storing them. The prompt library
creation helper must validate overrides before returning a composed prompt library.

For complete prompt libraries, validation must check:

1. Every required group exists.
2. Every required prompt function exists in every group.
3. Every prompt function is callable.

For prompt overrides, validation must check:

1. Every override group is a known group.
2. Every override function name is known in that group.
3. Every override value is callable.

Validation failures must raise `ValueError` with a message that names the invalid group or
function.

## 6. Runtime Dependency Flow

### 6.1. Graphiti Client State

Graphiti stores a configured chat prompt library in two places:

```text
self.prompt_library
self.clients.prompt_library
```

These fields stay ``None`` when the caller does not pass a library or runtime. Configured
chat prompt libraries are stored in both fields.

### 6.2. GraphitiClients

The Graphiti client dependency bundle must include the prompt library:

```text
GraphitiClients:
    driver
    llm_client
    embedder
    cross_encoder
    tracer
    prompt_library: optional
    llm_runtime: optional
```

### 6.3. Runtime Access Rule

Prompt-consuming call sites pass the legacy prompt builder and the optional ``clients`` bundle
to ``generate_prompt_response``. The helper uses ``clients.complete_prompt`` only when ``clients``
is a ``GraphitiClients`` instance with a runtime or chat prompt library. Otherwise, it calls the
legacy module-level prompt builder at call time.

Direct calls to ``GraphitiClients.complete_prompt`` use ``default_chat_prompt_library`` when no
chat prompt library is configured.

## 7. Required Refactoring

### 7.1. Graphiti Constructor

The Graphiti constructor must:

1. Add a `prompt_library` parameter.
2. Resolve the instance prompt library before creating `GraphitiClients`.
3. Store the resolved prompt library on `self.prompt_library`.
4. Add the resolved prompt library to `GraphitiClients`.
5. Document the parameter in the constructor docstring.

### 7.2. Graphiti Direct Prompt Calls

Graphiti methods that build prompts must use ``generate_prompt_response``.

The saga summary flow passes the module-level legacy builder:

```text
generate_prompt_response(
    self.llm_client,
    "summarize_sagas.summarize_saga",
    prompt_library.summarize_sagas.summarize_saga,
    context,
    clients=self.clients,
)
```

### 7.3. Node Maintenance Operations

Node maintenance operations must pass the module-level legacy builder and the optional
``clients`` bundle to ``generate_prompt_response``.

Required changes:

1. `extract_nodes` keeps accepting `clients`.
2. `_extract_nodes_single` and `_call_extraction_llm` pass ``clients`` through.
3. `_call_extraction_llm` reads the module-level builder for `extract_message`,
   `extract_text`, or `extract_json` at call time.
4. `_resolve_with_llm`, `_extract_entity_attributes`, and `_process_summary_flight` pass the
   legacy module-level builder to the helper.

### 7.4. Edge Maintenance Operations

Edge maintenance operations must pass the module-level legacy builder and the optional
``clients`` bundle to ``generate_prompt_response``.

Required changes:

1. `extract_edges` passes the module-level `extract_edges.edge` builder.
2. `resolve_extracted_edges` passes `clients` to each `resolve_extracted_edge` call.
3. `resolve_extracted_edge` passes the legacy builders for edge deduplication and attributes.
4. `_extract_edge_timestamps` passes the legacy timestamp builder.

### 7.5. Combined Extraction Operations

Combined extraction must pass the module-level legacy builder and the optional ``clients``
bundle to ``generate_prompt_response``.

Required changes:

1. `extract_nodes_and_edges` passes the module-level combined extraction builder.
2. Batch timestamp extraction passes the module-level batch timestamp builder.

### 7.6. Community Operations

Community operations keep their existing positional arguments and add an optional keyword-only
``clients`` parameter to each prompt-consuming helper.

Required changes:

1. Each helper passes the legacy builder to ``generate_prompt_response``.
2. Nested helper calls pass ``clients`` onward.
3. Graphiti call sites keep the main positional arguments and add ``clients=clients``.

### 7.7. Bulk Utilities

Bulk utilities pass ``GraphitiClients`` to node and edge maintenance functions. Those functions
use the configured chat prompt library or runtime when present, and keep the legacy prompt
path otherwise.

### 7.8. MCP Server Adoption

The MCP server may continue constructing Graphiti without prompt configuration. This preserves
current behavior.

If the MCP server later exposes prompt configuration, it must pass a `prompt_library` value to
the Graphiti constructor. If it accepts partial overrides, it must first compose them with
`create_prompt_library(overrides)`. It must not mutate the module-level default prompt library.

### 7.9. Graph Service Adoption

The graph service may continue constructing Graphiti without prompt configuration. This
preserves current behavior.

If service-level prompt configuration is added later, it must pass a `prompt_library` value to
the Graphiti constructor. If it accepts partial overrides, it must first compose them with
`create_prompt_library(overrides)`. It must not mutate the module-level default prompt library.

### 7.10. Eval Prompt Adoption

Eval prompt functions remain part of the prompt library and remain available from the default
prompt library. Runtime Graphiti code does not currently call eval prompts. Eval code that
directly imports the default prompt library may continue to do so.

## 8. Error Handling

### 8.1. Unknown Override Group

If prompt overrides include an unknown group, `create_prompt_library` must raise `ValueError`.

Error message format:

```text
Unknown prompt group: <group>
```

### 8.2. Unknown Override Function

If prompt overrides include an unknown function in a known group, `create_prompt_library` must
raise `ValueError`.

Error message format:

```text
Unknown prompt function for group <group>: <function>
```

### 8.3. Non-Callable Override

If prompt overrides include a non-callable value, `create_prompt_library` must raise
`ValueError`.

Error message format:

```text
Prompt override must be callable: <group>.<function>
```

### 8.4. Incomplete Complete Library

If a complete prompt library is missing a required group or function, Graphiti must raise
`ValueError`.

Error message formats:

```text
Prompt library missing group: <group>
Prompt library missing function: <group>.<function>
Prompt library function must be callable: <group>.<function>
```

## 9. Compatibility

### 9.1. Existing Constructors

Existing Graphiti constructor calls must continue to work without modification.

### 9.2. Existing Prompt Imports

Existing imports of the default prompt library must continue to work:

```text
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.lib import prompt_library
```

### 9.3. Existing Prompt Names

Existing `prompt_name` strings passed to `generate_response` must not change.

### 9.4. Existing Tests

Tests that monkeypatch module-level legacy prompt imports must continue to work because
call sites read the builder at call time. Tests that directly render default prompts may
continue using the module-level default prompt library.

### 9.5. Type Checking

The prompt library protocol and override types must be exported so downstream users can type
custom prompt libraries. The implementation must pass existing type checks.

## 10. Documentation

### 10.1. Constructor Documentation

The Graphiti constructor docstring must document:

1. `prompt_library`
2. Default behavior
3. How to use `create_prompt_library` for partial overrides
4. The requirement that custom prompts preserve response schemas

### 10.2. README Documentation

The README must include a configurable prompts example showing:

1. A custom prompt function.
2. Partial override construction with `create_prompt_library`.
3. Passing the composed prompt library to the Graphiti constructor.
4. A note that prompt outputs must match the expected response model.

### 10.3. API Export Documentation

The prompt package documentation must list the public prompt customization exports.

## 11. Implementation Algorithm

### 11.1. Create Prompt Library From Overrides

Pseudocode:

```text
function create_prompt_library(overrides = null):
    base = DefaultChatPromptLibrary()

    if overrides is null or overrides is empty:
        return base

    for group_name, group_overrides in overrides:
        if group_name not in PROMPT_GROUPS:
            raise ValueError("Unknown prompt group: " + group_name)

        for function_name, function in group_overrides:
            if function_name not in PROMPT_GROUPS[group_name]:
                raise ValueError(
                    "Unknown prompt function for group " + group_name + ": " + function_name
                )

            if not callable(function):
                raise ValueError(
                    "Prompt override must be callable: " + group_name + "." + function_name
                )

            validate_chat_prompt_function(function)

    return ComposedChatPromptLibrary(base, overrides)
```

### 11.2. Resolve Constructor Prompt Library

Pseudocode:

```text
function resolve_prompt_library(prompt_library):
    if prompt_library is not null:
        validate_prompt_library(prompt_library)
        return prompt_library

    return null
```

### 11.3. Validate Complete Prompt Library

Pseudocode:

```text
function validate_prompt_library(library):
    for group_name, function_names in PROMPT_GROUPS:
        if not has_attribute(library, group_name):
            raise ValueError("Prompt library missing group: " + group_name)

        group = get_attribute(library, group_name)

        for function_name in function_names:
            if not has_attribute(group, function_name):
                raise ValueError(
                    "Prompt library missing function: " + group_name + "." + function_name
                )

            function = get_attribute(group, function_name)
            if not callable(function):
                raise ValueError(
                    "Prompt library function must be callable: "
                    + group_name
                    + "."
                    + function_name
                )
```

## 12. Acceptance Criteria

### 12.1. Default Behavior Acceptance

A Graphiti instance created without prompt configuration uses the built-in default prompts.

### 12.2. Partial Override Acceptance

A Graphiti instance created with a prompt library composed from a partial override uses the
custom prompt only for the specified group and function, and uses defaults for every other
prompt.

### 12.3. Complete Library Acceptance

A Graphiti instance created with a complete prompt library uses that library for every runtime
prompt call.

### 12.4. Instance Isolation Acceptance

Two Graphiti instances with different prompt libraries can generate prompts concurrently
without cross-contamination.

### 12.5. Runtime Coverage Acceptance

Every runtime prompt call in Graphiti core uses the instance prompt library, not the
module-level default prompt library.

### 12.6. Backward Compatibility Acceptance

Existing public imports and constructor calls continue to work.

### 12.7. Validation Acceptance

Invalid prompt libraries and invalid overrides fail during construction or helper invocation
with deterministic `ValueError` messages.

### 12.8. Documentation Acceptance

Constructor and README documentation explain custom prompt configuration and schema
compatibility requirements.
