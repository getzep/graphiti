# LLM Runtime

Specification	LLM Runtime
Category	Graphiti Core
Drafted At	2026-08-07
Authors
Paul Paliychuk

## 1. Overview

`LLMRuntime` is an opt-in object that routes prompts to models across one or
more provider transports. It contains:

1. A required default `LLMModel` (omitting it is a type error)
2. Optional `PromptRoutes` (per-group `LLMModel` or nested group class)
3. Optional `LLMPromptOverrides` (nested group classes of builders)
4. A prompt library (`ChatPrompt` builders). Schemas live in the immutable
   `BUILTIN_PROMPT_SPECS` registry and are not user-configurable.

`LLMTransport` wraps one `LLMClient` and an optional set of model IDs. Each
`LLMModel` binds one model ID to a transport. The model ID must appear in the
transport's `models` set when the set is configured.

There are no caller-invented model nicknames. Bind `LLMModel` instances to local
Python variables and pass those variables into `PromptRoutes`.

Unknown prompt names are constructor / type errors on the nested dataclasses.
Graphiti keeps the legacy module-level prompt path when no chat library or runtime is configured.
The package-level `prompt_library` also keeps its `list[Message]` builders.
The runtime uses the separate `ChatPromptLibrary` API.

## 2. Constructor precedence

```text
Graphiti(..., llm_client=..., prompt_library=..., llm_runtime=...)
```

- `llm_runtime` with `llm_client` or `prompt_library` → `ValueError`
- Only `llm_runtime` → runtime owns the transports and prompts
- Only `prompt_library` → legacy `llm_client` + configured chat library
- Neither → legacy `llm_client` + module-level prompt builders; Graphiti keeps
  `prompt_library` unset

## 3. Builder resolution

For prompt `P` routed to model `M`:

1. `M.prompt_overrides` for `P` if present
2. Else general `prompt_overrides` for `P`
3. Else default `ChatPromptLibrary` method

Builders must return `ChatPrompt`. Schemas are never overridable.

## 4. Facade

`GraphitiClients.complete_prompt` routes to `LLMRuntime.complete` when a runtime is set.
`model_size` and `attribute_extraction` are forwarded on both paths. The runtime
selects the model ID from the prompt route or the default model. For string-model
transports, `ModelSize.small` does not select `LLMConfig.small_model`. Route a
prompt to a smaller model when the prompt requires one. Legacy calls without a
runtime keep `LLMConfig.small_model` behavior.

GLiNER2 binds model objects at initialization. It does not support per-prompt
model routing.

## 5. Public API

```text
LLMTransport[M](
  client: LLMClient,
  *,
  models: Sequence[M] | None = None,
)

LLMTransport.model(
  id: M,
  *,
  prompt_overrides: LLMPromptOverrides | None = None,
  max_tokens: int | None = None,
)

LLMModel(
  id: str,
  transport: LLMTransport[Any],
  prompt_overrides: LLMPromptOverrides | None = None,
  max_tokens: int | None = None,
)

PromptRoutes(
  extract_nodes: LLMModel | PromptRoutes.ExtractNodes | None = None,
  ...
)

LLMPromptOverrides(
  extract_nodes: LLMPromptOverrides.ExtractNodes | None = None,
  ...
)

LLMRuntime(
  model: LLMModel,
  *,
  routes: PromptRoutes | None = None,
  prompt_overrides: LLMPromptOverrides | None = None,
  library: ChatPromptLibrary | None = None,
)
```

Omitting `model` is a type error. Reuse the same `LLMModel` instance on several
routes (a local variable, not a Graphiti nickname).

Override callables must return `ChatPrompt`. `LLMModel.id` is an exact provider
model ID. `LLMModel` has no `small_id` field.

The runtime passes `model` and `model_size` to the selected transport client.
It does not pass a `small_model` keyword. Each prompt uses the selected model's
transport. The runtime does not clone or mutate transports, and concurrent
calls are not serialized.

Use a `Literal` type to restrict model IDs during static checking:

```python
from typing import Literal

from graphiti_core.llm_client import LLMTransport, OpenAIClient

OpenAIModels = Literal['gpt-5.1', 'gpt-5-nano']
openai = LLMTransport[OpenAIModels](
    OpenAIClient(),
    models=['gpt-5.1', 'gpt-5-nano'],
)
main = openai.model('gpt-5.1')
```

Create one transport for each provider client. A route can select a model from
any configured transport. A model ID must match the provider client used by its
transport.
