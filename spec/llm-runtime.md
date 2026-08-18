# LLM Runtime

Specification	LLM Runtime
Category	Graphiti Core
Drafted At	2026-08-07
Authors
Paul Paliychuk

## 1. Overview

`LLMRuntime` is an opt-in object that couples a single `LLMClient` transport with:

1. A required default `LLMModel` (omitting it is a type error)
2. Optional `PromptRoutes` (per-group `LLMModel` or nested group class)
3. Optional `LLMPromptOverrides` (nested group classes of builders)
4. A prompt library (`ChatPrompt` builders). Schemas live in the immutable
   `BUILTIN_PROMPT_SPECS` registry and are not user-configurable.

There are no caller-invented model nicknames. Bind `LLMModel` instances to local
Python variables and pass those variables into `PromptRoutes`. Multi-provider
routing is out of scope for v1. Putting a Claude model id on an OpenAI client
is unsupported.

Unknown prompt names are constructor / type errors on the nested dataclasses.
The legacy `Graphiti(llm_client=..., prompt_library=...)` path is unchanged.

## 2. Constructor precedence

```text
Graphiti(..., llm_client=..., prompt_library=..., llm_runtime=...)
```

- `llm_runtime` with `llm_client` or `prompt_library` → `ValueError`
- Only `llm_runtime` → runtime owns the transport and prompts
- Only `prompt_library` (or neither) → legacy `llm_client` + library path

## 3. Builder resolution

For prompt `P` routed to model `M`:

1. `M.prompt_overrides` for `P` if present
2. Else general `prompt_overrides` for `P`
3. Else default library ABC / duck-typed method

Builders must return `ChatPrompt`. Schemas are never overridable.

## 4. Facade

`GraphitiClients.complete_prompt` routes to `LLMRuntime.complete` when a runtime is set.
`model_size` and `attribute_extraction` are forwarded on both paths. On the default
model, omitting `LLMModel.small_id` keeps the transport's `small_model` (`small_model`
is passed as `None`). Routed models without `small_id` pin small to their own `id`.

## 5. Public API

```text
LLMModel(
  id: str,
  small_id: str | None = None,
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
  transport: LLMClient,
  model: LLMModel,
  routes: PromptRoutes | None = None,
  prompt_overrides: LLMPromptOverrides | None = None,
  library: PromptLibrary | None = None,
)
```

Omitting `model` is a type error. Reuse the same `LLMModel` instance on several
routes (a local variable, not a Graphiti nickname).

Override callables must return `ChatPrompt`. `LLMModel.id` is an exact provider
model id; reserved fields (temperature, structured_output_mode, …) may be added
later without changing call sites.

Per-prompt model selection passes ``model`` / ``small_model`` into
`LLMClient.generate_response` for that call. The original transport is not
cloned or mutated, and concurrent calls are not serialized. v1 requires a
transport that selects models via the ``model`` / ``small_model`` string
attributes. Providers that bind a model object at init (for example GLiNER2)
cannot be routed per prompt. Putting a Claude id on an OpenAI client is
unsupported.
