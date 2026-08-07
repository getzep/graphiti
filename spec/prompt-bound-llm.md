# Prompt-Bound LLM

Specification	Prompt-Bound LLM
Category	Graphiti Core
Drafted At	2026-08-07
Authors
Paul Paliychuk

## 1. Overview

`PromptBoundLLM` is an opt-in bundle that couples a single `LLMClient` transport with:

1. A prompt library (`ChatPrompt` builders + fixed `PromptSpec` schemas)
2. Named model slots (`LLMModelConfig`) on that one provider
3. Per-prompt model routing (`prompt_models`)
4. Optional general and model-specific prompt text overrides

Multi-provider routing is out of scope for v1. Putting a Claude model id on an OpenAI client is unsupported.

## 2. Constructor precedence

```text
Graphiti(..., prompt_library=..., prompt_bound_llm=...)
```

- Both `prompt_library` and `prompt_bound_llm` → `ValueError`
- Only `prompt_bound_llm` → bundle owns prompts; `self.prompt_library` is populated from the bundle
- Only `prompt_library` (or neither) → legacy `llm_client` + library path

## 3. Builder resolution

For prompt `P` routed to model id `M`:

1. `model_prompt_overrides[M][group][fn]` if present
2. Else general `prompt_overrides[group][fn]`
3. Else default library ABC / duck-typed method

Builders must return `ChatPrompt`. Schemas are never overridable.

## 4. Facade

`GraphitiClients.complete_prompt` routes to `PromptBoundLLM.complete` when a bundle is set.
On the bundle path, `model_size` is ignored; `attribute_extraction` and dynamic `response_model` still apply.
