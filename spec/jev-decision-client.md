# Jev Decision Client

Specification	Jev Decision Client
Category	Graphiti Core
Drafted At	2026-10-09
Authors
Preston Rasmussen

## 1. Overview

This spec adds `JevClient`, an `LLMClient` for the TypeSafe Jev decision
model. Jev is not a chat model. Jev answers typed questions about a shared
state document (the SystemOne protocol). The client lets a Graphiti user send
node dedupe and edge resolution to Jev through `LLMRuntime` routes. All other
prompts stay on the default model.

The design ports the Clef 27B dedupe route of the Zep platform. That route
uses the same SystemOne protocol and the same two prompts:

- `dedupe_nodes.nodes` uses the n6 question design.
- `dedupe_edges.resolve_edge` uses the e2 question design.

## 2. Scope

### 2.1. In scope

- `graphiti_core/llm_client/jev_client.py` with `JevClient`,
  `jev_prompt_overrides()`, and the question builders and answer mappers.
- Exports from `graphiti_core.llm_client`.
- Unit tests with a mocked HTTP transport.
- A README section next to the `LLMRuntime` section.

### 2.2. Out of scope

- A change to `LLMRuntime`, `PromptRoutes`, or the default prompts.
- A route for any prompt other than the two dedupe prompts.
- A change to the default behavior of `Graphiti`.

## 3. Blast radius

- The client is opt-in. A user who does not construct `JevClient` and route a
  prompt to it gets no change.
- No new required dependency. The client uses `httpx`, which `openai` (a
  core dependency) already installs.
- The Zep platform does not change until a later vendored sync, and the facts
  service does not route to `JevClient`.

## 4. Usage

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import (
    JevClient,
    LLMRuntime,
    LLMTransport,
    OpenAIClient,
    PromptRoutes,
    jev_prompt_overrides,
)

openai = LLMTransport(OpenAIClient(), models=['gpt-5.1'])
jev = LLMTransport(JevClient(), models=['jev-latest'])
jev_dedupe = jev.model('jev-latest', prompt_overrides=jev_prompt_overrides())

runtime = LLMRuntime(
    model=openai.model('gpt-5.1'),
    routes=PromptRoutes(
        dedupe_nodes=PromptRoutes.DedupeNodes(nodes=jev_dedupe),
        dedupe_edges=PromptRoutes.DedupeEdges(resolve_edge=jev_dedupe),
    ),
)
graphiti = Graphiti(..., llm_runtime=runtime)
```

`add_episode` and `add_episode_bulk` both use the route, because both call
the dedupe prompts with the runtime clients.

## 5. Protocol

- Request: `POST {base_url}` with `Authorization: Bearer <api_key>` and the
  body `{"model", "state", "questions"}`. The default `base_url` is
  `https://api.typesafe.ai/v1/systemone`. The default model is `jev-latest`.
- `questions` is an object of named questions. Each question has `type`
  (`choice` or `noul`), `instructions`, and `criteria` (option name to
  description).
- Response: `{"model", "answers", "usage"}`. A `noul` answer has the
  probability `noul`. A `choice` answer has `choice`, `confidence`, and
  `probabilities`.

Observed response (`jev-latest` served `jev-1.13.0`):

```json
{"model": "jev-1.13.0",
 "answers": {"dup0": {"type": "noul", "noul": 0.91},
             "pick": {"type": "choice", "choice": "0", "confidence": 1.0,
                      "probabilities": {"0": 1.0, "1": 0.0, "none": 0.0}}},
 "usage": {"input_tokens": 414, "output_tokens": 55}}
```

## 6. Prompt overrides

`jev_prompt_overrides()` returns an `LLMPromptOverrides` for the two prompts.
Each builder renders the default chat prompt, then converts the rendered user
message into one state document and a set of questions. The builder returns a
`ChatPrompt` with the JSON of `state` and `questions` as the user message.
`JevClient` reads that JSON.

- `dedupe_nodes.nodes` (n6): the state is the rendered user message up to
  `</EXISTING ENTITIES>`, plus the node rules. The builder makes one `choice`
  question for each extracted entity. The options are the `candidate_id` of
  each existing entity (name-only label) and `none`.
- `dedupe_edges.resolve_edge` (e2): the state lists the NEW FACT, the
  EXISTING FACTS, and the FACT INVALIDATION CANDIDATES, plus the duplicate and
  contradiction rules. The builder makes one `noul` duplicate question for
  each EXISTING FACT and one `noul` contradiction question for each fact in
  both lists.

The rule text and the question text are the Clef 27B text with no change.

## 7. Answer mapping

| Prompt | Response model | Rule |
|---|---|---|
| `dedupe_nodes.nodes` | `NodeResolutions` | The most probable option for each entity. `none` gives `duplicate_candidate_id=-1`. |
| `dedupe_edges.resolve_edge` | `EdgeDuplicate` | `duplicate_facts`: probability >= `duplicate_threshold`. `contradicted_facts`: probability >= `contradiction_threshold`. |

The thresholds are `JevClient` constructor arguments. Both defaults are 0.8.

## 8. Client behavior

- `JevClient` overrides `generate_response`. The base method appends the JSON
  schema and language instructions to the messages, which would break the
  JSON payload.
- The client serves only the two dedupe prompts. Any other `prompt_name`
  raises `ValueError` that names the prompt.
- The `model` keyword from `LLMRuntime.complete` selects the Jev model.
- HTTP 429, HTTP 5xx, and timeouts get a retry with exponential backoff.
- An input-too-long response (HTTP 400 `max_tokens_exceeded`) makes the
  client split the questions into two halves and send each half again. When
  one question is still too long, the client removes PREVIOUS MESSAGES from
  the state, then halves the CURRENT MESSAGE. When the request is still too
  long, the client raises `JevInputTooLongError`. The client has no fallback
  model. After the retries fail, the client raises the last error.
- The client adds the response `usage` to `token_tracker` with the prompt
  name.
- The client opens a tracer span for each call, as the base client does.

## 9. Tests

| Area | Test |
|---|---|
| Builders | n6 and e2 give the expected state and questions for a fixed context. |
| Mapping | Node argmax and `none`; edge thresholds at the boundary values. |
| Client | The request body and headers; retry on 429 and 5xx; split and shrink on input-too-long; `ValueError` for another prompt. |
| Runtime | An `LLMRuntime` with the routes in section 4 sends the two dedupe prompts to Jev and all other prompts to the default model. |
