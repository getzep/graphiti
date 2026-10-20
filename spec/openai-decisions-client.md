# OpenAI Decisions client for node and edge dedupe

# 1. Overview

This change adds `OpenAIDecisionsClient`. The client sends the two Graphiti
dedupe decisions to the OpenAI Decisions API (`POST /v1/decisions`) with the
model `gpt-6-luna`. The change also replaces the decision prompts of
`jev_prompt_overrides()` with the Zep n6/e2 decision prompts, and it adds a
FalkorDB example for the new client.

A decision model answers typed questions with probabilities. It does not
generate text. Thus, a user must not set a decision client as the default
model, and must not send extraction, summary, or other generation prompts to
it.

## 1.1. In scope

- `graphiti_core/llm_client/openai_decisions_client.py` with
  `OpenAIDecisionsClient` and `openai_decisions_prompt_overrides()`.
- A shared module for the n6/e2 decision prompts and the answer assembly.
  `JevClient` and `OpenAIDecisionsClient` will both use it.
- `jev_prompt_overrides()` will use the Zep n6/e2 text: the Zep duplicate
  rules, the zep e2 duplicate question, and the zep payload encoding.
- Default thresholds for both clients: duplicate 0.7, contradiction 0.8.
- `examples/openai_decisions/` with a FalkorDB script, a README, and an
  `.env.example` file.
- Unit tests with a mock transport. Updates to the root README and to
  `spec/jev-decision-client.md`.

## 1.2. Out of scope

- The default Graphiti prompts. The C2 extraction prompt has a separate spec
  and a separate PR.
- The temporal invalidation bug.
- The vendored hyperedge and subtype prompts.

# 2. Usage

The usage will be the same as the Jev usage in `spec/jev-decision-client.md`
section 4. `OpenAIDecisionsClient` and `openai_decisions_prompt_overrides()`
replace `JevClient` and `jev_prompt_overrides()` on the two routes. The default
model stays `gpt-6-luna` through `OpenAIClient` (Chat Completions).

# 3. Protocol

## 3.1. Request

```json
{"model": "gpt-6-luna",
 "input": "<instructions and state text>",
 "questions": [
   {"name": "pick", "type": "choice", "instructions": "...",
    "choices": [{"value": "c0", "description": "..."}]},
   {"name": "dup0", "type": "predicate", "instructions": "..."}]}
```

- A Graphiti `choice` question becomes an OpenAI `choice` question.
- A Graphiti `predicate` or `noul` question becomes an OpenAI `predicate`
  question.
- The client reads the API key from `OPENAI_API_KEY`.
- The base URL precedence is the `base_url` argument, `LLMConfig.base_url`,
  `OPENAI_BASE_URL`, and `https://api.openai.com/v1`.
- The client joins non-empty instructions and state with two newline
  characters. When instructions are empty, the input equals the state.

## 3.2. Response

```json
{"model": "gpt-6-luna",
 "answers": [
   {"type": "predicate", "name": "con0", "probability": 0.04},
   {"type": "choice", "name": "pick", "choice": "c1",
    "probabilities": [{"value": "c0", "probability": 0.0},
                      {"value": "c1", "probability": 1.0}],
    "confidence": 1.0}],
 "usage": {"input_tokens": 289, "output_tokens": 0, "total_tokens": 289}}
```

- The client will validate each answer: the name must be a question name, the
  type must agree with the question, and a choice must be one of the allowed
  values.
- A predicate answer must have `probability`. A choice answer must have
  `probabilities`, or `choice` with `confidence`.
- A missing answer or an answer that is not valid is an error. The client
  will raise it, and the client will not use a default value.
- The client will record the token usage in the same way as the other
  Graphiti clients.

# 4. Prompts and answer mapping

- Node dedupe (n6): one `choice` question. The options are the candidates and
  `none`. The argmax option is the duplicate. `none` maps to
  `duplicate_candidate_id=-1`.
- Edge resolution (e2): one `predicate` duplicate question and one `predicate`
  contradiction question for each candidate fact. A probability of 0.7 or more
  marks a duplicate. A probability of 0.8 or more marks a contradiction.
- The shared prompt text uses the Zep n6/e2 decision prompts. The node rules,
  node question, and contradiction rules use the Zep text.
- The duplicate rules treat an existing fact as a duplicate when it states
  all information in the new fact, even if the existing fact has more detail.
  A new fact that adds information is not a duplicate.
- The edge duplicate question uses the Zep wording. The payload includes the
  Zep keys `instructions`, `state`, and `questions`.

A 1,000,000-character input returned HTTP 200 with 125,151 input tokens. A
10,000,000-character input was rejected with HTTP 400 and this response:

```json
{"error":{"message":"Decision input exceeds the token limit.","type":"invalid_request_error","param":null,"code":null}}
```

The client treats a response as too long only when the status is 400 and
`error.code` is `context_length_exceeded` or `string_above_max_length`, or when
`error.type` is `invalid_request_error` and `error.message` contains
`exceeds the token limit`, without regard to letter case. It will split
questions, remove PREVIOUS MESSAGES, halve CURRENT MESSAGE, and then raise
`OpenAIDecisionsInputTooLongError` if one question still does not fit.

# 5. Blast radius

- Current Graphiti users: no change. The client and the overrides are new
  opt-in exports.
- Jev users: the Jev prompt text and the Jev duplicate threshold (0.8 to 0.7)
  will change. `JevClient` is not in a release yet, so no released user is
  affected.

# 6. Tests

- Request mapping: choice, predicate, and noul questions. The state text and
  the instructions go into `input`.
- Response mapping: predicate probability, choice probabilities, choice with
  confidence, and the `type` field.
- Errors: an unknown answer name, a choice that is not allowed, a missing
  answer, an HTTP error, and an input that is too long.
- Prompt overrides: the n6/e2 render for both clients is byte-identical to a
  fixture of the zep render.
- Thresholds: 0.69 and 0.70 for duplicates, 0.79 and 0.80 for contradictions.
- Too-long fallback: test both supported error codes and the observed response
  body. Test a different HTTP 400 `invalid_request_error` to confirm that the
  client does not split questions or retry it.
