# Requirements — cross-encoder provider auto-select

## Outcome

`Graphiti(...)` and the MCP server no longer require `OPENAI_API_KEY` when the user has configured an Anthropic, Azure, Gemini, Groq, or local LLM. The default cross-encoder follows the LLM provider unless the caller explicitly overrides it. Resolves #1441.

## Users affected

Anyone running Graphiti or the MCP server with a non-OpenAI LLM today — they currently must set a dummy `OPENAI_API_KEY` or instantiate `OpenAIRerankerClient` manually. The MCP server pain is most visible because users discover it after deploying with Anthropic / Gemini and get an opaque key-missing error on the first reranking call.

## In scope

- A resolver `default_cross_encoder_for(llm_client: LLMClient) -> CrossEncoderClient` in `graphiti_core/cross_encoder/default_resolver.py`.
- `Graphiti.__init__` uses the resolver when `cross_encoder is None`.
- A `cross_encoder=False` sentinel disables reranking entirely (search returns un-reranked but ordered results).
- New `LLMClient.provider: str` class attribute, populated on each concrete provider (`'openai'`, `'azure_openai'`, `'anthropic'`, `'gemini'`, `'groq'`, `'openai_generic'`, `'gliner2'`).
- Provider → default reranker mapping:
  - `openai` → `OpenAIRerankerClient`
  - `azure_openai` → `OpenAIRerankerClient` (reuses the Azure-wrapped client where present)
  - `gemini` → `GeminiRerankerClient`
  - `anthropic`, `groq`, `openai_generic`, `gliner2` → `BGERerankerClient` (local, no extra API key required)
- MCP server initialization reads `cross_encoder.provider` from `config.yaml` (default `auto`) and applies the same resolver.
- `tests/cross_encoder/test_default_resolution.py` covers each provider.
- Update Ollama, Gemini, Azure sections of `README.md` to drop the manual `OpenAIRerankerClient` boilerplate.

## Out of scope

- Building new reranker clients. Reuse existing `OpenAI`, `Gemini`, `BGE`.
- Auto-detecting from environment variables when no LLM client is provided — explicit `llm_client` is the only signal; magic is brittle.
- Reranker quality benchmarking — that is owned by the LongMemEval harness spec.

## Decisions

- BGE local fallback for Anthropic / Groq / Ollama. Reason: avoids requiring a second API key; `sentence-transformers` is already an optional extra and most users have it after `pip install graphiti-core[sentence-transformers]`.
- `cross_encoder=False` over `disable_reranking=True`. Reason: matches `cross_encoder=SomeClient()` shape — one parameter, one meaning.
- Default change goes out in 0.30 as a minor, not a breaking change. Reason: OpenAI users still get `OpenAIRerankerClient`; the only behavior change is "no longer crashes without OPENAI_API_KEY for non-OpenAI users."
