# Plan — cross-encoder provider auto-select

## Approach

Single new helper module `graphiti_core/cross_encoder/default_resolver.py`. Each concrete `LLMClient` subclass exposes a `provider: str` class attribute. `Graphiti.__init__` checks `cross_encoder is None` and calls `default_cross_encoder_for(self.llm_client)` instead of unconditionally instantiating `OpenAIRerankerClient()`. The resolver dispatches by provider string. A `cross_encoder=False` value is treated as "no reranking" and `Graphiti` short-circuits the reranker pass in `search/search.py`.

## Steps

1. Add `LLMClient.provider: str` (class-level) in `graphiti_core/llm_client/client.py`. Populate on each concrete provider: `OpenAIClient`, `AzureOpenAILLMClient`, `AnthropicClient`, `GeminiClient`, `GroqClient`, `OpenAIGenericClient`, `GLiNER2Client`.
2. Create `graphiti_core/cross_encoder/default_resolver.py` exporting `default_cross_encoder_for(llm_client) -> CrossEncoderClient`. Dispatch table per provider. Lazy-import each reranker so users without `[google-genai]` or `[sentence-transformers]` don't get import errors at startup.
3. Update `graphiti_core/graphiti.py` `Graphiti.__init__`: replace `self.cross_encoder = cross_encoder or OpenAIRerankerClient()` with the resolver call. Honor `cross_encoder=False` by setting `self.cross_encoder = None` and gating the reranker pass.
4. Update the `Graphiti.__init__` type signature to `cross_encoder: CrossEncoderClient | Literal[False] | None = None`.
5. In `graphiti_core/search/search.py`, treat `cross_encoder is None` as "skip cross-encoder rerank, keep RRF/MMR ordering." Add the guard at the dispatch site, not at every callsite.
6. Update `mcp_server/src/graphiti_mcp_server.py` initialization: instantiate `llm_client` first, pass it to `Graphiti`, rely on the resolver. Add `cross_encoder.provider: auto|openai|gemini|bge|none` field in `mcp_server/config/config.yaml` (default `auto`).
7. Add `tests/cross_encoder/test_default_resolution.py` — parametrized per provider, asserts the resolved class and that no `OPENAI_API_KEY` env var is read when provider != openai/azure_openai.
8. Add `tests/cross_encoder/test_no_reranking.py` — `cross_encoder=False` short-circuits and `search_()` returns un-reranked but ordered results.
9. Update `README.md` Ollama / Gemini / Azure / Anthropic sections to drop `OpenAIRerankerClient` boilerplate. Update `CLAUDE.md` LLM-provider section. Reference #1441 in PR description.

## Dependencies / order

Step 1 before 2. Step 2 before 3 and 6. Step 5 in same PR as 3 (single coherent change). Tests after the code.
