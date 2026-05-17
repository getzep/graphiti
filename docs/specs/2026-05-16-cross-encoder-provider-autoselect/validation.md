# Validation — cross-encoder provider auto-select

## Automated tests

- `tests/cross_encoder/test_default_resolution.py` — for each provider in {openai, azure_openai, anthropic, gemini, groq, openai_generic, gliner2}, assert `default_cross_encoder_for(<provider client>)` returns the expected class.
- `tests/test_graphiti_mock.py::test_init_anthropic_without_openai_key` — unset `OPENAI_API_KEY`, instantiate `Graphiti(llm_client=AnthropicClient(...))`, expect no key lookup and no exception.
- `tests/test_graphiti_mock.py::test_init_gemini_uses_gemini_reranker` — instantiate with `GeminiClient`, assert `graphiti.cross_encoder.__class__.__name__ == 'GeminiRerankerClient'`.
- `tests/cross_encoder/test_no_reranking.py` — `cross_encoder=False`, run `search_()`, assert result is non-empty and ordered, and that no reranker class is touched.
- `mcp_server/tests/test_cross_encoder_config.py` — `cross_encoder.provider: auto` with Anthropic LLM → BGE; explicit `cross_encoder.provider: gemini` → GeminiRerankerClient.

## Smoke checks

```bash
unset OPENAI_API_KEY
uv run python -c "
from graphiti_core import Graphiti
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
g = Graphiti(
  'bolt://localhost:7687', 'neo4j', 'pw',
  llm_client=AnthropicClient(LLMConfig(api_key='sk-test', model='claude-haiku-4-5-latest')),
)
print(type(g.cross_encoder).__name__)
"
# Expect: BGERerankerClient (or whatever Anthropic's chosen default ends up being)
```

## Manual criteria

- README's Ollama and Gemini sections compile in copy-paste form without setting `OPENAI_API_KEY`.
- MCP server with `LLM_PROVIDER=anthropic` starts cleanly and serves `search_memory_facts` without an OpenAI key.

## AI eval plan

- **Success criteria**: per-provider hybrid-search recall@10 on the LongMemEval regression subset (50 questions) drops by no more than 2 points versus the OpenAI-reranker baseline. BGE local fallback is permitted to be slower (higher p99 latency) but must not drop recall.
- **Eval dataset**: LongMemEval regression subset from `2026-05-16-longmemeval-bench-harness`.
- **Regression set**: per-provider triple `(llm, embedder, reranker)` matrix run via the LongMemEval CLI.
- **Cadence**: per-PR on the 50-question subset when this PR lands and on subsequent reranker-touching PRs.

## Risks & rollback

- **Failure modes**: BGE local reranker has higher per-query latency and `search_()` feels slower for the Anthropic default; default change surprises users who depended on `OpenAIRerankerClient` instantiation as a side-effect; BGE model download fails in air-gapped environments; the `LLMClient.provider` attribute clashes with a downstream subclass.
- **Rollback**: revert the `Graphiti.__init__` change; explicit `cross_encoder=OpenAIRerankerClient()` continues to work; the resolver module is dead code if unused.

## Open questions

- Anthropic default: BGE local vs Gemini reranker (cheap, fast log-probs) — which is better when only `ANTHROPIC_API_KEY` is present? Decide before merge; lean BGE for "no second key" purity.
- Should `cross_encoder=False` be a separate `disable_reranking=True` flag instead? UX call; the Literal approach is more compact but `False` is unusual as a sentinel.
- Azure OpenAI reranker: do we ship a dedicated `AzureOpenAIRerankerClient`, or wrap the existing `AsyncOpenAI` client with `OpenAIRerankerClient`? Decide based on whether Azure deployments expose the rerank endpoint at the same path.
