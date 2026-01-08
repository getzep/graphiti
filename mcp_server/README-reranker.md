# Reranker Configuration Guide

## What is Reranker?

A reranker is a component that improves search results by reordering them based on relevance to the query. Graphiti supports multiple reranking strategies, from local algorithms that require no API calls to advanced cross-encoder models that use LLM APIs.

## Supported Reranker Types

### Local Algorithms (No API Calls Required)

These rerankers run entirely locally and don't require any API keys:

- **`rrf`** (Reciprocal Rank Fusion): Default reranker that combines results from multiple search methods. Fast and effective for most use cases.
- **`mmr`** (Maximal Marginal Relevance): Reduces redundancy in results by maximizing diversity. Requires embedding vectors.
- **`node_distance`**: Reranks based on graph distance from a center node. Requires `center_node_uuid` parameter.
- **`episode_mentions`**: Reranks based on how many episodes mention each entity.

### Cross-Encoder (API-Based)

Cross-encoder rerankers use LLM APIs to score and rerank results:

- **`openai`**: OpenAI compatible API (supports OpenAI, Alibaba DashScope, Ollama, etc.)
- **`gemini`**: Google Gemini API
- **`sentence_transformers`**: Local BGE model (no API key required, but requires `sentence-transformers` package)

## Configuration

### Basic Configuration

Add a `reranker` section to your `config.yaml`:

```yaml
reranker:
  enabled: true
  type: "rrf"  # Options: rrf, mmr, node_distance, episode_mentions, cross_encoder
  provider: "openai"  # When type=cross_encoder: openai, gemini, sentence_transformers
  model: "gpt-4.1-nano"  # Model name for cross_encoder
  
  local:
    type: "rrf"  # Local reranker type
    mmr_lambda: 0.5  # MMR lambda parameter (0.0-1.0)
  
  providers:
    openai:
      api_key: ${RERANKER_OPENAI_API_KEY:}
      api_url: ${RERANKER_OPENAI_API_URL:https://api.openai.com/v1}
    
    gemini:
      api_key: ${GEMINI_API_KEY:}
```

### Configuration Examples

#### Option 1: Local RRF (Recommended for Development)

No API key required, fast and effective:

```yaml
reranker:
  enabled: true
  type: rrf
```

#### Option 2: Local MMR (Deduplication)

Requires embedding vectors but provides better diversity:

```yaml
reranker:
  enabled: true
  type: mmr
  local:
    mmr_lambda: 0.5  # Higher = more diversity, Lower = more relevance
```

#### Option 3: Alibaba DashScope qwen3-rerank

Use Alibaba's rerank model via OpenAI-compatible API:

```yaml
reranker:
  enabled: true
  type: cross_encoder
  provider: openai
  model: qwen3-rerank
  providers:
    openai:
      api_key: ${DASHSCOPE_API_KEY}
      api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### Option 4: OpenAI Official API

```yaml
reranker:
  enabled: true
  type: cross_encoder
  provider: openai
  model: gpt-4.1-nano
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
```

#### Option 5: Google Gemini

```yaml
reranker:
  enabled: true
  type: cross_encoder
  provider: gemini
  model: gemini-2.5-flash-lite
  providers:
    gemini:
      api_key: ${GEMINI_API_KEY}
```

#### Option 6: Local BGE Model

Requires `sentence-transformers` package but no API key:

```yaml
reranker:
  enabled: true
  type: cross_encoder
  provider: sentence_transformers
```

#### Option 7: Disable Reranker

```yaml
reranker:
  enabled: false
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_OPENAI_API_KEY` | OpenAI-compatible API key | - |
| `RERANKER_OPENAI_API_URL` | OpenAI-compatible API URL | `https://api.openai.com/v1` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `DASHSCOPE_API_KEY` | Alibaba DashScope API key | - |

## Command Line Arguments

You can override reranker settings via CLI:

```bash
python src/graphiti_mcp_server.py \
  --reranker-enabled true \
  --reranker-type cross_encoder \
  --reranker-provider openai \
  --reranker-model qwen3-rerank
```

## Alibaba DashScope qwen3-rerank Usage

Alibaba DashScope provides a dedicated rerank model `qwen3-rerank` that's optimized for Chinese and English text reranking.

### Setup

1. Get your API key from [Alibaba DashScope Console](https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2780056)
2. Set the environment variable:

   ```bash
   export DASHSCOPE_API_KEY=your-api-key
   ```

3. Configure in `config.yaml`:

   ```yaml
   reranker:
     enabled: true
     type: cross_encoder
     provider: openai
     model: qwen3-rerank
     providers:
       openai:
         api_key: ${DASHSCOPE_API_KEY}
         api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

### Benefits

- Optimized for Chinese and English text
- Lower latency than general-purpose LLMs
- Cost-effective for reranking tasks

## Comparison of Reranker Types

| Type | Speed | Quality | API Key Required | Best For |
|------|-------|---------|------------------|----------|
| `rrf` | Fast | Good | No | General use, development |
| `mmr` | Fast | Good | No | Reducing redundancy |
| `node_distance` | Fast | Good | No | Graph-based queries |
| `episode_mentions` | Fast | Good | No | Popular entities |
| `cross_encoder` (OpenAI) | Slow | Excellent | Yes | High-quality reranking |
| `cross_encoder` (Gemini) | Slow | Excellent | Yes | High-quality reranking |
| `cross_encoder` (BGE) | Medium | Very Good | No | Local high-quality reranking |

## Troubleshooting

### "Reranker API key not configured"

Make sure you've set the appropriate environment variable or configured it in the YAML file.

### "Reranker dependency not available"

For `sentence_transformers`, install the package:

```bash
pip install sentence-transformers
```

For `gemini`, install the package:

```bash
pip install graphiti-core[google-genai]
```

### Reranker not being used

Check that:

1. `reranker.enabled` is set to `true`
2. The search configuration uses `cross_encoder` reranker (for API-based rerankers)
3. The reranker client was successfully initialized (check logs)

## Performance Considerations

- **Local rerankers** (`rrf`, `mmr`, etc.) are fast and don't add latency
- **Cross-encoder rerankers** add API call latency but significantly improve result quality
- For production, consider using local rerankers for speed or cross-encoder for quality
- Alibaba DashScope `qwen3-rerank` is optimized for reranking and typically faster than general-purpose LLMs

## References

- [Alibaba DashScope Rerank Model](https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2780056)
- [MCP Server README](README.md)
