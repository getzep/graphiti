# GLiNER2 Hybrid LLM Client Example (Experimental)

> **Note:** The `GLiNER2Client` is experimental and may change in future releases.

This example demonstrates using [GLiNER2](https://github.com/fastino-ai/GLiNER2) as a hybrid LLM client for Graphiti. GLiNER2 handles entity extraction (NER) locally on CPU, while a general-purpose LLM client handles edge/fact extraction, deduplication, summarization, and other reasoning tasks.

- Paper: [GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface](https://arxiv.org/abs/2507.18546)
- Models on HuggingFace:
  - [fastino/gliner2-base-v1](https://huggingface.co/fastino/gliner2-base-v1) (205M params)
  - [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1) (340M params)
  - [fastino/gliner2-multi-v1](https://huggingface.co/fastino/gliner2-multi-v1) (multilingual)

## Prerequisites

- Python 3.11+
- Neo4j 5.26+ ([Neo4j Desktop](https://neo4j.com/download/) or Docker)
- An LLM provider API key (Google, OpenAI, Anthropic, etc.)

## Setup

```bash
# Install graphiti with the gliner2 extra
pip install graphiti-core[gliner2]

# Copy and configure environment variables
cp .env.example .env
```

The GLiNER2 model weights are downloaded automatically on first run.

## LLM and Embedding Providers

The example uses Google Gemini (`gemini-2.5-flash-lite`) for the LLM and embeddings, but `GLiNER2Client` accepts any Graphiti `LLMClient`. To swap providers, replace `GeminiClient` and `GeminiEmbedder` with the equivalent from another provider:

- `graphiti_core.llm_client.openai_client.OpenAIClient`
- `graphiti_core.llm_client.anthropic_client.AnthropicClient`
- `graphiti_core.llm_client.groq_client.GroqClient`
- `graphiti_core.embedder.openai.OpenAIEmbedder`
- `graphiti_core.embedder.voyage.VoyageAIEmbedder`

## Configuration

| Parameter | Description | Default |
|---|---|---|
| `threshold` | GLiNER2 confidence threshold (0.0-1.0). Higher values reduce spurious extractions. | `0.5` |
| `GLINER2_MODEL` | HuggingFace model ID | `fastino/gliner2-large-v1` |

## Running

```bash
python gliner2_neo4j.py
```
