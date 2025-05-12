<p align="center">
  <a href="https://www.getzep.com/">
    <img src="https://github.com/user-attachments/assets/119c5682-9654-4257-8922-56b7cb8ffd73" width="150" alt="Zep Logo">
  </a>
</p>

<h1 align="center">
Graphiti
</h1>
<h2 align="center"> Build Real-Time Knowledge Graphs for AI Agents</h2>
<div align="center">

[![Lint](https://github.com/getzep/Graphiti/actions/workflows/lint.yml/badge.svg?style=flat)](https://github.com/getzep/Graphiti/actions/workflows/lint.yml)
[![Unit Tests](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml)
[![MyPy Check](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml)

![GitHub Repo stars](https://img.shields.io/github/stars/getzep/graphiti)
[![Discord](https://dcbadge.vercel.app/api/server/W8Kw6bsgXQ?style=flat)](https://discord.com/invite/W8Kw6bsgXQ)
[![arXiv](https://img.shields.io/badge/arXiv-2501.13956-b31b1b.svg?style=flat)](https://arxiv.org/abs/2501.13956)
[![Release](https://img.shields.io/github/v/release/getzep/graphiti?style=flat&label=Release&color=limegreen)](https://github.com/getzep/graphiti/releases)

</div>
<div align="center">

<a href="https://trendshift.io/repositories/12986" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12986" alt="getzep%2Fgraphiti | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

:star: _Help us reach more developers and grow the Graphiti community. Star this repo!_

<br />

> [!TIP]
> Check out the new [MCP server for Graphiti](mcp_server/README.md)! Give Claude, Cursor, and other MCP clients powerful Knowledge Graph-based memory.

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents operating in dynamic environments. Unlike traditional retrieval-augmented generation (RAG) methods, Graphiti continuously integrates user interactions, structured and unstructured enterprise data, and external information into a coherent, queryable graph. The framework supports incremental data updates, efficient retrieval, and precise historical queries without requiring complete graph recomputation, making it suitable for developing interactive, context-aware AI applications.

Use Graphiti to:

- Integrate and maintain dynamic user interactions and business data.
- Facilitate state-based reasoning and task automation for agents.
- Query complex, evolving data with semantic, keyword, and graph-based search methods.

<br />

<p align="center">
    <img src="images/graphiti-graph-intro.gif" alt="Graphiti temporal walkthrough" width="700px">   
</p>

<br />

A knowledge graph is a network of interconnected facts, such as _"Kendra loves Adidas shoes."_ Each fact is a "triplet" represented by two entities, or
nodes ("Kendra", "Adidas shoes"), and their relationship, or edge ("loves"). Knowledge Graphs have been explored
extensively for information retrieval. What makes Graphiti unique is its ability to autonomously build a knowledge graph
while handling changing relationships and maintaining historical context.

## Graphiti and Zep Memory

Graphiti powers the core of [Zep's memory layer](https://www.getzep.com) for AI Agents.

Using Graphiti, we've demonstrated Zep is
the [State of the Art in Agent Memory](https://blog.getzep.com/state-of-the-art-agent-memory/).

Read our paper: [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956).

We're excited to open-source Graphiti, believing its potential reaches far beyond AI memory applications.

<p align="center">
    <a href="https://arxiv.org/abs/2501.13956"><img src="images/arxiv-screenshot.png" alt="Zep: A Temporal Knowledge Graph Architecture for Agent Memory" width="700px"></a>
</p>

## Why Graphiti?

Traditional RAG approaches often rely on batch processing and static data summarization, making them inefficient for frequently changing data. Graphiti addresses these challenges by providing:

- **Real-Time Incremental Updates:** Immediate integration of new data episodes without batch recomputation.
- **Bi-Temporal Data Model:** Explicit tracking of event occurrence and ingestion times, allowing accurate point-in-time queries.
- **Efficient Hybrid Retrieval:** Combines semantic embeddings, keyword (BM25), and graph traversal to achieve low-latency queries without reliance on LLM summarization.
- **Custom Entity Definitions:** Flexible ontology creation and support for developer-defined entities through straightforward Pydantic models.
- **Scalability:** Efficiently manages large datasets with parallel processing, suitable for enterprise environments.

<p align="center">
    <img src="/images/graphiti-intro-slides-stock-2.gif" alt="Graphiti structured + unstructured demo" width="700px">   
</p>

## Graphiti vs. GraphRAG

| Aspect                     | GraphRAG                              | Graphiti                                         |
| -------------------------- | ------------------------------------- | ------------------------------------------------ |
| **Primary Use**            | Static document summarization         | Dynamic data management                          |
| **Data Handling**          | Batch-oriented processing             | Continuous, incremental updates                  |
| **Knowledge Structure**    | Entity clusters & community summaries | Episodic data, semantic entities, communities    |
| **Retrieval Method**       | Sequential LLM summarization          | Hybrid semantic, keyword, and graph-based search |
| **Adaptability**           | Low                                   | High                                             |
| **Temporal Handling**      | Basic timestamp tracking              | Explicit bi-temporal tracking                    |
| **Contradiction Handling** | LLM-driven summarization judgments    | Temporal edge invalidation                       |
| **Query Latency**          | Seconds to tens of seconds            | Typically sub-second latency                     |
| **Custom Entity Types**    | No                                    | Yes, customizable                                |
| **Scalability**            | Moderate                              | High, optimized for large datasets               |

Graphiti is specifically designed to address the challenges of dynamic and frequently updated datasets, making it particularly suitable for applications requiring real-time interaction and precise historical queries.

## Installation

Requirements:

- Python 3.10 or higher
- Neo4j 5.26 or higher (serves as the embeddings storage backend)
- OpenAI API key (for LLM inference and embedding)

> [!IMPORTANT]
> Graphiti works best with LLM services that support Structured Output (such as OpenAI and Gemini).
> Using other services may result in incorrect output schemas and ingestion failures. This is particularly
> problematic when using smaller models.

Optional:

- Google Gemini, Anthropic, or Groq API key (for alternative LLM providers)

> [!TIP]
> The simplest way to install Neo4j is via [Neo4j Desktop](https://neo4j.com/download/). It provides a user-friendly
> interface to manage Neo4j instances and databases.

```bash
pip install graphiti-core
```

or

```bash
poetry add graphiti-core
```

You can also install optional LLM providers as extras:

```bash
# Install with Anthropic support
pip install graphiti-core[anthropic]

# Install with Groq support
pip install graphiti-core[groq]

# Install with Google Gemini support
pip install graphiti-core[google-genai]

# Install with multiple providers
pip install graphiti-core[anthropic,groq,google-genai]
```

## Quick Start

> [!IMPORTANT]
> Graphiti uses OpenAI for LLM inference and embedding. Ensure that an `OPENAI_API_KEY` is set in your environment.
> Support for Anthropic and Groq LLM inferences is available, too. Other LLM providers may be supported via OpenAI
> compatible APIs.

For a complete working example, see the [Quickstart Example](./examples/quickstart/README.md) in the examples directory. The quickstart demonstrates:

1. Connecting to a Neo4j database
2. Initializing Graphiti indices and constraints
3. Adding episodes to the graph (both text and structured JSON)
4. Searching for relationships (edges) using hybrid search
5. Reranking search results using graph distance
6. Searching for nodes using predefined search recipes

The example is fully documented with clear explanations of each functionality and includes a comprehensive README with setup instructions and next steps.

## MCP Server

The `mcp_server` directory contains a Model Context Protocol (MCP) server implementation for Graphiti. This server allows AI assistants to interact with Graphiti's knowledge graph capabilities through the MCP protocol.

Key features of the MCP server include:

- Episode management (add, retrieve, delete)
- Entity management and relationship handling
- Semantic and hybrid search capabilities
- Group management for organizing related data
- Graph maintenance operations

The MCP server can be deployed using Docker with Neo4j, making it easy to integrate Graphiti into your AI assistant workflows.

For detailed setup instructions and usage examples, see the [MCP server README](./mcp_server/README.md).

## REST Service

The `server` directory contains an API service for interacting with the Graphiti API. It is built using FastAPI.

Please see the [server README](./server/README.md) for more information.

## Optional Environment Variables

In addition to the Neo4j and OpenAi-compatible credentials, Graphiti also has a few optional environment variables.
If you are using one of our supported models, such as Anthropic or Voyage models, the necessary environment variables
must be set.

`USE_PARALLEL_RUNTIME` is an optional boolean variable that can be set to true if you wish
to enable Neo4j's parallel runtime feature for several of our search queries.
Note that this feature is not supported for Neo4j Community edition or for smaller AuraDB instances,
as such this feature is off by default.

## Using Graphiti with Azure OpenAI

Graphiti supports Azure OpenAI for both LLM inference and embeddings. To use Azure OpenAI, you'll need to configure both the LLM client and embedder with your Azure OpenAI credentials.

```python
from openai import AsyncAzureOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# Azure OpenAI configuration
api_key = "<your-api-key>"
api_version = "<your-api-version>"
azure_endpoint = "<your-azure-endpoint>"

# Create Azure OpenAI client for LLM
azure_openai_client = AsyncAzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# Initialize Graphiti with Azure OpenAI clients
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=OpenAIClient(
        client=azure_openai_client
    ),
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model="text-embedding-3-small"  # Use your Azure deployed embedding model name
        ),
        client=azure_openai_client
    ),
    # Optional: Configure the OpenAI cross encoder with Azure OpenAI
    cross_encoder=OpenAIRerankerClient(
        client=azure_openai_client
    )
)

# Now you can use Graphiti with Azure OpenAI
```

Make sure to replace the placeholder values with your actual Azure OpenAI credentials and specify the correct embedding model name that's deployed in your Azure OpenAI service.

## Using Graphiti with Google Gemini

Graphiti supports Google's Gemini models for both LLM inference and embeddings. To use Gemini, you'll need to configure both the LLM client and embedder with your Google API key.

Install Graphiti:

```bash
poetry add "graphiti-core[google-genai]"

# or

uv add "graphiti-core[google-genai]"
```

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

# Google API key configuration
api_key = "<your-google-api-key>"

# Initialize Graphiti with Gemini clients
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-2.0-flash"
        )
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model="embedding-001"
        )
    )
)

# Now you can use Graphiti with Google Gemini
```

## Documentation

- [Guides and API documentation](https://help.getzep.com/graphiti).
- [Quick Start](https://help.getzep.com/graphiti/graphiti/quick-start)
- [Building an agent with LangChain's LangGraph and Graphiti](https://help.getzep.com/graphiti/graphiti/lang-graph-agent)

## Status and Roadmap

Graphiti is under active development. We aim to maintain API stability while working on:

- [x] Supporting custom graph schemas:
  - Allow developers to provide their own defined node and edge classes when ingesting episodes
  - Enable more flexible knowledge representation tailored to specific use cases
- [x] Enhancing retrieval capabilities with more robust and configurable options
- [x] Graphiti MCP Server
- [ ] Expanding test coverage to ensure reliability and catch edge cases

## Contributing

We encourage and appreciate all forms of contributions, whether it's code, documentation, addressing GitHub Issues, or
answering questions in the Graphiti Discord channel. For detailed guidelines on code contributions, please refer
to [CONTRIBUTING](CONTRIBUTING.md).

## Support

Join the [Zep Discord server](https://discord.com/invite/W8Kw6bsgXQ) and make your way to the **#Graphiti** channel!
