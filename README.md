<p align="center">
  <a href="https://www.getzep.com/">
    <img src="https://github.com/user-attachments/assets/119c5682-9654-4257-8922-56b7cb8ffd73" width="150" alt="Zep Logo">
  </a>
</p>

<h1 align="center">
Graphiti
</h1>
<h2 align="center"> Build Real-Time Knowledge Graphs for AI Agents</h2>
<br />

[![Discord](https://dcbadge.vercel.app/api/server/W8Kw6bsgXQ?style=flat)](https://discord.com/invite/W8Kw6bsgXQ)
[![Lint](https://github.com/getzep/Graphiti/actions/workflows/lint.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/lint.yml)
[![Unit Tests](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml)
[![MyPy Check](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/getzep/Graphiti)

:star: _Help us reach more developers and grow the Graphiti community. Star this repo!_
<br />

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

A knowledge graph is a network of interconnected facts, such as _“Kendra loves Adidas shoes.”_ Each fact is a “triplet” represented by two entities, or
nodes (_”Kendra”_, _“Adidas shoes”_), and their relationship, or edge (_”loves”_). Knowledge Graphs have been explored
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

Optional:

- Anthropic or Groq API key (for alternative LLM providers)

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

## Quick Start

> [!IMPORTANT]
> Graphiti uses OpenAI for LLM inference and embedding. Ensure that an `OPENAI_API_KEY` is set in your environment.
> Support for Anthropic and Groq LLM inferences is available, too. Other LLM providers may be supported via OpenAI
> compatible APIs.

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

# Initialize Graphiti as Your Memory Layer
graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")

# Initialize the graph database with Graphiti's indices. This only needs to be done once.
graphiti.build_indices_and_constraints()

# Add episodes
episodes = [
    "Kamala Harris is the Attorney General of California. She was previously "
    "the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]
for i, episode in enumerate(episodes):
    await graphiti.add_episode(
        name=f"Freakonomics Radio {i}",
        episode_body=episode,
        source=EpisodeType.text,
        source_description="podcast",
        reference_time=datetime.now(timezone.utc)
    )

# Search the graph for semantic memory retrieval
# Execute a hybrid search combining semantic similarity and BM25 retrieval
# Results are combined and reranked using Reciprocal Rank Fusion
results = await graphiti.search('Who was the California Attorney General?')
[
    EntityEdge(
│   uuid = '3133258f738e487383f07b04e15d4ac0',
│   source_node_uuid = '2a85789b318d4e418050506879906e62',
│   target_node_uuid = 'baf7781f445945989d6e4f927f881556',
│   created_at = datetime.datetime(2024, 8, 26, 13, 13, 24, 861097),
│   name = 'HELD_POSITION',
# the fact reflects the updated state that Harris is
# no longer the AG of California
│   fact = 'Kamala Harris was the Attorney General of California',
│   fact_embedding = [
│   │   -0.009955154731869698,
│       ...
│   │   0.00784289836883545
│],
│   episodes = ['b43e98ad0a904088a76c67985caecc22'],
│   expired_at = datetime.datetime(2024, 8, 26, 20, 18, 1, 53812),
# These dates represent the date this edge was true.
│   valid_at = datetime.datetime(2011, 1, 3, 0, 0, tzinfo= < UTC >),
│   invalid_at = datetime.datetime(2017, 1, 3, 0, 0, tzinfo= < UTC >)
)
]

# Rerank search results based on graph distance
# Provide a node UUID to prioritize results closer to that node in the graph.
# Results are weighted by their proximity, with distant edges receiving lower scores.
await graphiti.search('Who was the California Attorney General?', center_node_uuid)

# Close the connection when chat state management is complete
graphiti.close()
```

## Graph Service

The `server` directory contains an API service for interacting with the Graphiti API. It is built using FastAPI.

Please see the [server README](./server/README.md) for more information.

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
