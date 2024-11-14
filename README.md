<div align="center">

<img width="350" alt="Graphiti-ts-small" src="https://github.com/user-attachments/assets/bbd02947-e435-4a05-b25a-bbbac36d52c8">

## Temporal Knowledge Graphs for Agentic Applications

<br />

[![Discord](https://dcbadge.vercel.app/api/server/W8Kw6bsgXQ?style=flat)](https://discord.com/invite/W8Kw6bsgXQ)
[![Lint](https://github.com/getzep/Graphiti/actions/workflows/lint.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/lint.yml)
[![Unit Tests](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml)
[![MyPy Check](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/getzep/Graphiti)

<br />

</div>

Graphiti builds dynamic, temporally aware Knowledge Graphs that represent complex, evolving relationships between
entities over time. Graphiti ingests both unstructured and structured data, and the resulting graph may be queried using
a fusion of time, full-text, semantic, and graph algorithm approaches.

<br />

<p align="center">
    <img src="/images/graphiti-graph-intro.gif" alt="Graphiti temporal walkthrough" width="700px">   
</p>

<br />

Graphiti helps you create and query Knowledge Graphs that evolve over time. A knowledge graph is a network of
interconnected facts, such as _“Kendra loves Adidas shoes.”_ Each fact is a “triplet” represented by two entities, or
nodes (_”Kendra”_, _“Adidas shoes”_), and their relationship, or edge (_”loves”_). Knowledge Graphs have been explored
extensively for information retrieval. What makes Graphiti unique is its ability to autonomously build a knowledge graph
while handling changing relationships and maintaining historical context.

With Graphiti, you can build LLM applications such as:

- Assistants that learn from user interactions, fusing personal knowledge with dynamic data from business systems like
  CRMs and billing platforms.
- Agents that autonomously execute complex tasks, reasoning with state changes from multiple dynamic sources.

Graphiti supports a wide range of applications in sales, customer service, health, finance, and more, enabling long-term
recall and state-based reasoning for both assistants and agents.

## Why Graphiti?

We were intrigued by Microsoft’s GraphRAG, which expanded on RAG text chunking by using a graph to better model a
document corpus and making this representation available via semantic and graph search techniques. However, GraphRAG did
not address our core problem: It's primarily designed for static documents and doesn't inherently handle temporal
aspects of data.

Graphiti is designed from the ground up to handle constantly changing information, hybrid semantic and graph search, and
scale:

- **Temporal Awareness:** Tracks changes in facts and relationships over time, enabling point-in-time queries. Graph
  edges include temporal metadata to record relationship lifecycles.
- **Episodic Processing:** Ingests data as discrete episodes, maintaining data provenance and allowing incremental
  entity and relationship extraction.
- **Hybrid Search:** Combines semantic and BM25 full-text search, with the ability to rerank results by distance from a
  central node e.g. “Kendra”.
- **Scalable:** Designed for processing large datasets, with parallelization of LLM calls for bulk processing while
  preserving the chronology of events.
- **Supports Varied Sources:** Can ingest both unstructured text and structured JSON data.

<p align="center">
    <img src="/images/graphiti-intro-slides-stock-2.gif" alt="Graphiti structured + unstructured demo" width="700px">   
</p>

## Graphiti and Zep Memory

Graphiti powers the core of [Zep's memory layer](https://www.getzep.com) for LLM-powered Assistants and Agents.

We're excited to open-source Graphiti, believing its potential reaches far beyond memory applications.

## Installation

Requirements:

- Python 3.10 or higher
- Neo4j 5.21 or higher
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
> Support for Anthropic and Groq LLM inferences is available, too.

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

# Initialize Graphiti
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

# Search the graph
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

# Close the connection
graphiti.close()
```

## Graph Service

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

## Documentation

- [Guides and API documentation](https://help.getzep.com/graphiti).
- [Quick Start](https://help.getzep.com/graphiti/graphiti/quick-start)
- [Building an agent with LangChain's LangGraph and Graphiti](https://help.getzep.com/graphiti/graphiti/lang-graph-agent)

## Status and Roadmap

Graphiti is under active development. We aim to maintain API stability while working on:

- [x] Implementing node and edge CRUD operations
- [ ] Improving performance and scalability
- [ ] Achieving good performance with different LLM and embedding models
- [x] Creating a dedicated embedder interface
- [ ] Supporting custom graph schemas:
    - Allow developers to provide their own defined node and edge classes when ingesting episodes
    - Enable more flexible knowledge representation tailored to specific use cases
- [x] Enhancing retrieval capabilities with more robust and configurable options
- [ ] Expanding test coverage to ensure reliability and catch edge cases

## Contributing

We encourage and appreciate all forms of contributions, whether it's code, documentation, addressing GitHub Issues, or
answering questions in the Graphiti Discord channel. For detailed guidelines on code contributions, please refer
to [CONTRIBUTING](CONTRIBUTING.md).

## Support

Join the [Zep Discord server](https://discord.com/invite/W8Kw6bsgXQ) and make your way to the **#Graphiti** channel!
