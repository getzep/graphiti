# Graphiti

<div align="center">

<img width="350" alt="Graphiti-ts-small" src="https://github.com/user-attachments/assets/bbd02947-e435-4a05-b25a-bbbac36d52c8">


<br />

[![Discord](https://dcbadge.vercel.app/api/server/W8Kw6bsgXQ?style=flat)](https://discord.com/invite/W8Kw6bsgXQ)
[![Lint](https://github.com/getzep/Graphiti/actions/workflows/lint.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/lint.yml)
[![Unit Tests](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml)
[![MyPy Check](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/typecheck.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/getzep/Graphiti)

<br />
</div>

Graphiti is an open-source framework for building dynamic, temporally-aware Knowledge Graphs that evolve over time. Designed for AI-driven applications, Graphiti enables the ingestion of both structured and unstructured data while supporting hybrid search approaches that combine semantic, full-text, and graph-based techniques.

## Why Graphiti?

- **Temporal Awareness**: Tracks relationships over time, allowing point-in-time queries.
- **Hybrid Search**: Combines semantic, BM25 full-text, and graph-based search to enhance retrieval.
- **Scalable Architecture**: Supports large-scale knowledge extraction and efficient query execution.
- **AI-Enhanced Reasoning**: Designed to power LLM applications with evolving knowledge states.
- **Flexible Data Ingestion**: Works with both unstructured text and structured JSON data.

## Knowledge Graph-Powered Memory

Graphiti serves as the core engine behind Zep, a novel memory layer service for AI agents. Unlike traditional retrieval-augmented generation (RAG) approaches that rely on static document retrieval, Graphiti dynamically integrates unstructured conversational data and structured business data while maintaining historical relationships This enables AI agents to:

- Maintain long-term recall and context across sessions.
- Reason with evolving data and temporal relationships.
- Improve response accuracy and reduce hallucinations in enterprise AI applications.

Graphiti has been rigorously evaluated against state-of-the-art AI memory systems, outperforming MemGPT in Deep Memory Retrieval (DMR) benchmarks and excelling in more complex evaluations such as LongMemEval[<sup>‡</sup>](https://arxiv.org/abs/2501.13956).

## Real-World Applications
<p align="center">
    <img src="/images/graphiti-graph-intro.gif" alt="Graphiti temporal walkthrough" width="700px">   
</p>
Graphiti helps you create and query Knowledge Graphs that evolve over time. A knowledge graph is a network of interconnected facts, such as "Kendra loves Adidas shoes." Each fact is a "triplet" represented by two entities, or nodes ("Kendra", "Adidas shoes"), and their relationship, or edge ("loves"). Knowledge Graphs have been explored extensively for information retrieval. What makes Graphiti unique is its ability to autonomously build a knowledge graph while handling changing relationships and maintaining historical context.

&nbsp;

<p align="center">
    <img src="/images/graphiti-intro-slides-stock-2.gif" alt="Graphiti structured + unstructured demo" width="700px">   
</p>
<br />

With Graphiti, you can build LLM applications such as:

- **Assistants** that learn from user interactions, fusing personal knowledge with dynamic data from business systems like CRMs and billing platforms.
- **Agents** that autonomously execute complex tasks, reasoning with state changes from multiple dynamic sources.

Graphiti supports a wide range of applications in **sales, customer service, health, finance, and more**, enabling long-term recall and state-based reasoning for both assistants and agents.

## Getting Started

### Requirements:

Ensure you have the following installed:

- Python 3.10 or higher
- Neo4j 5.21 or higher
- OpenAI API key (or Anthropic/Groq API key for alternative LLM providers)


 💡 **[Neo4j Desktop](https://neo4j.com/download/)** provides a user-friendly interface to manage Neo4j instances and databases.

### Installation:

To install Graphiti, use:
```bash
pip install graphiti-core
```

Or with Poetry:

```bash
poetry add graphiti-core
```
### Quick Start:
```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

# Initialize Graphiti
graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")

# Build indices (run once)
graphiti.build_indices_and_constraints()

# Add knowledge episodes
episodes = [
    "Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]
for i, episode in enumerate(episodes):
    await graphiti.add_episode(
        name=f"Episode {i}",
        episode_body=episode,
        source=EpisodeType.text,
        source_description="historical record",
        reference_time=datetime.now(timezone.utc)
    )

# Perform a hybrid search
results = await graphiti.search('Who was the California Attorney General?')
```

## Graphiti and Zep Memory

Graphiti powers the core of **[Zep's memory layer](https://www.getzep.com)** for LLM-powered Assistants and Agents.

We're excited to open-source Graphiti, believing its potential reaches far beyond memory applications.

## Documentation

Explore our full **[Guides and API Documentation](https://help.getzep.com/graphiti)** for detailed setup instructions and best practices.

📖 **[Quick Start](https://help.getzep.com/graphiti/graphiti/quick-start)**\
🚀 **[Building an agent with LangChain's LangGraph and Graphiti](https://help.getzep.com/graphiti/graphiti/lang-graph-agent)**

## Optional Environment Variables

In addition to the Neo4j and OpenAI-compatible credentials, Graphiti also has optional environment variables for enhanced functionality:

- `USE_PARALLEL_RUNTIME`: (Boolean) Enables Neo4j's parallel runtime feature for search queries. Note that this is not supported for Neo4j Community Edition or smaller AuraDB instances, so it is off by default.
- Additional model-related environment variables may be required for using supported models such as Anthropic or Voyage.

## Graph Service

The server directory contains an API service for interacting with the Graphiti API. It is built using FastAPI.

Please see the [server README](./server/README.md) for more information.

## Contributing

We encourage and appreciate all forms of contributions, whether it's code, documentation, addressing GitHub Issues, or answering questions in the Graphiti Discord channel. Check out our **[Contributing Guide](CONTRIBUTING.md)** for detailed instructions.

## Community & Support

💬 **[Join us on Discord](https://discord.com/invite/W8Kw6bsgXQ)**

If you find Graphiti useful, consider giving us a **⭐ on GitHub**. It helps us grow the community and improve the project!

## License

Graphiti is licensed under the [Apache 2.0 License](LICENSE).

## Roadmap

Graphiti is actively being developed. Upcoming features include:

- Enhanced scalability and performance optimizations
- **Supporting Custom Graph Schemas**:
  - Allow developers to define their own node and edge classes when ingesting episodes.
  - Enable more flexible knowledge representation tailored to specific use cases, ensuring adaptability to different domain-specific requirements.
- Expanded test coverage

## Learn More

For an in-depth look at Graphiti's architecture and performance benchmarks, check out our research paper:

📄 **[Using Knowledge Graphs to Power LLM-Agent Memory](https://arxiv.org/abs/2501.13956)**

