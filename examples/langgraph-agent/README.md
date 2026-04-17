# LangGraph Agent Example

This example demonstrates how to build an interactive sales agent using [LangGraph](https://github.com/langchain-ai/langgraph) and Graphiti. The agent uses Graphiti as its memory and knowledge layer to personalize responses based on prior conversations and product information.

## Overview

The Jupyter notebook (`agent.ipynb`) walks through building a **ShoeBot Sales Agent** that:

- Persists new chat turns to Graphiti and recalls relevant facts
- Uses a tool to query the Graphiti knowledge graph for shoe information
- Maintains agent state with an in-memory `MemorySaver`

## Prerequisites

- Python 3.10+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Neo4j database running locally or remotely
- Jupyter Notebook or JupyterLab

## Setup

1. Install the required dependencies:

```bash
pip install graphiti-core langchain-openai langgraph ipywidgets python-dotenv
```

2. Set up environment variables (or use a `.env` file):

```bash
export OPENAI_API_KEY=your_openai_api_key
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

3. _(Optional)_ To enable LangSmith tracing, set:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGCHAIN_PROJECT="Graphiti LangGraph Tutorial"
```

## Running the Example

```bash
jupyter notebook agent.ipynb
```

## How It Works

1. **Graphiti Initialization**: Connects to Neo4j and sets up Graphiti with indices and constraints.
2. **Product Data Loading**: Loads shoe product data into the Graphiti knowledge graph.
3. **Agent Construction**: Builds a LangGraph agent with:
   - A **search tool** that queries Graphiti for product information
   - **Memory persistence** that saves each conversation turn to Graphiti
   - **Fact recall** that retrieves relevant facts based on the latest user message
4. **Interactive Chat**: The agent responds to user queries using both its LLM capabilities and the knowledge stored in Graphiti.

## Key Concepts

- **Agent Memory with Knowledge Graphs**: Instead of simple chat history, the agent builds a rich knowledge graph that captures entities and relationships from conversations.
- **Tool-augmented Agents**: The agent uses a Graphiti search tool to retrieve product information on demand.
- **Framework Integration**: Shows how Graphiti integrates with the LangChain/LangGraph ecosystem.

## Related Examples

- `examples/ecommerce/` — E-commerce conversation without an agent framework
- `examples/quickstart/` — Basic Graphiti usage
- `examples/podcast/` — Processing conversational data
