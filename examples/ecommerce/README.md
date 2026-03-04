# E-commerce Conversation Example

This example demonstrates how to use Graphiti to build a knowledge graph from an e-commerce sales conversation. It showcases how Graphiti can extract product information, customer preferences, and conversational context from a chat between a customer and a sales assistant.

## Overview

The runner script (`runner.py`) simulates a conversation between a customer (John) and an Allbirds shoe sales assistant. The conversation covers product inquiries, preferences, and purchasing decisions, all of which are captured in the knowledge graph.

An interactive Jupyter notebook (`runner.ipynb`) is also available for step-by-step exploration.

## Prerequisites

- Python 3.10+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Neo4j database running locally or remotely

## Setup

1. Install the required dependencies:

```bash
pip install graphiti-core python-dotenv
```

2. Set up environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

## Running the Example

```bash
# Using the Python script
python runner.py

# Or using the Jupyter notebook
jupyter notebook runner.ipynb
```

## How It Works

1. **Conversation Ingestion**: Each message in the sales conversation is added to Graphiti as an episode of type `message`.

2. **Knowledge Extraction**: Graphiti automatically extracts:
   - **Customer entities**: The customer (John) and their attributes
   - **Product entities**: Shoes, materials (wool, tree fiber), styles
   - **Preferences**: Customer preferences for materials, colors, and styles
   - **Relationships**: Links between customers, products, and preferences

3. **Search**: After ingestion, the example performs searches to retrieve relevant facts from the knowledge graph.

## Key Concepts

- **Message-type Episodes**: Each chat message is ingested as an individual episode, preserving conversational flow.
- **Product Knowledge Graphs**: Demonstrates how Graphiti can build structured product knowledge from unstructured conversations.
- **Customer Preference Tracking**: Shows how Graphiti captures and links customer preferences to products over the course of a conversation.

## Use Cases

This pattern is applicable to:
- **Customer support bots** that need to remember user preferences across sessions
- **Sales assistants** that personalize recommendations based on conversation history
- **Product recommendation engines** built on conversational data

## Related Examples

- `examples/quickstart/` — Basic Graphiti usage
- `examples/langgraph-agent/` — Building an interactive sales agent with LangGraph and Graphiti
- `examples/podcast/` — Processing multi-speaker conversations
