# Wizard of Oz Example

This example demonstrates how to use Graphiti to build a knowledge graph from a literary text — _The Wonderful Wizard of Oz_ by L. Frank Baum. It showcases processing long-form narrative content and extracting entities and relationships across chapters.

## Overview

The example includes:

- **Text Parser** (`parser.py`): Splits the full text of _The Wizard of Oz_ into chapter-based episodes.
- **Runner** (`runner.py`): Ingests the parsed chapters into Graphiti and builds a knowledge graph of characters, locations, and events.

## Prerequisites

- Python 3.10+
- Anthropic API key (set as `ANTHROPIC_API_KEY` environment variable) — this example uses the Anthropic LLM client
- Neo4j database running locally or remotely

## Setup

1. Install the required dependencies:

```bash
pip install graphiti-core python-dotenv
```

2. Set up environment variables:

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

3. Ensure the text file (`woo.txt`) is in the same directory.

## Running the Example

```bash
python runner.py
```

## How It Works

1. **Text Parsing**: The parser splits the full text into chapters using regex-based chapter detection, extracting:
   - Chapter number
   - Chapter title
   - Chapter content

2. **Graph Construction**: Each chapter is added to Graphiti as an episode with simulated temporal spacing (one day per chapter).

3. **Knowledge Extraction**: Graphiti extracts characters (Dorothy, Toto, the Scarecrow, etc.), locations (Kansas, Oz, Emerald City), and the relationships between them.

## Key Concepts

- **Long-form Text Processing**: Demonstrates handling book-length content by splitting into logical episodes (chapters).
- **Anthropic LLM Client**: Shows how to use Graphiti with Anthropic's Claude instead of OpenAI.
- **Narrative Knowledge Graphs**: Builds a graph that captures story elements — characters, locations, events, and their evolving relationships.

## Related Examples

- `examples/podcast/` — Processing conversational transcripts
- `examples/quickstart/` — Basic Graphiti usage with OpenAI
- `examples/ecommerce/` — Domain-specific knowledge graphs
