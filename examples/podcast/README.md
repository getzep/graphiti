# Podcast Transcript Example

This example demonstrates how to use Graphiti to build a knowledge graph from a podcast transcript. It showcases Graphiti's ability to process conversational data with multiple speakers and temporal information.

## Overview

The example includes:

- **Transcript Parser** (`transcript_parser.py`): Parses a podcast transcript into structured messages with speaker information, timestamps, and roles.
- **Podcast Runner** (`podcast_runner.py`): Ingests the parsed transcript into Graphiti as a series of episodes and performs searches on the resulting knowledge graph.

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

3. Ensure the transcript file (`podcast_transcript.txt`) is in the same directory.

## Running the Example

```bash
python podcast_runner.py
```

## How It Works

1. **Transcript Parsing**: The `transcript_parser.py` module reads the raw transcript and extracts individual messages with:
   - Speaker name and role
   - Relative and absolute timestamps
   - Message content

2. **Graph Construction**: Each message is added to Graphiti as an episode, preserving the temporal ordering and speaker attribution.

3. **Knowledge Extraction**: Graphiti automatically extracts entities and relationships from the conversation, building a queryable knowledge graph.

## Key Concepts

- **Multi-speaker Episodes**: Demonstrates handling conversations with multiple participants, each with distinct roles.
- **Temporal Ordering**: Messages are ingested with accurate timestamps, enabling temporal queries on the knowledge graph.
- **Bulk Ingestion**: Uses `RawEpisode` for efficient batch processing of transcript data.

## Related Examples

- `examples/quickstart/` — Basic Graphiti usage
- `examples/wizard_of_oz/` — Processing literary text
- `examples/langgraph-agent/` — Building an interactive agent with Graphiti
