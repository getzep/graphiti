# Graphiti (LLM generated readme)

Graphiti is a Python library for building and managing knowledge graphs using Neo4j and OpenAI's language models. It provides a flexible framework for processing episodes of information, extracting semantic nodes and edges, and maintaining a dynamic graph structure.

## Features

- Asynchronous interaction with Neo4j database
- Integration with OpenAI's GPT models for natural language processing
- Automatic extraction of semantic nodes and edges from episodic data
- Temporal tracking of relationships and facts
- Flexible schema management

## Installation

(Add installation instructions here)

## Quick Start

```python
from graphiti import Graphiti

# Initialize Graphiti
graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")

# Process an episode
await graphiti.process_episode(
    name="Example Episode",
    episode_body="Alice met Bob at the coffee shop.",
    source_description="User input",
    reference_time=datetime.now()
)

# Retrieve recent episodes
recent_episodes = await graphiti.retrieve_episodes(last_n=5)

# Close the connection
graphiti.close()
```

## Documentation

(Add link to full documentation when available)

## Contributing

(Add contribution guidelines)

## License

(Add license information)
