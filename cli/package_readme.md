# Graphiti CLI

A supplementary command-line interface for direct interaction with Graphiti Core, providing data ingestion and graph search functionality. While the MCP server is the primary means of interacting with Graphiti, this CLI offers an alternative approach for certain workflows.

## Installation

```bash
pip install graphiti-cli
```

## Configuration

Set the following environment variables:

- `NEO4J_URI` (Default: bolt://localhost:7687)
- `NEO4J_USER` (Default: neo4j)
- `NEO4J_PASSWORD` (Default: demodemo)
- `OPENAI_API_KEY` (Optional, for embedding and search functionality)
- `MODEL_NAME` (Default: gpt-3.5-turbo)

## Basic Usage

```bash
# Check connection to Neo4j
graphiti check-connection

# Add JSON data from a file
graphiti add-json --json-file path/to/data.json --name "Dataset Name" 

# Search for nodes
graphiti search-nodes --query "search term"

# Get help
graphiti --help
```

For complete documentation, visit the [Graphiti documentation](https://help.getzep.com/graphiti/graphiti/overview). 