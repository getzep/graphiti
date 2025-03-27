# Graphiti MCP Server

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents operating in dynamic environments. Unlike traditional retrieval-augmented generation (RAG) methods, Graphiti continuously integrates user interactions, structured and unstructured enterprise data, and external information into a coherent, queryable graph. The framework supports incremental data updates, efficient retrieval, and precise historical queries without requiring complete graph recomputation, making it suitable for developing interactive, context-aware AI applications.

This is an experimental Model Context Protocol (MCP) server implementation for Graphiti. The MCP server exposes Graphiti's key functionality through the MCP protocol, allowing AI assistants to interact with Graphiti's knowledge graph capabilities.

## Features

The Graphiti MCP server exposes the following key high-level functions of Graphiti:

- **Episode Management**: Add, retrieve, and delete episodes (text, messages, or JSON data)
- **Entity Management**: Search and manage entity nodes and relationships in the knowledge graph
- **Search Capabilities**: Search for facts (edges) and node summaries using semantic and hybrid search
- **Group Management**: Organize and manage groups of related data with group_id filtering
- **Graph Maintenance**: Clear the graph and rebuild indices

## Installation

### Prerequisites

1. Ensure you have Python 3.10 or higher installed.
2. A running Neo4j database (version 5.26 or later required)
3. OpenAI API key for LLM operations

### Setup

1. Clone the repository and navigate to the mcp_server directory
2. Use `uv` to create a virtual environment and install dependencies:

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies in one step
uv sync
```

## Configuration

The server uses the following environment variables:

- `NEO4J_URI`: URI for the Neo4j database (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `demodemo`)
- `OPENAI_API_KEY`: OpenAI API key (required for LLM operations)
- `OPENAI_BASE_URL`: Optional base URL for OpenAI API
- `MODEL_NAME`: Optional model name to use for LLM inference

You can set these variables in a `.env` file in the project directory.

## Running the Server

To run the Graphiti MCP server directly using `uv`:

```bash
uv run graphiti_mcp_server.py
```

With options:

```bash
uv run graphiti_mcp_server.py --model gpt-4o --transport sse
```

Available arguments:

- `--model`: Specify the model name to use with the LLM client
- `--transport`: Choose the transport method (sse or stdio, default: sse)
- `--group-id`: Set a namespace for the graph (optional)
- `--destroy-graph`: Destroy all Graphiti graphs (use with caution)
- `--use-custom-entities`: Enable entity extraction using the predefined ENTITY_TYPES

### Docker Deployment

The Graphiti MCP server can be deployed using Docker. The Dockerfile uses `uv` for package management, ensuring consistent dependency installation.

#### Environment Configuration

Before running the Docker Compose setup, you need to configure the environment variables. You have two options:

1. **Using a .env file** (recommended):

   - Copy the provided `.env.example` file to create a `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to set your OpenAI API key and other configuration options:
     ```
     # Required for LLM operations
     OPENAI_API_KEY=your_openai_api_key_here
     MODEL_NAME=gpt-4o
     # Optional: OPENAI_BASE_URL only needed for non-standard OpenAI endpoints
     # OPENAI_BASE_URL=https://api.openai.com/v1
     ```
   - The Docker Compose setup is configured to use this file if it exists (it's optional)

2. **Using environment variables directly**:
   - You can also set the environment variables when running the Docker Compose command:
     ```bash
     OPENAI_API_KEY=your_key MODEL_NAME=gpt-4o docker compose up
     ```

#### Neo4j Configuration

The Docker Compose setup includes a Neo4j container with the following default configuration:

- Username: `neo4j`
- Password: `demodemo`
- URI: `bolt://neo4j:7687` (from within the Docker network)
- Memory settings optimized for development use

#### Running with Docker Compose

Start the services using Docker Compose:

```bash
docker compose up
```

Or if you're using an older version of Docker Compose:

```bash
docker-compose up
```

This will start both the Neo4j database and the Graphiti MCP server. The Docker setup:

- Uses `uv` for package management and running the server
- Installs dependencies from the `pyproject.toml` file
- Connects to the Neo4j container using the environment variables
- Exposes the server on port 8000 for HTTP-based SSE transport
- Includes a healthcheck for Neo4j to ensure it's fully operational before starting the MCP server

## Integrating with MCP Clients

### Configuration

To use the Graphiti MCP server with an MCP-compatible client, configure it to connect to the server:

```json
{
  "mcpServers": {
    "graphiti": {
      "transport": "stdio",
      "command": "uv",
      "args": [
        "run",
        "/ABSOLUTE/PATH/TO/graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "demodemo",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "MODEL_NAME": "gpt-4o"
      }
    }
  }
}
```

For SSE transport (HTTP-based), you can use this configuration:

```json
{
  "mcpServers": {
    "graphiti": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Or start the server with uv and connect to it:

```json
{
  "mcpServers": {
    "graphiti": {
      "command": "uv",
      "args": [
        "run",
        "/ABSOLUTE/PATH/TO/graphiti_mcp_server.py",
        "--transport",
        "sse"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "demodemo",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "MODEL_NAME": "gpt-4o"
      }
    }
  }
}
```

## Available Tools

The Graphiti MCP server exposes the following tools:

- `add_episode`: Add an episode to the knowledge graph (supports text, JSON, and message formats)
- `search_nodes`: Search the knowledge graph for relevant node summaries
- `search_facts`: Search the knowledge graph for relevant facts (edges between entities)
- `delete_entity_edge`: Delete an entity edge from the knowledge graph
- `delete_episode`: Delete an episode from the knowledge graph
- `get_entity_edge`: Get an entity edge by its UUID
- `get_episodes`: Get the most recent episodes for a specific group
- `clear_graph`: Clear all data from the knowledge graph and rebuild indices
- `get_status`: Get the status of the Graphiti MCP server and Neo4j connection

## Working with JSON Data

The Graphiti MCP server can process structured JSON data through the `add_episode` tool with `source="json"`. This allows you to automatically extract entities and relationships from structured data:

```
add_episode(
    name="Customer Profile",
    episode_body="{\"company\": {\"name\": \"Acme Technologies\"}, \"products\": [{\"id\": \"P001\", \"name\": \"CloudSync\"}, {\"id\": \"P002\", \"name\": \"DataMiner\"}]}",
    source="json",
    source_description="CRM data"
)
```

## Integrating with the Cursor IDE

To integrate the Graphiti MCP Server with the Cursor IDE, follow these steps:

1. Run the Graphiti MCP server using the SSE transport:

```bash
python graphiti_mcp_server.py --transport sse --use-custom-entities --group-id <your_group_id>
```

Hint: specify a `group_id` to retain prior graph data. If you do not specify a `group_id`, the server will create a new graph

2. Configure Cursor to connect to the Graphiti MCP server.

```json
{
  "mcpServers": {
    "Graphiti": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

3. Add the Graphiti rules to Cursor's User Rules. See [cursor_rules.md](cursor_rules.md) for details.

4. Kick off an agent session in Cursor.

The integration enables AI assistants in Cursor to maintain persistent memory through Graphiti's knowledge graph capabilities.

## Requirements

- Python 3.10 or higher
- Neo4j database (version 5.26 or later required)
- OpenAI API key (for LLM operations and embeddings)
- MCP-compatible client

## License

This project is licensed under the same license as the Graphiti project.
