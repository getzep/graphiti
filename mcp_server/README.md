# Graphiti MCP Server

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents
operating in dynamic environments. Unlike traditional retrieval-augmented generation (RAG) methods, Graphiti
continuously integrates user interactions, structured and unstructured enterprise data, and external information into a
coherent, queryable graph. The framework supports incremental data updates, efficient retrieval, and precise historical
queries without requiring complete graph recomputation, making it suitable for developing interactive, context-aware AI
applications.

This is an experimental Model Context Protocol (MCP) server implementation for Graphiti. The MCP server exposes
Graphiti's key functionality through the MCP protocol, allowing AI assistants to interact with Graphiti's knowledge
graph capabilities.

## Features

The Graphiti MCP server exposes the following key high-level functions of Graphiti:

- **Episode Management**: Add, retrieve, and delete episodes (text, messages, or JSON data)
- **Entity Management**: Search and manage entity nodes and relationships in the knowledge graph
- **Search Capabilities**: Search for facts (edges) and node summaries using semantic and hybrid search
- **Group Management**: Organize and manage groups of related data with group_id filtering
- **Multi-Project Support**: URL-based group_id switching for seamless project isolation
- **Graph Maintenance**: Clear the graph and rebuild indices

## Quick Start for Claude Desktop, Cursor, and other clients

1. Clone the Graphiti GitHub repo

```bash
git clone https://github.com/getzep/graphiti.git
```

or

```bash
gh repo clone getzep/graphiti
```

Note the full path to this directory.

```
cd graphiti && pwd
```

2. Install the [Graphiti prerequisites](#prerequisites).

3. Configure Claude, Cursor, or other MCP client to use [Graphiti with a `stdio` transport](#integrating-with-mcp-clients). See the client documentation on where to find their MCP configuration files.

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
- `MODEL_NAME`: OpenAI model name to use for LLM operations.
- `SMALL_MODEL_NAME`: OpenAI model name to use for smaller LLM operations.
- `LLM_TEMPERATURE`: Temperature for LLM responses (0.0-2.0).
- `AZURE_OPENAI_ENDPOINT`: Optional Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Optional Azure OpenAI deployment name
- `AZURE_OPENAI_API_VERSION`: Optional Azure OpenAI API version
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`: Optional Azure OpenAI embedding deployment name
- `AZURE_OPENAI_EMBEDDING_API_VERSION`: Optional Azure OpenAI API version
- `AZURE_OPENAI_USE_MANAGED_IDENTITY`: Optional use Azure Managed Identities for authentication

You can set these variables in a `.env` file in the project directory.

## Running the Server

To run the Graphiti MCP server directly using `uv`:

```bash
uv run graphiti_mcp_server.py
```

With options:

```bash
uv run graphiti_mcp_server.py --model gpt-4.1-mini --transport sse
```

Available arguments:

- `--model`: Overrides the `MODEL_NAME` environment variable.
- `--small-model`: Overrides the `SMALL_MODEL_NAME` environment variable.
- `--temperature`: Overrides the `LLM_TEMPERATURE` environment variable.
- `--transport`: Choose the transport method (sse or stdio, default: sse)
- `--group-id`: Set a namespace for the graph (optional). If not provided, defaults to "default".
- `--destroy-graph`: If set, destroys all Graphiti graphs on startup.
- `--use-custom-entities`: Enable entity extraction using the predefined ENTITY_TYPES

### Docker Deployment

The Graphiti MCP server can be deployed using Docker. The Dockerfile uses `uv` for package management, ensuring
consistent dependency installation.

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
     MODEL_NAME=gpt-4.1-mini
     # Optional: OPENAI_BASE_URL only needed for non-standard OpenAI endpoints
     # OPENAI_BASE_URL=https://api.openai.com/v1
     ```
   - The Docker Compose setup is configured to use this file if it exists (it's optional)

2. **Using environment variables directly**:
   - You can also set the environment variables when running the Docker Compose command:
     ```bash
     OPENAI_API_KEY=your_key MODEL_NAME=gpt-4.1-mini docker compose up
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
- **Supports URL-based group_id switching** for multi-project setups
- Includes a healthcheck for Neo4j to ensure it's fully operational before starting the MCP server

#### Multi-Project Access

Once running, you can access different project namespaces using URL query parameters:

- Default: `http://localhost:8000/sse`
- Project A: `http://localhost:8000/sse?group_id=project_a`
- Project B: `http://localhost:8000/sse?group_id=project_b`
- Personal: `http://localhost:8000/sse?group_id=personal`

Each URL provides completely isolated data storage while using the same server instance.

## Integrating with MCP Clients

### Configuration

To use the Graphiti MCP server with an MCP-compatible client, configure it to connect to the server:

> [!IMPORTANT]
> You will need the Python package manager, `uv` installed. Please refer to the [`uv` install instructions](https://docs.astral.sh/uv/getting-started/installation/).
>
> Ensure that you set the full path to the `uv` binary and your Graphiti project folder.

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "stdio",
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "run",
        "--isolated",
        "--directory",
        "/Users/<user>>/dev/zep/graphiti/mcp_server",
        "--project",
        ".",
        "graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OPENAI_API_KEY": "sk-XXXXXXXX",
        "MODEL_NAME": "gpt-4.1-mini"
      }
    }
  }
}
```

For SSE transport (HTTP-based), you can use this configuration:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### Multi-Project Setup with Group IDs

Graphiti supports isolating data between different projects using `group_id` namespaces. You can specify a group_id in several ways:

#### Option 1: URL Query Parameter (Recommended for SSE Transport)

**NEW FEATURE**: For the most seamless multi-project experience, you can now specify the group_id directly in the SSE URL query parameter:

```json
{
  "mcpServers": {
    "graphiti-project-a": {
      "transport": "sse",
      "url": "http://localhost:8000/sse?group_id=project_a"
    },
    "graphiti-project-b": {
      "transport": "sse",
      "url": "http://localhost:8000/sse?group_id=project_b"
    },
    "graphiti-personal": {
      "transport": "sse",
      "url": "http://localhost:8000/sse?group_id=personal"
    }
  }
}
```

**Benefits of URL-based group_id switching**:

- **Single server instance**: Use one Graphiti server for unlimited projects
- **Automatic data isolation**: Each project's data is completely separated
- **Zero configuration**: No need for multiple server instances or complex setup
- **Instant switching**: Change projects by simply switching MCP server connections
- **Backward compatible**: Works alongside existing group_id methods
- **Real-time**: Group switching takes effect immediately without server restart

**How it works**: The server automatically extracts the `group_id` from the URL query parameter and uses it for all operations in that connection. Each SSE connection maintains its own isolated namespace.

#### Option 2: Environment Variables (For STDIO Transport)

For STDIO transport, you can set the group_id via environment variables:

```json
{
  "mcpServers": {
    "graphiti-project-a": {
      "transport": "stdio",
      "command": "uv",
      "args": ["run", "graphiti_mcp_server.py", "--group-id", "project_a"],
      "env": {
        "OPENAI_API_KEY": "sk-XXXXXXXX"
      }
    }
  }
}
```

#### Option 3: Tool-Level Group ID

You can also specify group_id in individual tool calls:

```python
add_episode(
    name="Project Feature",
    episode_body="Implementation details...",
    group_id="project_a"  # This will override the default group_id
)
```

#### Verifying Group Isolation

You can verify that your groups are properly isolated by querying the Neo4j database:

```cypher
// See all groups and their data counts
MATCH (n) RETURN DISTINCT n.group_id as group_id, labels(n) as node_types, count(n) as count ORDER BY group_id

// See episodes for a specific group
MATCH (e:Episodic {group_id: "project_a"}) RETURN e.name, e.content
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

The Graphiti MCP server can process structured JSON data through the `add_episode` tool with `source="json"`. This
allows you to automatically extract entities and relationships from structured data:

```

add_episode(
name="Customer Profile",
episode_body="{\"company\": {\"name\": \"Acme Technologies\"}, \"products\": [{\"id\": \"P001\", \"name\": \"CloudSync\"}, {\"id\": \"P002\", \"name\": \"DataMiner\"}]}",
source="json",
source_description="CRM data"
)

```

## Cursor Setup Guide for Graphiti MCP Server

This section walks you through setting up Graphiti's MCP (Model Context Protocol) server and integrating it with Cursor for persistent agent memory.

### Prerequisites for Cursor Integration

- **Python 3.10+**
- **A Neo4j 5.26+ instance** (can be run via Docker)
- **An OpenAI API key** - Get one from [OpenAI's platform](https://platform.openai.com/api-keys)
- **Docker and Docker Compose** (recommended for easy setup)

### Step 1: Clone and Install

```bash
git clone https://github.com/getzep/graphiti.git
cd graphiti/mcp_server

# Install 'uv' if needed (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment & install dependencies
uv sync
```

### Step 2: Configure Environment

Create your environment configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```dotenv
# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
MODEL_NAME=gpt-4o-mini

# Optional: Embedder model (defaults to text-embedding-3-small)
EMBEDDER_MODEL_NAME=text-embedding-3-small
```

> **Note**: Replace `sk-your-openai-api-key-here` with your actual OpenAI API key.

### Step 3: Run the MCP Server

You have several options for running the server:

#### Option A: Docker Compose (Recommended)

This automatically sets up both Neo4j and the Graphiti MCP server:

```bash
# Make sure your .env file is configured
docker compose up -d
```

The server will be available at `http://localhost:8000/sse`

#### Option B: Local Development

If you have Neo4j running separately:

```bash
# For SSE transport (recommended for Cursor)
uv run graphiti_mcp_server.py --transport sse

# For StdIO transport (for direct CLI clients)
uv run graphiti_mcp_server.py --transport stdio
```

### Step 4: Configure Cursor

#### Add MCP Server to Cursor Config

1. Open your Cursor MCP configuration file:

   - **Windows**: `%APPDATA%\Cursor\User\globalStorage\mcp.json`
   - **macOS**: `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`
   - **Linux**: `~/.config/Cursor/User/globalStorage/mcp.json`

2. Add the Graphiti MCP server configuration:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

If you already have other MCP servers configured, just add the `"graphiti-memory"` entry to your existing `"mcpServers"` object.

#### Enable Graphiti Rules in Cursor

1. Copy the contents of `cursor_rules.md` from the repository
2. In Cursor, go to **Settings** → **Rules** → **User Rules**
3. Paste the Graphiti rules into your User Rules

These rules instruct Cursor to:

- Search for existing preferences and procedures before starting tasks
- Save new requirements, preferences, and procedures to memory
- Use the MCP tools (`add_episode`, `search_nodes`, `search_facts`) automatically

### Step 5: Verify Setup

1. **Check server status**: Visit `http://localhost:8000/sse` in your browser - you should see a connection attempt
2. **Check Docker logs** (if using Docker):
   ```bash
   docker compose logs graphiti-mcp
   ```
3. **Restart Cursor** to load the new MCP configuration

### Step 6: Test the Integration

1. Open a new Cursor session
2. Try asking Cursor to remember something:
   ```
   Remember that I prefer using TypeScript over JavaScript for new projects
   ```
3. Later, ask Cursor about your preferences:
   ```
   What programming language preferences do you know about me?
   ```

If everything is working correctly, Cursor should be able to store and retrieve your preferences using the Graphiti memory system.

### Troubleshooting Cursor Integration

#### Common Issues

1. **Connection refused errors**:

   - Ensure the MCP server is running on `http://localhost:8000/sse`
   - Check that Docker containers are up: `docker compose ps`

2. **Authentication errors**:

   - Verify your OpenAI API key is correct in the `.env` file
   - Check that you have sufficient OpenAI API credits

3. **Neo4j connection issues**:

   - Ensure Neo4j is running and accessible
   - Verify the connection details in your `.env` file

4. **Cursor not using MCP tools**:
   - Restart Cursor after adding the MCP configuration
   - Check that the Graphiti rules are properly added to User Rules
   - Verify the MCP server URL in your Cursor config

#### Checking Logs

**Docker setup**:

```bash
# Check Graphiti MCP server logs
docker compose logs graphiti-mcp

# Check Neo4j logs
docker compose logs neo4j
```

**Local setup**:
Check the terminal where you're running the MCP server for any error messages.

### Advanced Configuration for Cursor

#### Custom Entity Types

Enable custom entity extraction by running with the `--use-custom-entities` flag:

```bash
uv run graphiti_mcp_server.py --transport sse --use-custom-entities
```

This enables extraction of:

- **Requirements**: Project needs and specifications
- **Preferences**: User likes/dislikes
- **Procedures**: Step-by-step instructions

#### Different Models

You can specify different models in your `.env` file:

```dotenv
MODEL_NAME=gpt-4o
SMALL_MODEL_NAME=gpt-4o-mini
EMBEDDER_MODEL_NAME=text-embedding-3-large
```

### Next Steps with Cursor

Once everything is working:

1. **Start using memory**: Ask Cursor to remember your coding preferences, project requirements, and procedures
2. **Explore the tools**: The MCP server provides tools for searching nodes, facts, and managing episodes
3. **Customize rules**: Modify the Cursor rules to better fit your workflow

Your Cursor AI assistant now has persistent memory powered by Graphiti's knowledge graph! 🎉

## Integrating with Claude Desktop (Docker MCP Server)

The Graphiti MCP Server container uses the SSE MCP transport. Claude Desktop does not natively support SSE, so you'll need to use a gateway like `mcp-remote`.

1.  **Run the Graphiti MCP server using SSE transport**:

    ```bash
    docker compose up
    ```

2.  **(Optional) Install `mcp-remote` globally**:
    If you prefer to have `mcp-remote` installed globally, or if you encounter issues with `npx` fetching the package, you can install it globally. Otherwise, `npx` (used in the next step) will handle it for you.

    ```bash
    npm install -g mcp-remote
    ```

3.  **Configure Claude Desktop**:
    Open your Claude Desktop configuration file (usually `claude_desktop_config.json`) and add or modify the `mcpServers` section as follows:

    ```json
    {
      "mcpServers": {
        "graphiti-memory": {
          // You can choose a different name if you prefer
          "command": "npx", // Or the full path to mcp-remote if npx is not in your PATH
          "args": [
            "mcp-remote",
            "http://localhost:8000/sse" // Ensure this matches your Graphiti server's SSE endpoint
          ]
        }
      }
    }
    ```

    If you already have an `mcpServers` entry, add `graphiti-memory` (or your chosen name) as a new key within it.

4.  **Restart Claude Desktop** for the changes to take effect.

## Support

- **GitHub Issues**: [Graphiti Repository](https://github.com/getzep/graphiti/issues)
- **Documentation**: [Graphiti Docs](https://docs.getzep.com/graphiti/)
- **Community**: Join the discussion in the Graphiti community channels

## Requirements

- Python 3.10 or higher
- Neo4j database (version 5.26 or later required)
- OpenAI API key (for LLM operations and embeddings)
- MCP-compatible client

## License

This project is licensed under the same license as the parent Graphiti project.
