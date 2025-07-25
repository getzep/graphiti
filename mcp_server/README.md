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
- **Graph Maintenance**: Clear the graph and rebuild indices
- **Flexible Ollama Configuration**: Fully configurable LLM and embedding models via CLI arguments and environment variables

## Quick Start

### Clone the Graphiti GitHub repo

```bash
git clone https://github.com/getzep/graphiti.git
```

or

```bash
gh repo clone getzep/graphiti
```

### For Claude Desktop and other `stdio` only clients

1. Note the full path to this directory.

```
cd graphiti && pwd
```

2. Install the [Graphiti prerequisites](#prerequisites).

3. Configure Claude, Cursor, or other MCP client to use [Graphiti with a `stdio` transport](#integrating-with-mcp-clients). See the client documentation on where to find their MCP configuration files.

### For Cursor and other `sse`-enabled clients

1. Change directory to the `mcp_server` directory

`cd graphiti/mcp_server`

2. Start the service using Docker Compose

`docker compose up`

3. Point your MCP client to `http://localhost:8020/sse`

## Installation

### Prerequisites

1. Ensure you have Python 3.10 or higher installed.
2. A running Neo4j database (version 5.26 or later required)
3. **Ollama** installed and running (default) OR OpenAI API key for LLM operations

### Ollama Setup (Default)

The server now defaults to using Ollama for LLM operations and embeddings. To set up Ollama:

1. **Install Ollama**: Visit [https://ollama.ai](https://ollama.ai) for installation instructions
2. **Start Ollama**: Run `ollama serve` to start the Ollama server
3. **Pull required models**:
   ```bash
   ollama pull deepseek-r1:7b     # LLM model
   ollama pull nomic-embed-text   # Embedding model
   ```

The server will automatically connect to Ollama at `http://localhost:11434/v1` and use these models by default.

### OpenAI Setup (Alternative)

If you prefer to use OpenAI instead of Ollama:

1. Set the environment variable: `USE_OLLAMA=false`
2. Configure your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`
3. Optionally customize the models using `MODEL_NAME` and `SMALL_MODEL_NAME` environment variables

### Docker Setup

The project includes Docker Compose configuration for easy deployment. There are several ways to configure Ollama with Docker:

#### Option 1: Use Host Ollama (Recommended)

If you have Ollama running on your host machine:

1. **Start Ollama on your host**:
   ```bash
   ollama serve
   ```

2. **Pull required models**:
   ```bash
   ollama pull deepseek-r1:7b
   ollama pull nomic-embed-text
   ```

3. **Start the services**:
   ```bash
   docker compose up
   ```

The server will connect to your host Ollama instance using `host.docker.internal:11434`.

**Note**: The Graphiti core library requires an `OPENAI_API_KEY` environment variable even when using Ollama (for the reranker component). The Docker configuration includes a dummy API key (`abc`) for this purpose.

#### Option 2: Use Containerized Ollama

If you prefer to run Ollama in a container:

1. **Uncomment the Ollama service** in `docker-compose.yml`:
   ```yaml
   ollama:
     image: ollama/ollama:latest
     ports:
       - "11434:11434"
     volumes:
       - ollama_data:/root/.ollama
   ```

2. **Update the OLLAMA_BASE_URL** in the environment section:
   ```yaml
   environment:
     - OLLAMA_BASE_URL=http://ollama:11434/v1
   ```

3. **Uncomment the volume**:
   ```yaml
   volumes:
     ollama_data:
   ```

4. **Start the services**:
   ```bash
   docker compose up
   ```

5. **Pull models in the container**:
   ```bash
   docker compose exec ollama ollama pull deepseek-r1:7b
   docker compose exec ollama ollama pull nomic-embed-text
   ```

#### Option 3: Use OpenAI with Docker

To use OpenAI instead of Ollama in Docker:

1. **Create a `.env` file** with your OpenAI configuration:
   ```bash
   USE_OLLAMA=false
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL_NAME=gpt-4o-mini
   ```

2. **Update docker-compose.yml** to use the `.env` file:
   ```yaml
   env_file:
     - .env
   ```

3. **Start the services**:
   ```bash
   docker compose up
   ```

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mcp_server
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp sample_env.txt .env
   # Edit .env with your configuration
   ```

4. **Start Neo4j** (if using Docker):
   ```bash
   docker compose up neo4j -d
   ```

5. **Run the server**:
   ```bash
   uv run src/graphiti_mcp_server.py.py --transport sse
   ```

## Configuration

The server uses the following environment variables:

- `NEO4J_URI`: URI for the Neo4j database (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `demodemo`)

### LLM Configuration

The server now defaults to using **Ollama** for LLM operations and embeddings. You can configure it using these environment variables:

#### Ollama Configuration (Default)
- `USE_OLLAMA`: Use Ollama for LLM and embeddings (default: `true`)
- `OLLAMA_BASE_URL`: Ollama base URL (default: `http://localhost:11434/v1`)
- `OLLAMA_LLM_MODEL`: Ollama LLM model name (default: `deepseek-r1:7b`)
- `OLLAMA_EMBEDDING_MODEL`: Ollama embedding model name (default: `nomic-embed-text`)
- `OLLAMA_EMBEDDING_DIM`: Ollama embedding dimension (default: `768`)

#### OpenAI Configuration (Alternative)
To use OpenAI instead of Ollama, set `USE_OLLAMA=false` and configure:
- `OPENAI_API_KEY`: OpenAI API key (required for LLM operations)
- `OPENAI_BASE_URL`: Optional base URL for OpenAI API
- `MODEL_NAME`: OpenAI model name to use for LLM operations (default: `gpt-4.1-mini`)
- `SMALL_MODEL_NAME`: OpenAI model name to use for smaller LLM operations (default: `gpt-4.1-nano`)
- `LLM_TEMPERATURE`: Temperature for LLM responses (0.0-2.0)

#### Azure OpenAI Configuration (Alternative)
To use Azure OpenAI, set `USE_OLLAMA=false` and configure:
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI LLM endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Azure OpenAI LLM deployment name
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI LLM API version
- `AZURE_OPENAI_EMBEDDING_API_KEY`: Azure OpenAI Embedding deployment key (if different from `OPENAI_API_KEY`)
- `AZURE_OPENAI_EMBEDDING_ENDPOINT`: Azure OpenAI Embedding endpoint URL
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`: Azure OpenAI embedding deployment name
- `AZURE_OPENAI_EMBEDDING_API_VERSION`: Azure OpenAI API version
- `AZURE_OPENAI_USE_MANAGED_IDENTITY`: Use Azure Managed Identities for authentication

#### General Configuration
- `SEMAPHORE_LIMIT`: Episode processing concurrency. See [Concurrency and LLM Provider 429 Rate Limit Errors](#concurrency-and-llm-provider-429-rate-limit-errors)
- `MCP_SERVER_PORT`: Port for the MCP server when using SSE transport (default: 8020)

You can set these variables in a `.env` file in the project directory. A sample configuration file (`sample_env.txt`) is provided with all available options and their default values.

## Running the Server

To run the Graphiti MCP server directly using `uv`:

```bash
uv run src/graphiti_mcp_server.py.py
```

With options:

```bash
uv run src/graphiti_mcp_server.py.py --model gpt-4.1-mini --transport sse
```

Available arguments:

- `--model`: Overrides the `MODEL_NAME` environment variable (only when not using Ollama).
- `--small-model`: Overrides the `SMALL_MODEL_NAME` environment variable (only when not using Ollama).
- `--temperature`: Overrides the `LLM_TEMPERATURE` environment variable.
- `--transport`: Choose the transport method (sse or stdio, default: sse)
- `--port`: Port to bind the MCP server to (default: 8020)
- `--group-id`: Set a namespace for the graph (optional). If not provided, defaults to "default".
- `--destroy-graph`: If set, destroys all Graphiti graphs on startup.
- `--use-custom-entities`: Enable entity extraction using the predefined ENTITY_TYPES

#### Ollama Configuration Arguments
- `--use-ollama`: Use Ollama for LLM and embeddings (default: true)
- `--ollama-base-url`: Ollama base URL (default: http://localhost:11434/v1)
- `--ollama-llm-model`: Ollama LLM model name (default: deepseek-r1:7b)
- `--ollama-embedding-model`: Ollama embedding model name (default: nomic-embed-text)
- `--ollama-embedding-dim`: Ollama embedding dimension (default: 768)

### Ollama Configuration Examples

The Graphiti MCP server provides flexible configuration options for Ollama models. Here are some common use cases:

#### Basic Configuration Examples

**Use default models:**
```bash
# With default .env configuration
uv run src/graphiti_mcp_server.py.py

# Or explicitly set in .env file:
# USE_OLLAMA=true
# OLLAMA_LLM_MODEL=deepseek-r1:7b
# OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**Use a different LLM model:**
```bash
uv run src/graphiti_mcp_server.py.py --ollama-llm-model llama3.2:3b
```

**Use a different embedding model with custom dimension:**
```bash
uv run src/graphiti_mcp_server.py.py --ollama-embedding-model all-minilm-l6-v2 --ollama-embedding-dim 384
```

**Connect to a remote Ollama server:**
```bash
uv run src/graphiti_mcp_server.py.py --ollama-base-url http://remote-server:11434/v1 --ollama-llm-model llama3.2:8b
```

#### Environment Variable Configuration

You can also configure Ollama models using environment variables in a `.env` file:

```bash
# Create or edit .env file
nano .env
```

Add the following variables to your `.env` file:

```env
# Ollama Configuration
OLLAMA_LLM_MODEL=mistral:7b
OLLAMA_EMBEDDING_MODEL=all-minilm-l6-v2
OLLAMA_EMBEDDING_DIM=384
LLM_TEMPERATURE=0.1
```

Then run the server:

```bash
uv run src/graphiti_mcp_server.py.py
```

#### Configuration Priority

The configuration system follows this priority order (highest to lowest):

1. **CLI arguments** - Override all other settings
2. **Environment variables** - Provide defaults that can be overridden by CLI
3. **Default values** - Built-in defaults for all settings

#### Available Ollama Models

**Common LLM Models:**
- `deepseek-r1:7b` (default) - Good balance of performance and quality
- `llama3.2:3b` - Fast, smaller model for development
- `llama3.2:8b` - Higher quality, larger model
- `mistral:7b` - Excellent performance for many tasks
- `codellama:7b` - Specialized for code generation
- `phi3:3.8b` - Microsoft's efficient model

**Common Embedding Models:**
- `nomic-embed-text` (default) - High-quality embeddings
- `nomic-embed-text-v2` - Improved version of the default
- `all-minilm-l6-v2` - Fast, efficient embeddings
- `all-MiniLM-L6-v2` - Alternative spelling for the same model
- `text-embedding-ada-002` - OpenAI-compatible embeddings

#### Performance Considerations

- **Smaller models** (3B parameters) are faster but may have lower quality
- **Larger models** (7B+ parameters) provide better quality but require more resources
- **Embedding dimensions** affect both performance and storage requirements
- **Remote Ollama servers** can be used for distributed deployments

### Concurrency and LLM Provider 429 Rate Limit Errors

Graphiti's ingestion pipelines are designed for high concurrency, controlled by the `SEMAPHORE_LIMIT` environment variable.
By default, `SEMAPHORE_LIMIT` is set to `10` concurrent operations to help prevent `429` rate limit errors from your LLM provider. If you encounter such errors, try lowering this value.

If your LLM provider allows higher throughput, you can increase `SEMAPHORE_LIMIT` to boost episode ingestion performance.

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

A Graphiti MCP container is available at: `zepai/knowledge-graph-mcp`. The latest build of this container is used by the Compose setup below.

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
- Exposes the server on port 8020 for HTTP-based SSE transport
- Includes a healthcheck for Neo4j to ensure it's fully operational before starting the MCP server

## Integrating with MCP Clients

### Configuration

To use the Graphiti MCP server with an MCP-compatible client, configure it to connect to the server:

> [!IMPORTANT]
> You will need the Python package manager, `uv` installed. Please refer to the [`uv` install instructions](https://docs.astral.sh/uv/getting-started/installation/).
>
> Ensure that you set the full path to the `uv` binary and your Graphiti project folder.

#### Using Ollama (Default)

**Basic Ollama configuration:**
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
        "src/graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

**Custom Ollama models via CLI arguments:**
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
        "src/graphiti_mcp_server.py",
        "--transport",
        "stdio",
        "--ollama-llm-model",
        "llama3.2:3b",
        "--ollama-embedding-model",
        "all-minilm-l6-v2",
        "--ollama-embedding-dim",
        "384"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

**Custom Ollama models via environment variables:**
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
        "src/graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OLLAMA_LLM_MODEL": "mistral:7b",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text-v2",
        "OLLAMA_EMBEDDING_DIM": "768",
        "LLM_TEMPERATURE": "0.1"
      }
    }
  }
}
```

#### Using OpenAI (Alternative)

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
        "src/graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "USE_OLLAMA": "false",
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
      "url": "http://localhost:8020/sse"
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

## Integrating with the Cursor IDE

To integrate the Graphiti MCP Server with the Cursor IDE, follow these steps:

1. Run the Graphiti MCP server using the SSE transport:

```bash
python src/graphiti_mcp_server.py --transport sse --use-custom-entities --group-id <your_group_id>
```

Hint: specify a `group_id` to namespace graph data. If you do not specify a `group_id`, the server will use "default" as the group_id.

or

```bash
docker compose up
```

2. Configure Cursor to connect to the Graphiti MCP server.

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8020/sse"
    }
  }
}
```

3. Add the Graphiti rules to Cursor's User Rules. See [cursor_rules.md](cursor_rules.md) for details.

4. Kick off an agent session in Cursor.

The integration enables AI assistants in Cursor to maintain persistent memory through Graphiti's knowledge graph
capabilities.

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
            "http://localhost:8020/sse" // Ensure this matches your Graphiti server's SSE endpoint
          ]
        }
      }
    }
    ```

    If you already have an `mcpServers` entry, add `graphiti-memory` (or your chosen name) as a new key within it.

4.  **Restart Claude Desktop** for the changes to take effect.

## Troubleshooting

### Ollama Configuration Issues

**Server won't start with Ollama:**
- Ensure Ollama is installed and running: `ollama serve`
- Check that required models are pulled: `ollama list`
- Verify Ollama server is accessible: `curl http://localhost:11434/v1/models`
- Check your `.env` file has `USE_OLLAMA=true` (default)

**Model not found errors:**
- Pull the required model: `ollama pull <model-name>`
- Check model name spelling (case-sensitive)
- Verify model is available in Ollama library

**Embedding dimension mismatch:**
- Ensure `OLLAMA_EMBEDDING_DIM` matches your embedding model's output dimension
- Common dimensions: 384 (all-minilm-l6-v2), 768 (nomic-embed-text), 1536 (nomic-embed-text-v2)

**Performance issues:**
- Try smaller models for faster response times
- Adjust `SEMAPHORE_LIMIT` for concurrency control
- Consider using remote Ollama servers for distributed workloads

### General Issues

**Neo4j connection errors:**
- Verify Neo4j is running and accessible
- Check connection credentials and URI
- Ensure Neo4j version is 5.26 or later

**MCP client connection issues:**
- Verify transport method (stdio vs sse) matches client requirements
- Check port configuration for SSE transport
- Ensure firewall allows connections on configured ports

## Requirements

- Python 3.10 or higher
- Neo4j database (version 5.26 or later required)
- **Ollama** installed and running (default) OR OpenAI API key (for LLM operations)
- MCP-compatible client

## Telemetry

The Graphiti MCP server uses the Graphiti core library, which includes anonymous telemetry collection. When you initialize the Graphiti MCP server, anonymous usage statistics are collected to help improve the framework.

### What's Collected

- Anonymous identifier and system information (OS, Python version)
- Graphiti version and configuration choices (LLM provider, database backend, embedder type)
- **No personal data, API keys, or actual graph content is ever collected**

### How to Disable

To disable telemetry in the MCP server, set the environment variable:

```bash
export GRAPHITI_TELEMETRY_ENABLED=false
```

Or add it to your `.env` file:

```
GRAPHITI_TELEMETRY_ENABLED=false
```

For complete details about what's collected and why, see the [Telemetry section in the main Graphiti README](../README.md#telemetry).

## License

This project is licensed under the same license as the parent Graphiti project.
