# Graphiti MCP Server

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents operating in dynamic environments. Unlike traditional retrieval-augmented generation (RAG) methods, Graphiti continuously integrates user interactions, structured and unstructured enterprise data, and external information into a coherent, queryable graph. The framework supports incremental data updates, efficient retrieval, and precise historical queries without requiring complete graph recomputation, making it suitable for developing interactive, context-aware AI applications.

This is an experimental Model Context Protocol (MCP) server implementation for Graphiti. The MCP server exposes Graphiti's key functionality through the MCP protocol, allowing AI assistants to interact with Graphiti's knowledge graph capabilities.

## Features

The Graphiti MCP server provides comprehensive knowledge graph capabilities:

- **Episode Management**: Add, retrieve, and delete episodes (text, messages, or JSON data)
- **Entity Management**: Search and manage entity nodes and relationships in the knowledge graph
- **Search Capabilities**: Search for facts (edges) and node summaries using semantic and hybrid search
- **Group Management**: Organize and manage groups of related data with group_id filtering
- **Graph Maintenance**: Clear the graph and rebuild indices
- **Graph Database Support**: Multiple backend options including FalkorDB (default) and Neo4j
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, Gemini, Groq, and Azure OpenAI
- **Multiple Embedding Providers**: Support for OpenAI, Voyage, Sentence Transformers, and Gemini embeddings
- **Rich Entity Types**: Built-in entity types including Preferences, Requirements, Procedures, Locations, Events, Organizations, Documents, and more for structured knowledge extraction
- **HTTP Transport**: Default HTTP transport with MCP endpoint at `/mcp/` for broad client compatibility
- **Queue-based Processing**: Asynchronous episode processing with configurable concurrency limits
- **Project Isolation**: Automatic project-level memory isolation with `.graphiti.json` configuration (stdio mode)
- **Smart Memory Classification**: Intelligent routing of memories to appropriate knowledge spaces

## Documentation

- **[Architecture Design](docs/design/architecture.md)** - Overall architecture and design decisions for memory classification
- **[Implementation Plan](docs/design/implementation-plan.md)** - Detailed phased implementation plan
- **[Progress Tracking](docs/progress.md)** - Current implementation status and progress

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

   ```bash
   cd graphiti && pwd
   ```

2. Install the [Graphiti prerequisites](#prerequisites).

3. Configure Claude, Cursor, or other MCP client to use [Graphiti with a `stdio` transport](#integrating-with-mcp-clients). See the client documentation on where to find their MCP configuration files.

### For Cursor and other HTTP-enabled clients

1. Change directory to the `mcp_server` directory

   ```bash
   cd graphiti/mcp_server
   ```

2. Start the combined FalkorDB + MCP server using Docker Compose (recommended)

   ```bash
   docker compose up
   ```

   This starts both FalkorDB and the MCP server in a single container.

   **Alternative**: Run with separate containers using Neo4j:

   ```bash
   docker compose -f docker/docker-compose-neo4j.yml up
   ```

3. Point your MCP client to `http://localhost:8000/mcp/`

## Installation

### Prerequisites

1. Docker and Docker Compose (for the default FalkorDB setup)
2. OpenAI API key for LLM operations (or API keys for other supported LLM providers)
3. (Optional) Python 3.10+ if running the MCP server standalone with an external FalkorDB instance

### Setup

1. Clone the repository and navigate to the mcp_server directory
2. Use `uv` to create a virtual environment and install dependencies:

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies in one step
uv sync

# Optional: Install additional LLM providers (anthropic, gemini, groq, voyage, sentence-transformers)
uv sync --extra providers
```

## Project Isolation with .graphiti.json

> **Transport Mode Support**: Project isolation with `.graphiti.json` is supported for **stdio mode only**. For HTTP mode, use the `--group-id` parameter or server configuration.

The Graphiti MCP server supports automatic project isolation using `.graphiti.json` configuration files in stdio mode. This allows different projects to have separate knowledge graphs, preventing cross-project contamination when working with multiple projects simultaneously in IDEs like Cursor or Claude Desktop.

### Creating a Project Configuration

Create a `.graphiti.json` file in your project root:

```json
{
  "group_id": "my-awesome-project",
  "description": "My project knowledge graph"
}
```

### How It Works

1. When the MCP server starts, it searches upward from the working directory for `.graphiti.json`
2. If found, it uses the `group_id` from that file for all operations
3. All knowledge added to the graph is automatically associated with that project
4. Subdirectories within the project share the same `group_id`

### MCP Client Configuration

For stdio-based clients (Claude Desktop, Cursor), pass the project directory via environment variable:

```json
{
  "mcpServers": {
    "graphiti": {
      "transport": "stdio",
      "command": "uv",
      "args": [
        "--directory", "/path/to/graphiti/mcp_server",
        "run", "python", "src/graphiti_mcp_server.py",
        "--config", "config/config-local-ollama.yaml"
      ],
      "env": {
        "GRAPHITI_PROJECT_DIR": "${workspaceFolder}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

The `${workspaceFolder}` variable is automatically expanded by the MCP client to your project directory.

### The `GRAPHITI_PROJECT_DIR` Environment Variable

The `GRAPHITI_PROJECT_DIR` environment variable tells the MCP server **where to start searching** for the `.graphiti.json` configuration file.

**What it does:**

- **Startup only**: Used only during server initialization to locate the project config
- **Not stored**: The directory path is never stored in the knowledge graph
- **Points to config**: Tells the server where to begin the upward search for `.graphiti.json`

**What gets stored:**

- The `group_id` value from `.graphiti.json` is what actually tags and isolates your memory data
- This allows projects to be renamed or moved without losing their memory associations

**Typical configuration:**

```json
"env": {
  "GRAPHITI_PROJECT_DIR": "${workspaceFolder}"
}
```

When the MCP client (Cursor/Claude Desktop) expands `${workspaceFolder}` to `/workspace/my-app`, the server will:

1. Search for `/workspace/my-app/.graphiti.json`
2. Extract the `group_id` (e.g., `"my-app"`)
3. Use that `group_id` for all memory operations

### The `OPENAI_API_KEY` Environment Variable

> **Not required when using local rerankers (RRF, MMR, etc.)**

**When is OPENAI_API_KEY required?**

`OPENAI_API_KEY` is only required when using `cross_encoder` reranker type with OpenAI provider. The default configuration uses RRF (Reciprocal Rank Fusion), which is a local algorithm that doesn't make any API calls.

**Reranker Configuration:**

| Reranker Type | OPENAI_API_KEY Required | API Calls |
|---------------|-------------------------|-----------|
| `rrf` (default) | ❌ No | Local computation only |
| `mmr` | ❌ No | Local computation only |
| `node_distance` | ❌ No | Local computation only |
| `episode_mentions` | ❌ No | Local computation only |
| `cross_encoder` (OpenAI) | ✅ Yes | Makes OpenAI API calls |
| `cross_encoder` (Gemini) | ❌ No (uses GEMINI_API_KEY) | Makes Gemini API calls |

**Component API Key Usage:**

| Component | API Key Source | Required for Default Config? |
|-----------|---------------|-------------------------------|
| LLM Client | Config file `llm.providers.openai.api_key` | Only if using OpenAI LLM |
| Embedder Client | Config file `embedder.providers.openai.api_key` | Only if using OpenAI Embedder |
| Reranker (RRF/MMR) | None | ❌ Not required |
| Reranker (CrossEncoder) | `reranker.providers.openai.api_key` or `OPENAI_API_KEY` | ✅ Required |

**Note:** When using `cross_encoder` with OpenAI provider, you must provide an OpenAI API key through one of these sources:
- Set `OPENAI_API_KEY` environment variable (used as fallback)
- Set `RERANKER_OPENAI_API_KEY` environment variable
- Set `reranker.providers.openai.api_key` directly in config.yaml

Use local reranker types (RRF, MMR) to avoid the API key requirement.

### Fallback Behavior

| Scenario | Behavior |
|----------|----------|
| `.graphiti.json` found | Uses the `group_id` from the file |
| No `.graphiti.json` | Uses server default `group_id` (from config.yaml or "main") |
| Invalid `.graphiti.json` | Logs error and uses server default `group_id` |

### Benefits

- **Automatic Isolation**: Each project gets its own knowledge graph space
- **No Manual group_id**: Don't need to specify `group_id` in every tool call
- **Git-Like Discovery**: Works from any subdirectory within your project
- **Persistent Configuration**: The `.graphiti.json` file can be committed to your project

### Example: Monorepo Setup

For a monorepo with multiple sub-projects, each sub-project can have its own configuration:

```text
/my-monorepo/
  ├── .graphiti.json          {"group_id": "monorepo-root"}
  ├── frontend/
  │   └── .graphiti.json      {"group_id": "frontend-app"}
  └── backend/
      └── .graphiti.json      {"group_id": "backend-api"}
```

When working in `frontend/src/`, the `frontend-app` configuration is used. When working in `backend/`, the `backend-api` configuration is used.

## Smart Memory Classification

> **Feature Status**: ✅ Implemented (Phases 1-8 Complete)

The Smart Memory Classification feature automatically routes memories to appropriate knowledge spaces based on their content. This allows shared knowledge (user preferences, coding conventions, team procedures) to be accessible across multiple projects while keeping project-specific knowledge isolated.

### How It Works

1. **Classification**: When you add a memory, the system analyzes its content to determine if it's:
   - **Shared**: User preferences, coding conventions, team standards (stored in shared groups)
   - **Project-Specific**: API endpoints, project architecture, implementation details (stored only in project group)
   - **Mixed**: Contains both shared and project-specific elements (split and stored separately)

2. **Smart Write**: Memories are written to appropriate groups based on classification:
   - Shared memories are written to all shared groups
   - Project-specific memories are written only to the project group
   - Mixed memories are split - shared parts go to shared groups, project parts go to project group

3. **Smart Search**: When you search, the system automatically searches across both your project group AND shared groups

### Configuration

Enable shared memory classification in your `.graphiti.json`:

```json
{
  "group_id": "my-project",
  "description": "My project with shared knowledge",
  "shared_group_ids": ["user-common", "team-standards"],
  "shared_entity_types": ["Preference", "Procedure", "Requirement"],
  "shared_patterns": ["偏好", "习惯", "convention"],
  "write_strategy": "simple"
}
```

### Configuration Fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `group_id` | string | Your project's unique identifier (required) |
| `shared_group_ids` | array | List of shared group IDs for cross-project knowledge |
| `shared_entity_types` | array | Entity types that indicate shared knowledge |
| `shared_patterns` | array | Keywords/patterns that indicate shared knowledge |
| `write_strategy` | string | Strategy name: "simple" (rule-based) or "llm_based" (LLM classifier) |

### What Gets Classified as Shared

**Automatically Shared** (when matched):

- User preferences: "User preference: dark mode"
- Coding conventions: "Convention: 4-space indentation"
- Team procedures: "Procedure: run tests before committing"
- Requirements: "Requirement: must support Python 3.10+"

**Stays Project-Specific**:

- API endpoints: "The API endpoint is at /api/v1/users"
- Implementation details: "Project uses FastAPI framework"
- Project-specific files: "Config file is in config/settings.yaml"

**Mixed Content** (split automatically):

- "User prefers dark mode. Project uses React at /api/v1/users."
- Shared part: "User prefers dark mode" → stored in shared groups
- Project part: "Project uses React at /api/v1/users" → stored in project group

### Classification Strategies

The system supports multiple classification strategies:

#### 1. Rule-Based Classifier (Default)

Fast, pattern-based classification using keyword matching and entity type detection.

**Pros:**

- Zero additional LLM cost
- Fast response time
- Predictable behavior

**Cons:**

- Limited to predefined patterns
- May miss nuanced content

**How it works:**

- Checks for shared entity types (Preference, Procedure, Requirement)
- Matches against shared patterns (keywords like "convention", "标准", "偏好")
- Falls back to PROJECT_SPECIFIC if no patterns match

#### 2. LLM-Based Classifier

Intelligent classification using language model analysis for higher accuracy.

**Pros:**

- More accurate classification
- Understands context and nuance
- Can split mixed content automatically

**Cons:**

- Additional LLM API cost per classification
- Slower response time

**How it works:**

- Uses LLM to analyze content and determine category
- For MIXED category, splits content into shared and project-specific parts
- Falls back to PROJECT_SPECIFIC on errors

**Configuration:**

```json
{
  "group_id": "my-project",
  "shared_group_ids": ["user-common"],
  "shared_entity_types": ["Preference", "Procedure", "Requirement"],
  "write_strategy": "llm_based"
}
```

### Content Splitting for Mixed Memories

When a memory is classified as **MIXED**, the LLM-based classifier can automatically split it into:

1. **Shared Part**: Knowledge useful across projects (preferences, conventions, procedures)
2. **Project Part**: Project-specific knowledge (APIs, implementation details, project files)

**Example:**

Original memory:

```text
"User prefers dark mode UI. The API endpoint is at /api/v1/users for user data."
```

After splitting:

- **Shared Part** → stored in `user-common`: "User prefers dark mode UI"
- **Project Part** → stored in `my-project`: "The API endpoint is at /api/v1/users for user data"

**Benefits:**

- Shared groups remain clean and relevant
- Project groups contain only project-specific details
- Better search results across projects

### Example Scenarios

**Scenario 1: User Preferences (Rule-Based)**

```
You: "User preference: I prefer 4-space indentation"
→ Classified as: SHARED (matches "preference" keyword)
→ Stored in: my-project + user-common + team-standards
→ Benefit: All your projects will know your preference
```

**Scenario 2: Project-Specific Knowledge (Rule-Based)**

```
You: "The API endpoint is at /api/v1/users"
→ Classified as: PROJECT_SPECIFIC (no shared patterns matched)
→ Stored in: my-project (only)
→ Benefit: Other projects won't see this project-specific detail
```

**Scenario 3: Mixed Content (LLM-Based with Splitting)**

```
You: "User prefers dark mode for all interfaces. The API is at /api/v1/users."
→ Classified as: MIXED (LLM detects both shared and project-specific content)
→ Split into:
  - Shared: "User prefers dark mode for all interfaces"
  - Project: "The API is at /api/v1/users"
→ Stored in:
  - user-common: "User prefers dark mode for all interfaces"
  - my-project: "The API is at /api/v1/users"
→ Benefit: Other projects learn your preference but not the API endpoint
```

### Benefits

- **Consistent Preferences**: Your coding style and preferences are automatically shared across projects
- **Team Knowledge**: Team standards and procedures are accessible from any project
- **Automatic Isolation**: Project-specific details stay project-specific
- **No Manual Management**: The system automatically routes memories to the right places

### Example .graphiti.json Files

**Basic Setup** (minimal):

```json
{
  "group_id": "my-app"
}
```

**With Shared Knowledge**:

```json
{
  "group_id": "my-app",
  "shared_group_ids": ["my-shared-knowledge"]
}
```

**Full Configuration (Rule-Based)**:

```json
{
  "group_id": "my-app",
  "description": "My application project",
  "shared_group_ids": ["user-common", "team-standards"],
  "shared_entity_types": ["Preference", "Procedure", "Requirement"],
  "shared_patterns": ["convention", "guideline", "标准"],
  "write_strategy": "simple"
}
```

**With LLM-Based Classification**:

```json
{
  "group_id": "my-app",
  "description": "My application with LLM-based classification",
  "shared_group_ids": ["user-common", "team-standards"],
  "shared_entity_types": ["Preference", "Procedure", "Requirement"],
  "shared_patterns": ["convention", "guideline", "标准"],
  "write_strategy": "llm_based"
}
```

See `.graphiti.json.example` for a complete example.

## Configuration

The server can be configured using a `config.yaml` file, environment variables, or command-line arguments (in order of precedence).

### Default Configuration

The MCP server comes with sensible defaults:

- **Transport**: HTTP (accessible at `http://localhost:8000/mcp/`)
- **Database**: FalkorDB (combined in single container with MCP server)
- **LLM**: OpenAI with model gpt-5-mini
- **Embedder**: OpenAI text-embedding-3-small

### Database Configuration

#### FalkorDB (Default)

FalkorDB is a Redis-based graph database that comes bundled with the MCP server in a single Docker container. This is the default and recommended setup.

```yaml
database:
  provider: "falkordb"  # Default
  providers:
    falkordb:
      uri: "redis://localhost:6379"
      password: ""  # Optional
      database: "default_db"  # Optional
```

#### Neo4j

For production use or when you need a full-featured graph database, Neo4j is recommended:

```yaml
database:
  provider: "neo4j"
  providers:
    neo4j:
      uri: "bolt://localhost:7687"
      username: "neo4j"
      password: "your_password"
      database: "neo4j"  # Optional, defaults to "neo4j"
```

#### FalkorDB

FalkorDB is another graph database option based on Redis:

```yaml
database:
  provider: "falkordb"
  providers:
    falkordb:
      uri: "redis://localhost:6379"
      password: ""  # Optional
      database: "default_db"  # Optional
```

### Configuration File (config.yaml)

The server supports multiple LLM providers (OpenAI, Anthropic, Gemini, Groq) and embedders. Edit `config.yaml` to configure:

```yaml
server:
  transport: "http"  # Default. Options: stdio, http

llm:
  provider: "openai"  # or "anthropic", "gemini", "groq", "azure_openai"
  model: "gpt-4.1"  # Default model

database:
  provider: "falkordb"  # Default. Options: "falkordb", "neo4j"
```

### Using Ollama for Local LLM

To use Ollama with the MCP server, configure it as an OpenAI-compatible endpoint:

```yaml
llm:
  provider: "openai"
  model: "gpt-oss:120b"  # or your preferred Ollama model
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"  # dummy key required

embedder:
  provider: "sentence_transformers"  # recommended for local setup
  model: "all-MiniLM-L6-v2"
```

Make sure Ollama is running locally with: `ollama serve`

### Entity Types

Graphiti MCP Server includes built-in entity types for structured knowledge extraction. These entity types are always enabled and configured via the `entity_types` section in your `config.yaml`:

**Available Entity Types:**

- **Preference**: User preferences, choices, opinions, or selections (prioritized for user-specific information)
- **Requirement**: Specific needs, features, or functionality that must be fulfilled
- **Procedure**: Standard operating procedures and sequential instructions
- **Location**: Physical or virtual places where activities occur
- **Event**: Time-bound activities, occurrences, or experiences
- **Organization**: Companies, institutions, groups, or formal entities
- **Document**: Information content in various forms (books, articles, reports, videos, etc.)
- **Topic**: Subject of conversation, interest, or knowledge domain (used as a fallback)
- **Object**: Physical items, tools, devices, or possessions (used as a fallback)

These entity types are defined in `config.yaml` and can be customized by modifying the descriptions:

```yaml
graphiti:
  entity_types:
    - name: "Preference"
      description: "User preferences, choices, opinions, or selections"
    - name: "Requirement"
      description: "Specific needs, features, or functionality"
    # ... additional entity types
```

The MCP server automatically uses these entity types during episode ingestion to extract and structure information from conversations and documents.

### Environment Variables

The `config.yaml` file supports environment variable expansion using `${VAR_NAME}` or `${VAR_NAME:default}` syntax. Key variables:

- `NEO4J_URI`: URI for the Neo4j database (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `demodemo`)
- `OPENAI_API_KEY`: OpenAI API key (required for OpenAI LLM/embedder)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Claude models)
- `GOOGLE_API_KEY`: Google API key (for Gemini models)
- `GROQ_API_KEY`: Groq API key (for Groq models)
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT`: Azure OpenAI deployment name
- `AZURE_OPENAI_EMBEDDINGS_ENDPOINT`: Optional Azure OpenAI embeddings endpoint URL
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`: Optional Azure OpenAI embeddings deployment name
- `AZURE_OPENAI_API_VERSION`: Optional Azure OpenAI API version
- `USE_AZURE_AD`: Optional use Azure Managed Identities for authentication
- `SEMAPHORE_LIMIT`: Episode processing concurrency. See [Concurrency and LLM Provider 429 Rate Limit Errors](#concurrency-and-llm-provider-429-rate-limit-errors)

You can set these variables in a `.env` file in the project directory.

## Running the Server

### Default Setup (FalkorDB Combined Container)

To run the Graphiti MCP server with the default FalkorDB setup:

```bash
docker compose up
```

This starts a single container with:

- HTTP transport on `http://localhost:8000/mcp/`
- FalkorDB graph database on `localhost:6379`
- FalkorDB web UI on `http://localhost:3000`
- OpenAI LLM with gpt-5-mini model

### Running with Neo4j

#### Option 1: Using Docker Compose

The easiest way to run with Neo4j is using the provided Docker Compose configuration:

```bash
# This starts both Neo4j and the MCP server
docker compose -f docker/docker-compose.neo4j.yaml up
```

#### Option 2: Direct Execution with Existing Neo4j

If you have Neo4j already running:

```bash
# Set environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Run with Neo4j
uv run graphiti_mcp_server.py --database-provider neo4j
```

Or use the Neo4j configuration file:

```bash
uv run graphiti_mcp_server.py --config config/config-docker-neo4j.yaml
```

### Running with FalkorDB

#### Option 1: Using Docker Compose

```bash
# This starts both FalkorDB (Redis-based) and the MCP server
docker compose -f docker/docker-compose.falkordb.yaml up
```

#### Option 2: Direct Execution with Existing FalkorDB

```bash
# Set environment variables
export FALKORDB_URI="redis://localhost:6379"
export FALKORDB_PASSWORD=""  # If password protected

# Run with FalkorDB
uv run graphiti_mcp_server.py --database-provider falkordb
```

Or use the FalkorDB configuration file:

```bash
uv run graphiti_mcp_server.py --config config/config-docker-falkordb.yaml
```

### Available Command-Line Arguments

- `--config`: Path to YAML configuration file (default: config.yaml)
- `--llm-provider`: LLM provider to use (openai, anthropic, gemini, groq, azure_openai)
- `--embedder-provider`: Embedder provider to use (openai, azure_openai, gemini, voyage)
- `--database-provider`: Database provider to use (falkordb, neo4j) - default: falkordb
- `--model`: Model name to use with the LLM client
- `--temperature`: Temperature setting for the LLM (0.0-2.0)
- `--transport`: Choose the transport method (http or stdio, default: http)
- `--group-id`: Set a namespace for the graph (optional). If not provided, defaults to "main"
- `--destroy-graph`: If set, destroys all Graphiti graphs on startup

### Concurrency and LLM Provider 429 Rate Limit Errors

Graphiti's ingestion pipelines are designed for high concurrency, controlled by the `SEMAPHORE_LIMIT` environment variable. This setting determines how many episodes can be processed simultaneously. Since each episode involves multiple LLM calls (entity extraction, deduplication, summarization), the actual number of concurrent LLM requests will be several times higher.

**Default:** `SEMAPHORE_LIMIT=10` (suitable for OpenAI Tier 3, mid-tier Anthropic)

#### Tuning Guidelines by LLM Provider

**OpenAI:**

- Tier 1 (free): 3 RPM → `SEMAPHORE_LIMIT=1-2`
- Tier 2: 60 RPM → `SEMAPHORE_LIMIT=5-8`
- Tier 3: 500 RPM → `SEMAPHORE_LIMIT=10-15`
- Tier 4: 5,000 RPM → `SEMAPHORE_LIMIT=20-50`

**Anthropic:**

- Default tier: 50 RPM → `SEMAPHORE_LIMIT=5-8`
- High tier: 1,000 RPM → `SEMAPHORE_LIMIT=15-30`

**Azure OpenAI:**

- Consult your quota in Azure Portal and adjust accordingly
- Start conservative and increase gradually

**Ollama (local):**

- Hardware dependent → `SEMAPHORE_LIMIT=1-5`
- Monitor CPU/GPU usage and adjust

#### Symptoms

- **Too high**: 429 rate limit errors, increased API costs from parallel processing
- **Too low**: Slow episode throughput, underutilized API quota

#### Monitoring

- Watch logs for `429` rate limit errors
- Monitor episode processing times in server logs
- Check your LLM provider's dashboard for actual request rates
- Track token usage and costs

Set this in your `.env` file:

```bash
SEMAPHORE_LIMIT=10  # Adjust based on your LLM provider tier
```

### Docker Deployment

The Graphiti MCP server can be deployed using Docker with your choice of database backend. The Dockerfile uses `uv` for package management, ensuring consistent dependency installation.

A pre-built Graphiti MCP container is available at: `zepai/knowledge-graph-mcp`

#### Environment Configuration

Before running Docker Compose, configure your API keys using a `.env` file (recommended):

1. **Create a .env file in the mcp_server directory**:

   ```bash
   cd graphiti/mcp_server
   cp .env.example .env
   ```

2. **Edit the .env file** to set your API keys:

   ```bash
   # Required - at least one LLM provider API key
   OPENAI_API_KEY=your_openai_api_key_here

   # Optional - other LLM providers
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key

   # Optional - embedder providers
   VOYAGE_API_KEY=your_voyage_key
   ```

**Important**: The `.env` file must be in the `mcp_server/` directory (the parent of the `docker/` subdirectory).

#### Running with Docker Compose

**All commands must be run from the `mcp_server` directory** to ensure the `.env` file is loaded correctly:

```bash
cd graphiti/mcp_server
```

##### Option 1: FalkorDB Combined Container (Default)

Single container with both FalkorDB and MCP server - simplest option:

```bash
docker compose up
```

##### Option 2: Neo4j Database

Separate containers with Neo4j and MCP server:

```bash
docker compose -f docker/docker-compose-neo4j.yml up
```

Default Neo4j credentials:

- Username: `neo4j`
- Password: `demodemo`
- Bolt URI: `bolt://neo4j:7687`
- Browser UI: `http://localhost:7474`

##### Option 3: FalkorDB with Separate Containers

Alternative setup with separate FalkorDB and MCP server containers:

```bash
docker compose -f docker/docker-compose-falkordb.yml up
```

FalkorDB configuration:

- Redis port: `6379`
- Web UI: `http://localhost:3000`
- Connection: `redis://falkordb:6379`

#### Accessing the MCP Server

Once running, the MCP server is available at:

- **HTTP endpoint**: `http://localhost:8000/mcp/`
- **Health check**: `http://localhost:8000/health`

#### Running Docker Compose from a Different Directory

If you run Docker Compose from the `docker/` subdirectory instead of `mcp_server/`, you'll need to modify the `.env` file path in the compose file:

```yaml
# Change this line in the docker-compose file:
env_file:
  - path: ../.env    # When running from mcp_server/

# To this:
env_file:
  - path: .env       # When running from mcp_server/docker/
```

However, **running from the `mcp_server/` directory is recommended** to avoid confusion.

## Integrating with MCP Clients

### VS Code / GitHub Copilot

VS Code with GitHub Copilot Chat extension supports MCP servers. Add to your VS Code settings (`.vscode/mcp.json` or global settings):

```json
{
  "mcpServers": {
    "graphiti": {
      "uri": "http://localhost:8000/mcp/",
      "transport": {
        "type": "http"
      }
    }
  }
}
```

### Other MCP Clients

To use the Graphiti MCP server with other MCP-compatible clients, configure it to connect to the server:

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

For HTTP transport (default), you can use this configuration:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "http",
      "url": "http://localhost:8000/mcp/"
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

```text
add_episode(
name="Customer Profile",
episode_body="{\"company\": {\"name\": \"Acme Technologies\"}, \"products\": [{\"id\": \"P001\", \"name\": \"CloudSync\"}, {\"id\": \"P002\", \"name\": \"DataMiner\"}]}",
source="json",
source_description="CRM data"
)

```

## Integrating with the Cursor IDE

To integrate the Graphiti MCP Server with the Cursor IDE, follow these steps:

1. Run the Graphiti MCP server using the default HTTP transport:

   ```bash
   uv run graphiti_mcp_server.py --group-id <your_group_id>
   ```

   Hint: specify a `group_id` to namespace graph data. If you do not    specify a `group_id`, the server will use "main" as the group_id.

   or

   ```bash
   docker compose up
   ```

2. Configure Cursor to connect to the Graphiti MCP server.

   ```json
   {
     "mcpServers": {
       "graphiti-memory": {
         "url": "http://localhost:8000/mcp/"
       }
     }
   }
   ```

3. Add the Graphiti rules to Cursor's User Rules. See [cursor_rules.md](cursor_rules.md) for details.

4. Kick off an agent session in Cursor.

The integration enables AI assistants in Cursor to maintain persistent memory through Graphiti's knowledge graph
capabilities.

## Integrating with Claude Desktop (Docker MCP Server)

The Graphiti MCP Server uses HTTP transport (at endpoint `/mcp/`). Claude Desktop does not natively support HTTP transport, so you'll need to use a gateway like `mcp-remote`.

1. **Run the Graphiti MCP server**:

    ```bash
    docker compose up
    # Or run directly with uv:
    uv run graphiti_mcp_server.py
    ```

2. **(Optional) Install `mcp-remote` globally**:
    If you prefer to have `mcp-remote` installed globally, or if you encounter issues with `npx` fetching the package, you can install it globally. Otherwise, `npx` (used in the next step) will handle it for you.

    ```bash
    npm install -g mcp-remote
    ```

3. **Configure Claude Desktop**:
    Open your Claude Desktop configuration file (usually `claude_desktop_config.json`) and add or modify the `mcpServers` section as follows:

    ```json
    {
      "mcpServers": {
        "graphiti-memory": {
          // You can choose a different name if you prefer
          "command": "npx", // Or the full path to mcp-remote if npx is not in your PATH
          "args": [
            "mcp-remote",
            "http://localhost:8000/mcp/" // The Graphiti server's HTTP endpoint
          ]
        }
      }
    }
    ```

    If you already have an `mcpServers` entry, add `graphiti-memory` (or your chosen name) as a new key within it.

4. **Restart Claude Desktop** for the changes to take effect.

## Requirements

- Python 3.10 or higher
- OpenAI API key (for LLM operations and embeddings) or other LLM provider API keys
- MCP-compatible client
- Docker and Docker Compose (for the default FalkorDB combined container)
- (Optional) Neo4j database (version 5.26 or later) if not using the default FalkorDB setup

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

```text
GRAPHITI_TELEMETRY_ENABLED=false
```

For complete details about what's collected and why, see the [Telemetry section in the main Graphiti README](../README.md#telemetry).

## License

This project is licensed under the same license as the parent Graphiti project.
