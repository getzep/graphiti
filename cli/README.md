# Graphiti Core CLI

This CLI provides direct access to certain Graphiti Core functions, primarily focused on ingesting data, bypassing the MCP server layer.

## Prerequisites

1. **Dependencies Installed:** Ensure you have run `uv sync` in the project root directory (`graphiti`) to install `typer` and other necessary packages listed in the root `pyproject.toml`.
2. **Environment Variables:** The CLI reads Neo4j connection details and the OpenAI API key from environment variables or a `.env` file located in the project root (`graphiti/.env`). Ensure the following are set:
    * `NEO4J_URI` (Defaults to `bolt://localhost:7687`)
    * `NEO4J_USER` (Defaults to `neo4j`)
    * `NEO4J_PASSWORD` (Defaults to `demodemo`)
    * `OPENAI_API_KEY` (Optional, needed for embeddings if adding text later)
    * `MODEL_NAME` (Optional, defaults to `gpt-3.5-turbo` if `OPENAI_API_KEY` is set)
3. **(Recommended) Shell Alias:** For convenient use from any directory, add the following alias to your `~/.zshrc` (or equivalent shell config):

    ```bash
    alias graphiti-json='(cd /path/to/your/graphiti/repo && uv run cli/main.py "$@")'
    ```

    Replace `/path/to/your/graphiti/repo` with the actual absolute path (`/Users/dmieloch/mcp-servers/graphiti`). Remember to `source ~/.zshrc` after adding it.

## Simplified Command Script (Recommended)

We've created a simplified command script that uses Python 3.10+ as required by Graphiti. This is the recommended approach for using the CLI.

### Setup

1. Create a Python 3.10+ virtual environment:
   ```bash
   python3.10 -m venv .venv-py310
   source .venv-py310/bin/activate
   pip install graphiti-core typer
   ```

2. Copy the script to your bin directory:
   ```bash
   mkdir -p ~/bin
   cp cli/graphiti-cmd.sh ~/bin/graphiti
   chmod +x ~/bin/graphiti
   ```

3. Add ~/bin to your PATH:
   ```bash
   echo 'export PATH=$HOME/bin:$PATH' >> ~/.zshrc
   source ~/.zshrc
   ```

### Usage

```bash
# Show help
graphiti help

# Check connection
graphiti check-connection

# Search for nodes
graphiti search-nodes --query "your search query" 

# Add JSON from a file
graphiti add-json --json-file path/to/data.json --name "Episode Name"
```

For a complete list of commands, refer to `cli/GRAPHITI-COMMANDS.md`.

## Usage

### Checking Connection

Test if the CLI can connect to your Neo4j instance using the configured credentials.

**Command:**

```bash
# Using the new script:
graphiti check-connection

# Using alias:
graphiti-json check-connection

# Without alias (run from project root):
uv run cli/main.py check-connection
```

**Example Success Output:**

```
INFO:cli.main:Running check-connection command...
INFO:cli.main:Checking Neo4j connection...
INFO:cli.main:Attempting connection to bolt://localhost:7687...
INFO:cli.main:Successfully connected to Neo4j at bolt://localhost:7687
INFO:cli.main:Neo4j driver connection closed.
```

### Adding JSON from a File

Add JSON data from a file to Graphiti as an episode.

**Command:**

```bash
# Using the new script:
graphiti add-json --json-file path/to/data.json --name "Episode Name" [OPTIONS]

# Using alias:
graphiti-json add-json --json-file path/to/data.json --name "Episode Name" [OPTIONS]

# Without alias (run from project root):
uv run cli/main.py add-json --json-file path/to/data.json --name "Episode Name" [OPTIONS]
```

**Options:**
- `--json-file`, `-f`: Path to the JSON file (required)
- `--name`, `-n`: Name for the episode (required)
- `--desc`, `-d`: Description of the data source (optional)
- `--group-id`, `-g`: Group ID for the graph (optional)
- `--uuid`: UUID for the episode (optional)

**Example:**

```bash
graphiti add-json --json-file ./data/customer.json --name "Customer Data" --desc "Sales CRM Data" --group-id "sales-data"
```

### Adding JSON Directly as a String

Add JSON data directly as a string to Graphiti as an episode.

**Command:**

```bash
# Using alias:
graphiti-json add-json-string --json-data '{"key": "value"}' --name "Episode Name" [OPTIONS]

# Without alias (run from project root):
uv run cli/main.py add-json-string --json-data '{"key": "value"}' --name "Episode Name" [OPTIONS]
```

**Options:**
- `--json-data`, `-d`: JSON string to ingest (required, must be valid JSON)
- `--name`, `-n`: Name for the episode (required)
- `--desc`, `-s`: Description of the data source (optional)
- `--group-id`, `-g`: Group ID for the graph (optional)
- `--uuid`: UUID for the episode (optional)

**Example:**

```bash
graphiti-json add-json-string --json-data '{"company": "Acme", "revenue": 1000000}' --name "Company Data" --desc "Financial Data" --group-id "finance"
```

**Note:** When using the `add-json-string` command, ensure that your JSON string is properly quoted for your shell. In bash/zsh, single quotes are recommended to prevent special character interpretation.

### Project-Based Group IDs

By default, both CLI commands will now automatically generate a consistent group_id based on the workspace path:

1. The CLI first looks for a path provided via the `--workspace` parameter
2. If not found, it checks for the `CURSOR_WORKSPACE` environment variable 
3. If neither is available, it uses the current working directory

The group_id is generated by taking an MD5 hash of the path and using the first 8 characters in the format `cursor_<hash>`. This ensures knowledge is properly isolated by project and matches the same behavior used by the MCP server in Cursor.

**Example with automatic group_id:**

```bash
# The group_id will be automatically generated based on the current directory
graphiti-json add-json --json-file ./data/customer.json --name "Customer Data"

# Explicitly set the workspace path for group_id generation
graphiti-json add-json-string --json-data '{"key": "value"}' --name "Example" --workspace "/path/to/your/project"
```

**Example overriding the automatic group_id:**

```bash
# Explicitly provide a custom group_id (overrides automatic generation)
graphiti-json add-json --json-file ./data/customer.json --name "Customer Data" --group-id "custom_group"
```

## Testing

To run tests for the CLI:

```bash
# From the project root directory:
python -m pytest cli/tests/
```

This will run all tests for the CLI functionality including both file-based and direct string-based JSON ingestion.

## Installation Options

The CLI provides multiple installation options to fit different workflows:

### Option 1: Using pip with requirements.txt

```bash
# Install dependencies from the requirements file (from the cli directory)
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
# Install the CLI package in development mode with dev dependencies
# From the cli directory:
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

### Option 3: Using pyproject.toml with modern tools

```bash
# From the cli directory:
pip install .

# Or with uv
uv pip install .

# With development dependencies:
pip install ".[dev]"
```

## Packaging

To build the CLI as a standalone package:

```bash
# From the cli directory:
python -m build
```

This will create both wheel and source distributions in the `dist/` directory.