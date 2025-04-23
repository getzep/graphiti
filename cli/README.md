# Graphiti Core CLI

This CLI provides supplementary direct access to Graphiti Core functions, primarily focused on ingesting data and bypassing the MCP server layer. It is not intended to replace the MCP server as the primary means of interaction with Graphiti, but rather offers an alternative approach for certain workflows, testing, and diagnostics.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Standard Installation](#standard-installation)
  - [Development Setup](#development-setup)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Development & Packaging (Advanced)](#development--packaging-advanced)
- [Troubleshooting / FAQ](#troubleshooting--faq)

## Quick Start

1. Install the package and CLI:

   ```bash
   pip install graphiti-core
   ```

2. Set up your environment by creating a `.env` file in your project root or setting environment variables.
3. Test the connection:

   ```bash
   graphiti check-connection
   ```

   If successful, you should see output indicating a connection to Neo4j.

## Installation

### Standard Installation

For regular users who want to use the CLI, installation is simple:

```bash
# Install the package from PyPI
pip install graphiti-core
```

This will make the `graphiti` command available globally. Ensure you have Python 3.10+ installed.

Required environment variables:

- `NEO4J_URI` (Defaults to `bolt://localhost:7687`)
- `NEO4J_USER` (Defaults to `neo4j`)
- `NEO4J_PASSWORD` (Defaults to `demodemo`)
- `OPENAI_API_KEY` (Optional, needed for embeddings if adding text later)
- `MODEL_NAME` (Optional, defaults to `gpt-3.5-turbo` if `OPENAI_API_KEY` is set)

### Development Setup

If you're developing the CLI or contributing to it, you'll need a more detailed setup:

1. **Clone the Repository and Set Up Environment:**

   ```bash
   # Clone the repository
   git clone https://github.com/getzep/graphiti.git
   cd graphiti
   
   # Create a virtual environment with Python 3.10+
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Install in development mode
   pip install -e .
   ```

2. **Optional: Create a Command Script for Testing:**

   ```bash
   # Create a bin directory (if needed)
   mkdir -p ~/bin
   
   # Create the script
   cat > ~/bin/graphiti << 'EOF'
   #!/bin/bash
   # Get the directory of the script and the project root
   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
   PROJECT_ROOT="/path/to/your/graphiti/clone"
   
   # Execute the CLI
   cd "$PROJECT_ROOT" && python -m cli.main "$@"
   EOF
   
   # Make the script executable
   chmod +x ~/bin/graphiti
   
   # Ensure ~/bin is in your PATH
   echo 'export PATH=$HOME/bin:$PATH' >> ~/.zshrc
   source ~/.zshrc
   ```

   Make sure to replace `/path/to/your/graphiti/clone` with your actual repository path.

## Usage

Once the `graphiti` command script is set up, you can use it from anywhere:

```bash
graphiti <command> [options]
```

### Available Commands

- `help`: Show help and list all commands.
- `check-connection`: Test connection to Neo4j using environment variables.
- `add-json`: Add JSON data from a file as an episode.
  - `--json-file` / `-f` (Required): Path to the JSON file.
  - `--name` / `-n` (Required): Name for the episode.
  - `--desc` / `-d`: Description of the data source.
  - `--group-id` / `-g`: Group ID for the graph (overrides automatic project-based ID).
  - `--workspace`: Workspace path to use for automatic group ID generation (defaults to `CURSOR_WORKSPACE` env var or current dir).
  - `--uuid`: Specific UUID for the episode.
- `add-json-string`: Add JSON data directly as a string.
  - `--json-data` / `-j` (Required): JSON string to ingest (must be valid JSON, use single quotes in shell).
  - `--name` / `-n` (Required): Name for the episode.
  - `--desc` / `-s`: Description of the data source.
  - `--group-id` / `-g`: Group ID for the graph (overrides automatic project-based ID).
  - `--workspace`: Workspace path to use for automatic group ID generation (defaults to `CURSOR_WORKSPACE` env var or current dir).
  - `--uuid`: Specific UUID for the episode.
- `search-nodes`: Search for nodes (entities) by properties or keywords.
  - `--query` (Required): The search term.
  - `--group-id`: Filter search by group ID.
- `search-facts`: Search for facts (relationships) by properties or keywords.
  - `--query` (Required): The search term.
  - `--group-id`: Filter search by group ID.
- `get-entity-edge`: Get detailed information about a specific relationship (edge).
  - `--edge-uuid` (Required): The UUID of the edge.
- `get-episodes`: List recent episodes.
  - `--limit`: Number of episodes to list (default: 10).
  - `--group-id`: Filter episodes by group ID.
- `delete-entity-edge`: Delete a specific relationship (edge).
  - `--edge-uuid` (Required): The UUID of the edge to delete.
- `delete-episode`: Delete an episode and its associated data.
  - `--episode-uuid` (Required): The UUID of the episode to delete.
- `clear-graph`: **DANGER:** Clear ALL data from the graph (requires confirmation).
  - `--group-id`: Only clear data for a specific group ID. If omitted, clears the entire graph.
  - `--yes`: Bypass confirmation prompt (Use with extreme caution!).

### Examples

**Check Connection:**

```bash
graphiti check-connection
```

```txt
# Example Success Output:
INFO:cli.main:Running check-connection command...
INFO:cli.main:Checking Neo4j connection...
INFO:cli.main:Attempting connection to bolt://localhost:7687...
INFO:cli.main:Successfully connected to Neo4j at bolt://localhost:7687
INFO:cli.main:Neo4j driver connection closed.
```

**Add JSON from File:**

```bash
# Assumes data/customer.json exists relative to where you run the command
graphiti add-json --json-file ./data/customer.json --name "Customer Data Q3" --desc "Sales CRM Data"
```

**Add JSON as String:**

```bash
graphiti add-json-string --json-data '{"company": "Acme Corp", "revenue": 1500000}' --name "Acme Financials" --desc "Q3 Financial Data"
```

**Search for Nodes:**

```bash
graphiti search-nodes --query "Acme Corp"
```

**Search for Facts:**

```bash
graphiti search-facts --query "revenue"
```

**List Episodes:**

```bash
graphiti get-episodes --limit 5
```

**Delete an Episode:**

```bash
# First, find the episode UUID using get-episodes
# graphiti get-episodes
# Then, delete it
graphiti delete-episode --episode-uuid "paste-uuid-here"
```

**Clear Graph (Use Caution!):**

```bash
# Clear data only for the automatically detected project group
graphiti clear-graph

# Clear data for a specific group ID (requires confirmation)
graphiti clear-graph --group-id "custom_group"

# Clear ENTIRE graph (requires confirmation)
graphiti clear-graph

# Clear ENTIRE graph WITHOUT confirmation (VERY DANGEROUS)
# graphiti clear-graph --yes
```

### Project-Based Group IDs

By default, the CLI commands automatically generate a consistent `group_id` based on the workspace path to isolate knowledge graphs by project. This matches the behavior of the MCP server in Cursor. The detection order is:

1. Path provided via the `--workspace` parameter.
2. `CURSOR_WORKSPACE` environment variable.
3. Current working directory.

The `group_id` is generated by taking an MD5 hash of the path and using the first 8 characters in the format `cursor_<hash>`.

**Example with automatic `group_id`:**

```bash
# The group_id will be automatically generated based on the current directory
graphiti add-json --json-file ./data/project_data.json --name "Project Data"

# Explicitly set the workspace path for group_id generation
graphiti add-json-string --json-data '{"key": "value"}' --name "Example" --workspace "/path/to/your/project"
```

**Example overriding the automatic `group_id`:**

```bash
# Explicitly provide a custom group_id
graphiti add-json --json-file ./data/other_data.json --name "Other Data" --group-id "custom_group"
```

## Technical Details

The `graphiti` command script:

- Is a command-line entry point installed by the Python package
- When installed via pip, it's available globally 
- When installed for development, you can create a custom script as shown in the Development Setup section

## Testing

The CLI has both automated tests and an interactive test script.

### Automated Tests

CLI tests are located in the project's test suite structure. You can run:

```bash
# Run all project tests including CLI tests (from project root)
python -m pytest

# Run only CLI-specific tests
python -m pytest tests/cli/
```

Both approaches will execute the automated test suite for the CLI functionality, including file-based and string-based JSON ingestion.

### Interactive CLI Test Script

The repository includes an interactive bash script (`cli/test_commands.sh`) that tests all CLI commands with user guidance. This script:

- Tests connection to Neo4j
- Tests adding JSON from file and string
- Tests search functionality for nodes and relationships
- Tests retrieving edge and episode information
- Tests deletion operations (with safeguards)
- Tests the clear-graph interface without actual deletion

To run the interactive test script:

```bash
# Make the script executable (first time only)
chmod +x cli/test_commands.sh

# Run the test script from the project root
./cli/test_commands.sh
```

The script will:

1. Guide you through each test with prompts
2. Create clearly labeled test data (prefixed with `TEST_SAFE_TO_DELETE`)
3. Ask for confirmation before any destructive operations
4. Clean up test files after completion

Note: You may need to edit the script to update the `GRAPHITI_DIR` variable to match your installation path.

## Development & Packaging (Advanced)

These sections are primarily for developers contributing to the CLI itself.

### Installation for Development

You can install the CLI package in development mode along with development dependencies:

```bash
# From the project root directory (with virtual environment activated):
uv pip install -e ".[dev]"
```

This uses the `[project.optional-dependencies]` defined in the root `pyproject.toml`.

### Packaging

To build the CLI as a standalone package (less common for this internal tool):

```bash
# From the project root directory:
python -m build
```

This will create wheel and source distributions in the `dist/` directory, based on the root `pyproject.toml`.

## Troubleshooting / FAQ

**Q:** The `graphiti` command is not found or not working.
**A:** Ensure that the package was installed correctly using `pip install graphiti-core`. If you're using a custom script, make sure it's in a directory listed in your PATH and has execute permissions.

**Q:** I get connection errors when running `graphiti check-connection`.
**A:** Double-check that your `.env` file (or your environment variables) includes the correct settings for `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.

**Q:** I'm experiencing issues with the virtual environment or package installations.
**A:** Make sure you're using Python 3.10+ and that your virtual environment is activated (if using one) before running commands. Verify that all required packages are installed.

If you encounter additional issues, please refer to the full project documentation or reach out to the maintainers for help.

## Command Help

Each command has detailed help available. To see options for a specific command:

```bash
# Get general help
graphiti help

# Get help for a specific command
graphiti add-json --help
graphiti search-nodes --help
```

## CLI Architecture

The CLI is built using [Typer](https://typer.tiangolo.com/) and is organized with a clean command structure in `main.py`. Commands are grouped by functionality:

- Core connection commands
- Data ingestion commands
- Search and retrieval commands
- Management and deletion commands

### CLI Tests

The automated tests for the CLI are organized in the `tests/cli/` directory and include:

- Unit tests for individual commands
- Integration tests for data ingestion
- Mock tests for Neo4j interactions

### Extending the CLI

To add new commands to the CLI:

1. Identify the appropriate command group in `main.py`
2. Add your command function with appropriate Typer decorators
3. Implement your command logic, leveraging existing utility functions
4. Add tests for your command
5. Update the documentation in this README
