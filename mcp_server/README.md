# Graphiti MCP Server

This is a Model Context Protocol (MCP) server implementation for Graphiti, a dynamic, temporally aware Knowledge Graph system. The MCP server exposes Graphiti's key functionality through the Anthropic MCP protocol, allowing AI assistants like Claude to interact with Graphiti's knowledge graph capabilities.

## Features

The Graphiti MCP server exposes the following key high-level functions of Graphiti:

- **Episode Management**: Add, retrieve, and delete episodes (text, messages, or JSON data)
- **Entity Management**: Add and manage entity nodes in the knowledge graph
- **Search Capabilities**: Search for facts (edges) and node summaries using semantic and full-text search
- **Group Management**: Organize and manage groups of related data
- **Graph Maintenance**: Clear the graph and rebuild indices

## Installation

1. Ensure you have Python 3.10 or higher installed.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The server uses the following environment variables:

- `NEO4J_URI`: URI for the Neo4j database (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `password`)
- `ANTHROPIC_API_KEY`: Anthropic API key (preferred LLM provider)
- `OPENAI_API_KEY`: OpenAI API key (fallback LLM provider)
- `OPENAI_BASE_URL`: Optional base URL for OpenAI API
- `MODEL_NAME`: Optional model name to use for LLM inference

## Running the Server

To run the Graphiti MCP server:

```bash
python graphiti_mcp_server.py
```

You can specify an LLM client to use (default is Anthropic if the API key is available, otherwise OpenAI):

```bash
python graphiti_mcp_server.py --llm-client anthropic
# or
python graphiti_mcp_server.py --llm-client openai
```

You can also specify a model name:

```bash
python graphiti_mcp_server.py --model claude-3-opus-20240229
```

## Integrating with Claude Desktop

To use the Graphiti MCP server with Claude Desktop:

1. Make sure you have Claude Desktop installed and updated to the latest version.
2. Configure Claude Desktop to use the Graphiti MCP server by editing the configuration file:

```json
{
  "mcpServers": {
    "graphiti": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/graphiti_mcp_server.py"]
    }
  }
}
```

Replace `/ABSOLUTE/PATH/TO/` with the absolute path to the `graphiti_mcp_server.py` file.

## Available Tools

The Graphiti MCP server exposes the following tools:

- `add_episode`: Add an episode to the knowledge graph
- `search_facts`: Search the knowledge graph for relevant facts (edges between entities)
- `search_nodes`: Search the knowledge graph for relevant node summaries
- `add_entity_node`: Add an entity node to the knowledge graph
- `delete_entity_edge`: Delete an entity edge from the knowledge graph
- `delete_group`: Delete a group and all its associated nodes and edges
- `delete_episode`: Delete an episode from the knowledge graph
- `get_entity_edge`: Get an entity edge by its UUID
- `get_episodes`: Get the most recent episodes for a specific group
- `clear_graph`: Clear all data from the knowledge graph and rebuild indices

## Available Resources

- `graphiti/status`: Get the status of the Graphiti MCP server and Neo4j connection

## Example Usage with Claude

Once the Graphiti MCP server is running and configured with Claude Desktop, you can use it like this:

```
User: Add an episode about Kamala Harris to the knowledge graph.

Claude: I'll help you add an episode about Kamala Harris to the knowledge graph. Let me do that for you.

[Claude uses the add_episode tool to add the information]

User: Now search for information about Kamala Harris.

Claude: I'll search the knowledge graph for information about Kamala Harris.

[Claude uses the search_facts tool to find relevant facts]

User: Can you show me a summary of entities related to Kamala Harris?

Claude: I'll search for node summaries related to Kamala Harris.

[Claude uses the search_nodes tool to find relevant entity summaries]
```

See the `example_usage.py` file for a more detailed example.

## Requirements

- Python 3.10 or higher
- Neo4j database
- OpenAI API key (for LLM inference and embedding)
- MCP-compatible client (like Claude Desktop)

## License

This project is licensed under the same license as the Graphiti project.
