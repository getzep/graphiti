# Graphiti CLI Quick Reference for Agents

## Basic Command Pattern

```bash
graphiti [command] [options]
```

## Essential Commands

### Add Data

```bash
# Add JSON data from a file
graphiti add-json --json-file /path/to/file.json --name "Entry Name" [--desc "Description"]

# Add JSON directly as a string
graphiti add-json-string --json-data '{"key":"value"}' --name "Entry Name" [--desc "Description"]
```

### Search

```bash
# Search for nodes (entities)
graphiti search-nodes --query "search term" [--group-id "group-id"]

# Search for facts (relationships)
graphiti search-facts --query "search term" [--group-id "group-id"]

# Get detailed information about a relationship
graphiti get-entity-edge --uuid "uuid-from-search"
```

### Other Operations

```bash
# List recent episodes
graphiti get-episodes --last 10 [--group-id "group-id"]

# Delete episode
graphiti delete-episode --uuid "episode-uuid" --confirm

# Delete entity edge (relationship)
graphiti delete-entity-edge --uuid "edge-uuid" --confirm

# Clear graph data (USE WITH CAUTION)
graphiti clear-graph --confirm --force
```

### Check Connection

```bash
# Test Neo4j connection
graphiti check-connection
```
