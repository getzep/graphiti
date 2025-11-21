# VS Code Models Integration Example

This example demonstrates how to use Graphiti with VS Code's built-in AI models and embeddings.

## Prerequisites

1. **VS Code with AI Extensions**: Make sure you have VS Code with compatible language model extensions:
   - GitHub Copilot
   - Azure OpenAI extension
   - Any other VS Code language model provider

2. **Neo4j Database**: Running Neo4j instance (can be local or remote)

3. **Python Dependencies**:
   ```bash
   pip install "graphiti-core[vscodemodels]"
   ```

## Environment Setup

Set up your environment variables:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional VS Code Configuration
VSCODE_LLM_MODEL=gpt-4o-mini
VSCODE_EMBEDDING_MODEL=embedding-001
VSCODE_EMBEDDING_DIM=1024
USE_VSCODE_MODELS=true
```

## Running the Example

```bash
python basic_usage.py
```

## What the Example Does

1. **Initializes VS Code Clients**:
   - Creates a `VSCodeClient` for language model operations
   - Creates a `VSCodeEmbedder` for embedding generation
   - Both clients automatically detect available VS Code models

2. **Creates Graphiti Instance**:
   - Connects to Neo4j database
   - Uses VS Code models for all AI operations

3. **Adds Knowledge Episodes**:
   - Adds sample data about a fictional company "TechCorp"
   - Each episode is processed and added to the knowledge graph

4. **Performs Search**:
   - Searches the knowledge graph for information about TechCorp
   - Returns relevant facts and relationships

## Expected Output

```
Adding episodes to the knowledge graph...
✓ Added episode 1
✓ Added episode 2
✓ Added episode 3
✓ Added episode 4

Searching for information about TechCorp...
Search Results:
1. John is a software engineer who works at TechCorp and specializes in Python development...
2. Sarah is the CTO at TechCorp and has been leading the engineering team for 5 years...
3. TechCorp is developing a new AI-powered application using machine learning...
4. John and Sarah collaborate on the AI project with John handling backend implementation...

Example completed successfully!
VS Code models integration is working properly.
```

## Key Features Demonstrated

- **Zero External Dependencies**: No API keys required, uses VS Code's built-in AI
- **Automatic Model Detection**: Detects available VS Code models automatically
- **Intelligent Fallbacks**: Falls back gracefully when VS Code models are unavailable
- **Semantic Search**: Performs hybrid search across the knowledge graph
- **Relationship Extraction**: Automatically extracts entities and relationships from text

## Troubleshooting

**Models not detected**: 
- Ensure VS Code language model extensions are installed and active
- Check that you're running the script within VS Code or with VS Code in your PATH

**Connection errors**:
- Verify Neo4j is running and accessible
- Check NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables

**Embedding dimension mismatch**:
- Set VSCODE_EMBEDDING_DIM to match your model's output dimension
- Default is 1024 for consistent similarity preservation