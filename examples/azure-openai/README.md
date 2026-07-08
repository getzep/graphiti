# Azure OpenAI with Neo4j Example

This example demonstrates how to use Graphiti with Azure OpenAI and Neo4j to build a knowledge graph.

## Prerequisites

- Python 3.10+
- Neo4j database (running locally or remotely)
- Azure OpenAI subscription with deployed models

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and fill in your credentials:

```bash
cd examples/azure-openai
cp .env.example .env
```

Edit `.env` with your actual values:

```env
# Neo4j connection settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 3. Azure OpenAI Model Deployments

This example requires two Azure OpenAI model deployments:

1. **Chat Completion Model**: Used for entity extraction and relationship analysis
   - Set the deployment name in `AZURE_OPENAI_DEPLOYMENT`

2. **Embedding Model**: Used for semantic search
   - Set the deployment name in `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

### 4. Neo4j Setup

Make sure Neo4j is running and accessible at the URI specified in your `.env` file.

For local development:
- Download and install [Neo4j Desktop](https://neo4j.com/download/)
- Create a new database
- Start the database
- Use the credentials in your `.env` file

## Running the Example

```bash
cd examples/azure-openai
uv run azure_openai_neo4j.py
```

## What This Example Does

1. **Initialization**: Sets up connections to Neo4j and Azure OpenAI
2. **Adding Episodes**: Ingests text and JSON data about California politics
3. **Basic Search**: Performs hybrid search combining semantic similarity and BM25 retrieval
4. **Center Node Search**: Reranks results based on graph distance to a specific node
5. **Cleanup**: Properly closes database connections

## Key Concepts

### Azure OpenAI Integration

The example shows how to configure Graphiti to use Azure OpenAI with the OpenAI v1 API:

```python
# Initialize Azure OpenAI client using the standard OpenAI client
# with Azure's v1 API endpoint
azure_client = AsyncOpenAI(
    base_url=f"{azure_endpoint}/openai/v1/",
    api_key=azure_api_key,
)

# Create LLM and Embedder clients
llm_client = AzureOpenAILLMClient(
    azure_client=azure_client,
    config=LLMConfig(model=azure_deployment, small_model=azure_deployment)
)
embedder_client = AzureOpenAIEmbedderClient(
    azure_client=azure_client,
    model=azure_embedding_deployment
)

# Initialize Graphiti with custom clients
graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=llm_client,
    embedder=embedder_client,
)
```

**Note**: This example uses Azure OpenAI's v1 API compatibility layer, which allows using the standard `AsyncOpenAI` client. The endpoint format is `https://your-resource-name.openai.azure.com/openai/v1/`.

### Episodes

Episodes are the primary units of information in Graphiti. They can be:
- **Text**: Raw text content (e.g., transcripts, documents)
- **JSON**: Structured data with key-value pairs

### Hybrid Search

Graphiti combines multiple search strategies:
- **Semantic Search**: Uses embeddings to find semantically similar content
- **BM25**: Keyword-based text retrieval
- **Graph Traversal**: Leverages relationships between entities

## Troubleshooting

### Azure OpenAI API Errors

- Verify your endpoint URL is correct (should end in `.openai.azure.com`)
- Check that your API key is valid
- Ensure your deployment names match actual deployments in Azure
- Verify API version is supported by your deployment

### Neo4j Connection Issues

- Ensure Neo4j is running
- Check firewall settings
- Verify credentials are correct
- Check URI format (should be `bolt://` or `neo4j://`)

## Next Steps

- Explore other search recipes in `graphiti_core/search/search_config_recipes.py`
- Try different episode types and content
- Experiment with custom entity definitions
- Add more episodes to build a larger knowledge graph

## Related Examples

- `examples/quickstart/` - Basic Graphiti usage with OpenAI
- `examples/podcast/` - Processing longer content
- `examples/ecommerce/` - Domain-specific knowledge graphs
