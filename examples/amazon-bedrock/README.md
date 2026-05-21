# Amazon Bedrock + Neo4j Example

This example demonstrates how to use Graphiti with Amazon Bedrock for LLM inference, embeddings, and reranking, combined with Neo4j as the graph database.

## Features Demonstrated

- **Amazon Bedrock LLM Client**: Uses Claude models for text generation and structured output
- **Amazon Bedrock Embedder**: Uses Titan embedding models for semantic search
- **Amazon Bedrock Reranker**: Uses Cohere or Amazon rerank models for result reranking
- **Neo4j Integration**: Stores and queries the knowledge graph
- **Hybrid Search**: Combines semantic similarity and BM25 text retrieval
- **Graph-based Reranking**: Reorders results based on graph distance

## Prerequisites

### 1. Neo4j Database

Install and start Neo4j:
- Download [Neo4j Desktop](https://neo4j.com/download/)
- Create a new database with username `neo4j` and password `password`
- Start the database

### 2. AWS Account and Bedrock Access

You need:
- An AWS account with Bedrock access
- Model access enabled for the models you want to use:
  - Claude models (e.g., `us.anthropic.claude-sonnet-4-20250514-v1:0`)
  - Titan embedding models (e.g., `amazon.titan-embed-text-v2:0`)
  - Rerank models (e.g., `cohere.rerank-v3-5:0`)

### 3. AWS Credentials

Configure AWS credentials using one of these methods:

**Option 1: AWS CLI**
```bash
aws configure
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

### 4. Python Dependencies

Sync dependencies with Amazon Bedrock support:
```bash
uv sync --extra bedrock
```

## Setup

1. **Navigate to the example directory:**
   ```bash
   cd examples/amazon-bedrock
   ```

2. **Copy and configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```


## Configuration

### Environment Variables

The example uses these environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_LLM_MODEL` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Claude model for LLM |
| `BEDROCK_EMBEDDING_MODEL` | `amazon.titan-embed-text-v2:0` | Titan model for embeddings |
| `BEDROCK_RERANKER_MODEL` | `cohere.rerank-v3-5:0` | Rerank model |

### Model Availability by Region

Different Bedrock models are available in different AWS regions. Please check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) for the latest information on model availability by region.

## Running the Example

```bash
uv run python amazon_bedrock_neo4j.py
```

## What the Example Does

1. **Initializes Graphiti** with Amazon Bedrock clients and Neo4j
2. **Adds Episodes** containing information about California politics
3. **Performs Hybrid Search** using semantic similarity and BM25
4. **Demonstrates Reranking** using Amazon Bedrock reranker
5. **Shows Integration Details** including model and region information

## Expected Output

```
Added episode: California Politics 0 (text)
Added episode: California Politics 1 (text)
Added episode: California Politics 2 (json)

Searching for: 'Who was the California Attorney General?'

Search Results:
UUID: [uuid]
Fact: Kamala Harris holds the position of Attorney General of California
Valid from: [timestamp]
---

...

Reranking search results based on graph distance:
Using center node UUID: [uuid]
Reranker model: cohere.rerank-v3-5:0

Reranked Search Results (using Amazon Bedrock reranker):
UUID: [uuid]
Fact: Kamala Harris is the Attorney General of California
Valid from: [timestamp]
---

...

Connection closed
```

## Troubleshooting

### Common Issues

**1. AWS Credentials Not Found**
```
NoCredentialsError: Unable to locate credentials
```
- Ensure AWS credentials are properly configured
- Check that your AWS profile has the correct permissions

**2. Model Access Denied**
```
AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel
```
- Request access to the specific models in the AWS Bedrock console
- Ensure your AWS account has Bedrock permissions

**3. Model Not Available in Region**
```
ValidationException: The model ID is not supported in this region
```
- Check model availability in your selected region
- Update the region or model in your configuration

**4. Neo4j Connection Failed**
```
ServiceUnavailable: Failed to establish connection
```
- Ensure Neo4j is running
- Check the connection URI, username, and password

### Getting Help

- Check the [Graphiti documentation](https://help.getzep.com/graphiti)
- Join the [Zep Discord server](https://discord.com/invite/W8Kw6bsgXQ) #graphiti channel
- Review AWS Bedrock documentation for model access and permissions

## Next Steps

- Explore different Amazon Bedrock models
- Try different AWS regions
- Experiment with custom entity types
- Integrate with your own data sources
- Scale up with larger datasets