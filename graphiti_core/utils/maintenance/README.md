# Field System Components - Usage Guide

This README provides comprehensive guidance for using the field system components in the Graphiti project, including MongoDB-Neo4j hybrid architecture for field and cluster management.

## Overview

The field system in Graphiti provides advanced field-level analysis and relationship management capabilities for structured and semi-structured data. It combines Neo4j for relationship storage with MongoDB for metadata management, creating a hybrid architecture that supports both graph operations and efficient metadata queries.

### Architecture Components

1. **Field Nodes (`field_nodes.py`)**

   - FieldNode: Represents individual data fields with metadata
   - ClusterNode: Represents clusters of related fields
   - MongoDB synchronization for cluster metadata

2. **Field Edges (`field_edges.py`)**

   - BelongsToEdge: Connects fields to their clusters
   - FieldRelationshipEdge: Represents relationships between fields
   - MongoDB cluster validation

3. **Bulk Operations (`field_nodes_operations.py`, `field_edges_operations.py`)**
   - Efficient bulk processing for large datasets
   - MongoDB synchronization helpers
   - Analytics and pattern detection

## Prerequisites

### Required Services

1. **Neo4j Database**

   ```bash
   # Using Docker (Recommended)
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/test \
     -e NEO4J_PLUGINS='["apoc"]' \
     neo4j:latest
   ```

2. **MongoDB (Optional but Recommended)**

   ```bash
   # Using Docker
   docker run -d \
     --name mongodb \
     -p 27017:27017 \
     -e MONGO_INITDB_ROOT_USERNAME=admin \
     -e MONGO_INITDB_ROOT_PASSWORD=password \
     mongo:latest
   ```

3. **Python Environment**

   ```bash
   # Using Poetry (Recommended)
   poetry install

   # Using UV
   uv sync

   # Using pip
   pip install -r requirements.txt
   ```

### Environment Configuration

Set the following environment variables:

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="test"

# MongoDB Configuration (optional)
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="graphiti_metadata"

# Embedder Configuration
export OPENAI_API_KEY="your_openai_api_key"  # or other embedder
```

## Basic Usage

### 1. Setting Up the Field System

```python
import asyncio
from neo4j import AsyncGraphDatabase
from graphiti_core.field_nodes import FieldNode, ClusterNode
from graphiti_core.field_edges import BelongsToEdge, FieldRelationshipEdge
from graphiti_core.embedder import OpenAIEmbedder

async def setup_field_system():
    # Initialize Neo4j driver
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "test")
    )

    # Initialize embedder
    embedder = OpenAIEmbedder()

    return driver, embedder
```

### 2. Creating Field Nodes

```python
async def create_field_nodes(driver, embedder):
    # Create a field node
    field_node = FieldNode(
        name="user_id",
        description="Unique identifier for users in the system",
        data_type="INTEGER",
        cluster_id="user_data_cluster",
        embedder=embedder
    )

    # Save to Neo4j (with MongoDB sync)
    await field_node.save(driver)

    # Create a cluster node
    cluster_node = ClusterNode(
        name="User Data Cluster",
        description="Contains all user-related data fields",
        cluster_id="user_data_cluster",
        embedder=embedder
    )

    # Save cluster (with MongoDB metadata)
    await cluster_node.save(driver)

    return field_node, cluster_node
```

### 3. Creating Field Relationships

```python
async def create_field_relationships(driver, field1_uuid, field2_uuid, cluster_id):
    # Create belongs-to relationship
    belongs_to_edge = BelongsToEdge(
        source_node_uuid=field1_uuid,
        target_node_uuid=cluster_id,
        cluster_partition_id=cluster_id
    )

    # Save with MongoDB validation
    await belongs_to_edge.save(driver)

    # Create field relationship
    field_relationship = FieldRelationshipEdge(
        source_node_uuid=field1_uuid,
        target_node_uuid=field2_uuid,
        name="CORRELATES_WITH",
        description="These fields frequently appear together",
        confidence=0.85,
        cluster_partition_id=cluster_id,
        embedder=embedder
    )

    # Save relationship
    await field_relationship.save(driver)

    return belongs_to_edge, field_relationship
```

### 4. Bulk Operations

```python
from graphiti_core.utils.maintenance.field_nodes_operations import (
    save_field_nodes_bulk,
    create_cluster_nodes_bulk
)
from graphiti_core.utils.maintenance.field_edges_operations import (
    save_field_relationship_edges_bulk,
    build_field_cluster_membership
)

async def bulk_field_operations(driver, embedder):
    # Create multiple field nodes
    field_nodes = [
        FieldNode(
            name=f"field_{i}",
            description=f"Description for field {i}",
            cluster_id="bulk_cluster",
            embedder=embedder
        )
        for i in range(100)
    ]

    # Save in bulk (with MongoDB sync)
    saved_fields = await save_field_nodes_bulk(driver, field_nodes)

    # Create cluster membership relationships
    field_uuids = [field.uuid for field in saved_fields]
    cluster_uuid = "bulk_cluster"

    membership_edges = await build_field_cluster_membership(
        driver, field_uuids, cluster_uuid
    )

    return saved_fields, membership_edges
```

## Advanced Features

### 1. Field Analytics

```python
from graphiti_core.utils.maintenance.field_nodes_operations import (
    analyze_field_distribution,
    detect_field_similarities,
    validate_field_constraints
)

async def field_analytics(driver, cluster_id):
    # Analyze field distribution
    distribution = await analyze_field_distribution(driver, cluster_id)
    print(f"Total fields: {distribution['total_fields']}")
    print(f"Data types: {distribution['data_type_distribution']}")

    # Detect similar fields
    similarities = await detect_field_similarities(
        driver, cluster_id, threshold=0.8
    )
    print(f"Found {len(similarities)} similar field pairs")

    # Validate constraints
    validation = await validate_field_constraints(driver, cluster_id)
    print(f"Validation passed: {validation['is_valid']}")

    return distribution, similarities, validation
```

### 2. Relationship Analysis

```python
from graphiti_core.utils.maintenance.field_edges_operations import (
    analyze_field_relationship_patterns,
    find_bidirectional_relationships,
    get_field_relationship_network
)

async def relationship_analytics(driver, cluster_id):
    # Analyze relationship patterns
    patterns = await analyze_field_relationship_patterns(driver, cluster_id)
    print(f"Total relationships: {patterns['total_relationships']}")
    print(f"Relationship types: {patterns['relationship_types']}")

    # Find bidirectional relationships
    bidirectional = await find_bidirectional_relationships(driver, cluster_id)
    print(f"Bidirectional relationships: {len(bidirectional)}")

    # Get field network for a specific field
    field_uuid = "some_field_uuid"
    network = await get_field_relationship_network(
        driver, field_uuid, cluster_id, max_depth=2
    )
    print(f"Network size: {network['network_size']}")

    return patterns, bidirectional, network
```

### 3. Data Quality Validation

```python
async def data_quality_checks(driver, cluster_id):
    # Validate relationship constraints
    constraint_validation = await validate_relationship_constraints(driver, cluster_id)

    if not constraint_validation['is_valid']:
        print("Data quality issues found:")
        for error in constraint_validation['validation_errors']:
            print(f"  - {error}")

    # Check field consistency
    field_validation = await validate_field_constraints(driver, cluster_id)

    if field_validation['constraint_violations']:
        print("Field constraint violations:")
        for violation in field_validation['constraint_violations']:
            print(f"  - {violation}")

    return constraint_validation, field_validation
```

## MongoDB Integration

### Cluster Metadata Management

The field system automatically synchronizes with MongoDB for cluster metadata:

```python
from graphiti_core.cluster_metadata.service import ClusterMetadataService

async def mongodb_operations():
    # Initialize cluster service
    cluster_service = ClusterMetadataService()

    # Create cluster in MongoDB
    cluster_data = await cluster_service.create_cluster(
        cluster_id="new_cluster",
        organization_id="org_123",
        field_count=0
    )

    # Validate cluster exists
    exists = await cluster_service.validate_cluster_exists("new_cluster")
    print(f"Cluster exists: {exists}")

    # Increment field count
    await cluster_service.increment_field_count("new_cluster", 5)

    # Get cluster info
    cluster_info = await cluster_service.get_cluster("new_cluster")
    print(f"Field count: {cluster_info['field_count']}")
```

### Error Handling

The system gracefully handles MongoDB unavailability:

```python
async def robust_field_operations(driver, embedder):
    try:
        # Field operations continue even if MongoDB is down
        field_node = FieldNode(
            name="robust_field",
            description="Field that works without MongoDB",
            cluster_id="test_cluster",
            embedder=embedder
        )

        # This will succeed even if MongoDB sync fails
        await field_node.save(driver)
        print("Field saved successfully (with or without MongoDB)")

    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Performance Optimization

### 1. Batch Processing

```python
async def optimized_bulk_processing(driver, embedder, field_data_list):
    # Process in batches to avoid memory issues
    batch_size = 50
    all_saved_fields = []

    for i in range(0, len(field_data_list), batch_size):
        batch = field_data_list[i:i + batch_size]

        # Create field nodes for batch
        field_nodes = [
            FieldNode(**field_data, embedder=embedder)
            for field_data in batch
        ]

        # Save batch
        saved_batch = await save_field_nodes_bulk(driver, field_nodes)
        all_saved_fields.extend(saved_batch)

        print(f"Processed batch {i//batch_size + 1}")

    return all_saved_fields
```

### 2. Efficient Querying

```python
from graphiti_core.utils.maintenance.field_nodes_operations import (
    get_fields_by_cluster,
    search_fields_by_pattern
)

async def efficient_queries(driver, cluster_id):
    # Get all fields in cluster (single query)
    cluster_fields = await get_fields_by_cluster(driver, cluster_id)

    # Search fields by pattern
    pattern_matches = await search_fields_by_pattern(
        driver, cluster_id, name_pattern="user_*"
    )

    # Get fields with specific data types
    typed_fields = await get_fields_by_data_type(
        driver, cluster_id, ["INTEGER", "STRING"]
    )

    return cluster_fields, pattern_matches, typed_fields
```

## Error Handling and Debugging

### Common Error Scenarios

1. **MongoDB Connection Issues**

   ```python
   # Graceful handling of MongoDB errors
   try:
       await field_node.save(driver)
   except Exception as e:
       if "MongoDB" in str(e):
           print("MongoDB unavailable, continuing with Neo4j only")
       else:
           raise
   ```

2. **Neo4j Connection Issues**

   ```python
   # Retry logic for Neo4j operations
   import asyncio

   async def retry_neo4j_operation(operation, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await operation()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise
               await asyncio.sleep(2 ** attempt)  # Exponential backoff
   ```

3. **Validation Errors**

   ```python
   from graphiti_core.cluster_metadata.exceptions import (
       ClusterNotFoundError,
       ClusterValidationError
   )

   try:
       await field_relationship.save(driver)
   except ClusterNotFoundError:
       print("Cluster doesn't exist in MongoDB, creating it...")
       # Handle cluster creation
   except ClusterValidationError as e:
       print(f"Validation failed: {e}")
   ```

### Debugging Tips

1. **Enable Debug Logging**

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Database States**

   ```python
   # Neo4j state
   async with driver.session() as session:
       result = await session.run("MATCH (n) RETURN count(n) as node_count")
       record = await result.single()
       print(f"Neo4j nodes: {record['node_count']}")

   # MongoDB state (if available)
   cluster_service = ClusterMetadataService()
   clusters = await cluster_service.list_clusters()
   print(f"MongoDB clusters: {len(clusters)}")
   ```

3. **Validate Data Consistency**

   ```python
   async def check_consistency(driver, cluster_id):
       # Check Neo4j vs MongoDB field counts
       neo4j_count = await get_field_count_neo4j(driver, cluster_id)
       mongo_count = await get_field_count_mongodb(cluster_id)

       if neo4j_count != mongo_count:
           print(f"Inconsistency detected: Neo4j={neo4j_count}, MongoDB={mongo_count}")
   ```

## Best Practices

### 1. Resource Management

```python
async def proper_resource_management():
    driver = None
    try:
        # Initialize resources
        driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test"))
        embedder = OpenAIEmbedder()

        # Perform operations
        await field_operations(driver, embedder)

    finally:
        # Clean up resources
        if driver:
            await driver.close()
```

### 2. Error Recovery

```python
async def robust_field_creation(driver, field_data, embedder):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            field_node = FieldNode(**field_data, embedder=embedder)
            await field_node.save(driver)
            return field_node
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to create field after {max_retries} attempts: {e}")
                raise
            await asyncio.sleep(2 ** attempt)
```

### 3. Monitoring and Metrics

```python
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def timed_operation(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"{operation_name} took {duration:.2f} seconds")

# Usage
async def monitored_bulk_operation(driver, field_nodes):
    async with timed_operation("Bulk field save"):
        return await save_field_nodes_bulk(driver, field_nodes)
```

## Configuration

### Environment-Specific Settings

Create a configuration file for different environments:

```python
# config.py
import os

class Config:
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "graphiti_metadata")

    EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Performance settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

class ProductionConfig(Config):
    NEO4J_URI = os.getenv("PROD_NEO4J_URI")
    MONGODB_URI = os.getenv("PROD_MONGODB_URI")

class TestConfig(Config):
    NEO4J_URI = "bolt://localhost:7687"
    MONGODB_URI = "mongodb://localhost:27017"
    MONGODB_DATABASE = "graphiti_test"
```

## Support and Troubleshooting

### Common Issues

1. **Connection timeouts**: Increase timeout values in configuration
2. **Memory issues with large datasets**: Reduce batch sizes
3. **MongoDB sync failures**: Check MongoDB connectivity and permissions
4. **Embedding API limits**: Implement rate limiting and retry logic

### Getting Help

1. Check the test files for usage examples
2. Review the main Graphiti documentation
3. Enable debug logging for detailed error information
4. Open an issue in the project repository for bugs or feature requests

For additional support, consult the project's main documentation or community resources.
