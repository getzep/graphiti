# Graphiti Structure Update Requirements Document

---

## 1. Executive Summary

This document outlines the requirements for modifying the open-source Graphiti project to support domain-specific knowledge graph structure for intelligent Splunk query generation. The modification will transform Graphiti from a general-purpose temporal knowledge graph into a specialized security field relationship engine using a **hybrid MongoDB + Neo4j architecture**. We are currently building a SPLUNK agent and we have a problem of selecting appropriate fields for specific macro we need to update graphiti to create a system that we can store inside it the fields related with specific macro with specific relationship structure and predefined schema for each node.

## Document Information

- **Project**: Splunk Security Query Generation Agent
- **Component**: Graphiti Knowledge Graph Structure Modification
- **Version**: 2.0 (Updated for Hybrid Architecture)
- **Date**: January 15, 2025
- **Author**: AI Engineering Team

## **🏗️ HYBRID ARCHITECTURE OVERVIEW**

### **MongoDB Layer: Cluster Metadata Management**

- **Purpose**: Fast cluster/sub-cluster discovery and configuration management
- **Collection**: `cluster_metadata`
- **Benefits**: Flexible schema, fast lookups, easy configuration updates

### **Neo4j Layer: Field Relationships and Graph Data**

- **Purpose**: Field nodes, relationships, embeddings, complex graph queries
- **Benefits**: Graph traversal, vector similarity search, relationship analysis

### **Integration**: Unified API layer coordinates between both databases

## What we trying to achieve

---

Disconnected Subgraphs with Sub-labels in Neo4j Community Edition for Graphiti with MongoDB-based Cluster Registry

**Summary**: Enable the creation of disconnected subgraphs (referred to as "clusters") using a hybrid architecture where cluster metadata is managed in MongoDB for fast access, while field relationships remain in Neo4j for graph operations. Store audit data for different organizations (e.g., `linux_audit_GFH` and `linux_audit_batelco`) with no relationships between them.

**Requirements**:

1. **MongoDB Cluster Registry**:
   - Store cluster metadata in `cluster_metadata` collection
   - Enable fast cluster/sub-cluster discovery and validation
   - Support dynamic cluster configuration updates
2. **Neo4j Field Storage**:

   - Create field nodes with cluster labels for isolation
   - Store field relationships and embeddings in Neo4j
   - Maintain graph structure for complex queries

3. **Hybrid Integration**:
   - Unified API layer coordinates between MongoDB and Neo4j
   - Cluster validation against MongoDB registry
   - Field operations use Neo4j for graph functionality

## **📊 MONGODB CLUSTER METADATA COLLECTION**

### **Collection: `cluster_metadata`**

**Document Structure:**

```json
{
  "_id": "linux_audit_batelco", // Unique cluster identifier
  "cluster_name": "linux_audit_batelco", // Full cluster name
  "macro_name": "linux_audit", // Macro identifier (linux_audit, azure_ad, etc.)
  "organization": "batelco", // Organization name (batelco, sico, gfh, etc.)
  "description": "Batelco Linux audit field cluster for security monitoring",
  "sub_clusters": [
    {
      "name": "UserAuthentication",
      "description": "User authentication and session management fields",
      "field_count": 12,
      "created_at": "2025-01-15T10:30:00Z",
      "last_updated": "2025-01-15T10:30:00Z"
    },
    {
      "name": "NetworkAccess",
      "description": "Network access and connection tracking fields",
      "field_count": 8,
      "created_at": "2025-01-15T10:30:00Z",
      "last_updated": "2025-01-15T10:30:00Z"
    }
  ],
  "status": "active", // active, inactive, deprecated
  "total_fields": 45,
  "created_at": "2025-01-15T10:30:00Z",
  "last_updated": "2025-01-15T10:30:00Z",
  "created_by": "system",
  "metadata": {
    "version": "1.0",
    "soc_validated": true,
    "compliance_tags": ["PCI-DSS", "SOX"],
    "data_retention_days": 365
  }
}
```

### **MongoDB Service Integration**

**Service Class Usage:**

```python
# Using the provided MongoDB service
from app.services.mongo_service import get_mongo_connection

class ClusterMetadataService:
    def __init__(self):
        self.collection_name = "cluster_metadata"

    async def get_collection(self):
        """Get cluster_metadata collection"""
        db = await get_mongo_connection()
        return db[self.collection_name]

    async def get_cluster(self, cluster_id: str) -> dict:
        """Get cluster metadata by ID"""
        collection = await self.get_collection()
        return await collection.find_one({"_id": cluster_id})

    async def list_clusters(self, organization: str = None) -> list:
        """List all clusters, optionally filtered by organization"""
        collection = await self.get_collection()
        filter_query = {"organization": organization} if organization else {}
        cursor = collection.find(filter_query)
        return await cursor.to_list(length=None)

    async def get_available_sub_clusters(self, cluster_id: str) -> list:
        """Get sub-clusters for a specific cluster"""
        cluster = await self.get_cluster(cluster_id)
        return cluster.get("sub_clusters", []) if cluster else []
```

## **🔗 NEO4J FIELD STORAGE STRUCTURE**

### **FieldNode in Neo4j**

```python
class FieldNode(Node):
    """Field node stored in Neo4j with cluster labels"""
    field_name: str = Field(description="Security field name (e.g., 'AUID')")
    description: str = Field(description="Field description for AI context")
    examples: list[str] = Field(description="Example values", default_factory=list)
    data_type: str = Field(description="Field data type")
    count: int = Field(description="Occurrence count in events", default=0)
    distinct_count: int = Field(description="Distinct value count", default=0)
    embedding: list[float] | None = Field(description="Vector for semantic search")

    # Temporal validation
    validated_at: datetime = Field(default_factory=utc_now)
    invalidated_at: datetime | None = Field(default=None)
    last_updated: datetime = Field(default_factory=utc_now)

    # Cluster reference (NOT stored as properties - derived from labels)
    @property
    def cluster_id(self) -> str:
        """Extract cluster ID from Neo4j labels"""
        for label in self.labels:
            if label != "Field" and ":" not in label:
                return label
        return "unknown"
```

### **Neo4j Label Structure**

```cypher
-- Field with cluster and sub-cluster labels
(:Field:linux_audit_batelco:UserAuthentication:NetworkAccess {
    uuid: "field-uuid-123",
    field_name: "AUID",
    description: "Audit User ID for tracking user sessions",
    examples: ["1000", "1001", "0"],
    data_type: "integer",
    count: 15420,
    distinct_count: 847,
    validated_at: "2025-01-15T10:30:00Z",
    last_updated: "2025-01-15T10:30:00Z"
})
```

## **🔄 HYBRID WORKFLOW INTEGRATION**

### **Field Creation Workflow**

```python
async def create_field_with_cluster_validation(field_data: FieldCreateRequest):
    # 1. Validate cluster exists in MongoDB
    cluster_service = ClusterMetadataService()
    cluster = await cluster_service.get_cluster(field_data.cluster_id)
    if not cluster:
        raise ClusterNotFoundError(field_data.cluster_id)

    # 2. Validate sub-clusters against MongoDB registry
    available_subs = await cluster_service.get_available_sub_clusters(field_data.cluster_id)
    invalid_subs = set(field_data.sub_clusters) - {sub["name"] for sub in available_subs}
    if invalid_subs:
        raise InvalidSubClusterError(invalid_subs)

    # 3. Create FieldNode in Neo4j with proper labels
    field_node = FieldNode(
        field_name=field_data.field_name,
        description=field_data.description,
        labels=["Field", cluster["cluster_name"]] +
               [f"{cluster['cluster_name']}:{sub}" for sub in field_data.sub_clusters]
    )

    # 4. Generate relationships using Neo4j graph data
    await generate_field_relationships(field_node)

    # 5. Update MongoDB cluster statistics
    await cluster_service.increment_field_count(field_data.cluster_id, field_data.sub_clusters)

    return field_node
```

### **Splunk Agent Integration Workflow**

```python
async def splunk_agent_field_search(question: str, macro: str, organization: str):
    cluster_id = f"{macro}_{organization}"

    # 1. Fast cluster validation via MongoDB
    cluster_service = ClusterMetadataService()
    cluster = await cluster_service.get_cluster(cluster_id)
    if not cluster:
        raise ClusterNotFoundError(cluster_id)

    # 2. Get available sub-clusters from MongoDB
    sub_clusters = await cluster_service.get_available_sub_clusters(cluster_id)

    # 3. Semantic search in Neo4j using cluster labels
    question_embedding = await embed_question(question)
    fields = await search_fields_by_embedding(
        cluster_labels=[cluster_id],
        query_embedding=question_embedding,
        limit=10
    )

    # 4. LLM decides if needs sub-cluster filtering
    if needs_sub_cluster_filtering:
        relevant_sub = await select_relevant_sub_cluster(question, sub_clusters)
        fields = await search_fields_by_sub_cluster(
            cluster_id=cluster_id,
            sub_cluster=relevant_sub["name"],
            query_embedding=question_embedding
        )

    return fields
```

## **📚 UPDATED IMPLEMENTATION APPROACH**

### **Phase 1: MongoDB Integration**

1. **Setup MongoDB service** using provided service class
2. **Create cluster metadata management** with CRUD operations
3. **Implement cluster validation** against MongoDB registry
4. **Add sub-cluster management** with statistics tracking

### **Phase 2: Neo4j Field System**

1. **Keep current FieldNode structure** but validate against MongoDB
2. **Use Neo4j labels** for cluster/sub-cluster assignment
3. **Maintain graph relationships** in Neo4j for complex queries
4. **Store embeddings** in Neo4j for similarity search

### **Phase 3: Hybrid API Layer**

1. **Unified FastAPI endpoints** coordinate both databases
2. **Cluster operations** primarily use MongoDB
3. **Field operations** use MongoDB validation + Neo4j storage
4. **Search operations** combine both databases

### **Phase 4: MCP Integration**

1. **Update MCP tools** to use hybrid approach
2. **Fast cluster discovery** via MongoDB
3. **Field search** via Neo4j with cluster validation
4. **Splunk agent integration** using both databases

## **💾 DATABASE RESPONSIBILITIES**

### **MongoDB Responsibilities:**

- ✅ Cluster metadata storage and retrieval
- ✅ Sub-cluster configuration management
- ✅ Fast cluster/organization lookups
- ✅ Cluster statistics and field counts
- ✅ SOC validation status tracking
- ✅ Compliance and metadata management

### **Neo4j Responsibilities:**

- ✅ Field node storage with properties
- ✅ Field-to-field relationships
- ✅ Embedding storage and similarity search
- ✅ Graph traversal and complex queries
- ✅ Label-based cluster isolation
- ✅ Temporal relationship tracking

### **Integration Layer Responsibilities:**

- ✅ Coordinate between both databases
- ✅ Validate cluster existence before field creation
- ✅ Maintain data consistency across systems
- ✅ Provide unified API interface
- ✅ Handle error scenarios and rollbacks

## **🔧 BENEFITS OF HYBRID APPROACH**

### **Performance Benefits:**

- **Fast cluster lookups** from MongoDB (sub-millisecond)
- **Efficient field searches** using Neo4j vector similarity
- **Reduced Neo4j query complexity** for metadata operations
- **Better scalability** for cluster management

### **Operational Benefits:**

- **Easier cluster configuration** via MongoDB documents
- **Flexible metadata schema** without Neo4j migrations
- **Better separation of concerns** between metadata and graph data
- **Simplified backup and maintenance** strategies

### **Development Benefits:**

- **Familiar MongoDB operations** for cluster management
- **Leverage existing Graphiti patterns** for field relationships
- **Clear architectural boundaries** between systems
- **Easier testing and validation** of components

This hybrid approach provides the best of both worlds: MongoDB's flexibility for metadata management and Neo4j's power for graph relationships, while maintaining clean separation of concerns and optimal performance characteristics.
