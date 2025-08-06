# Graphiti Field-Based System Implementation Todo List

## **HYBRID MONGODB + NEO4J ARCHITECTURE**

## Document Information

- **Project**: Splunk Security Query Generation Agent - Graphiti Implementation
- **Architecture**: Hybrid MongoDB (Metadata) + Neo4j (Graph)
- **Version**: 2.0 (Updated for Hybrid Architecture)
- **Date**: January 15, 2025
- **Status**: Ready for Implementation

---

# **🏗️ ARCHITECTURE OVERVIEW**

## **MongoDB Layer: Cluster Metadata Management**

- **Collection**: `cluster_metadata`
- **Purpose**: Fast cluster discovery, sub-cluster configuration, metadata management
- **Service**: Using provided MongoDB async service class

## **Neo4j Layer: Field Graph Storage**

- **Purpose**: Field nodes, relationships, embeddings, graph queries
- **Labels**: Cluster-based isolation using Neo4j labels
- **Integration**: Validated against MongoDB cluster registry

---

# PHASE 1: MONGODB CLUSTER METADATA SYSTEM (Weeks 1-2)

## Task 1.1: Setup MongoDB Cluster Metadata Service

**Priority**: CRITICAL
**Estimated Time**: 2-3 days

### **Files to Understand First:**

- **PROVIDED**: MongoDB service class (app/services/mongo_service.py)
- `graphiti_core/helpers.py` - Current validation patterns
- `graphiti_core/errors.py` - Error handling patterns

### **Files to Create:**

- [ ] **CREATE**: `graphiti_core/cluster_metadata/cluster_service.py`

  - [ ] Implement `ClusterMetadataService` using provided MongoDB service
  - [ ] Add CRUD operations for cluster_metadata collection
  - [ ] Implement cluster validation and existence checking
  - [ ] Add sub-cluster management operations

- [ ] **CREATE**: `graphiti_core/cluster_metadata/models.py`
  - [ ] Define `ClusterMetadata` Pydantic model
  - [ ] Define `SubCluster` model with statistics
  - [ ] Add validation rules for cluster patterns
  - [ ] Include temporal fields (created_at, last_updated)

### **MongoDB Document Schema to Implement:**

```json
{
  "_id": "linux_audit_batelco",
  "cluster_name": "linux_audit_batelco",
  "macro_name": "linux_audit",
  "organization": "batelco",
  "description": "Batelco Linux audit field cluster",
  "sub_clusters": [
    {
      "name": "UserAuthentication",
      "description": "User authentication fields",
      "field_count": 12,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "status": "active",
  "total_fields": 45,
  "created_at": "2025-01-15T10:30:00Z",
  "last_updated": "2025-01-15T10:30:00Z"
}
```

### **Core Methods to Implement:**

```python
class ClusterMetadataService:
    async def get_cluster(self, cluster_id: str) -> dict
    async def list_clusters(self, organization: str = None) -> list
    async def create_cluster(self, cluster_data: dict) -> dict
    async def update_cluster(self, cluster_id: str, updates: dict) -> dict
    async def delete_cluster(self, cluster_id: str) -> bool
    async def get_available_sub_clusters(self, cluster_id: str) -> list
    async def add_sub_cluster(self, cluster_id: str, sub_cluster: dict) -> dict
    async def increment_field_count(self, cluster_id: str, sub_clusters: list) -> dict
```

---

## Task 1.2: Create Cluster Validation and Management

**Priority**: CRITICAL  
**Estimated Time**: 2-3 days

### **Files to Create:**

- [ ] **CREATE**: `graphiti_core/cluster_metadata/validators.py`

  - [ ] Implement cluster name pattern validation (`{macro}_{organization}`)
  - [ ] Add sub-cluster name validation
  - [ ] Create organization lookup/validation
  - [ ] Add duplicate cluster prevention

- [ ] **CREATE**: `graphiti_core/cluster_metadata/exceptions.py`
  - [ ] Define `ClusterNotFoundError`
  - [ ] Define `InvalidClusterPatternError`
  - [ ] Define `DuplicateClusterError`
  - [ ] Define `InvalidSubClusterError`

### **Validation Rules to Implement:**

- Cluster names must follow `{macro_name}_{organization}` pattern
- Sub-cluster names must be alphanumeric with underscores
- Organizations must be from approved list (batelco, sico, gfh, etc.)
- Macro names must be from approved list (linux_audit, azure_ad, etc.)

---

## Task 1.3: MongoDB Integration Layer

**Priority**: HIGH
**Estimated Time**: 2-3 days

### **Files to Create:**

- [ ] **CREATE**: `graphiti_core/cluster_metadata/integration.py`

  - [ ] Create hybrid coordinator between MongoDB and Neo4j
  - [ ] Implement cluster existence validation before field operations
  - [ ] Add error handling for MongoDB connection issues
  - [ ] Create cluster statistics synchronization

- [ ] **CREATE**: `graphiti_core/cluster_metadata/__init__.py`
  - [ ] Export main service classes
  - [ ] Initialize MongoDB connection on import
  - [ ] Set up logging for cluster operations

### **Integration Patterns:**

```python
class HybridClusterManager:
    def __init__(self, mongo_service: ClusterMetadataService, neo4j_driver: GraphDriver):
        self.mongo_service = mongo_service
        self.neo4j_driver = neo4j_driver

    async def validate_cluster_for_field_creation(self, cluster_id: str) -> bool:
        # Check MongoDB first, then validate Neo4j constraints

    async def sync_field_count_to_mongodb(self, cluster_id: str):
        # Count fields in Neo4j and update MongoDB statistics
```

---

# PHASE 2: NEO4J FIELD SYSTEM INTEGRATION (Weeks 3-4)

## Task 2.1: Update FieldNode for Hybrid Architecture

**Priority**: CRITICAL
**Estimated Time**: 3-4 days

### **Files to Understand First:**

- `graphiti_core/nodes.py` - Current EntityNode patterns
- `graphiti_core/models/nodes/node_db_queries.py` - Database operations
- **CREATED**: `graphiti_core/cluster_metadata/cluster_service.py`

### **Files to Create/Modify:**

- [ ] **UPDATE**: `graphiti_core/field_nodes.py` (if exists) or **CREATE**

  - [ ] Implement FieldNode with cluster validation against MongoDB
  - [ ] Remove cluster metadata properties (use labels only)
  - [ ] Add methods to extract cluster info from Neo4j labels
  - [ ] Integrate with MongoDB cluster validation

- [ ] **CREATE**: `graphiti_core/models/nodes/field_db_queries.py`
  - [ ] Create field save queries with cluster label validation
  - [ ] Add field uniqueness constraints within clusters
  - [ ] Implement field search queries by cluster labels

### **Updated FieldNode Structure:**

```python
class FieldNode(Node):
    # Core field properties (stored in Neo4j)
    field_name: str = Field(description="Security field name")
    description: str = Field(description="Field description for embedding")
    examples: list[str] = Field(default_factory=list)
    data_type: str = Field(description="Field data type")
    count: int = Field(default=0)
    distinct_count: int = Field(default=0)
    embedding: list[float] | None = Field(default=None)

    # Temporal fields
    validated_at: datetime = Field(default_factory=utc_now)
    invalidated_at: datetime | None = Field(default=None)
    last_updated: datetime = Field(default_factory=utc_now)

    # NO cluster properties stored - derived from Neo4j labels
    @property
    def cluster_id(self) -> str:
        """Extract cluster ID from labels"""
        for label in self.labels:
            if label != "Field" and ":" not in label:
                return label
        return "unknown"

    async def validate_cluster_exists(self, cluster_service: ClusterMetadataService):
        """Validate cluster exists in MongoDB before saving"""
        cluster = await cluster_service.get_cluster(self.cluster_id)
        if not cluster:
            raise ClusterNotFoundError(self.cluster_id)
```

---

## Task 2.2: Implement Hybrid Field Creation Workflow

**Priority**: CRITICAL
**Estimated Time**: 3-4 days

### **Files to Create:**

- [ ] **CREATE**: `graphiti_core/field_processing/hybrid_field_manager.py`
  - [ ] Coordinate between MongoDB validation and Neo4j storage
  - [ ] Implement complete field creation workflow
  - [ ] Add rollback procedures for failed operations
  - [ ] Include relationship generation with cluster validation

### **Hybrid Field Creation Process:**

```python
class HybridFieldManager:
    async def create_field_with_validation(self, field_data: FieldCreateRequest) -> FieldNode:
        # 1. Validate cluster exists in MongoDB
        cluster = await self.cluster_service.get_cluster(field_data.cluster_id)
        if not cluster:
            raise ClusterNotFoundError(field_data.cluster_id)

        # 2. Validate sub-clusters against MongoDB registry
        available_subs = await self.cluster_service.get_available_sub_clusters(field_data.cluster_id)
        self._validate_sub_clusters(field_data.sub_clusters, available_subs)

        # 3. Create FieldNode in Neo4j with proper labels
        field_node = self._create_field_node_with_labels(field_data, cluster)

        # 4. Generate embeddings and relationships in Neo4j
        await self._process_field_embeddings_and_relationships(field_node)

        # 5. Update MongoDB cluster statistics
        await self.cluster_service.increment_field_count(
            field_data.cluster_id,
            field_data.sub_clusters
        )

        return field_node
```

---

## Task 2.3: Update Field Relationship System

**Priority**: HIGH
**Estimated Time**: 2-3 days

### **Files to Understand First:**

- `graphiti_core/edges.py` - Current relationship patterns
- `graphiti_core/utils/maintenance/edge_operations.py` - Relationship processing

### **Files to Create/Modify:**

- [ ] **CREATE**: `graphiti_core/field_processing/field_relationships.py`
  - [ ] Implement field-to-field relationships within clusters only
  - [ ] Add cluster isolation validation for relationships
  - [ ] Create predefined relationship patterns
  - [ ] Add AI-generated relationship creation with cluster constraints

### **Cluster-Constrained Relationships:**

```python
class FieldRelationshipManager:
    async def create_relationships_within_cluster(
        self,
        field: FieldNode,
        cluster_id: str
    ) -> list[FieldRelationship]:
        # 1. Validate cluster exists in MongoDB
        cluster = await self.cluster_service.get_cluster(cluster_id)

        # 2. Find related fields ONLY within same cluster
        related_fields = await self._find_fields_in_cluster(cluster_id)

        # 3. Generate relationships constrained to cluster
        relationships = await self._generate_cluster_constrained_relationships(
            field, related_fields
        )

        return relationships
```

---

# PHASE 3: API LAYER AND INTEGRATION (Weeks 5-6)

## Task 3.1: Create Hybrid FastAPI Endpoints

**Priority**: CRITICAL
**Estimated Time**: 4-5 days

### **Files to Understand First:**

- `server/graph_service/routers/ingest.py` - Current API patterns
- **CREATED**: `graphiti_core/cluster_metadata/cluster_service.py`
- **CREATED**: `graphiti_core/field_processing/hybrid_field_manager.py`

### **Files to Create:**

- [ ] **CREATE**: `server/graph_service/routers/clusters.py`

  - [ ] `POST /clusters/` - Create cluster in MongoDB
  - [ ] `GET /clusters/` - List clusters from MongoDB
  - [ ] `GET /clusters/{cluster_id}` - Get cluster details from MongoDB
  - [ ] `PUT /clusters/{cluster_id}` - Update cluster in MongoDB
  - [ ] `DELETE /clusters/{cluster_id}` - Delete cluster (MongoDB + Neo4j cleanup)

- [ ] **CREATE**: `server/graph_service/routers/subclusters.py`

  - [ ] `POST /clusters/{cluster_id}/subclusters/` - Add sub-cluster to MongoDB
  - [ ] `GET /clusters/{cluster_id}/subclusters/` - List sub-clusters from MongoDB
  - [ ] `PUT /clusters/{cluster_id}/subclusters/{subcluster_name}` - Update sub-cluster
  - [ ] `DELETE /clusters/{cluster_id}/subclusters/{subcluster_name}` - Remove sub-cluster

- [ ] **CREATE**: `server/graph_service/routers/fields.py`
  - [ ] `POST /clusters/{cluster_id}/fields/` - Create field (MongoDB validation + Neo4j storage)
  - [ ] `GET /clusters/{cluster_id}/fields/` - List fields (Neo4j query with MongoDB metadata)
  - [ ] `GET /clusters/{cluster_id}/fields/{field_id}` - Get field details
  - [ ] `PUT /clusters/{cluster_id}/fields/{field_id}` - Update field
  - [ ] `DELETE /clusters/{cluster_id}/fields/{field_id}` - Delete field

### **API Response Format:**

```json
{
  "status": "created",
  "field": {
    "uuid": "field-uuid-123",
    "field_name": "ARCH",
    "cluster_id": "linux_audit_batelco",
    "sub_clusters": ["SystemInfo"],
    "created_at": "2025-01-15T10:30:00Z",
    "relationships_created": 3
  },
  "cluster_metadata": {
    "total_fields": 46,
    "sub_cluster_counts": {
      "SystemInfo": 12,
      "UserAuthentication": 8
    }
  },
  "validation_status": "passed"
}
```

---

## Task 3.2: Create Service Layer for Hybrid Operations

**Priority**: HIGH
**Estimated Time**: 3-4 days

### **Files to Create:**

- [ ] **CREATE**: `server/graph_service/services/hybrid_cluster_service.py`

  - [ ] Coordinate cluster operations between MongoDB and Neo4j
  - [ ] Handle cluster deletion with field cleanup
  - [ ] Provide unified cluster statistics
  - [ ] Manage cluster status and lifecycle

- [ ] **CREATE**: `server/graph_service/services/hybrid_field_service.py`
  - [ ] Implement field CRUD with hybrid validation
  - [ ] Coordinate field statistics between databases
  - [ ] Handle field updates with relationship regeneration
  - [ ] Provide field search with cluster filtering

### **Service Integration Pattern:**

```python
class HybridFieldService:
    def __init__(self):
        self.cluster_service = ClusterMetadataService()
        self.field_manager = HybridFieldManager()
        self.neo4j_driver = get_neo4j_driver()

    async def create_field(self, cluster_id: str, field_data: dict) -> dict:
        # 1. MongoDB validation
        # 2. Neo4j field creation
        # 3. Relationship generation
        # 4. Statistics update
        # 5. Return unified response
```

---

## Task 3.3: Update MCP Server for Hybrid Architecture

**Priority**: CRITICAL
**Estimated Time**: 3-4 days

### **Files to Understand First:**

- `mcp_server/graphiti_mcp_server.py` - Current MCP implementation
- **CREATED**: Hybrid service classes

### **Files to Modify:**

- [ ] **UPDATE**: `mcp_server/graphiti_mcp_server.py`
  - [ ] Add MongoDB cluster service initialization
  - [ ] Update field search tools to use cluster validation
  - [ ] Add cluster discovery tools
  - [ ] Update memory search for hybrid architecture

### **New MCP Tools for Hybrid Architecture:**

```python
@mcp_tool
async def list_available_clusters(organization: str = None) -> List[ClusterInfo]:
    """List clusters from MongoDB with optional organization filter"""

@mcp_tool
async def get_cluster_subclusters(cluster_id: str) -> List[SubClusterInfo]:
    """Get sub-clusters for cluster from MongoDB"""

@mcp_tool
async def search_fields_in_cluster(
    cluster_id: str,
    question: str,
    sub_cluster: Optional[str] = None,
    limit: int = 10
) -> List[FieldMetadata]:
    """Search fields in Neo4j with MongoDB cluster validation"""

@mcp_tool
async def get_cluster_statistics(cluster_id: str) -> ClusterStats:
    """Get comprehensive cluster statistics from both databases"""
```

---

# PHASE 4: TESTING AND DEPLOYMENT (Weeks 7-8)

## Task 4.1: Create Hybrid Architecture Tests

**Priority**: HIGH
**Estimated Time**: 4-5 days

### **Files to Create:**

- [ ] **CREATE**: `tests/cluster_metadata/test_cluster_service.py`

  - [ ] Test MongoDB cluster CRUD operations
  - [ ] Test cluster validation logic
  - [ ] Test sub-cluster management
  - [ ] Test error handling for MongoDB operations

- [ ] **CREATE**: `tests/field_processing/test_hybrid_field_manager.py`

  - [ ] Test field creation with MongoDB validation
  - [ ] Test cluster constraint enforcement
  - [ ] Test rollback procedures
  - [ ] Test statistics synchronization

- [ ] **CREATE**: `tests/integration/test_hybrid_api.py`
  - [ ] Test complete API workflows
  - [ ] Test MongoDB + Neo4j coordination
  - [ ] Test error scenarios and recovery
  - [ ] Test performance under load

### **Test Coverage Requirements:**

- [ ] MongoDB cluster operations (100%)
- [ ] Neo4j field operations with cluster validation (100%)
- [ ] API endpoint functionality (95%)
- [ ] Error handling and rollback procedures (90%)
- [ ] MCP tool integration (95%)

---

## Task 4.2: Performance Testing and Optimization

**Priority**: MEDIUM
**Estimated Time**: 2-3 days

### **Files to Create:**

- [ ] **CREATE**: `tests/performance/test_hybrid_performance.py`
  - [ ] Test MongoDB cluster lookup performance
  - [ ] Test Neo4j field search with cluster labels
  - [ ] Test hybrid field creation throughput
  - [ ] Test API response times

### **Performance Targets:**

- MongoDB cluster lookup: < 10ms
- Neo4j field search: < 100ms for 1000 fields
- Field creation (hybrid): < 500ms
- API endpoints: < 200ms response time

---

## Task 4.3: Documentation and Migration Tools

**Priority**: MEDIUM
**Estimated Time**: 3-4 days

### **Files to Create:**

- [ ] **CREATE**: `docs/hybrid_architecture_guide.md`

  - [ ] Complete hybrid architecture documentation
  - [ ] MongoDB cluster management guide
  - [ ] Neo4j field operations guide
  - [ ] API usage examples

- [ ] **CREATE**: `scripts/migrate_to_hybrid.py`

  - [ ] Migrate existing data to hybrid architecture
  - [ ] Create cluster metadata in MongoDB
  - [ ] Preserve existing field data in Neo4j
  - [ ] Validate migration success

- [ ] **CREATE**: `examples/hybrid_usage/`
  - [ ] Cluster management examples
  - [ ] Field creation examples
  - [ ] Splunk agent integration examples

---

# **🔄 UPDATED IMPLEMENTATION PRIORITY MATRIX**

## 🔴 CRITICAL (Must Complete First)

1. **Task 1.1**: MongoDB Cluster Metadata Service
2. **Task 1.2**: Cluster Validation and Management
3. **Task 2.1**: FieldNode Hybrid Integration
4. **Task 2.2**: Hybrid Field Creation Workflow

## 🟡 HIGH (Complete After Critical)

1. **Task 1.3**: MongoDB Integration Layer
2. **Task 3.1**: Hybrid FastAPI Endpoints
3. **Task 3.3**: MCP Server Updates
4. **Task 2.3**: Field Relationship System

## 🟢 MEDIUM (Complete After High)

1. **Task 3.2**: Service Layer Implementation
2. **Task 4.1**: Hybrid Architecture Tests
3. **Task 4.2**: Performance Testing

## 🔵 LOW (Complete Last)

1. **Task 4.3**: Documentation and Migration

---

# **🔗 UPDATED DEPENDENCY GRAPH**

```
Task 1.1 (MongoDB Service) → Task 1.2 (Validation) → Task 2.1 (FieldNode)
       ↓                           ↓                        ↓
Task 1.3 (Integration) → Task 2.2 (Hybrid Creation) → Task 3.1 (API)
       ↓                           ↓                        ↓
Task 2.3 (Relationships) → Task 3.2 (Services) → Task 3.3 (MCP)
       ↓                           ↓                        ↓
Task 4.1 (Tests) → Task 4.2 (Performance) → Task 4.3 (Documentation)
```

---

# **💡 HYBRID ARCHITECTURE BENEFITS**

## **Performance Benefits:**

- **MongoDB**: Sub-10ms cluster lookups vs 100ms+ Neo4j queries
- **Neo4j**: Optimized for field relationships and embeddings
- **Separation**: Reduced query complexity in both databases

## **Operational Benefits:**

- **Easy cluster management** via MongoDB documents
- **Flexible metadata** without Neo4j schema migrations
- **Independent scaling** of metadata vs graph operations
- **Clear backup strategies** for each database type

## **Development Benefits:**

- **Familiar MongoDB patterns** for CRUD operations
- **Leverage existing Graphiti** Neo4j functionality
- **Clear separation of concerns** between systems
- **Easier testing** with isolated components

**TOTAL ESTIMATED TIME: 8 weeks with 4-person team**
**ARCHITECTURE: Hybrid MongoDB (Metadata) + Neo4j (Graph)**
