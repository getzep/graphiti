# Graphiti Hybrid Architecture Project Status Tracker (Revised)

## 📋 Project Overview

This document tracks all modifications and additions to the Graphiti project to implement a **Hybrid MongoDB + Neo4j Architecture** for Splunk Security Query Generation. The project transforms Graphiti from a general-purpose temporal knowledge graph into a specialized security field relationship engine. This revised architecture follows the principles of a robust labeled property graph model, leveraging relationships to define hierarchy for improved flexibility and scalability.

### 🏷️ **General Project Rules**

The application uses **Neo4j nodes and relationships** to organize fields into a hierarchical cluster structure. This approach separates data (properties on nodes) from its structure (relationships between nodes), a fundamental principle of a scalable graph model.

#### **Core Node Labels**

- `Field`: Represents an individual security field (e.g., `src_ip`, `user_id`).
- `Cluster`: Represents a primary organizational/macro cluster (e.g., `linux_audit_batelco`).

#### **Core Relationship Types**

- `BELONGS_TO`: Connects a `Field` node to its member `Cluster` nodes.
- `FIELD_RELATES_TO`: Connects `Field` nodes relationships to other `Field` nodes within same `Cluster` only.

#### **Simplified Node & Relationship Example**

Instead of using compound labels, the hierarchy is defined by relationships:

```cypher
// Create the primary cluster node
CREATE (c:Cluster {
  name: 'linux_audit_batelco',
  organization: 'batelco',
  macro_type: 'linux_audit'
})

// Create a field and link it directly to the primary cluster
CREATE (f:Field {
  name: 'user_id'
  // Add additional properties here if needed
})
CREATE (f)-[:BELONGS_TO]->(c)
```

### 📐 **Data Organization Rules**

#### **1. Cross-Organization Field Replication**

The same field can exist across different primary clusters by having separate relationships:

- The `src_ip` field is a `Field` node.
- The `src_ip` `Field` node `BELONGS_TO` that `Cluster` named `linux_audit_batelco`.
- The `src_ip` `Field` node `FIELD_RELATES_TO` the `user_id` `Field` node within the same `Cluster`.

**Relationship Constraints:**

- **No Duplicate Fields per Cluster**: A `Cluster` should not contain duplicate `Field` nodes. For example, a `Field` named `user_id` should exist only once within the same `Cluster` named `linux_audit_batelco`.
- **Flexible Field-Cluster Membership**: A `Field` can exist on multiple `Cluster` nodes, But in isolated way like field named `user_id` can exist in `linux_audit_batelco` and `linux_audit_sico` clusters, but each cluster has its own isolated `Field` node and property.
- **Cluster Fields** each field will belong to one cluster , and the same field can exist in multiple clusters, but each cluster has its own isolated `Field` node and properties.
- **Field Relationships**: Fields can have relationships with other fields within the same cluster, but not across clusters. For example, `src_ip` can relate to `user_id` within the `linux_audit_batelco` cluster, but cannot relate to `user_id` in the `linux_audit_sico` cluster.

Each instance of the relationship is isolated and managed independently.

#### **2. Relationship-Based Isolation**

- **Primary clusters** are isolated by having no relationships between their `Cluster` nodes, maintaining data separation.
- **Query isolation** is achieved by traversing relationships from a specific `Cluster` node, ensuring data separation without relying on labels.
- **Field relationships** are defined by `BELONGS_TO` relationships, allowing fields to be grouped within their respective clusters without duplication.
- **Field relationship direction** in neo4j is typically unidirectional, meaning that a `Field` with highest event counts will have relationships to other `Field` nodes, that related with this field. For example field call it `src_ip` related to field `user_id` we will check which field has the highest event counts if `src_ip` has the highest event counts, then we will create a relationship from `src_ip` to `user_id` with the direction of `src_ip` -> `user_id`.

#### **4. Field Uniqueness Rules**

- **Primary Cluster Assignment**: A `Field` node can have one or more `[:BELONGS_TO]` relationships to `Cluster` node and filed like `src_ip` can belongs to multiple cluster but with differnt `Field` node in each cluster and different properties.
- **Label Requirements**: Every field MUST have the `Field` label.
- **No Duplication Within Cluster**: A `Field` node represents a unique field name. Duplication is avoided by ensuring only one `Field` node exists for a given `name`. If `src_ip` exists in a cluster, there is only one `Field` node for `src_ip`.
- **Unique Field Identification**: Each `Field` node has a unique `uuid`. This `uuid` is tied to the field's properties, not its cluster membership the relationships `[:BELONGS_TO]` will aslo have unique properties within `Cluster`.

#### **5. Relationship Integrity Constraints**

- **Relationship Type Definitions**:

  - **BELONGS_TO**: Establishes direct field membership within clusters. This relationship connects Field nodes directly to Cluster nodes, enabling straightforward field categorization within organizational boundaries. Each Field has exactly one BELONGS_TO relationship to its primary Cluster.

  - **FIELD_RELATES_TO**: Creates dynamic semantic relationships between Field nodes within the same Cluster. These relationships capture domain-specific connections like correlation, similarity, or derivation between security fields. All FIELD_RELATES_TO relationships must respect cluster isolation constraints, ensuring fields can only relate to other fields within their organizational boundary.

- **Field-to-Field Relationship Rules**:
  - A `Field` can have relationships with **1 to many** other `Field` nodes within the same `Cluster` only
  - **Cross-Cluster Field Relationships**: Not allowed - Fields cannot have direct relationships with Fields from different `Cluster` nodes
  - **Relationship Types**: FIELD_RELATES_TO with specific semantic names (e.g., 'CORRELATES_WITH', 'SIMILAR_TO', 'DERIVED_FROM')

---

### **Main Project Schemas**

#### Node Schema

Updated global Node class aligned with our project-specific requirements and relationship-based architecture.

```python
class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())
    validated_at: datetime = Field(default_factory=lambda: utc_now())             # Validation timestamp (default: utc_now())
    invalidated_at: datetime | None = Field(default=None)                         # Invalidation timestamp (default: None)
    last_updated: datetime = Field(default_factory=lambda: utc_now())             # Last update timestamp (default: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        """Delete node and all its relationships from Neo4j database"""
        result = await driver.execute_query(
            """
            MATCH (n:Field|Cluster {uuid: $uuid})
            DETACH DELETE n
            """,
            uuid=self.uuid,
        )

        logger.debug(f'Deleted Node: {self.uuid}')
        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Get node by UUID - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement get_by_uuid")

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Get multiple nodes by UUIDs - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement get_by_uuids")

    def update_temporal_fields(self):
        """Update last_updated timestamp and optionally validated_at"""
        self.last_updated = utc_now()
        self.validated_at = utc_now()
```

#### Field Schema

```python
class FieldNode(Node):
    """
    Specialized node for security audit fields.
    Relationships are used for clustering and hierarchy.
    Inherits temporal validation fields from Node base class.
    """

    # ==================== INHERITED FROM NODE CLASS ====================
    # uuid: str, name: str, labels: list[str] = ['Field'], created_at: datetime
    # validated_at: datetime, invalidated_at: datetime | None, last_updated: datetime

    # ==================== FIELD-SPECIFIC PROPERTIES ====================

    # Core Field Properties
    description: str                   # Field description for AI context and embedding
    examples: list[str] = Field(default_factory=list)  # Example values for validation
    data_type: str                     # Field data type (string, integer, etc.)
    count: int = 0                     # Field occurrences in events
    distinct_count: int = 0            # Distinct value count in events

    # Cluster Tracking (Relationship-based)
    primary_cluster_id: str          # Primary cluster UUID

    # Semantic Search Support
    embedding: list[float] | None = None  # Vector for semantic search on description

    def __init__(self, **data):
        # Set default label for Field nodes
        if 'labels' not in data:
            data['labels'] = ['Field']
        super().__init__(**data)

    # ==================== FIELD-SPECIFIC METHODS ====================

    async def save(self, driver: GraphDriver):
        """Save Field node to Neo4j database with relationship-based approach"""
        # Update temporal fields before saving
        self.update_temporal_fields()

        field_data = {
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'examples': self.examples,
            'data_type': self.data_type,
            'count': self.count,
            'distinct_count': self.distinct_count,
            'primary_cluster_id': self.primary_cluster_id,
            'embedding': self.embedding,
            'validated_at': self.validated_at,
            'invalidated_at': self.invalidated_at,
            'last_updated': self.last_updated,
            'created_at': self.created_at,
        }

        result = await driver.execute_query(
            """
            MERGE (f:Field {uuid: $uuid})
            SET f += $field_data
            RETURN f
            """,
            uuid=self.uuid,
            field_data=field_data,
        )
        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve Field node by UUID"""
        records, _, _ = await driver.execute_query(
            """
            MATCH (f:Field {uuid: $uuid})
            RETURN f
            """,
            uuid=uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(uuid)

        field_data = records[0]['f']
        return cls(**field_data)

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple Field nodes by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            """
            MATCH (f:Field)
            WHERE f.uuid IN $uuids
            RETURN f
            """,
            uuids=uuids,
            routing_='r',
        )

        return [cls(**record['f']) for record in records]

    # ==================== CLUSTER INFORMATION (RELATIONSHIPS) ====================
    # Hierarchy is defined by relationships, not labels or properties on this node.
```

#### Cluster/Sub-Cluster Schemas

```python
class ClusterNode(Node):
    """
    Primary cluster node for organizational isolation and macro-level grouping.
    Represents organizational/macro clusters like 'linux_audit_batelco'.
    Inherits temporal validation fields from Node base class.
    """

    # ==================== INHERITED FROM NODE CLASS ====================
    # uuid: str, name: str, labels: list[str] = ['Cluster'], created_at: datetime
    # validated_at: datetime, invalidated_at: datetime | None, last_updated: datetime

    # ==================== CLUSTER-SPECIFIC PROPERTIES ====================

    # Core Cluster Properties
    organization: str                  # Organization identifier (e.g., 'batelco', 'sico')
    macro_name: str                    # Refers to the macro name with this organization

    def __init__(self, **data):
        # Set default label for Cluster nodes
        if 'labels' not in data:
            data['labels'] = ['Cluster']
        super().__init__(**data)

    async def save(self, driver: GraphDriver):
        """Save Cluster node to Neo4j database"""
        # Update temporal fields before saving
        self.update_temporal_fields()

        cluster_data = {
            'uuid': self.uuid,
            'name': self.name,
            'organization': self.organization,
            'macro_name': self.macro_name,
            'validated_at': self.validated_at,
            'invalidated_at': self.invalidated_at,
            'last_updated': self.last_updated,
            'created_at': self.created_at,
        }

        result = await driver.execute_query(
            """
            MERGE (c:Cluster {uuid: $uuid})
            SET c += $cluster_data
            RETURN c
            """,
            uuid=self.uuid,
            cluster_data=cluster_data,
        )
        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve Cluster node by UUID"""
        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Cluster {uuid: $uuid})
            RETURN c
            """,
            uuid=uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(uuid)

        cluster_data = records[0]['c']
        return cls(**cluster_data)

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple Cluster nodes by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Cluster)
            WHERE c.uuid IN $uuids
            RETURN c
            """,
            uuids=uuids,
            routing_='r',
        )

        return [cls(**record['c']) for record in records]

    # ==================== RELATIONSHIP INFORMATION ====================
    # Fields connected directly via BELONGS_TO relationships, not properties on this node.
```

#### Relationship Schemas

Complete Edge base class implementation with abstract methods for type safety and consistent implementation across all relationship types.

```python
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
from graphiti_core.utils.utc_helpers import utc_now
from graphiti_core.errors import EdgeNotFoundError

class Edge(BaseModel, ABC):
    """
    Abstract base class for all edge/relationship types in the graph.
    Enforces consistent implementation across all relationship schemas.
    Provides type safety through abstract methods that must be implemented by subclasses.
    """
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    source_node_uuid: str              # UUID of the source node
    target_node_uuid: str              # UUID of the target node
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver):
        """Save edge to Neo4j database - must be implemented by each edge type"""
        ...

    async def delete(self, driver: GraphDriver):
        """Delete edge from Neo4j database using generic relationship deletion"""
        result = await driver.execute_query(
            """
            MATCH ()-[e {uuid: $uuid}]->()
            DELETE e
            RETURN count(e) as deleted_count
            """,
            uuid=self.uuid,
        )

        deleted_count = result.records[0]['deleted_count'] if result.records else 0
        if deleted_count == 0:
            raise EdgeNotFoundError(f"Edge with UUID {self.uuid} not found")

        logger.debug(f'Deleted Edge: {self.uuid}')
        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.uuid == other.uuid
        return False

    @classmethod
    @abstractmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve edge by UUID - must be implemented by each edge type for type safety"""
        ...

    @classmethod
    @abstractmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple edges by UUIDs - must be implemented by each edge type"""
        ...

    def update_temporal_fields(self):
        """Update temporal tracking fields"""
        # For edges, we typically don't update created_at, but subclasses can override
        pass
```

```python
class BelongsToEdge(Edge):
    """
    Edge representing Field belonging to Cluster relationship.
    Connects Field nodes directly to their organizational Cluster.
    Enforces cluster isolation constraints.
    """

    # ==================== INHERITED FROM EDGE CLASS ====================
    # uuid: str, source_node_uuid: str, target_node_uuid: str, created_at: datetime

    # ==================== EDGE-SPECIFIC PROPERTIES ====================
    cluster_partition_id: str          # UUID of the Cluster for isolation tracking

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def save(self, driver: GraphDriver):
        """Save BELONGS_TO relationship to Neo4j database"""
        # Create the BELONGS_TO relationship directly from Field to Cluster
        result = await driver.execute_query(
            """
            MATCH (f:Field {uuid: $source_uuid})
            MATCH (c:Cluster {uuid: $target_uuid})
            CREATE (f)-[r:BELONGS_TO {
                uuid: $edge_uuid,
                created_at: $created_at,
                cluster_partition_id: $cluster_partition_id
            }]->(c)
            RETURN r
            """,
            source_uuid=self.source_node_uuid,
            target_uuid=self.target_node_uuid,
            edge_uuid=self.uuid,
            created_at=self.created_at,
            cluster_partition_id=self.cluster_partition_id,
        )
        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve BELONGS_TO edge by UUID and return full Edge object"""
        records, _, _ = await driver.execute_query(
            """
            MATCH (f:Field)-[r:BELONGS_TO {uuid: $uuid}]->(c:Cluster)
            RETURN r, f.uuid as source_uuid, c.uuid as target_uuid
            """,
            uuid=uuid,
            routing_='r',
        )

        if not records:
            raise EdgeNotFoundError(f"BelongsToEdge with UUID {uuid} not found")

        edge_data = records[0]['r']
        return cls(
            uuid=edge_data['uuid'],
            source_node_uuid=records[0]['source_uuid'],
            target_node_uuid=records[0]['target_uuid'],
            created_at=edge_data['created_at'],
            cluster_partition_id=edge_data['cluster_partition_id']
        )

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple BELONGS_TO edges by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            """
            MATCH (f:Field)-[r:BELONGS_TO]->(c:Cluster)
            WHERE r.uuid IN $uuids
            RETURN r, f.uuid as source_uuid, c.uuid as target_uuid
            """,
            uuids=uuids,
            routing_='r',
        )

        return [
            cls(
                uuid=record['r']['uuid'],
                source_node_uuid=record['source_uuid'],
                target_node_uuid=record['target_uuid'],
                created_at=record['r']['created_at'],
                cluster_partition_id=record['r']['cluster_partition_id']
            )
            for record in records
        ]

    # ==================== CONSTRAINT VALIDATION ====================
    # Field-Cluster relationships must respect organizational isolation constraints
```

```python
class FieldRelationshipEdge(Edge):
    """
    Edge representing dynamic semantic relationships between Field nodes within the same Cluster.
    Captures domain-specific connections like correlation, similarity, or derivation between security fields.
    All FIELD_RELATES_TO relationships must respect cluster isolation constraints.
    """

    # ==================== INHERITED FROM EDGE CLASS ====================
    # uuid: str, source_node_uuid: str, target_node_uuid: str, created_at: datetime

    # ==================== FIELD RELATIONSHIP PROPERTIES ====================
    name: str                          # Relationship type (e.g., 'CORRELATES_WITH', 'SIMILAR_TO', 'DERIVED_FROM')
    description: str                   # Why this relationship exists (semantic context)
    confidence: float = 1.0            # Confidence score for the relationship (0.0 to 1.0)
    cluster_partition_id: str          # UUID of the Cluster (both fields must belong here)
    relationship_type: str = "FIELD_RELATES_TO"  # Base relationship type for Neo4j

    # Semantic Search Support
    description_embedding: list[float] | None = Field(default=None, description='embedding of description for semantic search')

    # Temporal Validation Fields
    valid_at: datetime | None = Field(default=None, description='when the relationship became valid')
    invalid_at: datetime | None = Field(default=None, description='when the relationship stopped being valid')

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def save(self, driver: GraphDriver):
        """Save Field-to-Field relationship to Neo4j database with cluster isolation validation"""
        # Validate that both fields belong to the same cluster
        validation_result = await driver.execute_query(
            """
            MATCH (f1:Field {uuid: $source_uuid})-[:BELONGS_TO]->(c:Cluster {uuid: $cluster_id})
            MATCH (f2:Field {uuid: $target_uuid})-[:BELONGS_TO]->(c:Cluster {uuid: $cluster_id})
            RETURN count(*) as valid_relationship
            """,
            source_uuid=self.source_node_uuid,
            target_uuid=self.target_node_uuid,
            cluster_id=self.cluster_partition_id,
        )

        if not validation_result.records or validation_result.records[0]['valid_relationship'] == 0:
            raise ValueError("Field-to-Field relationships must be within the same Cluster")

        # Create the FIELD_RELATES_TO relationship
        result = await driver.execute_query(
            """
            MATCH (f1:Field {uuid: $source_uuid})
            MATCH (f2:Field {uuid: $target_uuid})
            CREATE (f1)-[r:FIELD_RELATES_TO {
                uuid: $edge_uuid,
                name: $name,
                description: $description,
                description_embedding: $description_embedding,
                confidence: $confidence,
                cluster_partition_id: $cluster_partition_id,
                created_at: $created_at,
                valid_at: $valid_at,
                invalid_at: $invalid_at
            }]->(f2)
            RETURN r
            """,
            source_uuid=self.source_node_uuid,
            target_uuid=self.target_node_uuid,
            edge_uuid=self.uuid,
            name=self.name,
            description=self.description,
            description_embedding=self.description_embedding,
            confidence=self.confidence,
            cluster_partition_id=self.cluster_partition_id,
            created_at=self.created_at,
            valid_at=self.valid_at,
            invalid_at=self.invalid_at,
        )
        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve Field relationship edge by UUID and return full Edge object"""
        records, _, _ = await driver.execute_query(
            """
            MATCH (f1:Field)-[r:FIELD_RELATES_TO {uuid: $uuid}]->(f2:Field)
            RETURN r, f1.uuid as source_uuid, f2.uuid as target_uuid
            """,
            uuid=uuid,
            routing_='r',
        )

        if not records:
            raise EdgeNotFoundError(f"FieldRelationshipEdge with UUID {uuid} not found")

        edge_data = records[0]['r']
        return cls(
            uuid=edge_data['uuid'],
            name=edge_data['name'],
            description=edge_data['description'],
            description_embedding=edge_data['description_embedding'],
            confidence=edge_data['confidence'],
            cluster_partition_id=edge_data['cluster_partition_id'],
            source_node_uuid=records[0]['source_uuid'],
            target_node_uuid=records[0]['target_uuid'],
            created_at=edge_data['created_at'],
            valid_at=edge_data['valid_at'],
            invalid_at=edge_data['invalid_at'],
        )

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple Field relationship edges by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            """
            MATCH (f1:Field)-[r:FIELD_RELATES_TO]->(f2:Field)
            WHERE r.uuid IN $uuids
            RETURN r, f1.uuid as source_uuid, f2.uuid as target_uuid
            """,
            uuids=uuids,
            routing_='r',
        )

        return [
            cls(
                uuid=record['r']['uuid'],
                name=record['r']['name'],
                description=record['r']['description'],
                description_embedding=record['r']['description_embedding'],
                confidence=record['r']['confidence'],
                cluster_partition_id=record['r']['cluster_partition_id'],
                source_node_uuid=record['source_uuid'],
                target_node_uuid=record['target_uuid'],
                created_at=record['r']['created_at'],
                valid_at=record['r']['valid_at'],
                invalid_at=record['r']['invalid_at'],
            )
            for record in records
        ]

    # ==================== CONSTRAINT VALIDATION ====================
    # Field-to-Field relationships must respect cluster isolation constraints
    # All fields in the relationship must belong to the same Cluster
```

---

## 📊 **Database Query Implementation**

### **Security Graph Queries Analysis**

**File**: `graphiti_core/graph_queries_security.py` (Neo4j Only - Initial Implementation)

Based on analysis of the current Graphiti `graph_queries.py` file, our security field relationship system requires specialized database query functions to support `Field`, `Cluster`, and `SubCluster` node relationships with proper constraint validation.

#### **Core Query Architecture Requirements**

**Key Differences from Current Graphiti Graph Queries**:

- **Current**: `group_id` for graph partitioning → **Our System**: `cluster_partition_id` for organizational isolation
- **Current**: General-purpose Entity indexes → **Our System**: Specialized Field-Cluster hierarchy indexes
- **Current**: `Entity`, `Episodic`, `Community` node types → **Our System**: `Field`, `Cluster` node types
- **Current**: Temporal knowledge graph operations → **Our System**: Security field organization and correlation queries

#### **Essential Database Query Functions Analysis**

| Function/Method                            | Current Graphiti Purpose                                  | Our Security System Purpose                                                                         | Implementation Priority |
| ------------------------------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------- |
| **Index Creation Functions**               |
| `get_security_range_indices()`             | Adapted from `get_range_indices()`                        | Create performance indexes for Field, Cluster nodes and BELONGS_TO, FIELD_RELATES_TO relationships  | **HIGH**                |
| `get_security_fulltext_indices()`          | Adapted from `get_fulltext_indices()`                     | Create fulltext search indexes for field descriptions, cluster names, and relationship descriptions | **HIGH**                |
| **Query Generation Functions**             |
| `get_security_nodes_query()`               | Adapted from `get_nodes_query()`                          | Search Field, Cluster nodes using fulltext search                                                   | **MEDIUM**              |
| `get_security_relationships_query()`       | Adapted from `get_relationships_query()`                  | Search security relationships using fulltext search                                                 | **MEDIUM**              |
| `get_security_vector_cosine_query()`       | Adapted from `get_vector_cosine_func_query()`             | Field relationship embedding similarity calculations                                                | **MEDIUM**              |
| **Bulk Operations Functions**              |
| `get_field_save_bulk_query()`              | Adapted from `get_entity_node_save_bulk_query()`          | Save multiple Field nodes efficiently with embedding support                                        | **HIGH**                |
| `get_cluster_save_bulk_query()`            | New function based on existing pattern                    | Save multiple Cluster nodes efficiently                                                             | **MEDIUM**              |
| `get_belongs_to_save_bulk_query()`         | New function based on `get_entity_edge_save_bulk_query()` | Save multiple BELONGS_TO relationships efficiently with constraint validation                       | **HIGH**                |
| `get_field_relationship_save_bulk_query()` | Adapted from `get_entity_edge_save_bulk_query()`          | Save multiple FIELD_RELATES_TO relationships with cluster isolation validation                      | **HIGH**                |

#### **Critical Database Constraint Functions**

| Constraint Function                  | Purpose                                                               | Implementation                                                          | Priority |
| ------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------- | -------- |
| `validate_cluster_isolation_query()` | Ensure Fields can only relate to other Fields within the same Cluster | Check cluster membership before creating FIELD_RELATES_TO relationships | **HIGH** |
| `validate_field_membership_query()`  | Ensure Fields belong to valid Clusters                                | Validate Field-Cluster relationship before creating BELONGS_TO          | **HIGH** |

#### **Query Pattern Adaptations**

**Current Graphiti Query Patterns** → **Our Security System Adaptations**:

```cypher
-- Current: Entity relationship search
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.group_id IN $group_ids

-- Our System: Field relationship search with cluster isolation
MATCH (f1:Field)-[e:FIELD_RELATES_TO]->(f2:Field)
WHERE e.cluster_partition_id = $cluster_id

-- Current: Community membership
MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity|Community)

-- Our System: Cluster hierarchy traversal
MATCH (c:Cluster)-[contains:CONTAINS]->(sc:SubCluster)-[belongs:BELONGS_TO]-(f:Field)
WHERE c.uuid = $cluster_uuid
```

#### **Performance Optimization Requirements**

Based on current Graphiti patterns, our system needs:

1. **Relationship Indexes**: Similar to current entity indexes but for security relationships
2. **Batch Operations**: Adapt `get_by_group_ids()` pattern for `cluster_partition_id` filtering
3. **Cursor-Based Pagination**: Implement `uuid_cursor` pattern for large result sets
4. **Embedding Integration**: Adapt `create_entity_edge_embeddings()` for Field relationship descriptions

#### **Implementation Sequence**

1. **Phase 1**: Index creation functions for security node and relationship types
2. **Phase 2**: Basic query generation functions for node and relationship search
3. **Phase 3**: Bulk operations for efficient data loading and updates
4. **Phase 4**: Constraint validation queries for relationship integrity
5. **Phase 5**: Performance optimization and embedding integration

This analysis ensures our new security graph query system maintains the robustness and performance characteristics of the current Graphiti implementation while providing the specialized functionality required for security field relationship management.

---

## 🎯 Architecture Summary

**New Security Edge File**: `graphiti_core/security_edges.py` (Neo4j Only - Initial Implementation)

**Key Differences from Current Graphiti Edges**:

- **Current**: `group_id` for graph partitioning → **Our System**: `cluster_partition_id` for organizational isolation
- **Current**: General-purpose Entity relationships → **Our System**: Specialized Field-Cluster hierarchy
- **Current**: `EntityEdge`, `EpisodicEdge`, `CommunityEdge` → **Our System**: `BelongsToEdge`, `FieldRelationshipEdge`
- **Current**: Temporal knowledge graph → **Our System**: Security field organization and correlation

##### **Essential Edge Functions Analysis**

| Function/Method                               | Current Graphiti Purpose                                                                                   | Our Security System Purpose                                                                                                | Implementation Priority |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| **Abstract Base Edge Class**                  |
| `Edge.__init__()`                             | Initialize base edge properties (`uuid`, `group_id`, `source_node_uuid`, `target_node_uuid`, `created_at`) | Initialize security edge properties (`uuid`, `cluster_partition_id`, `source_node_uuid`, `target_node_uuid`, `created_at`) | **HIGH**                |
| `Edge.save()`                                 | Abstract method - must be implemented by subclasses                                                        | Abstract method - ensures consistent save implementation across `BelongsToEdge`, `FieldRelationshipEdge`                   | **HIGH**                |
| `Edge.delete()`                               | Delete any edge type using generic relationship deletion                                                   | Delete security relationships (BELONGS_TO, FIELD_RELATES_TO) with constraint validation                                    | **HIGH**                |
| `Edge.get_by_uuid()`                          | Abstract method - retrieve edge by UUID                                                                    | Abstract method - type-safe retrieval for security edges                                                                   | **HIGH**                |
| `Edge.get_by_uuids()`                         | Abstract method - batch retrieval by UUIDs                                                                 | Abstract method - batch retrieval for performance optimization                                                             | **MEDIUM**              |
| **BelongsToEdge Implementation**              |
| `BelongsToEdge.save()`                        | N/A (new relationship type)                                                                                | Create `Field` → `Cluster` BELONGS_TO relationship with cluster isolation validation                                       | **HIGH**                |
| `BelongsToEdge.get_by_uuid()`                 | N/A (new relationship type)                                                                                | Retrieve specific Field-Cluster membership relationship                                                                    | **HIGH**                |
| `BelongsToEdge.get_by_uuids()`                | N/A (new relationship type)                                                                                | Batch retrieval of multiple Field-Cluster relationships                                                                    | **MEDIUM**              |
| `BelongsToEdge.get_by_group_ids()`            | N/A (adapt from EntityEdge pattern)                                                                        | Find all Field-Cluster relationships within specific organizational boundaries                                             | **MEDIUM**              |
| **FieldRelationshipEdge Implementation**      |
| `FieldRelationshipEdge.save()`                | Similar to `EntityEdge.save()` but specialized                                                             | Create `Field` → `Field` semantic relationships (CORRELATES_WITH, SIMILAR_TO, DERIVED_FROM) with cluster isolation         | **HIGH**                |
| `FieldRelationshipEdge.get_by_uuid()`         | Adapted from `EntityEdge.get_by_uuid()`                                                                    | Retrieve specific Field-to-Field relationship with semantic properties                                                     | **HIGH**                |
| `FieldRelationshipEdge.get_by_uuids()`        | Adapted from `EntityEdge.get_by_uuids()`                                                                   | Batch retrieval of Field relationships for correlation analysis                                                            | **HIGH**                |
| `FieldRelationshipEdge.get_by_group_ids()`    | Similar to `EntityEdge.get_by_group_ids()`                                                                 | Find Field relationships within specific clusters for security analysis                                                    | **HIGH**                |
| `FieldRelationshipEdge.get_by_node_uuid()`    | Similar to `EntityEdge.get_by_node_uuid()`                                                                 | Find all relationships for a specific Field (correlation discovery)                                                        | **HIGH**                |
| `FieldRelationshipEdge.load_fact_embedding()` | Adapted from `EntityEdge.load_fact_embedding()`                                                            | Load semantic embeddings for Field relationship descriptions                                                               | **MEDIUM**              |
| **Helper Functions**                          |
| `get_belongs_to_edge_from_record()`           | N/A (new function based on existing pattern)                                                               | Convert Neo4j record to BelongsToEdge object                                                                               | **HIGH**                |
| `get_field_relationship_edge_from_record()`   | Adapted from `get_entity_edge_from_record()`                                                               | Convert Neo4j record to FieldRelationshipEdge object with semantic properties                                              | **HIGH**                |
| `create_field_relationship_embeddings()`      | Adapted from `create_entity_edge_embeddings()`                                                             | Generate embeddings for Field relationship descriptions for semantic search                                                | **MEDIUM**              |

##### **Critical Constraint Validation Functions**

| Constraint Function            | Purpose                                                               | Implementation                                                          | Priority |
| ------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------- | -------- |
| `validate_cluster_isolation()` | Ensure Fields can only relate to other Fields within the same Cluster | Check cluster membership before creating FIELD_RELATES_TO relationships | **HIGH** |
| `validate_field_membership()`  | Ensure Fields belong to valid Clusters                                | Validate Field-Cluster relationship before creating BELONGS_TO          | **HIGH** |

##### **Query Pattern Adaptations**

**Current Graphiti Query Patterns** → **Our Security System Adaptations**:

```cypher
-- Current: Entity relationship search
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.group_id IN $group_ids

-- Our System: Field relationship search with cluster isolation
MATCH (f1:Field)-[e:FIELD_RELATES_TO]->(f2:Field)
WHERE e.cluster_partition_id = $cluster_id

-- Current: Community membership
MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity|Community)

-- Our System: Cluster field membership
MATCH (c:Cluster)<-[belongs:BELONGS_TO]-(f:Field)
WHERE c.uuid = $cluster_uuid
```

##### **Performance Optimization Requirements**

Based on current Graphiti patterns, our system needs:

1. **Relationship Indexes**: Similar to current entity indexes but for security relationships
2. **Batch Operations**: Adapt `get_by_group_ids()` pattern for `cluster_partition_id` filtering
3. **Cursor-Based Pagination**: Implement `uuid_cursor` pattern for large result sets
4. **Embedding Integration**: Adapt `create_entity_edge_embeddings()` for Field relationship descriptions

##### **Implementation Sequence**

1. **Phase 1**: Abstract `Edge` base class with security-specific properties
2. **Phase 2**: `BelongsToEdge` for basic Field-Cluster membership
3. **Phase 3**: `FieldRelationshipEdge` for semantic Field correlations
4. **Phase 4**: Constraint validation and helper functions
5. **Phase 5**: Performance optimization and batch operations

This analysis ensures our new security edge system maintains the robustness and performance characteristics of the current Graphiti implementation while providing the specialized functionality required for security field relationship management.

---

## 🔍 **Search System Integration & Migration Strategy**

### **Search System Analysis & Compatibility Assessment**

Based on comprehensive analysis of the Graphiti search folder (`graphiti_core/search/`), the existing search system is **exceptionally well-suited** for our security field relationship requirements and should be **extended rather than replaced**.

#### **Current Graphiti Search Architecture**

##### **Core Components:**

- **search.py** - Main search orchestrator with unified search interface
- **search_config.py** - Configuration system with search methods, rerankers, and result types
- **search_config_recipes.py** - Pre-configured search patterns (hybrid, cross-encoder, etc.)
- **search_utils.py** - Core search implementation functions (BM25, similarity, BFS)
- **search_filters.py** - Filter system for nodes, edges, and temporal constraints
- **search_helpers.py** - Result formatting and context generation

##### **Available Search Methods:**

- **BM25 (Full-text)**: Keyword-based search using Lucene syntax
- **Cosine Similarity**: Vector embedding-based semantic search
- **BFS (Breadth-First Search)**: Graph traversal-based relationship discovery

##### **Reranking Strategies:**

- **RRF**: Reciprocal Rank Fusion for combining multiple search results
- **MMR**: Maximal Marginal Relevance for diversity
- **Cross-Encoder**: AI-powered relevance scoring
- **Node Distance**: Proximity-based reranking
- **Episode Mentions**: Temporal frequency-based ranking

#### **Compatibility Assessment for Security System**

##### **✅ HIGHLY COMPATIBLE Components:**

1. **Search Configuration System**

   - The `SearchConfig` architecture is **perfectly adaptable**
   - Can easily extend to support `FieldNode`, `ClusterNode`
   - Filter system can be extended for security-specific constraints

2. **Core Search Methods**

   - **BM25**: Excellent for field description and cluster name searches
   - **Cosine Similarity**: Perfect for field relationship discovery via embeddings
   - **BFS**: Ideal for traversing `BELONGS_TO`, `FIELD_RELATES_TO` relationships

3. **Reranking Strategies**
   - **RRF**: Great for combining field name + description search results
   - **Cross-Encoder**: Excellent for semantic field relationship ranking
   - **Custom Distance**: Can implement cluster isolation-based ranking

##### **🔧 ADAPTATION REQUIRED:**

1. **Node Type Extensions** (Medium Effort):

   ```python
   # Current: EntityNode, CommunityNode, EpisodicNode
   # Need: FieldNode, ClusterNode
   ```

2. **Edge Type Extensions** (Medium Effort):

   ```python
   # Current: EntityEdge (RELATES_TO)
   # Need: BelongsToEdge, FieldRelationshipEdge
   ```

3. **Group ID → Cluster Partition ID** (Low Effort):
   ```python
   # Current: group_ids filtering
   # Need: cluster_partition_id filtering for organizational isolation
   ```

#### **🚀 RECOMMENDATION: Use Existing System as Service**

**✅ Strong Recommendation**: **Extend the existing search system** rather than building a separate one.

##### **Why This Approach is Optimal:**

1. **Architecture Alignment**:

   - Current search patterns **perfectly match** security field relationship needs
   - The hybrid search approach (BM25 + Semantic + Graph Traversal) is **ideal** for security field discovery
   - Existing reranking strategies provide **sophisticated relevance scoring**

2. **Implementation Efficiency**:

   - **80% of functionality already exists** and is production-tested
   - Only need to extend node/edge types and add security-specific filters
   - Can leverage **proven search recipes** for immediate functionality

3. **Advanced Features Available**:
   - **Cross-encoder reranking** for intelligent field relationship scoring
   - **MMR diversity** to avoid duplicate field results
   - **BFS graph traversal** for cluster hierarchy exploration
   - **Temporal filtering** for field validation states

### **📋 Search System Migration Implementation Plan**

#### **Phase 1: Core Extensions** (High Priority - 16-24 hours)

```python


## Security Prompt Implementation Status
*Date: December 19, 2024*

### Implementation Progress ✅ COMPLETE
**Status: All 6 major security prompt files successfully implemented**

#### Completed Security Prompt Files:
1. **extract_security_fields.py** ✅
   - 6 specialized functions for security field extraction
   - ExtractedSecurityField models with hierarchical context
   - Cluster isolation validation throughout extraction
   - Functions: extract_field_documentation, extract_field_specifications, extract_field_hierarchy, validate_field_extraction, classify_security_fields, extract_field_properties

2. **extract_field_relationships.py** ✅
   - 4 relationship extraction functions for BELONGS_TO, CONTAINS, FIELD_RELATES_TO
   - SecurityFieldRelationship models with semantic types
   - Cluster isolation constraints and validation
   - Functions: extract_belongs_to_relationships, extract_contains_relationships, extract_field_correlations, validate_cluster_isolation

3. **dedupe_security_fields.py** ✅
   - 3 deduplication functions for single fields, field lists, and batch processing
   - SecurityFieldDuplicate and SecurityFieldResolutions models
   - Cross-batch analysis with UUID preservation and conflict resolution
   - Functions: dedupe_single_field, dedupe_field_list, dedupe_fields_batch

4. **summarize_security_fields.py** ✅
   - 4 summarization functions for extraction results and organizational insights
   - SecurityFieldSummary and SecurityFieldInsights models
   - Coverage analysis and relationship pattern recognition
   - Functions: summarize_field_extraction, summarize_field_relationships, summarize_cluster_organization, analyze_field_coverage

5. **validate_field_relationships.py** ✅
   - 4 validation functions for relationship integrity and cluster compliance
   - ValidationResult and ValidationReport models
   - Comprehensive validation of cluster isolation, hierarchy consistency, semantic relationships
   - Functions: validate_cluster_isolation, validate_hierarchy_consistency, validate_semantic_relationships, validate_field_integrity

6. **models.py** ✅
   - Basic infrastructure with Message, PromptVersion, PromptFunction definitions
   - Foundation for all security prompt files
   - Proper typing and protocol definitions

#### Infrastructure Complete:
- **security_prompt/** folder created ✅
- **__init__.py** with proper exports ✅
- **Complete file structure** with all 6 transformation files ✅
- **Consistent architecture** across all prompt files ✅

#### Key Implementation Features:
- **Cluster Isolation**: All functions respect organizational boundaries
- **Hierarchical Context**: Field → SubCluster → Cluster relationships maintained
- **Semantic Validation**: Domain-aware relationship validation
- **Batch Processing**: Support for incremental field extraction
- **Quality Metrics**: Comprehensive validation and quality assessment
- **Deduplication Logic**: Cross-cluster and cross-batch duplicate resolution
- **Relationship Integrity**: BELONGS_TO, CONTAINS, FIELD_RELATES_TO validation

#### Transformation Analysis Summary:
- **Total Functions Implemented**: 21 specialized security prompt functions
- **Original Conversational Functions**: 6 functions analyzed for transformation
- **Transformation Complexity**: 70-95% complete redesign achieved
- **Security Domain Focus**: Successfully transformed from conversational entities to security field management
- **Organizational Compliance**: All functions implement proper cluster isolation and governance

### Next Phase: Integration and Testing
1. Integration layer for security prompt system
2. Testing with sample security field documentation
3. Performance optimization and embedding integration
4. Documentation updates for usage patterns
# 1. Extend SearchConfig for security nodes
class SecurityNodeSearchConfig(BaseModel):
    search_methods: list[SecurityNodeSearchMethod]  # field_search, cluster_search
    reranker: SecurityNodeReranker = SecurityNodeReranker.rrf

# 2. Add security-specific search methods
class SecuritySearchMethod(Enum):
    field_description_search = 'field_description_search'
    cluster_hierarchy_search = 'cluster_hierarchy_search'
    relationship_traversal = 'relationship_traversal'

# 3. Create security search filters
class SecuritySearchFilters(BaseModel):
    cluster_partition_ids: list[str] | None = None
    field_types: list[str] | None = None  # data_type filtering
    relationship_types: list[str] | None = None  # BELONGS_TO, FIELD_RELATES_TO
```

#### **Phase 2: Search Functions** (Medium Priority - 24-32 hours)

```python
# Adapt existing patterns for security nodes
async def field_fulltext_search(driver, query, filters, cluster_partition_ids, limit)
async def field_similarity_search(driver, query_vector, filters, cluster_partition_ids, limit)
async def security_relationship_search(driver, field_uuid, relationship_types, limit)
async def cluster_field_traversal(driver, cluster_uuid, max_depth, limit)
```

#### **Phase 3: Security-Specific Recipes** (Low Priority - 8-12 hours)

```python
# Pre-configured search patterns for security use cases
FIELD_RELATIONSHIP_DISCOVERY = SearchConfig(...)
CLUSTER_FIELD_EXPLORATION = SearchConfig(...)
SECURITY_CORRELATION_SEARCH = SearchConfig(...)
```

#### **🎯 Benefits of This Approach**

1. **Immediate Functionality**: Get sophisticated search capabilities from day one
2. **Production Ready**: Leverage battle-tested search algorithms and optimization
3. **Extensible**: Easy to add security-specific features without breaking existing functionality
4. **Performance**: Inherits existing indexing, pagination, and optimization strategies
5. **Maintenance**: Single codebase for all search functionality

#### **💡 Implementation Strategy**

1. **Start with Existing Recipes**: Use `NODE_HYBRID_SEARCH_RRF` as base pattern for field search
2. **Extend Gradually**: Add security node types one by one (Field → Cluster)
3. **Leverage Filters**: Use existing filter system for cluster isolation constraints
4. **Custom Rerankers**: Add security-specific ranking for field correlation scoring

#### **🎯 Expected Outcomes**

With **moderate extensions** (estimated 48-68 hours), you can have a **production-ready, sophisticated search service** that provides:

- **Multi-modal search** (keyword + semantic + graph traversal)
- **Advanced reranking** with AI-powered relevance scoring
- **Cluster isolation** through adapted filtering
- **Relationship discovery** via graph traversal
- **Performance optimization** through proven indexing strategies

This approach provides **maximum value with minimum risk** and leverages the substantial investment already made in the Graphiti search architecture.

---

# Implementation: Prompts System Transformation Analysis

## 📋 **Current Prompts System Analysis**

The existing Graphiti prompts system is designed for **episodic conversational knowledge extraction** and requires **comprehensive transformation (70% complete redesign)** to support security field relationship management.

### **Current Prompts Architecture Overview**

**Location**: `graphiti_core/prompts/`
**Total Files**: 12 prompt files focused on conversational entity extraction
**Core Functions**: Entity extraction from conversations, fact triple relationships, temporal episode processing

#### **Key Files Requiring Complete Transformation:**

| Current File          | Lines | Purpose                                                                 | Transformation Required          |
| --------------------- | ----- | ----------------------------------------------------------------------- | -------------------------------- |
| `extract_nodes.py`    | 279   | Extract entities from conversations with speaker identification         | **95% Complete Redesign**        |
| `extract_edges.py`    | 195   | Extract fact triples between conversational entities                    | **90% Complete Redesign**        |
| `dedupe_nodes.py`     | 209   | Deduplicate entities based on speaker identity and conversation context | **70% Significant Modification** |
| `summarize_nodes.py`  | 134   | Summarize entity characteristics from conversation history              | **85% Complete Redesign**        |
| `invalidate_edges.py` | ~150  | Detect contradictory relationships in conversational context            | **60% Moderate Modification**    |
| `dedupe_edges.py`     | ~120  | Deduplicate conversational relationship facts                           | **70% Significant Modification** |

### **Critical Transformation Requirements**

#### **1. Context Architecture Transformation**

**CURRENT (Conversational Context):**

```python
context = {
    'episode_content': conversation_messages,
    'previous_episodes': historical_conversations,
    'entity_types': conversational_entity_types,
    'extracted_entities': speaker_and_entities,
    'custom_prompt': additional_instructions
}
```

**REQUIRED (Security Field Context):**

```python
security_context = {
    'field_documentation': security_field_specifications,
    'field_relationships': existing_field_connections,
    'cluster_hierarchy': cluster_subcluster_structure,
    'security_field_types': ['Field', 'Cluster', 'SubCluster'],
    'relationship_types': ['BELONGS_TO', 'CONTAINS', 'FIELD_RELATES_TO'],
    'cluster_partition_id': organizational_boundary_id,
    'cluster_rules': organizational_isolation_rules,
    'extracted_security_fields': current_extraction_batch,
    'custom_field_prompt': security_specific_instructions
}
```

#### **2. Pydantic Models Complete Replacement**

**CURRENT (Conversational Models):**

```python
class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(description='ID of the classified entity type')

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity]
```

**REQUIRED (Security Field Models):**

```python
class ExtractedSecurityField(BaseModel):
    name: str = Field(..., description='Security field name (e.g., src_ip, user_id)')
    field_type: str = Field(..., description='Type: Field or Cluster')
    description: str = Field(..., description='Field description for AI context')
    data_type: str = Field(..., description='Data type (string, integer, ip_address)')
    examples: list[str] = Field(default_factory=list, description='Example values')
    cluster_context: str = Field(..., description='Organizational cluster membership')
    hierarchical_position: dict = Field(..., description='Position in cluster hierarchy')

class ExtractedSecurityFields(BaseModel):
    extracted_fields: list[ExtractedSecurityField]
```

#### **3. Function Signature Transformations**

**File-by-File Function Replacements:**

| Current Function       | New Function                     | Purpose Transformation                                                  |
| ---------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| `extract_message()`    | `extract_field_documentation()`  | Conversation parsing → Security field documentation parsing             |
| `extract_json()`       | `extract_field_specifications()` | JSON entity extraction → Structured field definition extraction         |
| `extract_text()`       | `extract_field_hierarchy()`      | Text entity extraction → Hierarchical field relationship extraction     |
| `reflexion()`          | `validate_field_extraction()`    | Conversation entity validation → Security field completeness validation |
| `classify_nodes()`     | `classify_security_fields()`     | Conversational entity classification → Field/Cluster classification     |
| `extract_attributes()` | `extract_field_properties()`     | Entity attribute extraction → Field specification property extraction   |

### **Detailed File Transformation Analysis**

#### **extract_nodes.py → extract_security_fields.py (279 lines → 350+ lines)**

**Complexity**: **95% Complete Redesign Required**

**Critical Changes Required:**

1. **System Prompt Transformation (Lines 70-75)**:

   ```python
   # CURRENT
   sys_prompt = """You are an AI assistant that extracts entity nodes from conversational messages.
   Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation."""

   # REQUIRED
   sys_prompt = """You are an AI assistant specialized in extracting security field definitions from documentation.
   Your primary task is to identify and classify Field and Cluster nodes from security field specifications."""
   ```

2. **Context Input Transformation (Lines 80-90)**:

   ```python
   # CURRENT
   <PREVIOUS MESSAGES>
   {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
   </PREVIOUS MESSAGES>
   <CURRENT MESSAGE>
   {context['episode_content']}
   </CURRENT MESSAGE>

   # REQUIRED
   <EXISTING FIELD RELATIONSHIPS>
   {json.dumps(context['field_relationships'], indent=2)}
   </EXISTING FIELD RELATIONSHIPS>
   <CURRENT FIELD DOCUMENTATION>
   {context['field_documentation']}
   </CURRENT FIELD DOCUMENTATION>
   <CLUSTER HIERARCHY CONTEXT>
   {json.dumps(context['cluster_hierarchy'], indent=2)}
   </CLUSTER HIERARCHY CONTEXT>
   ```

3. **Extraction Logic Complete Rewrite (Lines 100-140)**:

   ```python
   # CURRENT: Speaker + Entity extraction instructions
   1. **Speaker Extraction**: Always extract the speaker as the first entity node.
   2. **Entity Identification**: Extract all significant entities mentioned in the CURRENT MESSAGE.

   # REQUIRED: Security Field extraction instructions
   1. **Field Node Extraction**: Identify individual security fields with their data types and examples.
   2. **Cluster Node Identification**: Extract organizational/macro clusters with organization properties.
   3. **Hierarchical Validation**: Ensure Fields belong to valid Clusters within assigned organizational boundaries.
   ```

#### **extract_edges.py → extract_field_relationships.py (195 lines → 250+ lines)**

**Complexity**: **90% Complete Redesign Required**

**Critical Changes Required:**

1. **Edge Model Transformation**:

   ```python
   # CURRENT
   class Edge(BaseModel):
       relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
       source_entity_id: int = Field(..., description='The id of the source entity')
       target_entity_id: int = Field(..., description='The id of the target entity')
       fact: str = Field(..., description='')

   # REQUIRED
   class SecurityFieldRelationship(BaseModel):
       relationship_type: str = Field(..., description='BELONGS_TO or FIELD_RELATES_TO')
       source_field_uuid: str = Field(..., description='UUID of source security field')
       target_field_uuid: str = Field(..., description='UUID of target security field')
       cluster_partition_id: str = Field(..., description='Cluster isolation constraint')
       semantic_context: str = Field(..., description='Why this relationship exists')
       confidence: float = Field(default=1.0, description='Relationship confidence score')
   ```

2. **Relationship Detection Logic**:

   ```python
   # CURRENT: Fact triple extraction between conversational entities
   # REQUIRED: Security field relationship detection with cluster isolation validation

   Instructions:
   Extract hierarchical and semantic relationships between security fields and clusters.

   1. **BELONGS_TO Relationships**: Field → Cluster membership within organizational boundaries
   2. **FIELD_RELATES_TO Relationships**: Field → Field semantic connections within same Cluster
   3. **Cluster Isolation**: Validate all relationships respect organizational boundaries
   ```

#### **dedupe_nodes.py → dedupe_security_fields.py (209 lines → 220+ lines)**

**Complexity**: **70% Significant Modification Required**

**Critical Changes Required:**

1. **Deduplication Criteria Transformation**:

   ```python
   # CURRENT: Speaker identity + conversation context matching
   # REQUIRED: Semantic field similarity + hierarchical position-based consolidation

   Deduplication Guidelines:
   1. **Field Name Normalization**: Standardize field naming conventions (src_ip vs source_ip)
   2. **Semantic Similarity**: Use field descriptions for duplicate detection
   3. **Hierarchical Position**: Consider cluster/subcluster context in matching
   4. **Organizational Isolation**: Maintain separate field instances across different clusters
   ```

2. **Resolution Logic**:

   ```python
   # CURRENT: Merge conversational entity attributes
   # REQUIRED: Merge field definitions while preserving cluster hierarchy

   Resolution Process:
   1. Identify duplicate field names within same organizational cluster
   2. Compare field descriptions and data types for semantic similarity
   3. Merge field properties while preserving all cluster relationships
   4. Maintain field UUID uniqueness across organizational boundaries
   ```

#### **summarize_nodes.py → summarize_security_fields.py (134 lines → 160+ lines)**

**Complexity**: **85% Complete Redesign Required**

**Critical Changes Required:**

1. **Summarization Focus Transformation**:

   ```python
   # CURRENT: Person/entity characteristics from conversation history
   # REQUIRED: Field definition synthesis and relationship mapping

   Summarization Guidelines:
   1. **Field Specification Synthesis**: Consolidate field properties from multiple sources
   2. **Relationship Context**: Include hierarchical positioning and semantic connections
   3. **Organizational Context**: Maintain cluster/subcluster membership information
   4. **Technical Properties**: Summarize data types, examples, and validation rules
   ```

2. **Output Structure**:

   ```python
   # CURRENT: Conversational entity profile
   # REQUIRED: Comprehensive field specification with relationship context

   class SecurityFieldSummary(BaseModel):
       field_uuid: str
       name: str
       comprehensive_description: str  # Under 250 words
       data_type_summary: str
       example_values: list[str]
       cluster_membership: dict
       relationship_summary: str
       semantic_connections: list[str]
   ```

### **New Protocol Interfaces Required**

```python
class SecurityFieldPrompt(Protocol):
    extract_field_documentation: PromptVersion
    extract_field_specifications: PromptVersion
    extract_field_hierarchy: PromptVersion
    validate_field_extraction: PromptVersion
    classify_security_fields: PromptVersion
    extract_field_properties: PromptVersion

class SecurityFieldRelationshipPrompt(Protocol):
    extract_belongs_to_relationships: PromptVersion
    extract_field_correlations: PromptVersion
    validate_cluster_isolation: PromptVersion

class SecurityFieldValidationPrompt(Protocol):
    dedupe_security_fields: PromptVersion
    summarize_security_fields: PromptVersion
    validate_field_relationships: PromptVersion
    resolve_field_conflicts: PromptVersion
```

### **Implementation Priority & Effort Estimation**

| Transformation Component            | Effort (Hours) | Priority     | Dependencies                     |
| ----------------------------------- | -------------- | ------------ | -------------------------------- |
| **extract_security_fields.py**      | 16-20          | **CRITICAL** | Core schemas complete            |
| **extract_field_relationships.py**  | 14-18          | **HIGH**     | Field extraction complete        |
| **dedupe_security_fields.py**       | 8-12           | **HIGH**     | Field extraction complete        |
| **summarize_security_fields.py**    | 10-14          | **MEDIUM**   | Deduplication complete           |
| **validate_field_relationships.py** | 8-10           | **MEDIUM**   | Relationship extraction complete |
| **Security prompt integration**     | 6-8            | **LOW**      | All prompt files complete        |

**Total Estimated Effort**: 62-82 hours  
**Critical Path Duration**: 38-50 hours

### **Integration Requirements**

1. **Schema Alignment**: All prompt outputs must align with `FieldNode`, `ClusterNode`, `SubClusterNode` schemas
2. **Database Integration**: Extracted fields must integrate seamlessly with MongoDB/Neo4j hybrid architecture
3. **Search System Compatibility**: Prompt results must be compatible with extended search system requirements
4. **Validation Pipeline**: Comprehensive testing for security field extraction accuracy and relationship integrity
5. **Cluster Isolation**: All prompts must enforce organizational boundary constraints throughout extraction process

This comprehensive transformation converts Graphiti from conversational knowledge extraction to specialized security field relationship management while maintaining the robustness and extensibility of the original prompt architecture.

```txt
security_prompt/
├── __init__.py
├── models.py
├── extract_security_fields.py
├── extract_field_relationships.py
├── dedupe_security_fields.py
├── summarize_security_fields.py
└── validate_field_relationships.py
```

---

## 📝 **Recent Schema Updates**

_Date: August 6, 2025_

### Implementation Status Updates ✅ COMPLETE

#### Schema Refinements:

1. **FieldNode Schema Optimization** ✅

   - **Removed**: `field_name` property from FieldNode class
   - **Reason**: Redundant with inherited `name` property from Node base class
   - **Impact**: Simplified field identification using single `name` property
   - **Updated**: All FieldNode save operations, schema documentation, and security prompt models

2. **BELONGS_TO Relationship Cleanup** ✅
   - **Removed**: `relationship_type` property from all BELONGS_TO relationships
   - **Reason**: Relationship type is implicit in Neo4j relationship label `:BELONGS_TO`
   - **Impact**: Cleaner database queries and reduced redundant property storage
   - **Updated**: All database queries in field_db_queries.py and schema documentation

#### Database Query Alignment:

- **field_db_queries.py**: Updated to reflect FieldNode schema without `field_name`
- **project-status-tracker.md**: All examples and schemas updated for consistency
- **Security Prompt Models**: ExtractedSecurityField and SecurityFieldSummary updated

#### Final FieldNode Properties:

```python
class FieldNode(Node):
    # Inherited: uuid, name, labels, created_at, validated_at, invalidated_at, last_updated
    description: str
    examples: list[str]
    data_type: str
    count: int
    distinct_count: int
    primary_cluster_id: str
    embedding: list[float] | None
```

#### Final BELONGS_TO Relationship Properties:

```python
# Neo4j Relationship: (:Field)-[:BELONGS_TO]->(:Cluster)
{
    uuid: str,
    source_node_uuid: str,
    target_node_uuid: str,
    created_at: datetime,
    cluster_partition_id: str
}
```

These optimizations maintain full functionality while reducing schema complexity and improving query performance.
