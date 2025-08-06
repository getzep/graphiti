# Graphiti Hybrid Architecture - Project Index

## 🎯 **Project Context**

**Project Name**: Graphiti Hybrid MongoDB + Neo4j Architecture  
**Purpose**: Transform Graphiti from general-purpose temporal knowledge graph to specialized security field relationship engine for Splunk Security Query Generation  
**Architecture**: Hybrid MongoDB (metadata) + Neo4j (graph relationships) system  
**Current Status**: ⚠️ **Architectural Transition in Progress** (Label-based → Relationship-based)

## 📖 **Primary Documentation**

### **Main Project Document**: `project-status-tracker.md`

**Location**: `memory-bank/project-status-tracker.md`  
**Purpose**: Complete project documentation with all implementation details, schemas, and status tracking  
**Last Updated**: August 5, 2025

## 🧭 **Section Navigation Map**

### **🏷️ Foundation Sections** _(Lines 1-36)_

| Section                   | Lines | Purpose                        | Key Content                                            |
| ------------------------- | ----- | ------------------------------ | ------------------------------------------------------ |
| **Project Overview**      | 1-9   | High-level project description | Purpose, architecture goals, transformation objectives |
| **General Project Rules** | 10-36 | Core architecture principles   | Node labels, relationship types, hierarchy approach    |

### **📐 Data Organization Rules** _(Lines 37-128)_

| Rule Section                                | Lines  | Purpose                            | Key Constraints                             |
| ------------------------------------------- | ------ | ---------------------------------- | ------------------------------------------- |
| **1. Cross-Organization Field Replication** | 37-53  | Field sharing across clusters      | Relationship constraints, duplication rules |
| **2. Relationship-Based Isolation**         | 54-72  | Data separation strategies         | Query isolation, cluster disconnection      |
| **3. Field Uniqueness Rules**               | 73-80  | Field identification constraints   | UUID requirements, cluster membership       |
| **4. Relationship Integrity Constraints**   | 81-128 | Core relationship validation rules | Direct Field-Cluster relationships          |

### **🗂️ Main Project Schemas** _(Lines 129-920)_

| Schema Type              | Lines   | Content                                                       | Implementation Status |
| ------------------------ | ------- | ------------------------------------------------------------- | --------------------- |
| **Node Base Class**      | 129-170 | Unified Node class with temporal validation                   | ✅ Complete           |
| **Field Schema**         | 171-300 | FieldNode class definition with enhanced tracking             | ✅ Complete           |
| **Cluster Schemas**      | 301-480 | ClusterNode class with CRUD operations                        | ✅ Complete           |
| **Relationship Schemas** | 481-920 | Complete Edge base class and all relationship implementations | ✅ Complete           |

### **📊 Database Query Implementation** _(Lines 921-1020)_

| Query Analysis Section                     | Lines     | Content                                                        | Priority Level |
| ------------------------------------------ | --------- | -------------------------------------------------------------- | -------------- |
| **Security Graph Queries Analysis**        | 921-940   | File structure and core architecture requirements              | HIGH           |
| **Essential Database Query Functions**     | 941-990   | Index creation, query generation, and bulk operation functions | HIGH           |
| **Critical Database Constraint Functions** | 991-1000  | Constraint validation functions for relationship integrity     | HIGH           |
| **Query Pattern Adaptations**              | 1001-1010 | Current Graphiti → Security system query pattern conversions   | MEDIUM         |
| **Performance Optimization**               | 1011-1015 | Indexing, batch operations, and embedding integration          | MEDIUM         |
| **Implementation Sequence**                | 1016-1020 | 5-phase implementation plan for security query system          | HIGH           |

### **🎯 Architecture Summary** _(Lines 1021-1095)_

| Component                             | Lines     | Focus                                     | Details                                                                 |
| ------------------------------------- | --------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| **Essential Edge Functions Analysis** | 1021-1095 | Security edge implementation requirements | Complete analysis of Edge class functions and implementation priorities |

> **📍 Entry Point for LLMs**: This index provides a structured overview of the entire Graphiti Hybrid Architecture project, allowing quick navigation to specific sections and understanding of project context.

### **🔍 Search System Integration & Migration** _(Lines 1096-1200)_

| Migration Component               | Lines     | Content                                                   | Priority Level |
| --------------------------------- | --------- | --------------------------------------------------------- | -------------- |
| **Search System Analysis**        | 1096-1120 | Compatibility assessment and architecture evaluation      | HIGH           |
| **Current Search Architecture**   | 1121-1140 | Core components, search methods, and reranking strategies | MEDIUM         |
| **Compatibility Assessment**      | 1141-1160 | Security system alignment and adaptation requirements     | HIGH           |
| **Migration Implementation Plan** | 1161-1180 | 3-phase implementation strategy with effort estimates     | HIGH           |
| **Benefits & Expected Outcomes**  | 1181-1200 | Value proposition and performance expectations            | MEDIUM         |

### **🎭 Prompts System Transformation** _(Lines 1201-1390)_

| Transformation Component        | Lines     | Content                                                   | Priority Level |
| ------------------------------- | --------- | --------------------------------------------------------- | -------------- |
| **Current System Analysis**     | 1201-1220 | Overview of existing conversational prompts architecture  | HIGH           |
| **Transformation Requirements** | 1221-1280 | Context, models, and function signature changes required  | CRITICAL       |
| **File-by-File Analysis**       | 1281-1350 | Detailed transformation requirements for each prompt file | HIGH           |
| **Implementation Strategy**     | 1351-1370 | Priority matrix and effort estimation for prompt changes  | HIGH           |
| **Integration Requirements**    | 1371-1390 | Schema alignment and validation pipeline requirements     | MEDIUM         |

## 🔍 **Quick Reference Lookup**

### **Node Types & Relationships**

- **Field Node**: Individual security fields (e.g., `src_ip`, `user_id`)
- **Cluster Node**: Primary organizational clusters (e.g., `linux_audit_batelco`)
- **BELONGS_TO**: Field → Cluster relationship
- **FIELD_RELATES_TO**: Field → Field semantic relationships within same Cluster

### **Key Constraints**

1. **Cross-Cluster Isolation**: No direct relationships between different Clusters
2. **Field-Field Relationships**: Only within same Cluster (1-to-many)
3. **Direct Field-Cluster Membership**: Each Field belongs directly to one Cluster
4. **Proper Labeling**: Fields and Clusters should have labels of `Field` and `Cluster` respectively to ensure proper identification and isolation

### **🚀 Critical Implementation Steps**

#### **Prompts System Transformation (Priority: CRITICAL)**

| Phase       | Task                                                        | Effort | Status     | Dependencies             |
| ----------- | ----------------------------------------------------------- | ------ | ---------- | ------------------------ |
| **Phase 1** | Transform extract_nodes.py → extract_security_fields.py     | 16-20h | ⏳ Pending | ✅ Core schemas complete |
| **Phase 2** | Transform extract_edges.py → extract_field_relationships.py | 14-18h | ⏳ Pending | Phase 1 complete         |
| **Phase 3** | Transform deduplication and summarization prompts           | 18-26h | ⏳ Pending | Phase 2 complete         |

#### **Search System Migration (Priority: HIGH)**

| Phase       | Task                                    | Effort | Status     | Dependencies             |
| ----------- | --------------------------------------- | ------ | ---------- | ------------------------ |
| **Phase 1** | Extend SearchConfig for security nodes  | 16-24h | ⏳ Pending | ✅ Core schemas complete |
| **Phase 2** | Implement security search functions     | 24-32h | ⏳ Pending | Phase 1 complete         |
| **Phase 3** | Create security-specific search recipes | 8-12h  | ⏳ Pending | Phase 2 complete         |

#### **Key Implementation Actions Required:**

1. **🔧 Extend Search Configuration**:

   - Add `SecurityNodeSearchConfig`, `SecuritySearchMethod`, `SecuritySearchFilters`
   - Adapt `cluster_partition_id` filtering from existing `group_ids` pattern
   - Integrate `FieldNode`, `ClusterNode` support

2. **🔍 Implement Security Search Functions**:

   - `field_fulltext_search()` - Field description and name search
   - `field_similarity_search()` - Embedding-based field discovery
   - `cluster_field_traversal()` - BELONGS_TO relationship exploration
   - `security_relationship_search()` - FIELD_RELATES_TO semantic relationships

3. **📋 Create Search Recipes**:

   - `FIELD_RELATIONSHIP_DISCOVERY` - Find related fields within clusters
   - `CLUSTER_FIELD_EXPLORATION` - Explore direct field-cluster relationships
   - `SECURITY_CORRELATION_SEARCH` - Cross-field correlation analysis

4. **⚡ Performance Optimization**:
   - Leverage existing indexing strategies for security nodes
   - Implement cluster isolation constraints in search filters
   - Add custom rerankers for field correlation scoring

#### **Expected Benefits:**

- ✅ **80% of search functionality** already implemented and production-tested
- ✅ **Sophisticated multi-modal search** (keyword + semantic + graph traversal)
- ✅ **Advanced AI-powered reranking** with cross-encoder support
- ✅ **Cluster isolation** through proven filtering mechanisms
- ✅ **Performance optimization** via established indexing patterns

### **Implementation Priority Matrix**

| Priority   | Component                     | Status               | Effort (Hours) |
| ---------- | ----------------------------- | -------------------- | -------------- |
| **HIGH**   | Edge class architecture       | ✅ Complete          | 0              |
| **HIGH**   | Relationship schemas          | ✅ Complete          | 0              |
| **HIGH**   | Node class implementation     | ✅ Complete          | 0              |
| **HIGH**   | Main project schemas          | ✅ Complete          | 0              |
| **MEDIUM** | Database query implementation | Needs implementation | 24-36          |
| **MEDIUM** | Security graph queries        | Missing              | 12-16          |
| **LOW**    | Test updates                  | Needs updates        | 8-12           |

## 🚀 **For LLM Context Understanding**

### **Current Architecture State**

- **FROM**: Label-based clustering (`['Field', 'linux_audit_batelco', 'linux_audit_batelco:UserAuthentication']`)
- **TO**: Relationship-based hierarchy (`(Field)-[:BELONGS_TO]->(Cluster)`)

### **Database Responsibilities**

- **MongoDB**: Cluster metadata, fast lookups, business validation
- **Neo4j**: Graph relationships, field nodes, hierarchy traversal, embeddings

### **Critical Design Principles**

1. **Separation of Concerns**: Data (properties) vs Structure (relationships)
2. **Relationship-Based Isolation**: Using graph traversal for data separation
3. **Constraint Validation**: Relationship integrity is essential for data consistency
4. **Proper Node Labeling**: All nodes must have correct labels for identification and isolation

### **Recent Updates**

- **✅ Complete Core Architecture**: All main project schemas, node classes, and relationship schemas fully implemented
- **✅ Search System Integration Analysis**: Comprehensive evaluation of existing Graphiti search system compatibility with security field requirements
- **✅ Search Migration Strategy**: Detailed 3-phase implementation plan for extending existing search capabilities rather than building separate system
- **✅ Search System Compatibility Assessment**: Confirmed 80% functionality overlap with security requirements, requiring only targeted extensions
- **✅ Database Query Implementation Analysis**: Complete analysis of required security graph queries with function-by-function mapping
- **✅ Essential Edge Functions Analysis**: Complete analysis of Edge class functions and implementation priorities
- **✅ Abstract Edge Base Class**: Implemented with type safety and consistent patterns across all relationship types
- **✅ Full Relationship Implementations**: BelongsToEdge and FieldRelationshipEdge with complete CRUD operations
- **✅ Enhanced Node Classes**: Unified Node base class with temporal validation and proper inheritance
- **✅ Prompts System Transformation Analysis**: Comprehensive analysis of required changes to transform conversational prompts to security field extraction
- **✅ Architecture Simplification**: Removed SubCluster logic, simplified to direct Field-Cluster relationships
- **✅ Project Documentation Sync**: Updated project-index.md to accurately reflect simplified architecture

## 📝 **Usage Instructions for LLMs**

1. **Start Here**: Always reference this index first for project context
2. **Navigate Efficiently**: Use line number references to jump to specific sections in `project-status-tracker.md`
3. **Understand Status**: Check implementation status before making suggestions
4. **Follow Constraints**: All new implementations must follow the relationship integrity constraints
5. **Prioritize Work**: Focus on remaining MEDIUM and LOW priority components
6. **Maintain Consistency**: Always update both project-status-tracker.md and project-index.md when making changes

---

**📁 Related Files**:

- `memory-bank/project-status-tracker.md` - Complete project documentation
- `graphiti_core/field_nodes.py` - Core field implementation (needs refactoring)
- `graphiti_core/cluster_metadata/` - MongoDB integration modules (still valid)

**🔄 Last Index Update**: August 5, 2025  
**📊 Project Status**: Simplified Architecture Complete (Core Schemas)  
**🔄 Last Sync**: Updated to reflect simplified Field-Cluster architecture without SubCluster layer
