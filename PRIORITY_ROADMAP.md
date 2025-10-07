# Priority Roadmap - GitHub Issues

**Date**: 2025-10-07
**Purpose**: Identify and prioritize critical issues requiring immediate attention

---

## P0 - Critical (Fix Immediately)

### 1. MCP Server Protocol Deprecation
**Issue [#923](https://github.com/getzep/graphiti/issues/923)** - Switch from SSE to Streamable HTTP

- **Impact**: Using deprecated MCP protocol that will be removed
- **Affected Users**: All MCP server users
- **Risk**: Protocol will stop working when support is removed
- **Effort**: Medium
- **Component**: `component:mcp`

**Action**: Migrate to new protocol ASAP

---

### 2. Database Name Configuration Broken
**Issues [#851](https://github.com/getzep/graphiti/issues/851), #798** - Database name not passed/respected

- **Impact**: Cannot use non-default Neo4j databases for search operations
- **Affected Users**: Multi-database deployments, production environments
- **Root Cause**: Hardcoded 'neo4j' database name in search operations
- **Symptoms**:
  - Search only works with default 'neo4j' database
  - add_episode works with any database name (inconsistent)
- **Effort**: Low-Medium
- **Component**: `component:database`, `component:core`

**Action**: Fix database name propagation through entire codebase

---

### 3. Duplicate Entity Detection Broken
**Issue #875** (duplicate: [#963](https://github.com/getzep/graphiti/issues/963))

- **Impact**: Core deduplication functionality not working
- **Affected Users**: All users, especially with custom database names
- **Symptoms**: Same entities created multiple times instead of being merged
- **User Reports**: Tested with multiple LLM models, issue persists
- **Effort**: Medium-High
- **Component**: `component:core`

**Action**: Debug entity resolution and deduplication logic

---

## P1 - High Priority (Fix This Sprint)

### 4. Bulk Upload Completely Broken
**Issues [#882](https://github.com/getzep/graphiti/issues/882), [#879](https://github.com/getzep/graphiti/issues/879), #871** - Multiple bulk upload failures

- **Impact**: Bulk operations unusable
- **Affected Users**: Anyone trying to import large datasets
- **Symptoms**:
  - IndexError during node resolution ([#882](https://github.com/getzep/graphiti/issues/882))
  - ValidationError for missing 'duplicates' field ([#879](https://github.com/getzep/graphiti/issues/879))
  - Invalid JSON errors ([#871](https://github.com/getzep/graphiti/issues/871))
- **Effort**: Medium
- **Component**: `component:bulk`

**Action**: Fix validation schema and error handling in bulk pipeline

---

### 5. Datetime Comparison Issues
**Issues [#920](https://github.com/getzep/graphiti/issues/920), [#893](https://github.com/getzep/graphiti/issues/893), #606**

- **Impact**: Crashes when comparing dates
- **Status**: May be partially fixed by commit 73015e9
- **Affected Users**: All users with temporal queries
- **Effort**: Low (if recent fix is complete)
- **Component**: `component:core`, `component:database`

**Action**: Verify recent datetime UTC normalization fix covers all cases

---

### 6. Search Group ID Handling
**Issues [#810](https://github.com/getzep/graphiti/issues/810), [#838](https://github.com/getzep/graphiti/issues/838), #801** - Inconsistent group_id behavior

- **Impact**: Search returns empty results or behaves inconsistently
- **Affected Users**: Multi-tenant deployments using group_ids
- **Symptoms**:
  - Empty group_id handled differently in fulltext vs similarity search
  - Cannot search across all groups
- **Effort**: Low-Medium
- **Component**: `component:search`

**Action**: Standardize group_id handling across search methods

---

### 7. BFS Search Bugs
**Issues [#772](https://github.com/getzep/graphiti/issues/772), #789** - BFS traversal broken

- **Impact**: Graph traversal doesn't work as expected
- **Symptoms**:
  - max_depth parameter completely ignored ([#772](https://github.com/getzep/graphiti/issues/772))
  - Duplicate edges with swapped source/target ([#789](https://github.com/getzep/graphiti/issues/789))
- **Effort**: Medium
- **Component**: `component:search`

**Action**: Fix BFS implementation

---

## P2 - Medium Priority (Address Soon)

### 8. MCP Server Configuration Issues
**Issues [#945](https://github.com/getzep/graphiti/issues/945), [#840](https://github.com/getzep/graphiti/issues/840), [#848](https://github.com/getzep/graphiti/issues/848), #565** - Various MCP bugs

- [#945](https://github.com/getzep/graphiti/issues/945): Custom OPENAI_BASE_URL causes NaN embeddings
- [#840](https://github.com/getzep/graphiti/issues/840): "Failed to validate request" initialization timing
- [#848](https://github.com/getzep/graphiti/issues/848): clear_graph fails silently (async session bug)
- [#565](https://github.com/getzep/graphiti/issues/565): Cross-encoder ignores OPENAI_BASE_URL

**Impact**: MCP server unreliable with custom configurations
**Effort**: Low-Medium per issue
**Component**: `component:mcp`

---

### 9. LLM Provider Compatibility
**Issues [#878](https://github.com/getzep/graphiti/issues/878), [#902](https://github.com/getzep/graphiti/issues/902), [#912](https://github.com/getzep/graphiti/issues/912), #791** - Provider-specific bugs

- [#902](https://github.com/getzep/graphiti/issues/902): OpenAI reasoning.effort parameter breaks API
- [#878](https://github.com/getzep/graphiti/issues/878): GPT-5 doesn't support temperature parameter
- [#791](https://github.com/getzep/graphiti/issues/791): Small model setting ignored, always defaults to gpt-4.1-nano
- [#912](https://github.com/getzep/graphiti/issues/912): Pydantic validation errors with non-OpenAI models

**Impact**: Specific LLM models/providers broken
**Effort**: Low per issue
**Component**: `component:llm`

**Strategy**: Add provider capability detection and parameter filtering

---

### 10. FalkorDB Driver Issues
**Issues [#972](https://github.com/getzep/graphiti/issues/972), [#815](https://github.com/getzep/graphiti/issues/815), [#757](https://github.com/getzep/graphiti/issues/757), [#731](https://github.com/getzep/graphiti/issues/731), #749** - FalkorDB broken

- **Impact**: FalkorDB backend largely non-functional
- **Effort**: High (multiple issues)
- **Component**: `component:database`

**Decision Point**: Fix comprehensively or deprecate? Appears unmaintained.

---

### 11. Error Handling Improvements
**Issues [#937](https://github.com/getzep/graphiti/issues/937), #951** - Edge case crashes

- [#937](https://github.com/getzep/graphiti/issues/937): Empty query strings cause ArgumentError
- [#951](https://github.com/getzep/graphiti/issues/951): Incorrect import fallback for AsyncOpenSearch

**Impact**: Crashes on edge cases
**Effort**: Low per issue
**Component**: `component:core`

---

### 12. API Server Issues
**Issue [#566](https://github.com/getzep/graphiti/issues/566)** - /messages endpoint doesn't persist episodes

- **Impact**: Core API functionality broken
- **Effort**: Medium
- **Component**: `component:server`

**Action**: Debug why episodes aren't being persisted

---

### 13. Code Quality Issues
**Issues [#836](https://github.com/getzep/graphiti/issues/836), [#811](https://github.com/getzep/graphiti/issues/811), [#681](https://github.com/getzep/graphiti/issues/681), #451** - Technical debt

- [#836](https://github.com/getzep/graphiti/issues/836): update_communities broken (tuple unpacking error)
- [#811](https://github.com/getzep/graphiti/issues/811): Hoist hardcoded token constant
- [#681](https://github.com/getzep/graphiti/issues/681): Remove ghost variables
- [#451](https://github.com/getzep/graphiti/issues/451): Type bug in bulk_utils

**Impact**: Medium (functionality broken in [#836](https://github.com/getzep/graphiti/issues/836), others are maintenance)
**Effort**: Low per issue

---

## P3 - Lower Priority / Feature Requests

### 14. Feature Enhancements
- [#961](https://github.com/getzep/graphiti/issues/961) - Improve Episodes API (UUID, GET by ID, metadata)
- [#935](https://github.com/getzep/graphiti/issues/935) - Create episodes based on DOM
- [#934](https://github.com/getzep/graphiti/issues/934) - Flag contradictions on merging facts
- [#925](https://github.com/getzep/graphiti/issues/925) - Monitor LLM conversations
- [#819](https://github.com/getzep/graphiti/issues/819) - Count token usage
- [#747](https://github.com/getzep/graphiti/issues/747) - Progress reporting for bulk upload
- [#669](https://github.com/getzep/graphiti/issues/669) - Metadata on chunks for RAG
- [#465](https://github.com/getzep/graphiti/issues/465) - Support ignoring non-custom entities

### 15. New Provider Support
- [#907](https://github.com/getzep/graphiti/issues/907) - VSC Copilot models
- [#905](https://github.com/getzep/graphiti/issues/905) - ColbertV2 embeddings
- [#751](https://github.com/getzep/graphiti/issues/751) - Alibaba Cloud AI
- [#724](https://github.com/getzep/graphiti/issues/724) - Gemini with GCP credentials
- [#459](https://github.com/getzep/graphiti/issues/459) - Amazon Bedrock
- [#739](https://github.com/getzep/graphiti/issues/739) - zep-cloud based MCP

### 16. Documentation Improvements
- [#913](https://github.com/getzep/graphiti/issues/913) - How to update data in graph
- [#853](https://github.com/getzep/graphiti/issues/853) - VCS/extension setup
- [#828](https://github.com/getzep/graphiti/issues/828) - Flowchart documentation
- [#484](https://github.com/getzep/graphiti/issues/484) - OpenAIGenericClient documentation
- All "How to" questions

### 17. Alternative Database Support
- [#947](https://github.com/getzep/graphiti/issues/947) - Apache AGE
- [#933](https://github.com/getzep/graphiti/issues/933) - RDF support
- [#781](https://github.com/getzep/graphiti/issues/781) - NebulaGraph
- [#779](https://github.com/getzep/graphiti/issues/779) - Postgres/pgvector
- [#644](https://github.com/getzep/graphiti/issues/644) - AWS Neptune
- [#643](https://github.com/getzep/graphiti/issues/643) - Kuzu
- [#642](https://github.com/getzep/graphiti/issues/642) - MemGraph

---

## Immediate Action Plan (Next 2 Weeks)

### Week 1
1. **Fix [#923](https://github.com/getzep/graphiti/issues/923)** - MCP protocol migration (CRITICAL)
2. **Fix [#851](https://github.com/getzep/graphiti/issues/851)/[#798](https://github.com/getzep/graphiti/issues/798)** - Database name configuration
3. **Fix [#875](https://github.com/getzep/graphiti/issues/875)** - Duplicate entity detection
4. **Verify [#920](https://github.com/getzep/graphiti/issues/920)** - Datetime issues resolved by recent commit

### Week 2
5. **Fix [#882](https://github.com/getzep/graphiti/issues/882)/[#879](https://github.com/getzep/graphiti/issues/879)/[#871](https://github.com/getzep/graphiti/issues/871)** - Bulk upload pipeline
6. **Fix [#810](https://github.com/getzep/graphiti/issues/810)** - Group ID handling in search
7. **Fix [#772](https://github.com/getzep/graphiti/issues/772)/[#789](https://github.com/getzep/graphiti/issues/789)** - BFS search issues
8. **Close duplicates** - Process 6+ confirmed duplicates

### Ongoing
- **Investigate duplicate clusters** - Consolidate related issues
- **Document workarounds** - For issues that can't be fixed immediately
- **Provider compatibility matrix** - Document which LLMs work fully

---

## Impact Summary

**Critical Issues Blocking Production Use**:
- Database name configuration (multi-DB deployments)
- Duplicate entity detection (core functionality)
- Bulk upload failures (data ingestion at scale)
- MCP protocol deprecation (future compatibility)

**High-Impact Bugs Affecting Many Users**:
- Search group_id handling
- BFS traversal issues
- Datetime comparison problems
- Various MCP server configuration bugs

**Lower Impact**:
- Specific LLM provider incompatibilities (workarounds available)
- FalkorDB issues (alternative: Neo4j)
- Feature requests and enhancements

---

## Component Labels to Apply

Use these labels when triaging the issues above:

- `component:core` - Core library
- `component:mcp` - MCP server
- `component:database` - Database drivers
- `component:search` - Search functionality
- `component:bulk` - Bulk operations
- `component:server` - FastAPI server
- `component:llm` - LLM providers
- `component:docs` - Documentation
