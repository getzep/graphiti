# GitHub Issues Triage & Categorization

**Date**: 2025-10-07
**Total Open Issues**: 100+
**Analysis Scope**: Issues #451 - #972

## Issue Clusters

### 1. Database Driver Support

#### FalkorDB Issues (6 issues)
- **#972** - [BUG] Vector type mismatch in cosine distance operations
  - *Component*: `component:database`
  - *Status*: Open

- **#815** - [BUG] falkordb query
  - *Component*: `component:database`
  - *Status*: Open

- **#757** - [BUG] run quickstart_falkordb example, get query error
  - *Component*: `component:database`
  - *Status*: Open

- **#749** - [BUG] Official Docker Image doesn't support FalkorDB
  - *Component*: `component:database`, `component:mcp`
  - *Status*: Open

- **#731** - Episode insertion fails due to malformed Cypher query
  - *Component*: `component:database`
  - *Status*: Open

- **#719** - FalkorDB support in MCP-Server
  - *Component*: `component:mcp`, `component:database`
  - *Status*: Open

#### New Database Support Requests (6 issues)
- **#947** - Apache AGE Graph DB support
- **#933** - Support RDF
- **#781** - More graph database support (NebulaGraph mentioned)
- **#779** - Postgres with pgvector support
- **#644** - AWS Neptune driver
- **#643** - Kuzu driver support (may already be implemented?)
- **#642** - MemGraph driver

**Analysis**: FalkorDB has multiple critical bugs affecting basic operations. Consider deprecating or fixing comprehensively. High demand for alternative backends suggests need for driver abstraction layer.

---

### 2. MCP Server Issues (9 issues)

- **#923** - [BUG] Switch from SSE (deprecated) to Streamable HTTP
  - *Priority*: HIGH - using deprecated protocol
  - *Component*: `component:mcp`

- **#945** - [BUG] Custom OPENAI_BASE_URL causes NaN embeddings
  - *Component*: `component:mcp`, `component:llm`

- **#848** - [BUG] clear_graph tool fails silently (async session bug)
  - *Component*: `component:mcp`

- **#840** - [BUG] Failed to validate request (initialization timing)
  - *Component*: `component:mcp`

- **#723** - DEFAULT_MAX_TOKENS 8192 too restrictive
  - *Component*: `component:mcp`

- **#578** - MCP server with Gemini model
  - *Component*: `component:mcp`, `component:llm`

- **#565** - Cross-encoder ignores OPENAI_BASE_URL
  - *Component*: `component:mcp`, `component:llm`

- **#509** - MCP server add_nodes tool call doesn't work
  - *Component*: `component:mcp`

**Duplicates in this cluster**:
- **#867** + **#831** - GPT-oss:20 and 120B models (duplicate)
- **#787** - Rate limit even with SEMAPHORE_LIMIT=1 (marked duplicate)

**Analysis**: MCP server has protocol deprecation issue (#923) and multiple configuration/provider issues. Should be high priority for users relying on MCP integration.

---

### 3. LLM Provider Compatibility (8 issues)

- **#902** - [BUG] OpenAI internal call broken (reasoning.effort param)
  - *Component*: `component:llm`

- **#878** - [BUG] GPT-5 temperature parameter unsupported
  - *Component*: `component:llm`

- **#912** - [BUG] Pydantic validation error with deepseek-r1:7b
  - *Component*: `component:llm`

- **#791** - [BUG] Small model setting defaults to gpt-4.1-nano
  - *Component*: `component:llm`

- **#790** - [BUG] Failed to parse structured response with Gemini
  - *Component*: `component:llm`

- **#868** - [BUG] Cannot work with Ollama
  - *Component*: `component:llm`

- **#763** - [BUG] LLMConfig.max_tokens not respected
  - *Component*: `component:llm`

- **#760** - [BUG] Hallucinations with default models
  - *Component*: `component:llm`

**Additional LLM Provider Requests**:
- **#907** - VSC Copilot models
- **#751** - Alibaba Cloud AI model
- **#724** - Gemini with Google Cloud Credentials
- **#459** - Amazon Bedrock support

**Analysis**: Structured output compatibility is a recurring theme. Many issues stem from providers not supporting OpenAI's structured output format. Documentation should clearly state which providers are fully compatible.

---

### 4. Duplicate Entities (3 issues)

- **#963** - [BUG] Duplicate entities in Neo4j (marked duplicate)
  - *Component*: `component:core`
  - *Status*: Duplicate (see #875)

- **#875** - [BUG] Duplicate entities with custom db name
  - *Component*: `component:core`, `component:database`
  - *Priority*: HIGH - core deduplication functionality broken

- **#774** - Same Chinese text extracted as different facts
  - *Component*: `component:core`

**Analysis**: Core deduplication functionality appears broken, especially with custom database names. Critical for production use.

---

### 5. Bulk Upload Issues (3 issues)

- **#882** - [BUG] IndexError during node resolution
  - *Component*: `component:bulk`

- **#879** - ValidationError 'duplicates' field missing
  - *Component*: `component:bulk`

- **#871** - Invalid JSON and index errors
  - *Component*: `component:bulk`

**Related**:
- **#747** - Add progress reporting to bulk upload
- **#658** - Bulk ingestion not possible (may be duplicate)

**Analysis**: Bulk operations are fundamentally broken. Appears to be schema/validation issues in the bulk processing pipeline.

---

### 6. Search Issues (6 issues)

- **#810** - [BUG] Empty group_id handled inconsistently in search
  - *Component*: `component:search`

- **#801** - [BUG] episode_fulltext_search empty results (marked duplicate)
  - *Component*: `component:search`

- **#838** - Allow searching across all groups when group_ids is None
  - *Component*: `component:search`

- **#772** - [BUG] BFS max_depth parameter ignored
  - *Component*: `component:search`

- **#789** - [BUG] BFS returns duplicate edges with swapped source/target
  - *Component*: `component:search`

- **#777** - [BUG] MMR reranker RuntimeWarning, no results
  - *Component*: `component:search`

**Related**:
- **#488** - edge_search_filter_query_constructor creating incorrect query
- **#534** - retrieve_episodes always returns no results

**Analysis**: Search functionality has multiple bugs in filtering, BFS traversal, and reranking. Group ID handling is particularly problematic.

---

### 7. Database Configuration (4 issues)

- **#851** - [BUG] Search only connects to 'neo4j' db, add_episode works with any name
  - *Component*: `component:database`
  - *Priority*: HIGH - inconsistent behavior

- **#798** - [BUG] Database name not passed through Graphiti object
  - *Component*: `component:database`
  - *Priority*: HIGH - related to #851

- **#715** - [Feature] Configure Neo4j database name (multi-DB support)
  - *Component*: `component:database`

**Analysis**: Database name configuration is broken. Hardcoded 'neo4j' default causes issues. Related to commit mentioned in CLAUDE.md about hardcoded database names.

---

### 8. Datetime/Timezone Issues (4 issues)

- **#920** - [BUG] edge_operations.py timezone-naive/aware comparison (marked duplicate)
  - *Component*: `component:core`
  - *Status*: Duplicate (possibly fixed in commit 73015e9)

- **#893** - [BUG] Kuzu driver valid_at datetime format
  - *Component*: `component:database`

- **#606** - Add support for datetime fields in custom entities
  - *Component*: `component:core`

**Analysis**: Recent commit 73015e9 "Fix datetime comparison errors by normalizing to UTC" may have addressed some of these. Needs verification.

---

### 9. API/Server Issues (4 issues)

- **#961** - [Feature Request] Improve Episodes API (UUID, GET by ID, metadata)
  - *Component*: `component:server`

- **#921** - SDK client like zep-cloud/zep-python
  - *Component*: `component:server`

- **#566** - /messages endpoint doesn't persist episodes
  - *Component*: `component:server`

- **#904** - docker-compose.yml env vars override .env
  - *Component*: `component:server`, `component:mcp`

**Analysis**: API functionality gaps and configuration issues. Episode persistence bug is critical.

---

### 10. Error Handling & Validation (3 issues)

- **#941** - TaskGroup errors (marked duplicate)
  - *Component*: `component:core`

- **#937** - ArgumentError with empty query entities
  - *Component*: `component:core`

- **#951** - Incorrect try import for AsyncOpenSearch
  - *Component*: `component:database`

**Analysis**: Input validation needs improvement to handle edge cases like empty strings.

---

### 11. Feature Requests - Core Functionality (10 issues)

- **#935** - Create episodes based on DOM structure
- **#934** - Flag contradictions on merging facts
- **#925** - Monitor LLM conversations
- **#905** - ColbertV2 embeddings with Fastembed
- **#864** - How to forget knowledge
- **#819** - Count token usage
- **#669** - Metadata on chunks for RAG
- **#465** - Support ignoring non-custom entities
- **#467** - LLM inference expenses are high

**Analysis**: Feature requests range from observability (#925, #819) to advanced RAG features (#669, #905). Token cost reduction (#467) is recurring concern.

---

### 12. Documentation/Questions (9 issues)

- **#913** - How to update data in graph
- **#909** - Reproduce LongMemEval results
- **#869** - Portuguese: Change OpenAI model
- **#853** - Document VCS/extension setup
- **#828** - Get a flowchart
- **#701** - Chinese video tutorial (informational)
- **#484** - OpenAIGenericClient documentation
- **#530** - Cursor AI + OpenRouter setup
- **#517** - OpenRouter and Voyage setup

*Component*: `component:docs`

**Analysis**: Many questions indicate documentation gaps, especially around custom LLM provider setup and basic operations.

---

### 13. Code Quality/Refactoring (5 issues)

- **#836** - [BUG] update_communities broken
  - *Component*: `component:core`

- **#811** - Refactor: hoist EXTRACT_EDGES_MAX_TOKENS constant
  - *Component*: `component:core`

- **#681** - Refactor: no internal ghost variables
  - *Component*: `component:core`

- **#451** - Minor type bug in bulk_utils
  - *Component*: `component:bulk`

- **#717** - Adopt uv workspace for monorepo
  - *Component*: `component:core`, `component:server`, `component:mcp`

**Analysis**: Technical debt items. #717 (uv workspace) would improve developer experience for monorepo.

---

### 14. Cloud/Alternative Providers (2 issues)

- **#739** - MCP server based on zep-cloud
  - *Component*: `component:mcp`

- **#538** - Azure Cosmos DB version (informational)

---

### 15. Embeddings/Reranking (3 issues)

- **#728** - Voyage embedder installation issue
  - *Component*: `component:core`

- **#485** - Better Embedder error information
  - *Component*: `component:core`

- **#543** - Hard-coded model in OpenAIRerankerClient
  - *Component*: `component:core`

---

### 16. Miscellaneous Bugs (5 issues)

- **#800** - Blank disconnected nodes in sample project
  - *Component*: `component:mcp`

- **#686** - Relationships not extracted automatically
  - *Component*: `component:core`

- **#687** - Unable to add episodes with OpenAI agents SDK
  - *Component*: `component:core`

- **#587** - Node type not being set
  - *Component*: `component:core`

---

## Summary Statistics

- **Total Clustered Issues**: 100+
- **Major Clusters**: 16
- **Confirmed Duplicates**: 6 (marked)
- **High Priority Bugs**: ~15-20
- **Feature Requests**: ~20
- **Documentation Gaps**: ~10

## Component Breakdown

- `component:core` - 25+ issues
- `component:database` - 20+ issues (includes all DB drivers)
- `component:mcp` - 12+ issues
- `component:llm` - 12+ issues
- `component:search` - 8+ issues
- `component:bulk` - 5+ issues
- `component:server` - 4+ issues
- `component:docs` - 10+ issues
