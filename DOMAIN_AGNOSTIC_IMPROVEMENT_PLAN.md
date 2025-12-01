# Graphiti Domain-Agnostic Improvement Plan

**Date**: 2025-11-30
**Status**: Draft
**Last Pull**: `422558d` (main branch)

---

## Executive Summary

This document outlines a strategic plan to make Graphiti more domain-agnostic and adaptable to diverse use cases beyond conversational AI. The current architecture, while powerful, contains several domain-specific assumptions (primarily around messaging/conversational data) that limit its applicability to other domains such as scientific research, legal documents, IoT data, healthcare records, financial transactions, etc.

---

## Current Architecture Analysis

### Key Components Review

1. **NER & Entity Extraction** (`graphiti_core/utils/maintenance/node_operations.py`, `graphiti_core/prompts/extract_nodes.py`)
   - Hardcoded prompts for three episode types: message, text, JSON
   - Domain-specific language (e.g., "speaker", "conversation")
   - Entity type classification tightly coupled with extraction logic

2. **LLM Client Configuration** (`graphiti_core/llm_client/config.py`, `graphiti_core/graphiti.py`)
   - Defaults to OpenAI across all components
   - No centralized model selection strategy
   - Temperature (1.0) and max_tokens (8192) hardcoded as defaults

3. **Episode Types** (`graphiti_core/nodes.py`)
   - Limited to: message, text, JSON
   - Each type requires separate prompt functions
   - No extensibility mechanism for custom episode types

4. **Prompt System** (`graphiti_core/prompts/`)
   - Prompts are Python functions, not configurable data
   - No template engine or override mechanism
   - Domain assumptions embedded in prompt text

5. **Search & Retrieval** (`graphiti_core/search/`)
   - Flexible but complex configuration
   - Limited domain-specific search recipes
   - No semantic domain adapters

---

## Identified Issues from GitHub (Top 20)

### High-Impact Issues Related to Domain Agnostic Goals:

1. **#1087**: Embedding truncation reduces retrieval quality for text-embedding-3-small
2. **#1074**: Neo4j quickstart returns no results with OpenAI-compatible LLM + Ollama embeddings
3. **#1007**: OpenAIGenericClient outputs unstable for vllm serving gpt-oss-20b
4. **#1006**: OpenAIRerankerClient does not support AzureOpenAILLMClient
5. **#1004**: Azure OpenAI is not supported
6. **#995**: Docker container does not support Azure OpenAI
7. **#1077**: Support for Google Cloud Spanner Graph
8. **#947**: Support for Apache AGE as Graph DB
9. **#1016**: Support episode vector
10. **#961**: Improve Episodes API - return UUID, support GET by ID, custom metadata

---

## Improvement Directives

### 1. **Configurable Prompt System** 🔴 **Priority: CRITICAL**

#### Objective
Replace hardcoded prompt functions with a templatable, extensible prompt system that supports domain customization.

#### Implementation Plan

**Phase 1: Prompt Template Engine**
- Create `PromptTemplate` class with variable interpolation
- Support multiple template formats (Jinja2, mustache, or custom)
- Add prompt registry for registration and lookup

```python
# Example API
class PromptTemplate:
    def __init__(self, template: str, variables: dict[str, str]):
        self.template = template
        self.variables = variables

    def render(self, context: dict[str, Any]) -> str:
        # Template rendering logic
        pass

class PromptRegistry:
    def register(self, name: str, template: PromptTemplate) -> None:
        pass

    def get(self, name: str) -> PromptTemplate:
        pass

    def override(self, name: str, template: PromptTemplate) -> None:
        pass
```

**Phase 2: Refactor Existing Prompts**
- Convert all prompt functions in `graphiti_core/prompts/` to templates
- Maintain backward compatibility with existing API
- Add domain-specific prompt overrides

**Phase 3: Documentation & Examples**
- Create prompt customization guide
- Provide domain-specific examples (legal, scientific, financial)
- Add prompt testing utilities

#### Priority Rationale
- **Impact**: Enables all domain customization downstream
- **Complexity**: Medium - requires careful refactoring
- **Dependencies**: None - can be done independently

#### Blockers
- **Breaking Changes**: Need to maintain backward compatibility
- **LLM Provider Compatibility**: Different providers may require different prompt formats
- **Testing**: Need comprehensive test suite for prompt variations

#### Success Metrics
- Users can customize prompts without code changes
- 5+ domain-specific prompt examples documented
- No regression in existing use cases

---

### 2. **Pluggable NER & Entity Extraction Pipeline** 🔴 **Priority: CRITICAL**

#### Objective
Make the entity extraction pipeline modular and extensible for different domain requirements.

#### Implementation Plan

**Phase 1: Extraction Strategy Interface**
- Define `ExtractionStrategy` protocol/abstract class
- Support custom entity extractors (LLM-based, rule-based, hybrid)
- Allow domain-specific entity type systems

```python
class ExtractionStrategy(Protocol):
    async def extract_entities(
        self,
        episode: EpisodicNode,
        context: dict[str, Any],
        entity_types: dict[str, type[BaseModel]] | None = None,
    ) -> list[EntityNode]:
        ...

    async def extract_relations(
        self,
        episode: EpisodicNode,
        entities: list[EntityNode],
        context: dict[str, Any],
    ) -> list[EntityEdge]:
        ...
```

**Phase 2: Domain-Specific Extractors**
- Create extractors for common domains:
  - `ScientificPaperExtractor`: Extracts researchers, institutions, findings, citations
  - `LegalDocumentExtractor`: Extracts parties, cases, statutes, precedents
  - `FinancialExtractor`: Extracts companies, transactions, indicators
  - `IoTEventExtractor`: Extracts devices, sensors, readings, locations
  - `HealthcareExtractor`: Extracts patients, conditions, treatments, providers

**Phase 3: Extractor Composition**
- Allow chaining multiple extractors
- Support fallback strategies
- Enable parallel extraction with merging

#### Priority Rationale
- **Impact**: Directly addresses domain specificity in core extraction
- **Complexity**: High - touches critical path
- **Dependencies**: Depends on Directive #1 (prompts)

#### Blockers
- **Performance**: Multiple extractors may impact latency
- **Conflict Resolution**: Different extractors may produce conflicting entities
- **Schema Validation**: Need flexible validation for diverse entity types

#### Success Metrics
- 3+ domain-specific extractors implemented
- 50%+ reduction in domain customization code
- No performance degradation for default use case

---

### 3. **Centralized Configuration Management** 🟡 **Priority: HIGH**

#### Objective
Create a unified configuration system for LLM clients, embedders, and other components.

#### Implementation Plan

**Phase 1: Configuration Schema**
- Create `GraphitiConfig` with hierarchical structure
- Support environment variables, config files (YAML/TOML), and programmatic config
- Add validation with Pydantic

```python
class LLMProviderConfig(BaseModel):
    provider: Literal["openai", "anthropic", "gemini", "groq", "custom"]
    model: str
    small_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 1.0
    max_tokens: int = 8192

class EmbedderConfig(BaseModel):
    provider: Literal["openai", "voyage", "gemini", "custom"]
    model: str
    api_key: str | None = None
    embedding_dim: int | None = None

class GraphitiConfig(BaseModel):
    llm: LLMProviderConfig
    embedder: EmbedderConfig
    database: DatabaseConfig
    extraction: ExtractionConfig
    search: SearchConfig
```

**Phase 2: Config Loading & Merging**
- Support config file discovery (`.graphiti.yaml`, `graphiti.config.toml`)
- Merge configs from multiple sources (file < env < code)
- Add config validation and helpful error messages

**Phase 3: Domain-Specific Presets**
- Create preset configs for common use cases
- Support config inheritance and composition

```yaml
# Example: .graphiti.yaml
extends: "presets/scientific-research"

llm:
  provider: anthropic
  model: claude-sonnet-4-5-latest
  temperature: 0.3

extraction:
  entity_types:
    - Researcher
    - Institution
    - Finding
    - Methodology

  extractors:
    - type: llm
      prompt: prompts/scientific_entities.yaml
    - type: regex
      patterns: prompts/scientific_patterns.yaml
```

#### Priority Rationale
- **Impact**: Simplifies deployment and customization
- **Complexity**: Medium
- **Dependencies**: None

#### Blockers
- **Backward Compatibility**: Must support existing initialization patterns
- **Security**: API keys and credentials management
- **Validation**: Complex validation rules across providers

#### Success Metrics
- Single config file for complete setup
- Zero hardcoded defaults in core code
- 10+ domain preset configs available

---

### 4. **Extensible Episode Type System** 🟡 **Priority: HIGH**

#### Objective
Allow users to define custom episode types with associated extraction logic.

#### Implementation Plan

**Phase 1: Episode Type Registry**
- Create `EpisodeTypeRegistry` for dynamic episode types
- Support custom episode type definitions with Pydantic

```python
class EpisodeTypeDefinition(BaseModel):
    name: str
    description: str
    content_schema: type[BaseModel] | None = None
    extraction_strategy: str | ExtractionStrategy
    prompt_template: str | None = None

class EpisodeTypeRegistry:
    def register(self, episode_type: EpisodeTypeDefinition) -> None:
        pass

    def get(self, name: str) -> EpisodeTypeDefinition:
        pass
```

**Phase 2: Dynamic Dispatch**
- Modify `extract_nodes()` to dispatch based on episode type
- Support fallback to default extraction for undefined types

**Phase 3: Common Episode Types**
- Provide built-in types for common domains:
  - `scientific_paper`
  - `legal_document`
  - `financial_report`
  - `iot_event`
  - `healthcare_record`
  - `email`
  - `api_log`

#### Priority Rationale
- **Impact**: Removes major extensibility bottleneck
- **Complexity**: Medium
- **Dependencies**: Depends on Directive #2 (extractors)

#### Blockers
- **Type Safety**: Ensuring type safety with dynamic types
- **Validation**: Schema validation for custom content
- **Migration**: Migrating existing message/text/JSON types

#### Success Metrics
- Users can add episode types without code changes
- 5+ built-in episode types for different domains
- Clear migration path from existing types

---

### 5. **Domain-Specific Search Strategies** 🟢 **Priority: MEDIUM**

#### Objective
Provide domain-optimized search configurations and strategies.

#### Implementation Plan

**Phase 1: Search Strategy Templates**
- Create domain-specific search configs in `search_config_recipes.py`
- Optimize for domain characteristics (e.g., temporal for financial, spatial for IoT)

```python
# Examples
FINANCIAL_TEMPORAL_SEARCH = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.cosine_similarity,
            EdgeSearchMethod.bm25,
        ],
        reranker=EdgeReranker.episode_mentions,
    ),
    # Prioritize recent events
    # ... domain-specific configuration
)

SCIENTIFIC_CITATION_SEARCH = SearchConfig(
    # Optimize for citation networks
    # ... domain-specific configuration
)
```

**Phase 2: Semantic Domain Adapters**
- Create domain-specific query expansion
- Add domain vocabulary mapping
- Support domain-specific relevance scoring

**Phase 3: Search Analytics**
- Track search performance by domain
- Provide domain-specific search insights
- Auto-tune search configs based on usage

#### Priority Rationale
- **Impact**: Improves search quality for specific domains
- **Complexity**: Low-Medium
- **Dependencies**: None - additive feature

#### Blockers
- **Domain Expertise**: Requires deep understanding of each domain
- **Evaluation**: Need domain-specific test datasets
- **Maintenance**: Each domain strategy needs ongoing optimization

#### Success Metrics
- 5+ domain-optimized search strategies
- Measurable improvement in domain-specific retrieval quality
- Search strategy recommendation system

---

### 6. **Multi-Provider LLM & Embedder Support Enhancement** 🟢 **Priority: MEDIUM**

#### Objective
Improve support for diverse LLM and embedding providers, addressing current issues with Azure, Anthropic, and local models.

#### Implementation Plan

**Phase 1: Provider Abstraction Improvements**
- Enhance `LLMClient` interface for provider-specific features
- Better handling of structured output across providers (#1007)
- Unified error handling and retries

**Phase 2: Provider-Specific Optimizations**
- Azure OpenAI full support (#1004, #995, #1006)
- Anthropic optimization for structured output
- Local model support (Ollama, vLLM) (#1074, #1007)
- Google Cloud Vertex AI integration

**Phase 3: Embedder Flexibility**
- Support mixed embedding strategies (different models for nodes vs edges)
- Domain-specific embedding fine-tuning
- Embedding dimension adaptation (#1087)

#### Priority Rationale
- **Impact**: Addresses multiple GitHub issues, improves flexibility
- **Complexity**: Medium-High (provider-specific quirks)
- **Dependencies**: Related to Directive #3 (config)

#### Blockers
- **Provider API Changes**: External dependencies on provider APIs
- **Testing**: Requires access to multiple provider accounts
- **Cost**: Testing across providers can be expensive

#### Success Metrics
- All providers in CLAUDE.md fully supported
- Resolution of issues #1004, #1006, #1007, #1074, #995
- Provider switching with zero code changes

---

### 7. **Enhanced Metadata & Custom Attributes** 🟢 **Priority: MEDIUM**

#### Objective
Support domain-specific metadata on all graph elements (nodes, edges, episodes).

#### Implementation Plan

**Phase 1: Flexible Metadata Schema**
- Add `custom_metadata: dict[str, Any]` to all core types
- Support typed metadata with Pydantic models
- Index metadata for searchability

**Phase 2: Domain-Specific Attributes**
- Support custom attributes per domain
- Attribute extraction from episodes
- Attribute-based filtering in search

**Phase 3: Metadata API Improvements**
- Episode API enhancements (#961)
- Metadata update operations
- Bulk metadata operations

#### Priority Rationale
- **Impact**: Enables rich domain modeling
- **Complexity**: Low-Medium
- **Dependencies**: Database schema changes

#### Blockers
- **Schema Migration**: Existing graphs need migration
- **Index Performance**: Metadata indexing may impact performance
- **Validation**: Complex validation for diverse metadata

#### Success Metrics
- Custom metadata on all graph elements
- Metadata-based search and filtering
- Resolution of issue #961

---

### 8. **Database Provider Expansion** 🔵 **Priority: LOW**

#### Objective
Support additional graph databases to meet diverse deployment requirements.

#### Implementation Plan

**Phase 1: Abstract Driver Interface**
- Enhance `GraphDriver` abstraction
- Standardize query translation layer
- Support for property graph vs RDF models

**Phase 2: New Drivers**
- Google Cloud Spanner Graph (#1077)
- Apache AGE (#947)
- Amazon Neptune improvements (#1082)
- TigerGraph, NebulaGraph

**Phase 3: Driver Selection Guide**
- Performance comparison matrix
- Use case recommendations
- Migration tools between drivers

#### Priority Rationale
- **Impact**: Addresses specific GitHub requests, increases deployment options
- **Complexity**: High (each driver is significant work)
- **Dependencies**: None

#### Blockers
- **Maintenance Burden**: Each driver requires ongoing support
- **Feature Parity**: Different databases have different capabilities
- **Testing**: Complex integration testing for each database

#### Success Metrics
- 2+ new database drivers
- Resolution of issues #1077, #947
- Database migration tools

---

### 9. **Documentation & Examples for Domain Adaptation** 🟡 **Priority: HIGH**

#### Objective
Comprehensive documentation showing how to adapt Graphiti to different domains.

#### Implementation Plan

**Phase 1: Domain Adaptation Guide**
- Step-by-step guide for domain customization
- Decision tree for configuration choices
- Best practices for each domain type

**Phase 2: Complete Domain Examples**
- Scientific Research knowledge graph
- Legal Document analysis
- Financial Transaction network
- IoT Event processing
- Healthcare Records integration

**Phase 3: Tutorial Series**
- Video walkthroughs
- Interactive Jupyter notebooks
- Code generation tools for domain setup

#### Priority Rationale
- **Impact**: Critical for adoption in new domains
- **Complexity**: Medium (requires domain expertise)
- **Dependencies**: Depends on implementation of above directives

#### Blockers
- **Domain Expertise**: Need experts for each domain
- **Maintenance**: Examples need to stay current with codebase
- **Quality**: Need real-world datasets and validation

#### Success Metrics
- 5+ complete domain examples
- Documentation coverage >80%
- User-contributed domain examples

---

### 10. **Testing & Evaluation Framework for Domains** 🟢 **Priority: MEDIUM**

#### Objective
Create domain-specific test datasets and evaluation metrics.

#### Implementation Plan

**Phase 1: Domain Test Datasets**
- Curate/generate test data for each domain
- Include ground truth annotations
- Support for evaluation benchmarks

**Phase 2: Evaluation Metrics**
- Domain-specific quality metrics
- Extraction accuracy measurements
- Search relevance evaluation

**Phase 3: Continuous Evaluation**
- Automated testing across domains
- Performance regression detection
- Quality dashboards

#### Priority Rationale
- **Impact**: Ensures quality across domains
- **Complexity**: Medium
- **Dependencies**: Depends on domain implementations

#### Blockers
- **Data Acquisition**: Domain datasets can be hard to obtain
- **Annotation**: Ground truth annotation is expensive
- **Standardization**: Metrics vary significantly by domain

#### Success Metrics
- Test coverage >70% across domains
- Automated evaluation pipeline
- Public benchmark results

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
**Critical Infrastructure**
- [ ] Directive #1: Configurable Prompt System
- [ ] Directive #3: Centralized Configuration Management
- [ ] Directive #9: Initial documentation framework

**Estimated Effort**: 2-3 engineers, 3 months

### Phase 2: Core Extensibility (Months 4-6)
**Domain Adaptation**
- [ ] Directive #2: Pluggable NER Pipeline
- [ ] Directive #4: Extensible Episode Types
- [ ] Directive #7: Enhanced Metadata

**Estimated Effort**: 2-3 engineers, 3 months

### Phase 3: Provider & Database Support (Months 7-9)
**Infrastructure Expansion**
- [ ] Directive #6: Multi-Provider LLM Support
- [ ] Directive #8: Database Provider Expansion (Phase 1)

**Estimated Effort**: 2 engineers, 3 months

### Phase 4: Domain Optimization (Months 10-12)
**Domain-Specific Features**
- [ ] Directive #5: Domain-Specific Search
- [ ] Directive #10: Testing & Evaluation Framework
- [ ] Directive #9: Complete domain examples

**Estimated Effort**: 2-3 engineers, 3 months

---

## Risk Assessment

### High Risk
1. **Breaking Changes**: Refactoring may break existing integrations
   - *Mitigation*: Semantic versioning, deprecation warnings, migration guides

2. **Performance Regression**: More abstraction may impact performance
   - *Mitigation*: Continuous benchmarking, performance budgets

3. **Complexity Creep**: Too much configurability can confuse users
   - *Mitigation*: Sensible defaults, progressive disclosure, presets

### Medium Risk
1. **Provider API Changes**: External dependencies may change
   - *Mitigation*: Abstract interfaces, version pinning, adapter pattern

2. **Maintenance Burden**: More features = more maintenance
   - *Mitigation*: Automated testing, clear ownership, deprecation policy

3. **Documentation Debt**: Fast development may outpace docs
   - *Mitigation*: Docs-as-code, automated doc generation, examples as tests

### Low Risk
1. **Community Adoption**: Users may not need all domains
   - *Mitigation*: Modular architecture, optional components

---

## Success Criteria

### Technical Metrics
- [ ] Zero hardcoded domain assumptions in core library
- [ ] 5+ domain-specific configurations available
- [ ] All GitHub issues (#1004, #1006, #1007, #1074, #995, #1077, #947, #961) resolved
- [ ] Test coverage >75% across all domains
- [ ] Performance within 10% of current baseline

### User Experience Metrics
- [ ] Domain setup time <30 minutes (from docs)
- [ ] Config-driven customization (no code changes for 80% of use cases)
- [ ] 3+ community-contributed domain adaptations

### Business Metrics
- [ ] Adoption in 3+ new domains (outside conversational AI)
- [ ] 50%+ reduction in customization support requests
- [ ] Documentation satisfaction >4.0/5.0

---

## Appendix A: Affected Files

### Core Files Requiring Changes

**High Priority**
- `graphiti_core/graphiti.py` - Main class, initialization
- `graphiti_core/llm_client/config.py` - Configuration system
- `graphiti_core/prompts/extract_nodes.py` - NER prompts
- `graphiti_core/prompts/extract_edges.py` - Relation extraction prompts
- `graphiti_core/utils/maintenance/node_operations.py` - Extraction logic

**Medium Priority**
- `graphiti_core/nodes.py` - Episode type definitions
- `graphiti_core/search/search_config.py` - Search configuration
- `graphiti_core/search/search_config_recipes.py` - Search recipes
- `server/graph_service/config.py` - Server configuration

**Low Priority**
- `graphiti_core/driver/*.py` - Database drivers
- `graphiti_core/embedder/*.py` - Embedder clients

---

## Appendix B: Related GitHub Issues

### Directly Addressed
- #1087: Embedding truncation
- #1074: No results with Ollama embeddings
- #1007: Unstable outputs with vLLM
- #1006: AzureOpenAI reranker support
- #1004: Azure OpenAI support
- #995: Docker Azure OpenAI support
- #1077: Google Cloud Spanner Graph support
- #947: Apache AGE support
- #961: Episodes API improvements
- #1082: Neptune driver issues

### Indirectly Improved
- #1083: Orphaned entities cleanup
- #1062: Stale data in MCP server
- #1021: Incomplete graph structure
- #1018: Search with group_ids
- #1012: group_id and Anthropic issues
- #992: OOM in build_communities
- #963: Duplicate entities

---

## Appendix C: Backward Compatibility Strategy

### Deprecation Policy
1. **Feature Deprecation**: 2 minor versions notice
2. **API Changes**: Maintain old API with deprecation warnings
3. **Configuration**: Support both old and new config formats during transition

### Migration Support
- Automated migration scripts for major changes
- Detailed migration guides for each release
- Migration validation tools

### Version Support
- LTS releases for enterprise users
- Security patches for N-2 versions
- Clear EOL policy

---

## Next Steps

1. **Review & Approval**: Circulate this plan for stakeholder feedback
2. **Prioritization**: Finalize directive priorities based on business needs
3. **Resource Allocation**: Assign engineering teams to Phase 1 directives
4. **Kickoff**: Begin implementation of Directive #1 (Prompt System)

---

**Document Maintainer**: Claude (AI Assistant)
**Last Updated**: 2025-11-30
**Next Review**: After Phase 1 completion
