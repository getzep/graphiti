# Duplicate GitHub Issues

**Date**: 2025-10-07

## Confirmed Duplicates (Already Marked)

### [#963](https://github.com/getzep/graphiti/issues/963) - Duplicate Entities in Neo4j
- **Status**: Marked as duplicate
- **Original Issue**: [#875](https://github.com/getzep/graphiti/issues/875) - Duplicate Entities in Neo4j with custom db name
- **Labels**: bug, duplicate
- **Action**: Can be closed with reference to [#875](https://github.com/getzep/graphiti/issues/875)
- **Notes**: Same core issue - deduplication not working properly

---

### [#941](https://github.com/getzep/graphiti/issues/941) - TaskGroup Errors
- **Title**: ERROR：unhandled errors in a TaskGroup (1 sub-exception)
- **Status**: Marked as duplicate
- **Labels**: duplicate
- **Notes**: Reporter mentioned issue was previously reported in [#353](https://github.com/getzep/graphiti/issues/353)
- **Action**: Close with reference to original issue

---

### [#920](https://github.com/getzep/graphiti/issues/920) - Timezone-Naive/Aware Datetime Comparison
- **Title**: [BUG] edge_operations.py is unable to compare a timezone-naive datetime with a timezone-aware one
- **Status**: Marked as duplicate
- **Labels**: bug, duplicate
- **Original Issue**: Likely resolved by commit 73015e9 "Fix datetime comparison errors by normalizing to UTC"
- **Action**: Close with reference to recent fix. Ask reporter to verify on latest version.

---

### [#867](https://github.com/getzep/graphiti/issues/867) - MCP with GPT-oss Models
- **Title**: MPC does not work with GPT-oss:20 or 120B
- **Status**: Marked as duplicate
- **Duplicate Of**: [#831](https://github.com/getzep/graphiti/issues/831) - [BUG] GPT-oss:20 and 120
- **Labels**: bug, duplicate
- **Action**: Close with reference to [#831](https://github.com/getzep/graphiti/issues/831)

---

### [#801](https://github.com/getzep/graphiti/issues/801) - Empty Fulltext Search Results
- **Title**: [BUG] empty-result bug in episode_fulltext_search
- **Status**: Marked as duplicate
- **Labels**: bug, duplicate
- **Original Issue**: Likely [#810](https://github.com/getzep/graphiti/issues/810) - Empty group_id handling issues
- **Action**: Close with reference to [#810](https://github.com/getzep/graphiti/issues/810) or related search issue

---

### [#787](https://github.com/getzep/graphiti/issues/787) - Rate Limit with SEMAPHORE_LIMIT
- **Title**: [BUG] Got rate limit even SEMAPHORE_LIMIT=1 in mcp server
- **Status**: Marked as duplicate
- **Labels**: bug, duplicate
- **Action**: Close with reference to original rate limiting issue

---

## Potential Duplicates (Require Investigation)

### Database Configuration Issues (Likely Related)

[#851](https://github.com/getzep/graphiti/issues/851), [#798](https://github.com/getzep/graphiti/issues/798), [#715](https://github.com/getzep/graphiti/issues/715) - All relate to database name handling
- [#851](https://github.com/getzep/graphiti/issues/851) - Search only connects to 'neo4j' db
- [#798](https://github.com/getzep/graphiti/issues/798) - Database name not passed through Graphiti object
- [#715](https://github.com/getzep/graphiti/issues/715) - Feature request to configure Neo4j database name

**Analysis**: These may be describing the same underlying issue. [#715](https://github.com/getzep/graphiti/issues/715) appears to be the feature request, while [#851](https://github.com/getzep/graphiti/issues/851) and [#798](https://github.com/getzep/graphiti/issues/798) are bugs from the missing feature.

**Recommendation**: Fix the root cause and consolidate. Keep [#715](https://github.com/getzep/graphiti/issues/715) as the tracking issue if implementing multi-DB support properly.

---

### Bulk Upload Failures

[#882](https://github.com/getzep/graphiti/issues/882), [#879](https://github.com/getzep/graphiti/issues/879), [#871](https://github.com/getzep/graphiti/issues/871), [#658](https://github.com/getzep/graphiti/issues/658) - Bulk upload failures
- [#882](https://github.com/getzep/graphiti/issues/882) - IndexError during node resolution
- [#879](https://github.com/getzep/graphiti/issues/879) - ValidationError 'duplicates' field missing
- [#871](https://github.com/getzep/graphiti/issues/871) - Invalid JSON and index errors
- [#658](https://github.com/getzep/graphiti/issues/658) - "Bulk ingestion not possible"

**Analysis**: All appear to be bulk upload failures, potentially from same root cause in validation/schema handling.

**Recommendation**: Investigate if these are manifestations of same bug. If so, consolidate into single issue.

---

### FalkorDB Query Errors

[#815](https://github.com/getzep/graphiti/issues/815), [#757](https://github.com/getzep/graphiti/issues/757), [#731](https://github.com/getzep/graphiti/issues/731) - FalkorDB query errors
- [#815](https://github.com/getzep/graphiti/issues/815) - falkordb query error
- [#757](https://github.com/getzep/graphiti/issues/757) - quickstart_falkordb example query error
- [#731](https://github.com/getzep/graphiti/issues/731) - Malformed Cypher query on episode insertion

**Analysis**: May be same underlying issue with FalkorDB Cypher query generation.

**Recommendation**: Investigate if root cause is same. Could consolidate.

---

### MCP Server + Custom LLM Providers

[#565](https://github.com/getzep/graphiti/issues/565), [#945](https://github.com/getzep/graphiti/issues/945) - OPENAI_BASE_URL issues
- [#565](https://github.com/getzep/graphiti/issues/565) - Cross-encoder ignores OPENAI_BASE_URL
- [#945](https://github.com/getzep/graphiti/issues/945) - Custom OPENAI_BASE_URL causes NaN embeddings

**Analysis**: Both relate to custom OpenAI-compatible endpoints not being respected in MCP server.

**Recommendation**: Likely same root issue - configuration not properly passed through MCP components.

---

### Ollama Compatibility

[#868](https://github.com/getzep/graphiti/issues/868), [#831](https://github.com/getzep/graphiti/issues/831) - Ollama issues
- [#868](https://github.com/getzep/graphiti/issues/868) - Cannot get minimal example to work with Ollama
- [#831](https://github.com/getzep/graphiti/issues/831) - GPT-oss:20 and 120 (Ollama models)

**Analysis**: Both relate to Ollama model compatibility issues.

---

### Search Result Issues

[#534](https://github.com/getzep/graphiti/issues/534), [#801](https://github.com/getzep/graphiti/issues/801), [#810](https://github.com/getzep/graphiti/issues/810) - Search returning empty/no results
- [#534](https://github.com/getzep/graphiti/issues/534) - retrieve_episodes always returns no results
- [#801](https://github.com/getzep/graphiti/issues/801) - episode_fulltext_search empty results (marked duplicate)
- [#810](https://github.com/getzep/graphiti/issues/810) - Empty group_id handled inconsistently

**Analysis**: Likely all related to search filtering/group_id handling issues.

---

### Documentation/Setup Questions (Can Be Consolidated)

[#517](https://github.com/getzep/graphiti/issues/517), [#530](https://github.com/getzep/graphiti/issues/530) - OpenRouter setup questions
- [#517](https://github.com/getzep/graphiti/issues/517) - How to setup with OpenRouter and Voyage
- [#530](https://github.com/getzep/graphiti/issues/530) - Does this work with Cursor AI + OpenRouter ChatGPT

**Analysis**: Same topic - using OpenRouter as provider. Could close with documentation reference.

---

## Summary

- **Confirmed Duplicates (Already Marked)**: 6 issues ready to close
- **Potential Duplicate Clusters**: 7 clusters (20+ issues) requiring investigation
- **Estimated Consolidation**: Could reduce issue count by 15-25 through deduplication

## Recommended Actions

1. **Immediate**: Close the 6 confirmed duplicates with appropriate references
2. **Investigation**: Review potential duplicate clusters to confirm root causes
3. **Consolidation**: Create tracking issues for clusters where multiple issues stem from same bug
4. **Documentation**: Several "questions" can be closed once docs are updated
