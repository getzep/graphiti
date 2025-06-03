# Enhanced RAG Components

This document describes the enhanced Retrieval-Augmented Generation (RAG) components that improve information preservation and retrieval quality in the Graphiti knowledge graph system.

## Overview

The enhanced RAG system addresses key challenges in traditional knowledge graph RAG:

1. **Information Loss**: During text → graph → text conversion
2. **Context Preservation**: Maintaining nuanced information and relationships
3. **Retrieval Quality**: Better ranking and relevance of retrieved information
4. **Error Handling**: Robust fallback mechanisms for production use

## Components

### 1. Enhanced Cross-Encoder Client (`enhanced_client.py`)

Improves passage ranking with caching, batching, and error handling.

**Features:**
- **Caching**: Avoids recomputing rankings for repeated queries
- **Batch Processing**: Handles large passage lists efficiently  
- **Timeout Protection**: Prevents hanging operations
- **Fallback Ranking**: Simple similarity-based fallback when models fail
- **Composite Support**: Can combine multiple cross-encoder models

**Usage:**
```python
from graphiti_core.cross_encoder.enhanced_client import EnhancedCrossEncoderClient

class MyCrossEncoder(EnhancedCrossEncoderClient):
    async def _rank_batch(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        # Implement your ranking logic
        pass

client = MyCrossEncoder(
    cache_size=1000,
    batch_size=32, 
    timeout=10.0,
    fallback_enabled=True
)

ranked_passages = await client.rank(query, passages)
```

### 2. Enhanced Entity Extractor (`enhanced_entity_extractor.py`)

Extracts entities while preserving more contextual information.

**Features:**
- **Context Preservation**: Maintains original text snippets, temporal/spatial context
- **Rich Properties**: Extracts emotional context, confidence levels, and metadata
- **Enhanced Prompting**: More comprehensive extraction prompts
- **Fallback Extraction**: Graceful degradation when advanced extraction fails

**Usage:**
```python
from graphiti_core.utils.enhanced_entity_extractor import EnhancedEntityExtractor

extractor = EnhancedEntityExtractor(llm_client)
result = await extractor.extract_with_context_preservation(text)

# Create nodes that preserve context
nodes = extractor.create_context_preserving_nodes(result)
```

**Extraction Result Structure:**
```python
{
    "entities": [
        {
            "name": "entity name",
            "type": "PERSON|ORGANIZATION|...",
            "properties": {"key": "value"},
            "temporal_context": "when information",
            "spatial_context": "where information", 
            "emotional_context": "emotional context",
            "original_context": "original text snippet",
            "confidence": 0.95,
            "summary": "entity description"
        }
    ],
    "relationships": [...],
    "global_context": {
        "overall_theme": "main topic",
        "temporal_setting": "time period",
        "spatial_setting": "location",
        "mood_tone": "overall mood"
    }
}
```

### 3. Multi-Level Context Manager (`multi_level_context_manager.py`)

Manages context at multiple levels to preserve information during graph-to-text conversion.

**Features:**
- **Entity-Level Context**: Individual entity information with preserved context
- **Relationship-Level Context**: Detailed relationship information
- **Subgraph-Level Patterns**: Graph structure patterns and clusters
- **Temporal Sequencing**: Timeline construction from temporal information
- **Global Context Summary**: Overall context summary and complexity indicators

**Usage:**
```python
from graphiti_core.utils.multi_level_context_manager import MultiLevelContextManager

manager = MultiLevelContextManager()
context = manager.build_comprehensive_context(query, entities, edges)
```

**Context Output Structure:**
```
Query: [user question]

=== ENTITIES ===
• John Smith (PERSON)
  Summary: Software engineer at Microsoft
  Original Context: John Smith, a senior engineer...
  When: currently employed
  Properties: age: 30, department: AI

=== RELATIONSHIPS ===  
• John Smith works at Microsoft
  Details: Employment relationship since 2022
  Original Context: John has been working at Microsoft...

=== GRAPH PATTERNS ===
Central entities: Microsoft, AI Project
PERSON cluster: John Smith, Sarah Johnson
Common relationships: works at (3 instances)

=== TIMELINE ===
• 2022: John Smith joined Microsoft
• Yesterday: Meeting between John and Sarah

=== CONTEXT SUMMARY ===
Total entities: 5 (PERSON, ORGANIZATION, PROJECT)
Total relationships: 8 (works at, collaborates with, manages)
```

### 4. Compression-Aware Retrieval (`compression_aware_retrieval.py`)

Detects information loss and compensates with original text snippets.

**Features:**
- **Compression Ratio Estimation**: Measures information preservation 
- **Original Snippet Extraction**: Retrieves relevant original text
- **Hybrid Context Generation**: Combines structured graph data with original text
- **Quality Metrics**: Provides comprehensive quality assessment
- **Adaptive Context**: Adjusts context based on compression ratio

**Usage:**
```python
from graphiti_core.utils.compression_aware_retrieval import CompressionAwareRetrieval

retrieval = CompressionAwareRetrieval(compression_ratio_threshold=0.7)
result = await retrieval.retrieve_with_compression_awareness(query, search_function, config)

# Access comprehensive context
context = result.context
compression_ratio = result.compression_ratio  
original_snippets = result.original_snippets
```

**RetrievalResult Structure:**
```python
@dataclass
class RetrievalResult:
    entities: List[EntityNode]
    edges: List[EntityEdge] 
    context: str                    # Multi-level context
    compression_ratio: float        # 0.0-1.0, higher = better preservation
    original_snippets: List[str]    # Original text snippets
    metadata: Dict[str, Any]        # Processing metadata
```

### 5. Enhanced RAG Pipeline (`enhanced_rag_pipeline.py`)

Complete RAG pipeline with all enhancements integrated.

**Features:**
- **Compression-Aware Retrieval**: Automatic detection and compensation for information loss
- **Cross-Encoder Reranking**: Optional passage reranking for better relevance
- **Context Length Management**: Intelligent context truncation
- **Fallback Mechanisms**: Multiple levels of fallback for robustness
- **Quality Monitoring**: Comprehensive quality metrics and monitoring
- **Error Recovery**: Graceful handling of component failures

**Usage:**
```python
from graphiti_core.utils.enhanced_rag_pipeline import EnhancedRAGPipeline

pipeline = EnhancedRAGPipeline(
    llm_client=llm_client,
    search_function=search_function,
    cross_encoder_client=cross_encoder,
    compression_ratio_threshold=0.7,
    max_context_length=8000,
    enable_fallbacks=True
)

response = await pipeline.query(
    question="What is the relationship between John and Sarah?",
    max_entities=20,
    rerank_passages=True
)

print(f"Answer: {response.answer}")
print(f"Quality: {response.quality_metrics}")
print(f"Processing time: {response.processing_time}s")
```

**RAGResponse Structure:**
```python
@dataclass  
class RAGResponse:
    answer: str                    # Generated answer
    retrieval_result: RetrievalResult  # Detailed retrieval information
    quality_metrics: Dict[str, Any]    # Quality assessment
    processing_time: float             # Total processing time
    fallback_used: bool               # Whether fallback was used
```

## Quality Metrics

The enhanced system provides comprehensive quality metrics:

```python
{
    "compression_ratio": 0.75,           # Information preservation ratio
    "information_density": 15,           # Number of entities + edges  
    "context_length": 2500,              # Context length in characters
    "original_snippets_coverage": 0.8,   # Coverage of original snippets
    "hybrid_context_used": True,         # Whether hybrid context was used
    "processing_time": 1.2,              # Processing time in seconds
    "answer_length": 150,                # Answer length in characters
    "answer_has_uncertainty": False,     # Whether answer expresses uncertainty
    "context_utilization": 0.6,          # How well context was utilized
    "retrieval_success": True           # Whether retrieval found relevant info
}
```

## Migration Guide

### From Basic RAG to Enhanced RAG:

1. **Replace Cross-Encoder:**
```python
# Before
from graphiti_core.cross_encoder.client import CrossEncoderClient

# After  
from graphiti_core.cross_encoder.enhanced_client import EnhancedCrossEncoderClient
```

2. **Use Enhanced Pipeline:**
```python
# Before: Direct search + LLM generation
results = await graphiti.search(query)
context = build_context(results)
answer = await llm.generate(context)

# After: Enhanced pipeline
pipeline = EnhancedRAGPipeline(llm_client, search_function)
response = await pipeline.query(query)
answer = response.answer
```

3. **Leverage Context Preservation:**
```python
# Before: Basic entity extraction
entities = await extract_entities(text)

# After: Enhanced extraction with context
extractor = EnhancedEntityExtractor(llm_client)
result = await extractor.extract_with_context_preservation(text)
entities = extractor.create_context_preserving_nodes(result)
```

## Performance Considerations

1. **Caching**: Cross-encoder caching significantly reduces repeated computation
2. **Batching**: Batch processing improves throughput for large passage lists
3. **Context Length**: Monitor context length to avoid LLM token limits
4. **Fallbacks**: Enable fallbacks for production robustness
5. **Compression Ratio**: Higher thresholds reduce processing but may lose information

## Best Practices

1. **Tune Compression Threshold**: Start with 0.7, adjust based on your data
2. **Monitor Quality Metrics**: Use metrics to optimize system performance
3. **Enable Caching**: Use caching in production for better performance
4. **Test Fallbacks**: Ensure fallback mechanisms work with your data
5. **Context Management**: Monitor context length and truncation behavior
6. **Original Snippets**: Preserve original context during entity extraction

## Testing

Run the test suite:
```bash
pytest tests/test_enhanced_components.py -v
```

Run the demo:
```bash
python examples/enhanced_rag_demo.py
```

## Future Enhancements

1. **Adaptive Compression**: Dynamic compression ratio based on query complexity
2. **Multi-Modal Context**: Support for images and other media in context
3. **Incremental Learning**: Learning from user feedback to improve quality
4. **Distributed Caching**: Redis-based caching for multi-instance deployments
5. **Advanced Patterns**: More sophisticated graph pattern recognition
