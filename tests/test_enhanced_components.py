"""
Test suite for enhanced RAG components
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.cross_encoder.enhanced_client import EnhancedCrossEncoderClient, CompositeCrossEncoderClient
from graphiti_core.utils.enhanced_entity_extractor import EnhancedEntityExtractor
from graphiti_core.utils.multi_level_context_manager import MultiLevelContextManager
from graphiti_core.utils.compression_aware_retrieval import CompressionAwareRetrieval
from graphiti_core.utils.enhanced_rag_pipeline import EnhancedRAGPipeline


class MockCrossEncoderClient(EnhancedCrossEncoderClient):
    """Mock implementation for testing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
    
    async def _rank_batch(self, query: str, passages: List[str]) -> List[tuple[str, float]]:
        self.call_count += 1
        # Simple mock ranking based on string similarity
        results = []
        for passage in passages:
            score = len(set(query.lower().split()) & set(passage.lower().split())) / max(1, len(query.split()))
            results.append((passage, score))
        return sorted(results, key=lambda x: x[1], reverse=True)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = Mock()
    client.generate_response = AsyncMock(return_value='{"entities": [], "relationships": []}')
    return client


@pytest.fixture
def sample_entities():
    """Sample entities for testing"""
    return [
        EntityNode(
            name="John Smith",
            group_id="test_group_1",
            labels=["PERSON"],
            summary="Software engineer at Microsoft",
            attributes={
                "original_context": "John Smith, a software engineer at Microsoft, worked on the AI project",
                "temporal_context": "2024",
                "confidence": 0.9
            }
        ),
        EntityNode(
            name="Microsoft",
            group_id="test_group_2",
            labels=["ORGANIZATION"], 
            summary="Technology company",
            attributes={
                "original_context": "Microsoft is developing new AI technologies",
                "industry": "Technology"
            }
        )
    ]


@pytest.fixture
def sample_edges():
    """Sample edges for testing"""
    from datetime import datetime
    return [
        EntityEdge(
            source_node_uuid="test-uuid-1",
            target_node_uuid="test-uuid-2", 
            name="WORKS_AT",
            group_id="test_group_1",
            fact="John Smith works at Microsoft as a software engineer",
            created_at=datetime.utcnow(),
            attributes={
                "original_context": "John Smith works at Microsoft as a software engineer",
                "temporal_context": "Current employment"
            }
        )
    ]


class TestEnhancedCrossEncoderClient:
    """Test enhanced cross-encoder functionality"""
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test caching functionality"""
        client = MockCrossEncoderClient(cache_size=10)
        
        query = "test query"
        passages = ["passage 1", "passage 2"]
        
        # First call
        result1 = await client.rank(query, passages)
        assert client.call_count == 1
        
        # Second call with same input should use cache
        result2 = await client.rank(query, passages)
        assert client.call_count == 1  # No additional calls
        assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_batching(self):
        """Test batch processing"""
        client = MockCrossEncoderClient(batch_size=2)
        
        query = "test query"
        passages = ["passage 1", "passage 2", "passage 3", "passage 4"]
        
        result = await client.rank(query, passages)
        
        assert len(result) == 4
        assert all(isinstance(item[1], float) for item in result)
        # Should have made multiple batch calls
        assert client.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self):
        """Test fallback mechanism on timeout"""
        
        class TimeoutClient(EnhancedCrossEncoderClient):
            async def _rank_batch(self, query: str, passages: List[str]):
                await asyncio.sleep(1)  # Simulate slow operation
                return []
        
        client = TimeoutClient(timeout=0.1, fallback_enabled=True)
        
        query = "test query"
        passages = ["passage 1", "passage 2"]
        
        # Should fallback to simple ranking
        result = await client.rank(query, passages)
        assert len(result) == 2
        assert all(isinstance(item[1], float) for item in result)


class TestMultiLevelContextManager:
    """Test multi-level context management"""
    
    def test_entity_context_building(self, sample_entities):
        """Test entity context building"""
        manager = MultiLevelContextManager()
        
        entity = sample_entities[0]  # John Smith
        context = manager._build_entity_context(entity)
        
        assert "John Smith" in context
        assert "PERSON" in context
        assert "Software engineer at Microsoft" in context
        assert "original_context" not in context  # Should be processed, not raw
    
    def test_comprehensive_context_building(self, sample_entities, sample_edges):
        """Test comprehensive context building"""
        manager = MultiLevelContextManager()
        
        context = manager.build_comprehensive_context(
            "Tell me about John Smith",
            sample_entities,
            sample_edges
        )
        
        assert "Query: Tell me about John Smith" in context
        assert "=== ENTITIES ===" in context
        assert "=== RELATIONSHIPS ===" in context
        assert "John Smith" in context
        assert "Microsoft" in context
        assert "works at" in context.lower()


class TestCompressionAwareRetrieval:
    """Test compression-aware retrieval"""
    
    def test_compression_ratio_calculation(self, sample_entities, sample_edges):
        """Test compression ratio estimation"""
        retrieval = CompressionAwareRetrieval()
        
        ratio = retrieval._estimate_compression_ratio(sample_entities, sample_edges)
        
        assert 0.0 <= ratio <= 1.0
        # With original context preserved, ratio should be reasonable
        assert ratio > 0.1
    
    @pytest.mark.asyncio
    async def test_original_snippet_extraction(self, sample_entities, sample_edges):
        """Test extraction of original text snippets"""
        retrieval = CompressionAwareRetrieval(max_original_snippets=5)
        
        snippets = await retrieval._get_original_text_snippets(sample_entities, sample_edges)
        
        assert len(snippets) > 0
        assert any("John Smith" in snippet for snippet in snippets)
        assert any("Microsoft" in snippet for snippet in snippets)
    
    @pytest.mark.asyncio
    async def test_hybrid_context_generation(self, sample_entities, sample_edges):
        """Test hybrid context generation"""
        retrieval = CompressionAwareRetrieval()
        
        original_snippets = ["John Smith, a software engineer at Microsoft, worked on the AI project"]
        
        context = retrieval._merge_graph_and_text_context(
            sample_entities, 
            sample_edges,
            original_snippets,
            "Tell me about John Smith"
        )
        
        assert "Query: Tell me about John Smith" in context
        assert "=== STRUCTURED KNOWLEDGE ===" in context
        assert "=== ORIGINAL CONTEXT SNIPPETS ===" in context
        assert "John Smith, a software engineer" in context


class TestEnhancedEntityExtractor:
    """Test enhanced entity extraction"""
    
    @pytest.mark.asyncio
    async def test_context_preservation_extraction(self, mock_llm_client):
        """Test extraction with context preservation"""
        extractor = EnhancedEntityExtractor(mock_llm_client)
        
        # Mock LLM response
        mock_response = '''
        {
            "entities": [
                {
                    "name": "John Smith",
                    "type": "PERSON",
                    "properties": {"age": "30"},
                    "temporal_context": "currently",
                    "original_context": "John Smith is a 30-year-old engineer",
                    "confidence": 0.9,
                    "summary": "A software engineer"
                }
            ],
            "relationships": [],
            "global_context": {"theme": "professional profile"}
        }
        '''
        mock_llm_client.generate_response.return_value = mock_response
        
        text = "John Smith is a 30-year-old software engineer working at Microsoft."
        result = await extractor.extract_with_context_preservation(text)
        
        assert "entities" in result
        assert "extraction_metadata" in result
        assert len(result["entities"]) == 1
        
        entity = result["entities"][0]
        assert entity["name"] == "John Smith"
        assert entity["type"] == "PERSON"
        assert "context_window" in entity
        assert "extraction_confidence" in entity
    
    def test_create_context_preserving_nodes(self, mock_llm_client):
        """Test creation of context-preserving nodes"""
        extractor = EnhancedEntityExtractor(mock_llm_client)
        
        extraction_result = {
            "entities": [
                {
                    "name": "Test Entity",
                    "type": "PERSON",                "summary": "Test summary",
                    "properties": {"key": "value"},
                    "extraction_confidence": 0.8,
                    "context_window": "surrounding text"
                }
            ]
        }
        
        nodes = extractor.create_context_preserving_nodes(extraction_result)
        
        assert len(nodes) == 1
        node = nodes[0]
        assert node.name == "Test Entity"
        assert node.labels == ["PERSON"]
        assert node.summary == "Test summary"
        assert "extraction_confidence" in node.attributes
        assert "context_window" in node.attributes


class TestEnhancedRAGPipeline:
    """Test enhanced RAG pipeline"""
    
    @pytest.fixture
    def mock_search_function(self, sample_entities, sample_edges):
        """Mock search function"""
        async def search_func(query: str, config, limit: int = 10):
            return sample_entities, sample_edges
        return search_func
    
    @pytest.mark.asyncio
    async def test_basic_query(self, mock_llm_client, mock_search_function):
        """Test basic query functionality"""
        # Mock LLM response for the final answer
        mock_llm_client.generate_response.return_value = "John Smith is a software engineer at Microsoft."
        
        pipeline = EnhancedRAGPipeline(
            llm_client=mock_llm_client,
            search_function=mock_search_function,
            enable_fallbacks=True        )
        
        response = await pipeline.query("Tell me about John Smith")
        
        assert response.answer == "John Smith is a software engineer at Microsoft."
        assert response.retrieval_result is not None
        assert response.quality_metrics is not None
        assert response.processing_time >= 0  # Processing time should be non-negative
        assert len(response.retrieval_result.entities) > 0
    
    @pytest.mark.asyncio 
    async def test_fallback_mechanism(self, mock_llm_client):
        """Test fallback mechanism when primary search fails"""
        
        async def failing_search_function(query: str, config, limit: int = 10):
            raise Exception("Search failed")
        
        mock_llm_client.generate_response.return_value = "I apologize, but I'm unable to retrieve relevant information."
        
        pipeline = EnhancedRAGPipeline(
            llm_client=mock_llm_client,
            search_function=failing_search_function,
            enable_fallbacks=True
        )
        
        response = await pipeline.query("Test query")
        
        assert response.fallback_used == True
        assert "unable to retrieve" in response.answer
    
    def test_context_truncation(self, mock_llm_client, mock_search_function):
        """Test context truncation for long contexts"""
        pipeline = EnhancedRAGPipeline(
            llm_client=mock_llm_client,
            search_function=mock_search_function,
            max_context_length=100  # Very small limit for testing
        )
        
        # Create a retrieval result with long context
        from graphiti_core.utils.compression_aware_retrieval import RetrievalResult
        
        long_context = "Very long context " * 50  # Make it longer than limit
        retrieval_result = RetrievalResult(
            entities=[],
            edges=[],
            context=long_context,
            compression_ratio=1.0,
            original_snippets=[],
            metadata={}
        )
        
        truncated_result = pipeline._truncate_context(retrieval_result)
        
        assert len(truncated_result.context) <= pipeline.max_context_length
        assert truncated_result.metadata.get('context_truncated') == True


if __name__ == "__main__":
    pytest.main([__file__])
