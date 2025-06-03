"""
Example usage of the enhanced RAG components

This script demonstrates how to use the new enhanced components
for better context preservation and information retrieval.
"""

import asyncio
import logging
from typing import Optional

from graphiti_core.graphiti import Graphiti
from graphiti_core.search import search_config_recipes
from graphiti_core.cross_encoder.enhanced_client import EnhancedCrossEncoderClient, MockCrossEncoderClient
from graphiti_core.utils.enhanced_entity_extractor import EnhancedEntityExtractor
from graphiti_core.utils.enhanced_rag_pipeline import EnhancedRAGPipeline


# Mock Cross-Encoder for demonstration
class DemoCrossEncoderClient(EnhancedCrossEncoderClient):
    """Demo cross-encoder implementation"""
    
    async def _rank_batch(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Simple demonstration ranking based on keyword overlap"""
        results = []
        query_words = set(query.lower().split())
        
        for passage in passages:
            passage_words = set(passage.lower().split())
            overlap = len(query_words & passage_words)
            score = overlap / max(1, len(query_words))
            results.append((passage, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)


async def main():
    """Main demonstration function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting enhanced RAG demonstration...")
    
    # Initialize Graphiti
    graphiti = Graphiti()
    
    # Add some sample episodes to work with
    sample_episodes = [
        {
            "name": "meeting_notes_1",
            "content": """
            John Smith, a senior software engineer at Microsoft, met with Sarah Johnson 
            from the AI research team yesterday at 3 PM in the downtown Seattle office. 
            They discussed the new machine learning project that has been causing some 
            tension in the engineering team due to tight deadlines. John expressed concerns 
            about the project timeline, while Sarah was optimistic about the AI model's 
            potential impact on customer satisfaction.
            """
        },
        {
            "name": "project_update_1", 
            "content": """
            The AI project at Microsoft is progressing well despite initial concerns. 
            The team led by Sarah Johnson has made significant breakthroughs in natural 
            language processing. John Smith has been instrumental in optimizing the 
            model's performance, reducing inference time by 40%. The project is now 
            expected to launch in Q2 2024, ahead of the original Q3 timeline.
            """
        },
        {
            "name": "team_feedback_1",
            "content": """
            Team feedback on the AI project has been overwhelmingly positive. 
            Engineers appreciate the collaborative approach taken by Sarah Johnson 
            and John Smith. The project has improved team morale and demonstrated 
            the value of cross-functional collaboration between research and 
            engineering teams at Microsoft.
            """
        }
    ]
    
    # Add episodes to the knowledge graph
    logger.info("Adding sample episodes to knowledge graph...")
    for episode in sample_episodes:
        await graphiti.add_episode(
            episode_content=episode["content"],
            episode_name=episode["name"]
        )
    
    # Initialize enhanced components
    logger.info("Initializing enhanced RAG components...")
    
    # Enhanced cross-encoder (using demo implementation)
    cross_encoder = DemoCrossEncoderClient(
        cache_size=1000,
        batch_size=16,
        timeout=10.0,
        fallback_enabled=True
    )
    
    # Enhanced RAG pipeline
    async def search_wrapper(query: str, config, limit: int = 20):
        """Wrapper for graphiti search function"""
        results = await graphiti.search(query, config=config, limit=limit)
        
        # Extract entities and edges from search results
        entities = []
        edges = []
        
        for result in results:
            if hasattr(result, 'label'):  # It's likely a node
                entities.append(result)
            else:  # It might be an edge or other result type
                # Handle based on actual result structure
                pass
        
        return entities, edges
    
    enhanced_rag = EnhancedRAGPipeline(
        llm_client=graphiti.llm_client,
        search_function=search_wrapper,
        cross_encoder_client=cross_encoder,
        compression_ratio_threshold=0.7,
        max_context_length=4000,
        enable_fallbacks=True
    )
    
    # Example queries to demonstrate enhanced capabilities
    test_queries = [
        "What is the relationship between John Smith and Sarah Johnson?",
        "Tell me about the AI project at Microsoft",
        "What concerns were raised about the project timeline?",
        "How has the project affected team morale?",
        "What are the technical achievements of the project?"
    ]
    
    logger.info("Running enhanced RAG queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*60}")
        
        try:
            # Execute enhanced RAG query
            response = await enhanced_rag.query(
                question=query,
                search_config=search_config_recipes.NODE_HYBRID_SEARCH_RRF,
                max_entities=10,
                rerank_passages=True
            )
            
            # Display results
            print(f"ANSWER: {response.answer}")
            print(f"\nQUALITY METRICS:")
            for metric, value in response.quality_metrics.items():
                print(f"  {metric}: {value}")
            
            print(f"\nRETRIEVAL INFO:")
            print(f"  Entities found: {len(response.retrieval_result.entities)}")
            print(f"  Edges found: {len(response.retrieval_result.edges)}")
            print(f"  Compression ratio: {response.retrieval_result.compression_ratio:.3f}")
            print(f"  Original snippets: {len(response.retrieval_result.original_snippets)}")
            print(f"  Processing time: {response.processing_time:.3f}s")
            print(f"  Fallback used: {response.fallback_used}")
            
            if response.retrieval_result.original_snippets:
                print(f"\nORIGINAL CONTEXT SNIPPETS:")
                for j, snippet in enumerate(response.retrieval_result.original_snippets[:2], 1):
                    print(f"  {j}. {snippet[:100]}...")
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            print(f"ERROR: {str(e)}")
    
    # Demonstrate cache effectiveness
    print(f"\n{'='*60}")
    print("CACHE STATISTICS")
    print(f"{'='*60}")
    cache_stats = cross_encoder.get_cache_stats()
    print(f"Cache entries: {cache_stats['cache_size']}")
    print(f"Max cache size: {cache_stats['max_cache_size']}")
    
    # Test the same query again to show caching
    print(f"\nTesting cache with repeated query...")
    start_time = asyncio.get_event_loop().time()
    response = await enhanced_rag.query(test_queries[0])
    end_time = asyncio.get_event_loop().time()
    print(f"Cached query processing time: {end_time - start_time:.3f}s")
    
    logger.info("Enhanced RAG demonstration completed!")


async def demonstrate_enhanced_extraction():
    """Demonstrate enhanced entity extraction capabilities"""
    print(f"\n{'='*60}")
    print("ENHANCED ENTITY EXTRACTION DEMO")
    print(f"{'='*60}")
    
    # Initialize components
    graphiti = Graphiti()
    extractor = EnhancedEntityExtractor(graphiti.llm_client)
    
    # Sample text with rich context
    sample_text = """
    Dr. Emily Chen, a renowned AI researcher at Stanford University, announced 
    breakthrough results in neural language models during yesterday's conference 
    in San Francisco. The 45-year-old computer scientist, who has been working 
    on this project since 2022, expressed excitement about the potential 
    applications in healthcare and education. Her research team, which includes 
    researchers from Google and Microsoft, will present their findings at the 
    upcoming NeurIPS conference in December.
    """
    
    try:
        # Extract with enhanced context preservation
        result = await extractor.extract_with_context_preservation(sample_text)
        
        print(f"EXTRACTION RESULTS:")
        print(f"Entities found: {len(result.get('entities', []))}")
        print(f"Relationships found: {len(result.get('relationships', []))}")
        
        print(f"\nENTITIES WITH CONTEXT:")
        for entity in result.get('entities', []):
            print(f"  • {entity.get('name')} ({entity.get('type')})")
            print(f"    Summary: {entity.get('summary', 'N/A')}")
            print(f"    Temporal: {entity.get('temporal_context', 'N/A')}")
            print(f"    Original: {entity.get('original_context', 'N/A')[:100]}...")
            print(f"    Confidence: {entity.get('confidence', 'N/A')}")
            print()
        
        # Create context-preserving nodes
        nodes = extractor.create_context_preserving_nodes(result)
        print(f"CREATED {len(nodes)} CONTEXT-PRESERVING NODES")
        
        for node in nodes:
            print(f"  • {node.name} - Properties: {len(node.properties or {})}")
            
    except Exception as e:
        print(f"Extraction demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(demonstrate_enhanced_extraction())
