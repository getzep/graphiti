"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..search.search_config import SearchConfig
from ..search import search_config_recipes
from ..cross_encoder.enhanced_client import EnhancedCrossEncoderClient
from .compression_aware_retrieval import CompressionAwareRetrieval, RetrievalResult
from .multi_level_context_manager import MultiLevelContextManager


@dataclass
class RAGResponse:
    """Response from the enhanced RAG pipeline"""
    answer: str
    retrieval_result: RetrievalResult
    quality_metrics: Dict[str, Any]
    processing_time: float
    fallback_used: bool


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with compression awareness, fallback mechanisms, and quality monitoring
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        search_function,  # The actual search function from graphiti
        cross_encoder_client: Optional[EnhancedCrossEncoderClient] = None,
        compression_ratio_threshold: float = 0.7,
        max_context_length: int = 8000,
        enable_fallbacks: bool = True
    ):
        self.llm_client = llm_client
        self.search_function = search_function
        self.cross_encoder_client = cross_encoder_client
        self.compression_aware_retrieval = CompressionAwareRetrieval(
            compression_ratio_threshold=compression_ratio_threshold
        )
        self.context_manager = MultiLevelContextManager()
        self.max_context_length = max_context_length
        self.enable_fallbacks = enable_fallbacks
        self.logger = logging.getLogger(__name__)
    
    async def query(
        self, 
        question: str,
        search_config: Optional[SearchConfig] = None,
        max_entities: int = 20,
        rerank_passages: bool = True
    ) -> RAGResponse:
        """
        Execute enhanced RAG query with fallback mechanisms
        """
        start_time = asyncio.get_event_loop().time()
        fallback_used = False
        
        # Use default search config if not provided
        if search_config is None:
            search_config = search_config_recipes.NODE_HYBRID_SEARCH_RRF
        
        try:
            # Primary retrieval path
            retrieval_result = await self._retrieve_with_enhancement(
                question, search_config, max_entities, rerank_passages
            )
            
            # Generate response
            answer = await self._generate_response(question, retrieval_result.context)
            
        except Exception as e:
            self.logger.warning(f"Primary RAG pipeline failed: {str(e)}")
            
            if self.enable_fallbacks:
                # Fallback to simpler retrieval
                retrieval_result, answer, fallback_used = await self._fallback_query(
                    question, search_config, max_entities
                )
            else:
                raise
        
        # Calculate processing time and quality metrics
        processing_time = asyncio.get_event_loop().time() - start_time
        quality_metrics = self._calculate_quality_metrics(retrieval_result, answer, processing_time)
        
        return RAGResponse(
            answer=answer,
            retrieval_result=retrieval_result,
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            fallback_used=fallback_used
        )
    
    async def _retrieve_with_enhancement(
        self, 
        query: str, 
        search_config: SearchConfig,
        max_entities: int,
        rerank_passages: bool
    ) -> RetrievalResult:
        """
        Perform enhanced retrieval with compression awareness and reranking
        """
        # Wrapper function for the search to match expected interface
        async def search_wrapper(q: str, config: SearchConfig):
            # Call the actual search function and get results
            # This should return entities and edges
            results = await self.search_function(q, config, limit=max_entities)
            
            # Extract entities and edges from results
            # The exact format depends on the search function implementation
            if isinstance(results, dict):
                entities = results.get('entities', [])
                edges = results.get('edges', [])
            elif isinstance(results, (list, tuple)) and len(results) == 2:
                entities, edges = results
            else:
                # Assume it's a list of entities
                entities = results if isinstance(results, list) else [results]
                edges = []
            
            return entities, edges
        
        # Perform compression-aware retrieval
        retrieval_result = await self.compression_aware_retrieval.retrieve_with_compression_awareness(
            query, search_wrapper, search_config
        )
        
        # Optional: Rerank passages using cross-encoder
        if rerank_passages and self.cross_encoder_client and retrieval_result.entities:
            retrieval_result = await self._rerank_and_refine_context(query, retrieval_result)
        
        # Ensure context length is within limits
        if len(retrieval_result.context) > self.max_context_length:
            retrieval_result = self._truncate_context(retrieval_result)
        
        return retrieval_result
    
    async def _rerank_and_refine_context(
        self, 
        query: str, 
        retrieval_result: RetrievalResult
    ) -> RetrievalResult:
        """
        Use cross-encoder to rerank and refine the context
        """
        try:
            # Prepare passages for reranking
            passages = []
            
            # Entity summaries as passages
            for entity in retrieval_result.entities:
                if entity.summary:
                    passages.append(f"{entity.name}: {entity.summary}")
            
            # Edge summaries as passages
            for edge in retrieval_result.edges:
                if edge.summary:
                    passages.append(f"{edge.source_name} -> {edge.target_name}: {edge.summary}")
            
            # Original snippets as passages
            passages.extend(retrieval_result.original_snippets)
            
            if not passages:
                return retrieval_result
            
            # Rerank passages
            ranked_passages = await self.cross_encoder_client.rank(query, passages)
            
            # Take top-ranked passages and rebuild context
            top_passages = ranked_passages[:min(10, len(ranked_passages))]
            
            # Rebuild context with reranked information
            refined_context = self._build_reranked_context(
                query, 
                retrieval_result, 
                top_passages
            )
            
            # Update retrieval result
            retrieval_result.context = refined_context
            retrieval_result.metadata['reranked'] = True
            retrieval_result.metadata['rerank_score'] = top_passages[0][1] if top_passages else 0.0
            
        except Exception as e:
            self.logger.warning(f"Reranking failed: {str(e)}, using original context")
        
        return retrieval_result
    
    def _build_reranked_context(
        self, 
        query: str, 
        retrieval_result: RetrievalResult,
        top_passages: List[Tuple[str, float]]
    ) -> str:
        """
        Build context using reranked passages
        """
        context_parts = [f"Query: {query}\n"]
        
        context_parts.append("=== MOST RELEVANT INFORMATION ===")
        for i, (passage, score) in enumerate(top_passages[:5], 1):
            context_parts.append(f"{i}. {passage} (relevance: {score:.3f})")
        
        # Add some structured information
        if retrieval_result.entities:
            context_parts.append(f"\n=== ENTITIES ({len(retrieval_result.entities)}) ===")
            for entity in retrieval_result.entities[:5]:  # Top 5 entities
                context_parts.append(f"• {entity.name} ({entity.label})")
        
        if retrieval_result.edges:
            context_parts.append(f"\n=== RELATIONSHIPS ({len(retrieval_result.edges)}) ===")
            for edge in retrieval_result.edges[:5]:  # Top 5 relationships
                rel_type = edge.label.replace('_', ' ').lower()
                context_parts.append(f"• {edge.source_name} {rel_type} {edge.target_name}")
        
        return "\n".join(context_parts)
    
    def _truncate_context(self, retrieval_result: RetrievalResult) -> RetrievalResult:
        """
        Truncate context to fit within length limits while preserving important information
        """
        current_context = retrieval_result.context
        
        if len(current_context) <= self.max_context_length:
            return retrieval_result
        
        # Priority-based truncation
        lines = current_context.split('\n')
        important_lines = []
        total_length = 0
        
        # Always keep query line
        for line in lines:
            if line.startswith('Query:'):
                important_lines.append(line)
                total_length += len(line) + 1
                break
        
        # Prioritize sections
        section_priority = [
            '=== MOST RELEVANT INFORMATION ===',
            '=== ENTITIES ===',
            '=== RELATIONSHIPS ===',
            '=== GRAPH PATTERNS ===',
            '=== ORIGINAL CONTEXT SNIPPETS ==='
        ]
        
        for priority_section in section_priority:
            if total_length >= self.max_context_length * 0.8:  # Reserve 20% for other content
                break
                
            section_lines = []
            in_section = False
            
            for line in lines:
                if line.strip() == priority_section:
                    in_section = True
                    section_lines.append(line)
                elif line.startswith('===') and in_section:
                    break
                elif in_section:
                    if total_length + len(line) + 1 < self.max_context_length:
                        section_lines.append(line)
                        total_length += len(line) + 1
                    else:
                        section_lines.append("... (truncated)")
                        break
            
            important_lines.extend(section_lines)
        
        truncated_context = '\n'.join(important_lines)
        retrieval_result.context = truncated_context
        retrieval_result.metadata['context_truncated'] = True
        retrieval_result.metadata['original_context_length'] = len(current_context)
        
        return retrieval_result
    
    async def _generate_response(self, question: str, context: str) -> str:
        """
        Generate response using LLM with enhanced prompt
        """
        enhanced_prompt = f"""
You are a knowledgeable assistant answering questions based on a knowledge graph and contextual information.

CONTEXT INFORMATION:
{context}

QUESTION: {question}

Please provide a comprehensive and accurate answer based on the context information provided. 
- Use information from both the structured knowledge graph and any original context snippets
- If the information is insufficient, clearly state what is uncertain or missing
- Reference specific entities and relationships when relevant
- Maintain accuracy and avoid hallucination

ANSWER:"""

        try:
            messages = [{"role": "user", "content": enhanced_prompt}]
            response = await self.llm_client.generate_response(messages)
            return response
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    async def _fallback_query(
        self, 
        question: str, 
        search_config: SearchConfig,
        max_entities: int
    ) -> Tuple[RetrievalResult, str, bool]:
        """
        Fallback query method with simplified processing
        """
        self.logger.info("Using fallback query method")
        
        try:
            # Simple search without enhancements
            entities, edges = await self.search_function(question, search_config, limit=max_entities)
            
            # Build simple context
            simple_context = self.context_manager.build_comprehensive_context(
                question, entities, edges or []
            )
            
            # Create basic retrieval result
            retrieval_result = RetrievalResult(
                entities=entities,
                edges=edges or [],
                context=simple_context,
                compression_ratio=1.0,  # Assume perfect compression for fallback
                original_snippets=[],
                metadata={'fallback_used': True}
            )
            
            # Generate simple response
            answer = await self._generate_response(question, simple_context)
            
            return retrieval_result, answer, True
            
        except Exception as e:
            self.logger.error(f"Fallback query also failed: {str(e)}")
            
            # Ultimate fallback
            empty_result = RetrievalResult(
                entities=[], edges=[], context="", compression_ratio=0.0,
                original_snippets=[], metadata={'ultimate_fallback': True}
            )
            
            answer = f"I apologize, but I'm unable to retrieve relevant information to answer your question: {question}"
            
            return empty_result, answer, True
    
    def _calculate_quality_metrics(
        self, 
        retrieval_result: RetrievalResult, 
        answer: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics
        """
        base_metrics = self.compression_aware_retrieval.get_quality_metrics(retrieval_result)
        
        additional_metrics = {
            "processing_time": processing_time,
            "answer_length": len(answer),
            "answer_has_uncertainty": any(phrase in answer.lower() for phrase in [
                "i don't know", "uncertain", "unclear", "insufficient information"
            ]),
            "context_utilization": min(1.0, len(answer) / max(1, len(retrieval_result.context))),
            "retrieval_success": len(retrieval_result.entities) > 0 or len(retrieval_result.edges) > 0
        }
        
        return {**base_metrics, **additional_metrics}
