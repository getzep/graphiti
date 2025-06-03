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

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..nodes import EntityNode, EpisodicNode
from ..edges import EntityEdge
from ..search.search_config import SearchConfig
from .multi_level_context_manager import MultiLevelContextManager


@dataclass
class RetrievalResult:
    """Result of compression-aware retrieval"""
    entities: List[EntityNode]
    edges: List[EntityEdge]
    context: str
    compression_ratio: float
    original_snippets: List[str]
    metadata: Dict[str, Any]


class CompressionAwareRetrieval:
    """
    Retrieval system that is aware of information compression and loss
    """
    
    def __init__(
        self, 
        compression_ratio_threshold: float = 0.7,
        max_original_snippets: int = 5,
        snippet_max_length: int = 500
    ):
        self.compression_ratio_threshold = compression_ratio_threshold
        self.max_original_snippets = max_original_snippets
        self.snippet_max_length = snippet_max_length
        self.context_manager = MultiLevelContextManager()
        self.logger = logging.getLogger(__name__)
    
    async def retrieve_with_compression_awareness(
        self, 
        query: str,
        search_function,  # Function that performs the actual search
        search_config: SearchConfig
    ) -> RetrievalResult:
        """
        Retrieve information while being aware of compression losses
        """
        # 1. Get relevant subgraph using the provided search function
        entities, edges = await search_function(query, search_config)
        
        # 2. Estimate information loss
        compression_ratio = self._estimate_compression_ratio(entities, edges)
        
        # 3. Get original text snippets if compression is too high
        original_snippets = []
        if compression_ratio < self.compression_ratio_threshold:
            original_snippets = await self._get_original_text_snippets(entities, edges)
        
        # 4. Build context based on compression ratio
        if compression_ratio < self.compression_ratio_threshold and original_snippets:
            context = self._merge_graph_and_text_context(entities, edges, original_snippets, query)
            self.logger.info(f"Using hybrid context (compression ratio: {compression_ratio:.2f})")
        else:
            context = self.context_manager.build_comprehensive_context(query, entities, edges)
            self.logger.info(f"Using standard graph context (compression ratio: {compression_ratio:.2f})")
        
        # 5. Build metadata
        metadata = {
            "compression_ratio": compression_ratio,
            "hybrid_context_used": compression_ratio < self.compression_ratio_threshold,
            "original_snippets_count": len(original_snippets),
            "entities_count": len(entities),
            "edges_count": len(edges),
            "query": query
        }
        
        return RetrievalResult(
            entities=entities,
            edges=edges,
            context=context,
            compression_ratio=compression_ratio,
            original_snippets=original_snippets,
            metadata=metadata
        )
    
    def _estimate_compression_ratio(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge]
    ) -> float:
        """
        Estimate how much information is preserved in graph form
        """
        # Calculate original text length from preserved context
        original_text_length = 0
          # From entity original contexts
        for entity in entities:
            if entity.attributes and entity.attributes.get('original_context'):
                original_context = entity.attributes['original_context']
                if isinstance(original_context, str):
                    original_text_length += len(original_context)
            
            # From context windows
            if entity.attributes and entity.attributes.get('context_window'):
                context_window = entity.attributes['context_window']
                if isinstance(context_window, str):
                    original_text_length += len(context_window)
          # From edge original contexts
        for edge in edges:
            if edge.attributes and edge.attributes.get('original_context'):
                original_context = edge.attributes['original_context']
                if isinstance(original_context, str):
                    original_text_length += len(original_context)
        
        # Calculate graph representation length
        graph_representation_length = 0
        
        # Entity representation
        for entity in entities:
            graph_representation_length += len(entity.name or '')
            graph_representation_length += len(entity.summary or '')
            if entity.labels:
                graph_representation_length += len(entity.labels[0] or '')
        
        # Edge representation
        for edge in edges:
            graph_representation_length += len(edge.name or '')
            graph_representation_length += len(edge.fact or '')
        
        # Calculate compression ratio
        if original_text_length == 0:
            return 1.0  # Perfect preservation if no original context
        
        ratio = graph_representation_length / original_text_length
        return min(ratio, 1.0)  # Cap at 1.0
    
    async def _get_original_text_snippets(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge]
    ) -> List[str]:
        """
        Extract original text snippets that provide additional context
        """
        snippets = []
        seen_snippets = set()  # Avoid duplicates
        
        # Collect snippets from entities
        for entity in entities:
            if len(snippets) >= self.max_original_snippets:
                break
                
            snippet = self._extract_entity_snippet(entity)
            if snippet and snippet not in seen_snippets and len(snippet) <= self.snippet_max_length:
                snippets.append(snippet)
                seen_snippets.add(snippet)
          # Collect snippets from edges
        for edge in edges:
            if len(snippets) >= self.max_original_snippets:
                break
                
            snippet = self._extract_edge_snippet(edge)
            if snippet and snippet not in seen_snippets and len(snippet) <= self.snippet_max_length:
                snippets.append(snippet)
                seen_snippets.add(snippet)
        
        return snippets[:self.max_original_snippets]
    
    def _extract_entity_snippet(self, entity: EntityNode) -> Optional[str]:
        """Extract the most informative snippet for an entity"""
        if not entity.attributes:
            return None
          # Priority order for snippet sources
        snippet_sources = [
            'context_window',
            'original_context',
            'temporal_context',
            'spatial_context'
        ]
        
        for source in snippet_sources:
            snippet = entity.attributes.get(source)
            if snippet and isinstance(snippet, str) and len(snippet.strip()) > 10:
                return snippet.strip()
        
        return None
    
    def _extract_edge_snippet(self, edge: EntityEdge) -> Optional[str]:
        """Extract the most informative snippet for an edge"""
        if not edge.attributes:
            return None
        
        # Priority order for snippet sources
        snippet_sources = [
            'original_context',
            'contextual_details',
            'temporal_context'
        ]
        
        for source in snippet_sources:
            snippet = edge.attributes.get(source)
            if snippet and isinstance(snippet, str) and len(snippet.strip()) > 10:
                return snippet.strip()
        
        return None
    
    def _merge_graph_and_text_context(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge],
        original_snippets: List[str],
        query: str
    ) -> str:
        """
        Combine structured graph info with original text snippets
        """
        context_parts = []
        
        # Start with query
        context_parts.append(f"Query: {query}\n")
        
        # Add structured graph information
        graph_context = self.context_manager.build_comprehensive_context(query, entities, edges)
        context_parts.append("=== STRUCTURED KNOWLEDGE ===")
        context_parts.append(graph_context)
        
        # Add original context snippets for nuanced information
        if original_snippets:
            context_parts.append("\n=== ORIGINAL CONTEXT SNIPPETS ===")
            context_parts.append("The following snippets provide additional nuanced information:")
            
            for i, snippet in enumerate(original_snippets, 1):
                # Truncate if too long
                if len(snippet) > self.snippet_max_length:
                    snippet = snippet[:self.snippet_max_length] + "..."
                context_parts.append(f"{i}. {snippet}")
        
        # Add usage instructions
        context_parts.append("\n=== INSTRUCTIONS ===")
        context_parts.append(
            "Use the structured knowledge for factual information and relationships. "
            "Refer to the original context snippets for nuanced details, emotional context, "
            "and specific circumstances that may not be fully captured in the structured format."
        )
        
        return "\n".join(context_parts)
    
    def get_quality_metrics(self, result: RetrievalResult) -> Dict[str, Any]:
        """
        Calculate quality metrics for the retrieval result
        """
        return {
            "compression_ratio": result.compression_ratio,
            "information_density": len(result.entities) + len(result.edges),
            "context_length": len(result.context),
            "original_snippets_coverage": len(result.original_snippets) / max(1, self.max_original_snippets),
            "hybrid_context_used": result.compression_ratio < self.compression_ratio_threshold,
            "estimated_information_preservation": self._estimate_information_preservation(result)
        }
    
    def _estimate_information_preservation(self, result: RetrievalResult) -> float:
        """
        Estimate how well information is preserved overall
        """
        base_score = result.compression_ratio
        
        # Bonus for having original snippets when compression is low
        if result.compression_ratio < self.compression_ratio_threshold and result.original_snippets:
            snippet_bonus = min(0.3, len(result.original_snippets) * 0.1)
            base_score += snippet_bonus
        
        # Bonus for having rich structured information
        if result.entities and result.edges:
            structure_bonus = min(0.2, (len(result.entities) + len(result.edges)) * 0.01)
            base_score += structure_bonus
        
        return min(base_score, 1.0)
