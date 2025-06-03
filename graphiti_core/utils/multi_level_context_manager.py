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

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

from ..nodes import EntityNode
from ..edges import EntityEdge


class MultiLevelContextManager:
    """
    Manages context at multiple levels to preserve information during graph-to-text conversion
    """
    
    def __init__(self):
        self.context_levels = {
            'entity_level': {},      # Individual entity context
            'relationship_level': {},  # Pairwise relationships
            'subgraph_level': {},    # Local graph patterns
            'global_level': {}       # Overall graph context
        }
        self.logger = logging.getLogger(__name__)
    
    def build_comprehensive_context(
        self, 
        query: str, 
        relevant_entities: List[EntityNode],
        relevant_edges: Optional[List[EntityEdge]] = None
    ) -> str:
        """
        Build context that preserves information at multiple levels
        """
        context_parts = []
        relevant_edges = relevant_edges or []
        
        # Add query context
        context_parts.append(f"Query: {query}\n")
        
        # 1. Entity-level context
        entity_contexts = []
        for entity in relevant_entities:
            entity_context = self._build_entity_context(entity)
            entity_contexts.append(entity_context)
        
        if entity_contexts:
            context_parts.append("=== ENTITIES ===")
            context_parts.extend(entity_contexts)
        
        # 2. Relationship-level context
        if relevant_edges:
            relationship_contexts = []
            for edge in relevant_edges:
                rel_context = self._build_relationship_context(edge)
                relationship_contexts.append(rel_context)
            
            if relationship_contexts:
                context_parts.append("\n=== RELATIONSHIPS ===")
                context_parts.extend(relationship_contexts)
        
        # 3. Subgraph-level patterns
        patterns = self._identify_graph_patterns(relevant_entities, relevant_edges)
        if patterns:
            context_parts.append("\n=== GRAPH PATTERNS ===")
            context_parts.extend(patterns)
        
        # 4. Temporal sequencing
        temporal_sequence = self._build_temporal_sequence(relevant_entities, relevant_edges)
        if temporal_sequence:
            context_parts.append(f"\n=== TIMELINE ===\n{temporal_sequence}")
        
        # 5. Global context summary
        global_summary = self._build_global_context_summary(relevant_entities, relevant_edges)
        if global_summary:
            context_parts.append(f"\n=== CONTEXT SUMMARY ===\n{global_summary}")
        
        return "\n".join(context_parts)
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Build rich context for individual entity
        """
        context_lines = []
          # Basic entity information
        context_lines.append(f"• {entity.name} ({entity.labels[0] if entity.labels else 'UNKNOWN'})")
        
        # Summary
        if entity.summary:
            context_lines.append(f"  Summary: {entity.summary}")
        
        # Original context if preserved
        if entity.attributes and entity.attributes.get('original_context'):
            original_context = entity.attributes['original_context']
            if original_context and len(original_context.strip()) > 0:
                context_lines.append(f"  Original Context: {original_context}")
        
        # Temporal context
        if entity.attributes and entity.attributes.get('temporal_context'):
            temporal_context = entity.attributes['temporal_context']
            if temporal_context and len(temporal_context.strip()) > 0:
                context_lines.append(f"  When: {temporal_context}")
          # Spatial context
        if entity.attributes and entity.attributes.get('spatial_context'):
            spatial_context = entity.attributes['spatial_context']
            if spatial_context and len(spatial_context.strip()) > 0:
                context_lines.append(f"  Where: {spatial_context}")
        
        # Other properties
        if entity.attributes:
            filtered_props = {
                k: v for k, v in entity.attributes.items() 
                if k not in ['original_context', 'temporal_context', 'spatial_context', 'context_window', 'text_position']
                and v is not None and str(v).strip()
            }
            if filtered_props:
                props_str = ", ".join([f"{k}: {v}" for k, v in filtered_props.items()])
                context_lines.append(f"  Properties: {props_str}")
        
        return "\n".join(context_lines)
    
    def _build_relationship_context(self, edge: EntityEdge) -> str:
        """
        Build context for relationships
        """
        context_lines = []
          # Basic relationship
        rel_type = edge.name.replace('_', ' ').lower()
        context_lines.append(f"• Relationship: {rel_type}")
        
        # Fact details
        if hasattr(edge, 'fact') and edge.fact:
            context_lines.append(f"  Fact: {edge.fact}")
        
        # Original context
        if edge.attributes and edge.attributes.get('original_context'):
            original_context = edge.attributes['original_context']
            if original_context and len(original_context.strip()) > 0:
                context_lines.append(f"  Original Context: {original_context}")
          # Temporal context
        if edge.attributes and edge.attributes.get('temporal_context'):
            temporal_context = edge.attributes['temporal_context']
            if temporal_context and len(temporal_context.strip()) > 0:
                context_lines.append(f"  When: {temporal_context}")
        
        # Additional details
        if edge.attributes and edge.attributes.get('contextual_details'):
            details = edge.attributes['contextual_details']
            if details and len(details.strip()) > 0:
                context_lines.append(f"  Context: {details}")
        
        return "\n".join(context_lines)
    
    def _identify_graph_patterns(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge]
    ) -> List[str]:
        """
        Identify common graph patterns and structures
        """
        patterns = []
        
        if not entities or not edges:
            return patterns
          # Build adjacency for pattern detection
        # First create a UUID to name mapping for entities
        entity_uuid_to_name = {entity.uuid: entity.name for entity in entities}
        
        entity_connections = defaultdict(list)
        for edge in edges:
            source_name = entity_uuid_to_name.get(edge.source_node_uuid)
            target_name = entity_uuid_to_name.get(edge.target_node_uuid)
            
            if source_name and target_name:
                entity_connections[source_name].append((target_name, edge.name))
                entity_connections[target_name].append((source_name, edge.name))
        
        # Pattern 1: Highly connected entities (hubs)
        hub_entities = [
            name for name, connections in entity_connections.items()
            if len(connections) >= 3
        ]
        if hub_entities:
            patterns.append(f"Central entities: {', '.join(hub_entities)}")
          # Pattern 2: Entity clusters by type
        entity_types = defaultdict(list)
        for entity in entities:
            # Use the first label if multiple labels exist
            primary_label = entity.labels[0] if entity.labels else "UNKNOWN"
            entity_types[primary_label].append(entity.name)
        
        for entity_type, names in entity_types.items():
            if len(names) > 1:
                patterns.append(f"{entity_type} cluster: {', '.join(names)}")
          # Pattern 3: Relationship types
        relationship_types = defaultdict(int)
        for edge in edges:
            relationship_types[edge.name] += 1
        
        common_relationships = [
            f"{rel_type.replace('_', ' ').lower()} ({count} instances)"
            for rel_type, count in relationship_types.items()
            if count > 1
        ]
        if common_relationships:
            patterns.append(f"Common relationships: {', '.join(common_relationships)}")
        
        return patterns
    
    def _build_temporal_sequence(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge]
    ) -> Optional[str]:
        """
        Build temporal sequence from entities and relationships
        """
        temporal_events = []
          # Extract temporal information from entities
        for entity in entities:
            if entity.attributes and entity.attributes.get('temporal_context'):
                temporal_context = entity.attributes['temporal_context']
                if temporal_context and len(temporal_context.strip()) > 0:
                    primary_label = entity.labels[0] if entity.labels else "UNKNOWN"
                    temporal_events.append(f"{temporal_context}: {entity.name} ({primary_label})")
          # Extract temporal information from relationships
        # First create a UUID to name mapping for entities
        entity_uuid_to_name = {entity.uuid: entity.name for entity in entities}
        
        for edge in edges:
            if edge.attributes and edge.attributes.get('temporal_context'):
                temporal_context = edge.attributes['temporal_context']
                if temporal_context and len(temporal_context.strip()) > 0:
                    rel_type = edge.name.replace('_', ' ').lower() if edge.name else "related to"
                    source_name = entity_uuid_to_name.get(edge.source_node_uuid, "Unknown")
                    target_name = entity_uuid_to_name.get(edge.target_node_uuid, "Unknown")
                    temporal_events.append(f"{temporal_context}: {source_name} {rel_type} {target_name}")
        
        if temporal_events:
            # Simple chronological sorting (could be enhanced with actual date parsing)
            temporal_events.sort()
            return "\n".join([f"• {event}" for event in temporal_events])
        
        return None
    
    def _build_global_context_summary(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge]
    ) -> str:
        """
        Build a global summary of the context
        """
        summary_parts = []
          # Entity summary
        entity_count = len(entities)
        entity_types = set(entity.labels[0] if entity.labels else "UNKNOWN" for entity in entities)
        summary_parts.append(f"Total entities: {entity_count} ({', '.join(entity_types)})")
        
        # Relationship summary
        edge_count = len(edges)
        edge_types = set(edge.name.replace('_', ' ').lower() if edge.name else "unknown" for edge in edges)
        if edges:
            summary_parts.append(f"Total relationships: {edge_count} ({', '.join(edge_types)})")
        
        # Complexity indicators
        if entity_count > 10:
            summary_parts.append("Note: Large knowledge subgraph - information may be complex")
        
        return "\n".join(summary_parts)
    
    def preserve_graph_structure_context(
        self, 
        entities: List[EntityNode], 
        edges: List[EntityEdge]
    ) -> str:
        """
        Alternative approach that preserves more structural information in a structured format
        """
        context = {
            "entities": [
                {
                    "name": entity.name,
                    "type": entity.label,
                    "summary": entity.summary,
                    "properties": entity.properties
                }
                for entity in entities
            ],
            "relationships": [
                {
                    "source": edge.source_name,
                    "target": edge.target_name,
                    "type": edge.label,
                    "summary": edge.summary,
                    "properties": edge.properties
                }
                for edge in edges
            ]
        }
        
        return f"STRUCTURED_CONTEXT: {json.dumps(context, indent=2)}"
