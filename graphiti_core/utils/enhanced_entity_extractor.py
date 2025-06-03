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
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..nodes import EntityNode, EpisodicNode
from ..llm_client import LLMClient


class EnhancedEntityExtractor:
    """
    Enhanced entity extractor that preserves more contextual information
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    async def extract_with_context_preservation(
        self, 
        text: str, 
        episode_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities while preserving more contextual information
        """
        extraction_prompt = self._build_enhanced_extraction_prompt(text)
        
        try:
            result = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": extraction_prompt}]
            )
            
            parsed_result = self._parse_extraction_result(result, text)
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {str(e)}")
            # Fallback to basic extraction
            return await self._fallback_extraction(text)
    
    def _build_enhanced_extraction_prompt(self, text: str) -> str:
        """
        Build comprehensive extraction prompt
        """
        return f"""
        Extract entities and relationships from the following text, preserving as much contextual information as possible.
        
        Text: {text}
        
        For each entity, provide:
        1. Basic information (name, type/label)
        2. Contextual properties (any attributes mentioned)
        3. Temporal context (when something happened)
        4. Spatial context (where something happened)
        5. Emotional/situational context (mood, circumstances)
        6. Original text snippet where the entity appears
        7. Confidence level (0.0-1.0)
        
        For each relationship, provide:
        1. Source and target entities
        2. Relationship type
        3. Temporal context (when the relationship occurred)
        4. Spatial context (where the relationship occurred)
        5. Contextual details (how, why, circumstances)
        6. Original text snippet describing the relationship
        7. Confidence level (0.0-1.0)
        
        Return the result as a JSON object with the following structure:
        {{
            "entities": [
                {{
                    "name": "entity name",
                    "type": "PERSON|ORGANIZATION|LOCATION|EVENT|CONCEPT|OBJECT",
                    "properties": {{"key": "value"}},
                    "temporal_context": "when this entity was mentioned or relevant",
                    "spatial_context": "where this entity was mentioned or relevant",
                    "emotional_context": "emotional or situational context",
                    "original_context": "exact text snippet where entity appears",
                    "confidence": 0.95,
                    "summary": "brief description of the entity"
                }}
            ],
            "relationships": [
                {{
                    "source": "source entity name",
                    "target": "target entity name",
                    "type": "relationship type",
                    "properties": {{"key": "value"}},
                    "temporal_context": "when this relationship occurred",
                    "spatial_context": "where this relationship occurred",
                    "contextual_details": "additional context about the relationship",
                    "original_context": "exact text snippet describing the relationship",
                    "confidence": 0.90,
                    "summary": "brief description of the relationship"
                }}
            ],
            "global_context": {{
                "overall_theme": "main theme or topic of the text",
                "temporal_setting": "overall time period or setting",
                "spatial_setting": "overall location or setting",
                "mood_tone": "overall mood or tone of the text"
            }}
        }}
        
        Focus on accuracy and preserving nuanced information that might be lost in traditional extraction.
        """
    
    def _parse_extraction_result(self, result: str, original_text: str) -> Dict[str, Any]:
        """
        Parse and validate the extraction result
        """
        try:
            parsed = json.loads(result)
            
            # Validate and enhance the result
            enhanced_result = {
                "entities": self._enhance_entities(parsed.get("entities", []), original_text),
                "relationships": self._enhance_relationships(parsed.get("relationships", []), original_text),
                "global_context": parsed.get("global_context", {}),
                "extraction_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "original_text_length": len(original_text),
                    "entities_count": len(parsed.get("entities", [])),
                    "relationships_count": len(parsed.get("relationships", []))
                }
            }
            
            return enhanced_result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse extraction result: {str(e)}")
            return self._fallback_extraction_sync(original_text)
    
    def _enhance_entities(self, entities: List[Dict], original_text: str) -> List[Dict]:
        """
        Enhance entity information with additional context
        """
        enhanced_entities = []
        
        for entity in entities:
            enhanced_entity = {
                **entity,
                "extraction_confidence": entity.get("confidence", 0.5),
                "text_position": self._find_text_position(entity.get("name", ""), original_text),
                "context_window": self._extract_context_window(
                    entity.get("name", ""), 
                    original_text,
                    window_size=100
                )
            }
            
            # Ensure required fields
            if "properties" not in enhanced_entity:
                enhanced_entity["properties"] = {}
            
            # Add enhanced properties
            enhanced_entity["properties"].update({
                "original_context": enhanced_entity.get("original_context", ""),
                "temporal_context": enhanced_entity.get("temporal_context", ""),
                "spatial_context": enhanced_entity.get("spatial_context", ""),
                "emotional_context": enhanced_entity.get("emotional_context", "")
            })
            
            enhanced_entities.append(enhanced_entity)
        
        return enhanced_entities
    
    def _enhance_relationships(self, relationships: List[Dict], original_text: str) -> List[Dict]:
        """
        Enhance relationship information with additional context
        """
        enhanced_relationships = []
        
        for rel in relationships:
            enhanced_rel = {
                **rel,
                "extraction_confidence": rel.get("confidence", 0.5),
                "context_window": self._extract_context_window(
                    f"{rel.get('source', '')} {rel.get('target', '')}", 
                    original_text,
                    window_size=150
                )
            }
            
            # Ensure required fields
            if "properties" not in enhanced_rel:
                enhanced_rel["properties"] = {}
            
            # Add enhanced properties
            enhanced_rel["properties"].update({
                "original_context": enhanced_rel.get("original_context", ""),
                "temporal_context": enhanced_rel.get("temporal_context", ""),
                "spatial_context": enhanced_rel.get("spatial_context", ""),
                "contextual_details": enhanced_rel.get("contextual_details", "")
            })
            
            enhanced_relationships.append(enhanced_rel)
        
        return enhanced_relationships
    
    def _find_text_position(self, entity_name: str, text: str) -> Dict[str, int]:
        """
        Find the position of entity mention in text
        """
        try:
            start_pos = text.lower().find(entity_name.lower())
            if start_pos != -1:
                return {
                    "start": start_pos,
                    "end": start_pos + len(entity_name)
                }
        except Exception:
            pass
        
        return {"start": -1, "end": -1}
    
    def _extract_context_window(self, entity_text: str, full_text: str, window_size: int = 100) -> str:
        """
        Extract context window around entity mention
        """
        try:
            pos = full_text.lower().find(entity_text.lower())
            if pos == -1:
                return ""
            
            start = max(0, pos - window_size)
            end = min(len(full_text), pos + len(entity_text) + window_size)
            
            return full_text[start:end].strip()
        except Exception:
            return ""
    
    async def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """
        Fallback extraction method using simpler approach
        """
        # This would use the existing extraction logic as fallback
        return {
            "entities": [],
            "relationships": [],
            "global_context": {},
            "extraction_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "fallback_used": True,
                "original_text_length": len(text)
            }
        }
    
    def _fallback_extraction_sync(self, text: str) -> Dict[str, Any]:
        """
        Synchronous fallback extraction method using simpler approach
        """
        # This would use the existing extraction logic as fallback
        return {
            "entities": [],
            "relationships": [],  
            "global_context": {},
            "extraction_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "fallback_used": True,
                "original_text_length": len(text)
            }
        }
    
    def create_context_preserving_nodes(self, extraction_result: Dict[str, Any]) -> List[EntityNode]:
        """
        Create EntityNode objects that preserve extracted context
        """
        entities = []
        
        for entity_data in extraction_result.get("entities", []):
            # Create enhanced properties
            properties = entity_data.get("properties", {})
            properties.update({
                "extraction_confidence": entity_data.get("extraction_confidence", 0.5),
                "context_window": entity_data.get("context_window", ""),
                "text_position": json.dumps(entity_data.get("text_position", {})),
                "global_context": json.dumps(extraction_result.get("global_context", {}))
            })
              # Create EntityNode
            entity = EntityNode(
                name=entity_data.get("name", ""),
                group_id="default",  # Default group for extracted entities
                labels=[entity_data.get("type", "UNKNOWN")],
                summary=entity_data.get("summary", ""),
                attributes=properties
            )
            
            entities.append(entity)
        
        return entities
