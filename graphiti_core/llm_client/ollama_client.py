"""
Ollama Client for Graphiti
Provides local LLM support using Ollama instead of OpenAI
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
import httpx
from graphiti_core.llm_client.client import LLMClient


class OllamaClient(LLMClient):
    """
    Ollama client implementation for local LLM processing.
    Tested with qwen2.5:7b model in production environment.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",  # Not needed for Ollama but kept for interface compatibility
        timeout: int = 30
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Ollama model name (default: qwen2.5:7b)
            base_url: Ollama API URL (default: http://localhost:11434)
            api_key: Not used for Ollama, kept for compatibility
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using Ollama.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # Convert messages to Ollama format
        prompt = self._format_messages(messages)
        
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            request_body["options"]["num_predict"] = max_tokens
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=request_body
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except httpx.HTTPError as e:
            raise Exception(f"Ollama API error: {e}")
    
    async def extract_entities(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using Ollama.
        
        Args:
            text: Text to extract entities from
            entity_types: List of entity types to extract
            
        Returns:
            List of extracted entities
        """
        prompt = f"""Extract the following types of entities from the text: {', '.join(entity_types)}

Text: {text}

Return the entities as a JSON array with the format:
[{{"name": "entity_name", "type": "entity_type", "context": "relevant context"}}]

Only return the JSON array, no other text."""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.generate_response(messages, temperature=0.1)
            
            # Parse JSON response
            # Handle cases where model adds extra text
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            entities = json.loads(response)
            
            # Ensure it's a list
            if not isinstance(entities, list):
                entities = [entities]
            
            # Validate entity format
            validated_entities = []
            for entity in entities:
                if isinstance(entity, dict) and "name" in entity and "type" in entity:
                    # Ensure type is in our requested types
                    if entity["type"] in entity_types:
                        validated_entities.append(entity)
            
            return validated_entities
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try basic extraction
            return self._fallback_entity_extraction(text, entity_types)
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embeddings using Ollama.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding", [])
            
        except httpx.HTTPError as e:
            # If embeddings not supported, return empty
            print(f"Embedding generation not supported: {e}")
            return []
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for Ollama prompt.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            else:
                prompt += f"User: {content}\n\n"
        
        # Add final Assistant prompt
        if messages and messages[-1].get("role") != "assistant":
            prompt += "Assistant: "
        
        return prompt
    
    def _fallback_entity_extraction(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fallback entity extraction using simple pattern matching.
        
        Args:
            text: Text to extract from
            entity_types: Entity types to look for
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Simple heuristics for common entity types
        if "Person" in entity_types:
            # Look for capitalized words that might be names
            import re
            potential_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
            for name in potential_names[:3]:  # Limit to 3
                entities.append({
                    "name": name,
                    "type": "Person",
                    "context": text[:50]
                })
        
        if "Organization" in entity_types:
            # Look for company indicators
            org_patterns = [
                r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|LLC|Ltd|Company)\b',
                r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+ (?:Inc|Corp|LLC|Ltd)\b'
            ]
            for pattern in org_patterns:
                orgs = re.findall(pattern, text)
                for org in orgs[:2]:
                    entities.append({
                        "name": org,
                        "type": "Organization",
                        "context": text[:50]
                    })
        
        return entities
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()