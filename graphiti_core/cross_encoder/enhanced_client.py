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
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

from .client import CrossEncoderClient


class EnhancedCrossEncoderClient(CrossEncoderClient):
    """
    Enhanced CrossEncoder with caching, batch processing, and error handling
    """
    
    def __init__(
        self, 
        cache_size: int = 10000,
        batch_size: int = 32,
        timeout: float = 10.0,
        fallback_enabled: bool = True
    ):
        self.cache: Dict[str, List[Tuple[str, float]]] = {}
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.timeout = timeout
        self.fallback_enabled = fallback_enabled
        self.logger = logging.getLogger(__name__)
    
    def _create_cache_key(self, query: str, passages: List[str]) -> str:
        """Create a hash key for caching query-passages combinations"""
        content = f"{query}|{json.dumps(sorted(passages))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Rank passages with caching, batching, and error handling
        """
        if not passages:
            return []
            
        # Check cache first
        cache_key = self._create_cache_key(query, passages)
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[cache_key]
        
        try:
            # Apply timeout to prevent hanging
            result = await asyncio.wait_for(
                self._rank_with_batching(query, passages),
                timeout=self.timeout
            )
            
            # Cache the result
            self._update_cache(cache_key, result)
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"CrossEncoder timeout for query: {query[:50]}...")
            if self.fallback_enabled:
                return self._fallback_ranking(query, passages)
            raise
        except Exception as e:
            self.logger.error(f"CrossEncoder error: {str(e)}")
            if self.fallback_enabled:
                return self._fallback_ranking(query, passages)
            raise
    
    async def _rank_with_batching(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Process passages in batches for better efficiency
        """
        if len(passages) <= self.batch_size:
            return await self._rank_batch(query, passages)
        
        # Process in batches
        all_results = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i:i + self.batch_size]
            batch_results = await self._rank_batch(query, batch)
            all_results.extend(batch_results)
        
        # Sort all results by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results
    
    @abstractmethod
    async def _rank_batch(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Rank a single batch of passages - to be implemented by concrete classes
        """
        pass
    
    def _fallback_ranking(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Simple fallback ranking using basic text similarity
        """
        self.logger.info("Using fallback ranking method")
        
        results = []
        for passage in passages:
            # Simple similarity using SequenceMatcher
            similarity = SequenceMatcher(None, query.lower(), passage.lower()).ratio()
            results.append((passage, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _update_cache(self, cache_key: str, result: List[Tuple[str, float]]):
        """
        Update cache with size limit
        """
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size
        }


class CompositeCrossEncoderClient(EnhancedCrossEncoderClient):
    """
    A composite cross-encoder that can use multiple models and combine their results
    """
    
    def __init__(
        self,
        primary_client: CrossEncoderClient,
        fallback_clients: Optional[List[CrossEncoderClient]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.primary_client = primary_client
        self.fallback_clients = fallback_clients or []
    
    async def _rank_batch(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Use primary client, with fallback to other clients if needed
        """
        try:
            # Try primary client first
            return await self.primary_client.rank(query, passages)
        except Exception as e:
            self.logger.warning(f"Primary client failed: {str(e)}")
            
            # Try fallback clients
            for i, fallback_client in enumerate(self.fallback_clients):
                try:
                    self.logger.info(f"Trying fallback client {i}")
                    return await fallback_client.rank(query, passages)
                except Exception as fallback_e:
                    self.logger.warning(f"Fallback client {i} failed: {str(fallback_e)}")
                    continue
            
            # If all clients fail, use simple fallback
            return self._fallback_ranking(query, passages)
