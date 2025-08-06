"""
Cache service for cluster metadata validation.

This module provides a centralized caching solution with expiration mechanisms
for organization and macro validation results.
"""

import logging
from typing import Set, Optional, Union, Any, Dict, List
from diskcache import Cache
import os

logger = logging.getLogger(__name__)

"""
Cache service for cluster metadata validation.

This module provides a centralized caching solution with expiration mechanisms
for organization and macro validation results.
"""

import logging
from typing import Set, Optional, Union, Any, Dict, List
from diskcache import Cache
import os
import json

logger = logging.getLogger(__name__)

class ValidationCacheService:
    """
    Centralized cache service for validation results with expiration support.
    
    Uses diskcache for persistent, thread-safe caching with automatic expiration.
    """
    
    _instance: Optional['ValidationCacheService'] = None
    _cache_dir = './validation_cache'
    
    def __new__(cls) -> 'ValidationCacheService':
        """Singleton pattern to ensure single cache instance across the application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache service if not already initialized."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self._cache_dir, exist_ok=True)
            
            # Initialize cache with settings
            self.cache: Cache = Cache(
                directory=self._cache_dir,
                size_limit=200 * 1024 * 1024,  # 200MB size limit
                timeout=1,  # 1 second timeout for cache operations
            )
            
            # Cache expiration times (in seconds)
            self.ORGANIZATION_CACHE_TTL = 3600  # 1 hour
            self.MACRO_CACHE_TTL = 1800  # 30 minutes
            self.INTEGRATION_CACHE_TTL = 900  # 15 minutes
            
            self._initialized = True
            self._use_fallback = False
            logger.info(f"Validation cache service initialized at {self._cache_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            # Fallback to in-memory cache if disk cache fails
            self.cache = {}  # type: ignore
            self._use_fallback = True
            self._initialized = True
    
    def _make_org_key(self, organization: str) -> str:
        """Generate cache key for organization validation."""
        return f"org_valid:{organization.lower()}"
    
    def _make_macro_key(self, macro_name: str, organization: str) -> str:
        """Generate cache key for macro validation."""
        return f"macro_valid:{organization.lower()}:{macro_name.lower()}"
    
    def _make_integration_key(self, organization: str) -> str:
        """Generate cache key for integration validation."""
        return f"integration_valid:{organization.lower()}"
    
    def _safe_cache_get(self, key: str) -> Any:
        """Safely get value from cache, handling both diskcache and fallback."""
        try:
            if self._use_fallback:
                return self.cache.get(key)  # type: ignore
            else:
                return self.cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    def _safe_cache_set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Safely set value in cache, handling both diskcache and fallback."""
        try:
            if self._use_fallback:
                self.cache[key] = value  # type: ignore
            else:
                if expire:
                    self.cache.set(key, value, expire=expire)
                else:
                    self.cache.set(key, value)
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
    
    def _safe_cache_delete(self, key: str) -> None:
        """Safely delete value from cache."""
        try:
            if self._use_fallback:
                self.cache.pop(key, None)  # type: ignore
            else:
                if key in self.cache:
                    del self.cache[key]
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
    
    def _safe_cache_contains(self, key: str) -> bool:
        """Safely check if key exists in cache."""
        try:
            if self._use_fallback:
                return key in self.cache  # type: ignore
            else:
                return key in self.cache
        except Exception as e:
            logger.warning(f"Cache contains check failed for key {key}: {e}")
            return False
    
    def is_organization_cached(self, organization: str) -> bool:
        """Check if organization validation result is cached and valid."""
        key = self._make_org_key(organization)
        return self._safe_cache_contains(key)
    
    def cache_organization(self, organization: str, is_valid: bool) -> None:
        """Cache organization validation result with TTL."""
        key = self._make_org_key(organization)
        self._safe_cache_set(key, is_valid, self.ORGANIZATION_CACHE_TTL)
        logger.debug(f"Cached organization validation: {organization} = {is_valid}")
    
    def get_organization_validity(self, organization: str) -> Optional[bool]:
        """Get cached organization validation result."""
        key = self._make_org_key(organization)
        result = self._safe_cache_get(key)
        if result is not None:
            logger.debug(f"Cache hit for organization: {organization}")
            return bool(result) if result is not None else None
        return None
    
    def is_macro_cached(self, macro_name: str, organization: str) -> bool:
        """Check if macro validation result is cached and valid."""
        key = self._make_macro_key(macro_name, organization)
        return self._safe_cache_contains(key)
    
    def cache_macro(self, macro_name: str, organization: str, is_valid: bool) -> None:
        """Cache macro validation result with TTL."""
        key = self._make_macro_key(macro_name, organization)
        self._safe_cache_set(key, is_valid, self.MACRO_CACHE_TTL)
        logger.debug(f"Cached macro validation: {macro_name}@{organization} = {is_valid}")
    
    def get_macro_validity(self, macro_name: str, organization: str) -> Optional[bool]:
        """Get cached macro validation result."""
        key = self._make_macro_key(macro_name, organization)
        result = self._safe_cache_get(key)
        if result is not None:
            logger.debug(f"Cache hit for macro: {macro_name}@{organization}")
            return bool(result) if result is not None else None
        return None
    
    def cache_integration_organizations(self, organizations: Set[str]) -> None:
        """Cache the set of valid organizations from integrations."""
        key = "integration_orgs"
        org_list = list(organizations)
        self._safe_cache_set(key, json.dumps(org_list), self.INTEGRATION_CACHE_TTL)
        logger.debug(f"Cached {len(organizations)} integration organizations")
    
    def get_integration_organizations(self) -> Optional[Set[str]]:
        """Get cached set of valid organizations from integrations."""
        key = "integration_orgs"
        result = self._safe_cache_get(key)
        if result is not None:
            try:
                org_list = json.loads(str(result))
                if isinstance(org_list, list):
                    logger.debug(f"Cache hit for integration organizations: {len(org_list)} orgs")
                    return set(org_list)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to decode cached integration organizations: {e}")
        return None
    
    def invalidate_organization(self, organization: str) -> None:
        """Invalidate cached data for a specific organization."""
        org_key = self._make_org_key(organization)
        self._safe_cache_delete(org_key)
        logger.debug(f"Invalidated organization cache: {organization}")
        
        # Also invalidate integration cache since it might affect organization validity
        self.invalidate_integration_cache()
    
    def invalidate_integration_cache(self) -> None:
        """Invalidate the integration organizations cache."""
        key = "integration_orgs"
        self._safe_cache_delete(key)
        logger.debug("Invalidated integration organizations cache")
    
    def clear_all(self) -> None:
        """Clear all cached data."""
        try:
            if self._use_fallback:
                self.cache.clear()  # type: ignore
            else:
                self.cache.clear()
            logger.info("Cleared all validation cache data")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring and debugging."""
        try:
            if self._use_fallback:
                return {
                    'cache_type': 'fallback_dict',
                    'size': len(self.cache),  # type: ignore
                }
            else:
                # Get basic stats from diskcache
                cache_size = 0
                try:
                    # Use volume() method for diskcache size if available
                    if hasattr(self.cache, 'volume'):
                        cache_size = self.cache.volume()  # type: ignore
                    else:
                        cache_size = 0
                except Exception:
                    cache_size = 0
                
                return {
                    'cache_type': 'diskcache',
                    'directory': self._cache_dir,
                    'volume': cache_size,
                }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {'error': str(e)}


# Global cache service instance
_cache_service: Optional[ValidationCacheService] = None

def get_cache_service() -> ValidationCacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = ValidationCacheService()
    return _cache_service
